from abc import ABC, abstractmethod
import logging
from typing import Any, Dict, List, Optional, Tuple
import uuid

from elasticsearch import Elasticsearch

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def create_elasticsearch_client(
    url: Optional[str] = None,
    cloud_id: Optional[str] = None,
    api_key: Optional[str] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
    params: Optional[Dict[str, Any]] = None,
) -> Elasticsearch:
    if url and cloud_id:
        raise ValueError(
            "Both es_url and cloud_id are defined. Please provide only one."
        )

    connection_params: Dict[str, Any] = {}

    if url:
        connection_params["hosts"] = [url]
    elif cloud_id:
        connection_params["cloud_id"] = cloud_id
    else:
        raise ValueError("Please provide either elasticsearch_url or cloud_id.")

    if api_key:
        connection_params["api_key"] = api_key
    elif username and password:
        connection_params["basic_auth"] = (username, password)

    if params is not None:
        connection_params.update(params)

    es_client = Elasticsearch(**connection_params)

    es_client.info()  # test connection

    return es_client


class DenserKeywordSearch(ABC):
    """
    Denser keyword search interface.
    """

    def __init__(self, top_k: int = 50, weight: float = 0.5):
        self.top_k = top_k
        self.weight = weight

    @abstractmethod
    def create_index(
        self, index_name: str, search_fields: Dict[str, Any] = {}, **args: Any
    ):
        raise NotImplementedError

    @abstractmethod
    def add_documents(
        self,
        documents: List[Document],
        **kwargs: Any,
    ) -> List[str]:
        raise NotImplementedError

    @abstractmethod
    def retrieve(
        self,
        query: str,
        k: int = 4,
        filter: Dict[str, Any] = {},
    ) -> List[Tuple[Document, float]]:
        raise NotImplementedError

    @abstractmethod
    def get_index_mappings(self) -> Dict[Any, Any]:
        raise NotImplementedError

    @abstractmethod
    def get_categories(self, field: str, k: int = 10) -> List[Any]:
        raise NotImplementedError

    @abstractmethod
    def delete(
        self,
        ids: Optional[List[str]] = None,
        source_id: Optional[str] = None,
        **kwargs: str,
    ):
        raise NotImplementedError

    @abstractmethod
    def delete_all(self):
        raise NotImplementedError


class ElasticKeywordSearch(DenserKeywordSearch):
    """
    Elasticsearch keyword search class.
    """

    index_name: str
    """Index name for retrieval"""
    client: Elasticsearch
    """Elasticsearch client"""
    search_fields: Dict[str, Any]
    """Fields to be indexed"""
    analysis: Optional[str]
    """Analysis type"""

    def __init__(
        self,
        es_connection: Elasticsearch,
        analysis: Optional[str] = "default",
        **kwargs,
    ):
        super().__init__()
        self.analysis = analysis
        self.client = es_connection

    def create_index(self, index_name: str, search_fields: Dict[str, Any], **args: Any):
        # Define the index settings and mappings
        self.index_name = index_name
        self.search_fields = search_fields

        logger.info("ES analysis %s", self.analysis)
        if self.analysis == "default":
            settings = {
                "analysis": {"analyzer": {"default": {"type": "standard"}}},
                "similarity": {
                    "custom_bm25": {
                        "type": "BM25",
                        "k1": 1.2,
                        "b": 0.75,
                    }
                },
            }
            mappings = {
                "properties": {
                    "content": {
                        "type": "text",
                        "similarity": "custom_bm25",  # Use the custom BM25 similarity
                    },
                    "title": {
                        "type": "text",
                    },
                    "source": {
                        "type": "text",
                    },
                    "pid": {
                        "type": "text",
                    },
                }
            }
        else:  # ik
            settings = {
                "analysis": {
                    "analyzer": {
                        "ik_max_word": {"type": "custom", "tokenizer": "ik_max_word"},
                        "ik_smart": {"type": "custom", "tokenizer": "ik_smart"},
                    }
                },
                "similarity": {
                    "custom_bm25": {
                        "type": "BM25",
                        "k1": 1.2,
                        "b": 0.75,
                    }
                },
            }
            mappings = {
                "properties": {
                    "content": {
                        "type": "text",
                        "analyzer": "ik_max_word",
                        "similarity": "custom_bm25",  # Use the custom BM25 similarity
                    },
                    "title": {
                        "type": "text",
                        "analyzer": "ik_smart",
                    },
                    "source": {
                        "type": "text",
                    },
                    "pid": {
                        "type": "text",
                    },
                }
            }

        for key in self.search_fields:
            mappings["properties"][key] = self.search_fields[key]

        # Create the index with the specified settings and mappings
        if self.client.indices.exists(index=self.index_name):
            self.client.indices.delete(index=self.index_name)
        self.client.indices.create(
            index=self.index_name, mappings=mappings, settings=settings
        )

    def add_documents(
        self,
        documents: List[Document],
        refresh_indices: bool = True,
    ) -> List[str]:
        try:
            from elasticsearch.helpers import BulkIndexError, bulk
        except ImportError:
            raise ImportError(
                "Could not import elasticsearch python package. "
                "Please install it with `pip install elasticsearch`."
            )

        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        ids = [str(uuid.uuid4()) for _ in texts]
        requests = []

        for i, text in enumerate(texts):
            metadata = metadatas[i] if metadatas else {}

            request = {
                "_op_type": "index",
                "_index": self.index_name,
                "content": text,
                "title": metadata.get("title", ""),  # Index the title
                "_id": ids[i],
                "source": metadata.get("source"),
                "pid": metadata.get("pid"),
            }
            for filter in self.search_fields.keys():
                v = metadata.get(filter, "").strip()
                if v:
                    request[filter] = v
            requests.append(request)

        if len(requests) > 0:
            try:
                success, failed = bulk(
                    self.client,
                    requests,
                    stats_only=True,
                    refresh=refresh_indices,
                )
                logger.debug(
                    f"Added {success} and failed to add {failed} texts to index"
                )

                logger.debug(f"added texts {ids} to index")
                return ids
            except BulkIndexError as e:
                logger.error(f"Error adding texts: {e}")
                firstError = e.errors[0].get("index", {}).get("error", {})
                logger.error(f"First error reason: {firstError.get('reason')}")
                raise e
        else:
            logger.debug("No documents to add to index")
            return []

    def retrieve(
        self,
        query: str,
        k: int = 4,
        filter: Dict[str, Any] = {},
    ) -> List[Tuple[Document, float]]:
        assert self.client.indices.exists(index=self.index_name)

        query_dict = {
            "query": {
                "bool": {
                    "should": [
                        {
                            "match": {
                                "title": {
                                    "query": query,
                                    "boost": 2.0,
                                }
                            }
                        },
                        {
                            "match": {
                                "content": query,
                            },
                        },
                    ],
                    "must": [],
                }
            },
            "_source": True,
        }

        for field in filter:
            category_or_date = filter.get(field)
            if category_or_date:
                if isinstance(category_or_date, tuple):
                    query_dict["query"]["bool"]["must"].append(
                        {
                            "range": {
                                field: {
                                    "gte": category_or_date[0],
                                    "lte": category_or_date[1]
                                    if len(category_or_date) > 1
                                    else category_or_date[0],  # type: ignore
                                }
                            }
                        }
                    )
                else:
                    query_dict["query"]["bool"]["must"].append(
                        {"term": {field: category_or_date}}
                    )

        res = self.client.search(
            index=self.index_name,
            body=query_dict,
            size=k,
        )
        top_k_used = min(len(res["hits"]["hits"]), k)
        docs = []
        for id in range(top_k_used):
            _source = res["hits"]["hits"][id]["_source"]
            doc = Document(
                page_content=_source["content"],
                metadata={
                    "source": _source["source"],
                    "title": _source["title"],
                    "pid": _source["pid"],
                },
            )
            score = res["hits"]["hits"][id]["_score"]
            for field in filter:
                if _source.get(field):
                    doc.metadata[field] = _source.get(field)
            docs.append((doc, score))

        return docs

    def get_index_mappings(self):
        mapping = self.client.indices.get_mapping(index=self.index_name)

        # The mapping response structure can be quite nested, focusing on the 'properties' section
        properties = mapping[self.index_name]["mappings"]["properties"]

        # Function to recursively extract fields and types
        def extract_fields(fields_dict, parent_name=""):
            fields = {}
            for field_name, details in fields_dict.items():
                full_field_name = (
                    f"{parent_name}.{field_name}" if parent_name else field_name
                )
                if "properties" in details:
                    fields.update(
                        extract_fields(details["properties"], full_field_name)
                    )
                else:
                    fields[full_field_name] = details.get(
                        "type", "notype"
                    )  # Default 'notype' if no type is found
            return fields

        # Extract fields and types
        all_fields = extract_fields(properties)
        return all_fields

    def get_categories(self, field: str, k: int = 10):
        query = {
            "size": 0,  # No actual documents are needed, just the aggregation results
            "aggs": {
                "all_categories": {
                    "terms": {
                        "field": field,
                        "size": 1000,  # Adjust this value based on the expected number of unique categories
                    }
                }
            },
        }
        response = self.client.search(index=self.index_name, body=query)
        # Extract the aggregation results
        categories = response["aggregations"]["all_categories"]["buckets"]
        if k > 0:
            categories = categories[:k]
        res = [category["key"] for category in categories]
        return res

    def delete(
        self,
        ids: Optional[List[str]] = None,
        source_id: Optional[str] = None,
        **kwargs: str,
    ):
        if ids:
            for id in ids:
                self.client.delete(index=self.index_name, id=id)
        elif source_id:
            query = {"query": {"match": {"source_id": source_id}}}
            self.client.delete_by_query(index=self.index_name, body=query)
        else:
            raise ValueError("Please provide either ids or source to delete.")

    def delete_all(self):
        self.client.indices.delete(index=self.index_name)
