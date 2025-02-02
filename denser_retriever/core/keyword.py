from abc import ABC, abstractmethod
import logging
from typing import Any, Dict, List, Optional, Tuple, Callable
import uuid
import time
from dataclasses import dataclass

from elasticsearch import Elasticsearch

from langchain_core.documents import Document
from denser_retriever.core.filter import FieldMapper
from denser_retriever.core.utils import sigmoid

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

    @abstractmethod
    def create_index(self, index_name: str, search_fields: List[str], **args: Any):
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
        k: int = 100,
        filter: Dict[str, Any] = {},
        **kwargs: Any,
    ) -> Tuple[List[Tuple[Document, float]], Optional[Dict]]:
        raise NotImplementedError

    @abstractmethod
    def get_index_mappings(self) -> Dict[Any, Any]:
        raise NotImplementedError

    @abstractmethod
    def delete(
        self,
        ids: Optional[List[str]] = None,
        source_id: Optional[str] = None,
        source_url: Optional[str] = None,
        **kwargs: str,
    ):
        raise NotImplementedError

    @abstractmethod
    def delete_all(self, delete_index: bool = True):
        raise NotImplementedError

@dataclass
class ESIndexData:
    """Data class containing Elasticsearch index configuration."""
    index_name: str
    search_fields: FieldMapper
    date_fields: List[str]
    analysis: Optional[str]
    drop_old: bool

class ElasticKeywordSearch(DenserKeywordSearch):
    """
    Elasticsearch keyword search class.
    """
    def __init__(
        self,
        es_connection: Elasticsearch,
    ):
        self.client = es_connection

    def create_index(
        self,
        index_data: ESIndexData,
        **args: Any,
    ):
        logger.info("ES analysis %s", index_data.analysis)
        if index_data.analysis == "default":
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
                        "similarity": "custom_bm25",
                    },
                    "title": {
                        "type": "text",
                    },
                    "source": {
                        "type": "keyword",
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
                        "similarity": "custom_bm25",
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

        for key in index_data.search_fields.get_keys():
            mappings["properties"][key] = {
                "type": index_data.search_fields.get_field_type(key) or "text"
            }

        if self.client.indices.exists(index=index_data.index_name):
            if index_data.drop_old:
                self.client.indices.delete(index=index_data.index_name)

        if not self.client.indices.exists(index=index_data.index_name):
            self.client.indices.create(
                index=index_data.index_name,
                mappings=mappings,
                settings=settings
            )

    def add_documents(
            self,
            index_data: ESIndexData,
            documents: List[Document],
            refresh_indices: bool = True,
            progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None
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
        total_docs = len(texts)
        docs_processed = 0

        for i, text in enumerate(texts):
            metadata = metadatas[i] if metadatas else {}
            request = {
                "_op_type": "index",
                "_index": index_data.index_name,
                "content": text,
                "title": metadata.get("title", ""),
                "_id": ids[i],
                "source": metadata.get("source"),
                "pid": metadata.get("pid"),
            }
            for filter_key in metadata.keys():
                value = metadata.get(filter_key, "")
                if isinstance(value, list):
                    value = [str(v).strip() for v in value if v is not None]
                else:
                    if value is not None:
                        value = str(value).strip()
                if value:
                    request[filter_key] = value
            requests.append(request)

            docs_processed += 1
            if progress_callback and i % 1000 == 0:  # Update progress every 100 documents
                progress_callback({
                    "total_docs": total_docs,
                    "es_progress": (docs_processed / total_docs) * 100,
                    "vector_progress": 0,
                    "es_docs_processed": docs_processed,
                    "vector_docs_processed": 0,
                    "status": "elasticsearch_ingesting"
                })

        if len(requests) > 0:
            try:
                success, failed = bulk(
                    self.client,
                    requests,
                    stats_only=True,
                    refresh=refresh_indices,
                )
                logger.info(f"Added {success} and failed to add {failed} texts to index")

                if progress_callback:
                    progress_callback({
                        "total_docs": total_docs,
                        "es_progress": 100,
                        "vector_progress": 0,
                        "es_docs_processed": success,
                        "vector_docs_processed": 0,
                        "status": "elasticsearch_complete"
                    })

                return ids
            except BulkIndexError as e:
                logger.error(f"Error adding texts: {e}")
                firstError = e.errors[0].get("index", {}).get("error", {})
                logger.error(f"First error reason: {firstError.get('reason')}")
                raise e
        else:
            logger.info("No documents to add to index")
            return []

    def retrieve(
        self,
        index_data: ESIndexData,
        query: str,
        k: int = 100,
        filter: Dict[str, Any] = {},
        aggregation: bool = False,
        apply_sigmoid: bool = False,
    ) -> Tuple[List[Tuple[Document, float]], Dict]:
        assert self.client.indices.exists(index=index_data.index_name)
        start_time = time.time()

        query_dict = {
            "query": {
                "bool": {
                    "must": [
                        {
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
                                        }
                                    },
                                ],
                                "minimum_should_match": 1,
                            }
                        }
                    ]
                }
            },
            "_source": True,
            "aggs": {},
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
                                    else category_or_date[0],
                                }
                            }
                        }
                    )
                else:
                    query_dict["query"]["bool"]["must"].append(
                        {"term": {field: category_or_date}}
                    )

        if aggregation:
            for field in index_data.search_fields.get_keys():
                query_dict["aggs"][f"{field}_aggregation"] = {
                    "terms": {
                        "field": f"{field}",
                        "size": 50,
                    }
                }

        res = self.client.search(
            index=index_data.index_name,
            body=query_dict,
            size=k,
        )

        top_k_used = min(len(res["hits"]["hits"]), k)
        docs = []
        for id in range(top_k_used):
            _source = res["hits"]["hits"][id]["_source"]
            doc = Document(
                page_content=_source.pop("content"),
                metadata=_source,
            )
            score = res["hits"]["hits"][id]["_score"]
            docs.append((doc, sigmoid(score) if apply_sigmoid else score))

        aggregations = {}
        for field in index_data.search_fields.get_keys():
            field_agg = (
                res.get("aggregations", {})
                .get(f"{field}_aggregation", {})
                .get("buckets", [])
            )
            cat_keys = [cat["key"] for cat in field_agg]
            cat_counts = [cat["doc_count"] for cat in field_agg]
            if len(cat_keys) > 0:
                if field in index_data.date_fields:
                    sorted_data = sorted(
                        zip(cat_keys, cat_counts), key=lambda x: x[0], reverse=True
                    )
                    sorted_keys, sorted_counts = zip(*sorted_data)
                    cat_keys = list(sorted_keys)
                    cat_counts = list(sorted_counts)
                aggregations[field] = (cat_keys, cat_counts)

        retrieve_time_sec = time.time() - start_time
        logger.info(f"Keyword retrieve time: {retrieve_time_sec:.3f} sec.")
        logger.info(f"Retrieved {len(docs)} documents.")

        return docs, aggregations

    def get_index_mappings(self, index_data: ESIndexData):
        mapping = self.client.indices.get_mapping(index=index_data.index_name)
        properties = mapping[index_data.index_name]["mappings"]["properties"]

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
                    )
            return fields

        all_fields = extract_fields(properties)
        return all_fields

    def delete(
        self,
        index_data: ESIndexData,
        ids: Optional[List[str]] = None,
        source_id: Optional[str] = None,
        source_url: Optional[str] = None,
        **kwargs: str,
    ):
        if ids:
            query = {"query": {"terms": {"pid": ids}}}
        elif source_id:
            query = {"query": {"match": {"source": source_id}}}
        elif source_url:
            query = {"query": {"wildcard": {"source": f"*{source_url}*"}}}
        else:
            raise ValueError(
                "Please provide either ids, source_id, or source_url to delete."
            )
        result = self.client.delete_by_query(index=index_data.index_name, body=query)
        deleted_count = result.get("deleted", 0)

        self.client.indices.refresh(index=index_data.index_name)

        logger.info(
            f"Deleted {deleted_count} documents with {'ids' if ids else 'source_id' if source_id else 'source_url'}: {ids or source_id or source_url}"
        )

    def delete_all(self, index_data: ESIndexData, delete_index: bool = True):
        if delete_index:
            self.client.indices.delete(index=index_data.index_name)
        else:
            self.client.delete_by_query(
                index=index_data.index_name,
                query={"match_all": {}}
            )
            self.client.indices.refresh(index=index_data.index_name)

    def check_index(self, index_name: str):
        return bool(self.client.indices.exists(index=index_name))

    def get_count(self, index_name: str):
        result = self.client.count(index=index_name)
        return result['count']

    def list_pids(self, index_name: str) -> List[str]:
        query = {
            "query": {"match_all": {}},
            "_source": ["pid"],  # Only retrieve pid field
            "size": 10000  # Adjust based on your needs
        }
        try:
            response = self.client.search(
                index=index_name,
                body=query
            )

            pids = [hit["_source"]["pid"] for hit in response["hits"]["hits"]]
            logger.info(f"Retrieved {len(pids)} PIDs from index {index_name}")
            return pids

        except Exception as e:
            logger.error(f"Error retrieving PIDs from index {index_name}: {e}")
            raise