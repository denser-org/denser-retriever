from abc import ABC, abstractmethod
from typing import List
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from langchain_core.documents import Document
from denser_retriever.utils import sigmoid


class KeywordSearch(ABC):
    @abstractmethod
    def has_index(self, index_name: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def create_index(self, index_name: str, analysis: str):
        raise NotImplementedError

    @abstractmethod
    def drop_index(self, index_name: str):
        raise NotImplementedError

    @abstractmethod
    def indexing(
        self, index_name: str, pks: List[str], docs: List[Document]
    ) -> List[str]:
        raise NotImplementedError

    @abstractmethod
    def search(
        self,
        index_name: str,
        query: str,
        limit: int,
        apply_sigmoid: bool = False,
    ) -> List[tuple[Document, float]]:
        raise NotImplementedError

    @abstractmethod
    def delete(self, index_name: str, doc_ids: List[str]):
        raise NotImplementedError


class ElasticSearch(KeywordSearch):
    _client = None

    def __init__(
        self,
        hosts="localhost",
        port=9200,
        scheme="http",
        username=None,
        password=None,
    ):
        self._client = Elasticsearch(
            hosts=[{"host": hosts, "port": port, "scheme": scheme}],
            http_auth=(username, password) if username and password else None,
        )

    def has_index(self, index_name: str) -> bool:
        return self._client.indices.exists(index=index_name)

    def create_index(self, index_name: str, analysis: str):
        settings = {}
        mappings = {}

        if analysis == "default":
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
                    "metadata": {"type": "object", "dynamic": True, "properties": {}},
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
                    "metadata": {"type": "object", "dynamic": True, "properties": {}},
                }
            }

        self._client.indices.create(
            index=index_name, mappings=mappings, settings=settings
        )

    def drop_index(self, index_name: str):
        self._client.indices.delete(index=index_name)

    def indexing(self, index_name: str, pks: List[str], docs: List[Document]) -> list:
        if not docs:
            return []

        if not self.has_index(index_name):
            self.create_index(index_name, analysis="default")

        actions = []
        ret = []

        for id, doc in zip(pks, docs):
            doc_dict = {
                "page_content": doc.page_content,
                "metadata": doc.metadata or {},
            }
            action = {
                "_index": index_name,
                "_id": id,
                "_source": {
                    "content": doc.page_content,
                    "metadata": {**doc_dict, "id": id},
                },
            }
            actions.append(action)
            ret.append(id)

        bulk(self._client, actions, refresh=True)

        return ret

    def search(
        self,
        index_name: str,
        query: str,
        limit: int,
        apply_sigmoid: bool = False,
    ) -> List[tuple[Document, float]]:
        if not self.has_index(index_name):
            return []

        query_dict = {
            "query": {
                "bool": {
                    "must": [
                        {
                            "bool": {
                                "should": [
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

        result = self._client.search(
            index=index_name,
            body=query_dict,
            size=limit,
        )

        top_k_used = min(len(result["hits"]["hits"]), limit)

        ret = []
        for i in range(top_k_used):
            hit = result["hits"]["hits"][i]

            doc_dict = hit["_source"]["metadata"]
            doc = Document(**doc_dict)
            score = hit["_score"]

            ret.append((doc, sigmoid(score) if apply_sigmoid else score))

        return ret

    def delete(self, index_name: str, doc_ids: List[str]):
        if not self.has_index(index_name):
            return
        body = {"query": {"terms": {"_id": doc_ids}}}
        self._client.delete_by_query(index=index_name, body=body, refresh=True)

    def __del__(self):
        if self._client:
            self._client.close()
