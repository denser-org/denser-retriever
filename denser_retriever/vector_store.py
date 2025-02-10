from abc import ABC, abstractmethod
import json
from typing import Dict, List, Optional
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    Hit,
    connections,
    utility,
)
from langchain_core.documents import Document
from denser_retriever.utils import sigmoid


class VectorStore(ABC):
    @abstractmethod
    def has_collection(self, collection_name: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def create_collection(self, collection_name: str):
        raise NotImplementedError

    @abstractmethod
    def drop_collection(self, collection_name: str):
        raise NotImplementedError

    @abstractmethod
    def insert(
        self,
        collection_name: str,
        pks: List[str],
        docs: List[Document],
        embeddings: list,
    ) -> List[str]:
        raise NotImplementedError

    @abstractmethod
    def search(
        self,
        collection_name: str,
        embeddings: list,
        limit: int,
        search_params: Optional[Dict] = None,
        apply_sigmoid: bool = False,
    ) -> List[tuple[Document, float]]:
        raise NotImplementedError

    @abstractmethod
    def delete(self, collection_name: str, doc_ids: List[str]):
        raise NotImplementedError


class MilvusVectorStore(VectorStore):
    def __init__(
        self,
        alias="default",
        host="localhost",
        port="19530",
        user=None,
        password=None,
        dim=768,
    ):
        self._alias = alias
        self._dim = dim
        self._loaded_collections: Dict[str, Collection] = {}

        connections.connect(
            alias=alias, host=host, port=port, user=user, password=password
        )

    def has_collection(self, collection_name: str) -> bool:
        return utility.has_collection(collection_name)

    def create_collection(self, collection_name: str):
        if self.has_collection(collection_name):
            raise ValueError(f"Collection {collection_name} already exists")

        schema = CollectionSchema(
            [
                FieldSchema(
                    name="id",
                    dtype=DataType.VARCHAR,
                    auto_id=False,
                    is_primary=True,
                ),
                FieldSchema(
                    name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=self._dim
                ),
                FieldSchema(name="metadata", dtype=DataType.JSON),
            ]
        )

        collection = Collection(
            name=collection_name,
            schema=schema,
            consistency_level="Strong",
            using=self._alias,
        )
        collection.create_index(
            "embeddings",
            {
                "index_type": "FLAT",
                "metric_type": "L2",
            },
        )

    def drop_collection(self, collection_name: str):
        if self.has_collection(collection_name):
            if collection_name in self._loaded_collections:
                self._release_collection(collection_name)
            utility.drop_collection(collection_name)

    def insert(
        self,
        collection_name: str,
        pks: List[str],
        docs: List[Document],
        embeddings: list,
    ) -> List[str]:
        if not docs:
            return []

        if not self.has_collection(collection_name):
            self.create_collection(collection_name)

        collection = self._load_collection(collection_name)

        batch_data = []
        for id, doc, emb in zip(pks, docs, embeddings):
            doc_dict = {
                "page_content": doc.page_content,
                "metadata": {**doc.metadata, "id": id},
            }
            batch_data.append(
                {"id": id, "embeddings": emb, "metadata": json.dumps(doc_dict)}
            )

        ret = collection.insert(batch_data)
        collection.flush()

        return ret.primary_keys

    def search(
        self,
        collection_name: str,
        embeddings: list,
        limit: int,
        search_params: Optional[Dict] = None,
        apply_sigmoid: bool = False,
    ) -> List[tuple[Document, float]]:
        if not self.has_collection(collection_name):
            return []

        collection = self._load_collection(collection_name)

        result = collection.search(
            data=embeddings,
            anns_field="embeddings",
            param=search_params or {"metric_type": "L2", "params": {"nprobe": 10}},
            limit=limit,
            output_fields=["metadata"],
        )

        top_k_used = min(len(result[0]), limit)

        ret = []
        for i in range(top_k_used):

            hit: Hit = result[0][i]
            doc_dict = json.loads(hit.entity.get("metadata"))
            doc = Document(
                page_content=doc_dict["page_content"],
                metadata=doc_dict["metadata"] or {},
            )

            score = -hit.entity.distance
            ret.append((doc, sigmoid(score) if apply_sigmoid else score))

        return ret

    def delete(self, collection_name: str, doc_ids: list[str]):
        if not self.has_collection(collection_name):
            return

        collection = self._load_collection(collection_name)
        collection.delete(f"id in {doc_ids}")
        collection.flush()

    def _load_collection(self, collection_name: str) -> Collection:
        if collection_name not in self._loaded_collections:
            collection = Collection(name=collection_name, using=self._alias)
            collection.load()
            self._loaded_collections[collection_name] = collection
        return self._loaded_collections[collection_name]

    def _release_collection(self, collection_name: str):
        if collection_name in self._loaded_collections:
            collection = self._loaded_collections[collection_name]
            collection.release()
            del self._loaded_collections[collection_name]

    def __del__(self):
        """Cleanup method to release collections and disconnect from Milvus."""
        for collection_name in list(self._loaded_collections.keys()):
            self._release_collection(collection_name)
        connections.disconnect(self._alias)
