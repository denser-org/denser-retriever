import json
from typing import List
import uuid
from denser_retriever.constants import (
    DEFAULT_KEYWORD_TOP_K,
    DEFAULT_MAX_VECTOR_QUERY_LENGTH,
    DEFAULT_PRIMARY_KEY_FIELD,
    DEFAULT_RERANKER_TOP_K,
    DEFAULT_VECTOR_TOP_K,
)
from denser_retriever.embedding import EmbeddingModel, SentenceTransformerEmbeddings
from denser_retriever.keyword_search import KeywordSearch
from denser_retriever.fusion import FusionModel
from denser_retriever.reranker import Reranker
from denser_retriever.utils import (
    compute_document_features,
    create_instance,
    hybridCombine,
)
from denser_retriever.vector_store import VectorStore
from langchain_core.documents import Document


class DenserRetriever:
    def __init__(
        self,
        keyword_search: KeywordSearch = None,
        vector_store: VectorStore = None,
        embedding_model: EmbeddingModel = None,
        reranker: Reranker = None,
        fusion_model: FusionModel = None,
        vector_top_k: int = DEFAULT_VECTOR_TOP_K,
        keyword_top_k: int = DEFAULT_KEYWORD_TOP_K,
        reranker_top_k: int = DEFAULT_RERANKER_TOP_K,
        max_vector_query_length: int = DEFAULT_MAX_VECTOR_QUERY_LENGTH,
        primary_key_field: str = DEFAULT_PRIMARY_KEY_FIELD,
    ):
        if not keyword_search and not vector_store:
            raise ValueError(
                "At least one of keyword_search or vector_store must be provided."
            )

        self._keyword_search = keyword_search
        self._vector_store = vector_store
        self._embedding_model = (
            embedding_model
            or SentenceTransformerEmbeddings(
                model_name="Snowflake/snowflake-arctic-embed-m"
            )
            if (vector_store)
            else None
        )

        self._reranker = reranker
        self._fusion_model = fusion_model
        self._vector_top_k = vector_top_k
        self._keyword_top_k = keyword_top_k
        self._reranker_top_k = reranker_top_k
        self._primary_key_field = primary_key_field
        self._max_vector_query_length = max_vector_query_length

    def ingest(self, docs: List[Document], collection_name: str = "default"):
        if not docs:
            return []

        pks = [str(uuid.uuid4()) for _ in range(len(docs))]

        if self._keyword_search:
            pks = self._keyword_search.indexing(
                index_name=collection_name,
                primary_keys=pks,
                primary_key_field=self._primary_key_field,
                docs=docs,
            )

        if self._vector_store and self._embedding_model:
            embeddings = self._embedding_model.embed_documents(
                [doc.page_content for doc in docs]
            )

            pks = self._vector_store.insert(
                collection_name=collection_name,
                primary_keys=pks,
                primary_key_field=self._primary_key_field,
                docs=docs,
                embeddings=embeddings,
            )
            del embeddings

        return pks

    def retrieve(
        self,
        query: str,
        limit: int,
        collection_name: str = "default",
        ks_weight: float = 1.0,
        vs_weight: float = 1.0,
        rank_offset: int = 60,
    ):
        if not query or limit <= 0:
            return []

        ks_result = []  # keyword search results
        vs_result = []  # vector store search results

        if self._keyword_search:
            ks_result = self._keyword_search.search(
                collection_name, query, self._keyword_top_k
            )

        if self._vector_store and self._embedding_model:
            vector_query = query

            # Truncate query if it exceeds max_vector_query_length
            if self._max_vector_query_length > 0:
                vector_query = query[: self._max_vector_query_length]

            embeddings = self._embedding_model.embed_query(vector_query)
            vs_result = self._vector_store.search(
                collection_name, embeddings, self._vector_top_k
            )

        # Combine keyword search and vector store search results
        combine_result = hybridCombine(
            doc_lists=[ks_result, vs_result],
            weights=[ks_weight, vs_weight],
            rank_offset=rank_offset,
            max_rank=max(self._keyword_top_k, self._vector_top_k),
            key_field=self._primary_key_field,
        )

        # If reranker is provided, rerank the combined results
        if self._reranker:
            combine_result = self._reranker.rerank(
                [doc for doc, _ in combine_result], query
            )

        # If fusion model is provided, compute document features and predict scores
        if self._fusion_model:
            featured_docs, features = compute_document_features(
                ks_result, vs_result, combine_result, self._primary_key_field
            )
            scores = []
            for feature_list in features:
                scores.append(self._fusion_model.predict(feature_list))

            scored_docs = list(zip(featured_docs, scores))
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            return scored_docs[:limit]

        return combine_result[:limit]

    def delete(self, collection_name: str, pks: List[str]):
        if self._keyword_search:
            self._keyword_search.delete(collection_name, pks)
        if self._vector_store:
            self._vector_store.delete(collection_name, pks)

    def has_collection(self, collection_name: str):
        if self._keyword_search:
            return self._keyword_search.has_index(collection_name)
        if self._vector_store:
            return self._vector_store.has_collection(collection_name)
        return False

    def drop(self, collection_name: str):
        if self._keyword_search:
            self._keyword_search.drop_index(collection_name)
        if self._vector_store:
            self._vector_store.drop_collection(collection_name)

    @classmethod
    def from_config(cls, config_file_path: str):
        """Create a DenserRetriever instance from a JSON configuration file.

        Args:
            config_file_path: Path to JSON configuration file

        Returns:
            DenserRetriever: Configured retriever instance
        """

        with open(config_file_path, "r") as f:
            config: dict = json.load(f)

        components = {
            component: (
                create_instance(config[component]["class"], config[component]["params"])
                if component in config
                else None
            )
            for component in [
                "keyword_search",
                "vector_store",
                "embedding_model",
                "reranker",
                "fusion_model",
            ]
        }

        return cls(
            **components,
            vector_top_k=config.get("vector_top_k", DEFAULT_VECTOR_TOP_K),
            keyword_top_k=config.get("keyword_top_k", DEFAULT_KEYWORD_TOP_K),
            reranker_top_k=config.get("reranker_top_k", DEFAULT_RERANKER_TOP_K),
            primary_key_field=config.get(
                "primary_key_field", DEFAULT_PRIMARY_KEY_FIELD
            ),
            max_vector_query_length=config.get(
                "max_vector_query_length", DEFAULT_MAX_VECTOR_QUERY_LENGTH
            ),
        )
