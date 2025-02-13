import json
from typing import List
import uuid
from denser_retriever.constants import (
    DEFAULT_KEYWORD_TOP_K,
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
    remove_duplicates,
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
        primary_key_field: str = DEFAULT_PRIMARY_KEY_FIELD,
    ):
        if not keyword_search and not vector_store:
            raise ValueError(
                "At least one of keyword_search or vector_store must be provided."
            )

        self.keyword_search = keyword_search
        self.vector_store = vector_store
        self.embedding_model = (
            embedding_model
            or SentenceTransformerEmbeddings(
                model_name="Snowflake/snowflake-arctic-embed-m"
            )
            if (vector_store)
            else None
        )

        self.reranker = reranker
        self.fusion_model = fusion_model
        self.vector_top_k = vector_top_k
        self.keyword_top_k = keyword_top_k
        self.reranker_top_k = reranker_top_k
        self.primary_key_field = primary_key_field

    def ingest(self, docs: List[Document], collection_name="default"):
        if not docs:
            return []

        pks = [str(uuid.uuid4()) for _ in range(len(docs))]

        if self.keyword_search:
            self.keyword_search.indexing(
                index_name=collection_name,
                primary_keys=pks,
                primary_key_field=self.primary_key_field,
                docs=docs,
            )

        if self.vector_store and self.embedding_model:
            embeddings = self.embedding_model.embed_documents(
                [doc.page_content for doc in docs]
            )

            self.vector_store.insert(
                collection_name=collection_name,
                primary_keys=pks,
                primary_key_field=self.primary_key_field,
                docs=docs,
                embeddings=embeddings,
            )

        return pks

    def retrieve(self, query: str, limit: int, collection_name="default"):
        if not query or not limit:
            return []

        ks_docs = []  # keyword search results
        vs_docs = []  # vector store search results

        if self.keyword_search:
            ks_docs = self.keyword_search.search(
                collection_name, query, self.keyword_top_k
            )

        if self.vector_store and self.embedding_model:
            embeddings = self.embedding_model.embed_query(query)
            vs_docs = self.vector_store.search(
                collection_name, embeddings, self.vector_top_k
            )

        # If reranker is not provided, return the hybrid rerank results
        if not self.reranker:
            hybrid_reranked_docs = hybridCombine(
                ks_docs,
                vs_docs,
                max(self.keyword_top_k, self.vector_top_k),
                primary_key_field=self.primary_key_field,
            )
            return hybrid_reranked_docs[:limit]

        combined_docs = remove_duplicates(ks_docs + vs_docs, self.primary_key_field)
        reranked_docs = self.reranker.rerank([doc for doc, _ in combined_docs], query)

        # If fusion model is not provided, return the reranked results
        if not self.fusion_model:
            return reranked_docs[:limit]

        featured_docs, features = compute_document_features(
            ks_docs, vs_docs, reranked_docs, self.primary_key_field
        )

        scores = []
        for feature_list in features:
            scores.append(self.fusion_model.predict(feature_list))

        scored_docs = list(zip(featured_docs, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        return scored_docs[:limit]

    def delete(self, collection_name: str, pks: List[str]):
        if self.keyword_search:
            self.keyword_search.delete(collection_name, pks)
        if self.vector_store:
            self.vector_store.delete(collection_name, pks)

    def drop(self, collection_name: str):
        if self.keyword_search:
            self.keyword_search.drop_index(collection_name)
        if self.vector_store:
            self.vector_store.drop_collection(collection_name)

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

        # Initialize components based on config
        keyword_search = None
        if "keyword_search" in config:
            keyword_search = create_instance(
                config["keyword_search"]["class"], config["keyword_search"]["params"]
            )

        vector_store = None
        if "vector_store" in config:
            vector_store = create_instance(
                config["vector_store"]["class"], config["vector_store"]["params"]
            )

        embedding_model = None
        if "embedding_model" in config:
            embedding_model = create_instance(
                config["embedding_model"]["class"], config["embedding_model"]["params"]
            )

        reranker = None
        if "reranker" in config:
            reranker = create_instance(
                config["reranker"]["class"], config["reranker"]["params"]
            )

        fusion_model = None
        if "fusion_model" in config:
            fusion_model = create_instance(
                config["fusion_model"]["class"], config["fusion_model"]["params"]
            )

        return cls(
            keyword_search=keyword_search,
            vector_store=vector_store,
            embedding_model=embedding_model,
            reranker=reranker,
            fusion_model=fusion_model,
            vector_top_k=config.get("vector_top_k", DEFAULT_VECTOR_TOP_K),
            keyword_top_k=config.get("keyword_top_k", DEFAULT_KEYWORD_TOP_K),
            reranker_top_k=config.get("reranker_top_k", DEFAULT_RERANKER_TOP_K),
            primary_key_field=config.get(
                "primary_key_field", DEFAULT_PRIMARY_KEY_FIELD
            ),
        )
