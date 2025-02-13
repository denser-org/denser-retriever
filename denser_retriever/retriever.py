from typing import List
import uuid
from denser_retriever.embedding import EmbeddingModel, SentenceTransformerEmbeddings
from denser_retriever.keyword_search import KeywordSearch
from denser_retriever.fusion import FusionModel
from denser_retriever.reranker import Reranker
from denser_retriever.utils import (
    compute_document_features,
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
        vector_top_k: int = 100,
        keyword_top_k: int = 100,
        reranker_top_k: int = 100,
        primary_key_field: str = "id",
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
