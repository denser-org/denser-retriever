from asyncio.log import logger
from typing import Any, Dict, List, Optional, Tuple
import uuid

from langchain_core.documents import Document

from denser_retriever.core.embeddings import DenserEmbeddings
from denser_retriever.core.keyword import DenserKeywordSearch
from denser_retriever.core.reranker import DenserReranker
from denser_retriever.core.types import RetrievalResult, TokenMetrics
from denser_retriever.core.utils import docs_to_dict
from denser_retriever.core.vectordb.base import DenserVectorDB
from denser_retriever.config import CombineConfig
from denser_retriever.core.logistic_regression import LogisticRegression
from denser_retriever.core.utils import config_to_features
from denser_retriever.core.resource_tracking import ResourceTracker


class DenserRetriever:
    def __init__(
        self,
        index_name: str,
        keyword_search: Optional[DenserKeywordSearch],
        vector_db: Optional[DenserVectorDB],
        reranker: Optional[DenserReranker],
        embeddings: DenserEmbeddings,
        combine_config: CombineConfig,
        search_fields: List[str] = [],
        date_fields: List[str] = [],
    ):
        # config parameters
        self.index_name = index_name
        self.combine_method = combine_config.method
        # models
        self.embeddings = embeddings
        if combine_config.lr_config:
            self.lr_model = LogisticRegression(combine_config.lr_config.lr_model)
            self.lr_features = config_to_features[combine_config.lr_config.lr_features]
        else:
            self.lr_model = None
            self.lr_features = None
        self.keyword_search = keyword_search
        self.vector_db = vector_db
        self.reranker = reranker
        self.combine_config = combine_config

        # create index. If exists, remove them first if drop_old is true
        if self.vector_db:
            assert embeddings
            self.vector_db.create_index(index_name, embeddings, search_fields)
        if self.keyword_search:
            self.keyword_search.create_index(
                index_name=index_name,
                search_fields=search_fields,
                date_fields=date_fields,
            )

    def _ingest_elasticsearch(
        self,
        docs: List[Document],
        texts: List[str],
        es_storage_quota_gb: Optional[float],
    ) -> Tuple[int, float]:
        """Process Elasticsearch ingestion and return number of documents allowed and storage used."""
        if not self.keyword_search:
            return 0, 0.0

        metadata_fields = {
            "title": 200,  # Estimate 200 bytes per title
            "source": 500,  # Estimate 500 bytes per source
            "pid": 36,  # Fixed 36 bytes for UUID
        }

        es_size_gb = ResourceTracker.estimate_es_storage_gb(texts, metadata_fields)
        num_docs = len(docs)
        if es_storage_quota_gb and es_size_gb > es_storage_quota_gb:
            docs_allowed = int(num_docs * (es_storage_quota_gb / es_size_gb))
            logger.warning(
                f"ES storage quota would be exceeded. Limiting to {docs_allowed} documents"
            )
            es_size_gb = es_storage_quota_gb
        else:
            docs_allowed = num_docs

        logger.info(f"Adding {docs_allowed} documents to keyword search")
        self.keyword_search.add_documents(docs[:docs_allowed])
        return docs_allowed, es_size_gb

    def _ingest_vector_db(
        self,
        docs: List[Document],
        texts: List[str],
        vector_storage_quota_gb: Optional[float],
        vector_token_quota: Optional[int],
    ) -> Tuple[int, float, int]:
        """Process Vector DB ingestion and return docs allowed, storage used, and tokens used."""
        if not self.vector_db:
            return 0, 0.0, 0

        num_docs = len(docs)
        vector_size_gb = ResourceTracker.estimate_vector_storage_gb(texts)
        docs_allowed = num_docs
        # Check storage quota
        if vector_storage_quota_gb and vector_size_gb > vector_storage_quota_gb:
            docs_allowed = int(num_docs * (vector_storage_quota_gb / vector_size_gb))
            vector_size_gb = vector_storage_quota_gb
            logger.warning(
                f"Vector storage quota would be exceeded. Limiting to {docs_allowed} documents"
            )

        # Check token quota
        token_count = ResourceTracker.calculate_vector_token_count(texts[:docs_allowed])
        if vector_token_quota and token_count > vector_token_quota:
            tokens_per_doc = token_count / docs_allowed
            token_limited_docs = max(0, int(vector_token_quota / tokens_per_doc))

            if token_limited_docs < docs_allowed:
                docs_allowed = token_limited_docs
                token_count = int(docs_allowed * tokens_per_doc)
                vector_size_gb *= docs_allowed / num_docs
                logger.warning(
                    f"Vector token quota would be exceeded. Limiting to {docs_allowed} documents"
                )

        if docs_allowed > 0:
            self.vector_db.add_documents(documents=docs[:docs_allowed])

        return docs_allowed, vector_size_gb, token_count

    def ingest(
        self,
        docs: List[Document],
        overwrite_pid: bool = True,
        es_storage_quota_gb: Optional[float] = None,
        vector_storage_quota_gb: Optional[float] = None,
        vector_token_quota: Optional[int] = None,
    ) -> Tuple[List[str], Dict[str, float]]:
        """Ingest documents into elasticsearch and vector db with quota limits."""
        if overwrite_pid:
            for doc in docs:
                doc.metadata["pid"] = uuid.uuid4().hex

        texts = [doc.page_content for doc in docs]

        # Process elasticsearch and vector db ingestion
        es_docs_allowed, es_size_gb = self._ingest_elasticsearch(
            docs, texts, es_storage_quota_gb
        )
        vector_docs_allowed, vector_size_gb, token_count = self._ingest_vector_db(
            docs, texts, vector_storage_quota_gb, vector_token_quota
        )
        # Calculate max docs processed across both stores
        max_docs_processed = max(es_docs_allowed, vector_docs_allowed)

        metrics = {
            "es_docs": es_docs_allowed,
            "vector_docs": vector_docs_allowed,
            "es_storage_gb": es_size_gb,
            "vector_storage_gb": vector_size_gb,
            "vector_tokens": token_count,
        }
        return [doc.metadata["pid"] for doc in docs[:max_docs_processed]], metrics

    def retrieve(
        self,
        query: str,
        k: int,
        combine_config: CombineConfig,
        filter: Dict[str, Any] = {},
        aggregation: bool = False,
        usage: bool = False,
    ) -> RetrievalResult:
        logger.info(f"Retrieve query: {query} top_k: {k}")
        if self.combine_method == "vector":
            return self.retrieve_by_vector(
                query, k, combine_config, filter, aggregation, usage
            )
        elif self.combine_method == "hybrid":
            return self.retrieve_by_hybrid(
                query, k, combine_config, filter, aggregation, usage
            )
        elif self.combine_method == "reranker":
            return self.retrieve_by_reranker(
                query, k, combine_config, filter, aggregation, usage
            )
        elif self.combine_method == "fusion":
            return self.retrieve_by_fusion(
                query, k, combine_config, filter, aggregation, usage
            )
        else:
            raise ValueError(f"Unknown combine method {self.combine_method}")

    def retrieve_by_vector(
        self,
        query: str,
        k: int,
        combine_config: CombineConfig,
        filter: Dict[str, Any] = {},
        aggregation: bool = False,
        usage: bool = False,
    ) -> RetrievalResult:
        """Vector-only search using the vector database."""
        if not self.vector_db:
            raise ValueError("Vector database not initialized")

        # Get vector search results
        vs_docs = self.vector_db.similarity_search_with_score(query, k, filter=filter)

        # Calculate token usage
        metrics = None
        if usage:
            embedding_tokens = ResourceTracker.calculate_vector_search_token_count(
                query, [doc for doc, _ in vs_docs]
            )
            metrics = TokenMetrics(
                vector_tokens=embedding_tokens, total_tokens=embedding_tokens
            )

        return RetrievalResult(vs_docs, None, metrics)

    def retrieve_by_hybrid(
        self,
        query: str,
        k: int,
        combine_config: CombineConfig,
        filter: Dict[str, Any] = {},
        aggregation: bool = False,
        usage: bool = False,
    ) -> RetrievalResult:
        """Hybrid search using keyword and vector positions."""
        # Get keyword search results
        if not self.keyword_search:
            raise ValueError("Keyword search not initialized")
        ks_docs, aggregations = self.keyword_search.retrieve(
            query,
            combine_config.keyword_top_k,
            filter=filter,
            aggregation=aggregation,
            apply_sigmoid=False,
        )

        # Get vector search results
        if not self.vector_db:
            raise ValueError("Vector database not initialized")
        vs_docs = self.vector_db.similarity_search_with_score(
            query, combine_config.vector_top_k, filter=filter
        )

        # Calculate token usage
        metrics = None
        if usage:
            vector_tokens = ResourceTracker.calculate_vector_search_token_count(
                query, [doc for doc, _ in vs_docs]
            )
            keyword_tokens = len(query.split())
            metrics = TokenMetrics(
                vector_tokens=vector_tokens,
                keyword_tokens=keyword_tokens,
                total_tokens=vector_tokens + keyword_tokens,
            )

        # Extract position information and combine results
        _, _, ks_rank_dict = docs_to_dict(ks_docs)
        _, _, vs_rank_dict = docs_to_dict(vs_docs)

        # Combine results
        all_docs = {}
        for doc, _ in ks_docs + vs_docs:
            pid = doc.metadata["pid"]
            if pid not in all_docs:
                all_docs[pid] = doc

        hybrid_scores = {}
        max_rank = max(combine_config.keyword_top_k, combine_config.vector_top_k)

        for pid, doc in all_docs.items():
            ks_rank = ks_rank_dict.get(pid, max_rank + 1)
            vs_rank = vs_rank_dict.get(pid, max_rank + 1)

            hybrid_scores[pid] = 0.0
            if ks_rank <= max_rank:
                hybrid_scores[pid] += 1.0 / (ks_rank + 60)
            if vs_rank <= max_rank:
                hybrid_scores[pid] += 1.0 / (vs_rank + 60)

        scored_docs = [(all_docs[pid], score) for pid, score in hybrid_scores.items()]
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        return RetrievalResult(
            documents=scored_docs[:k], aggregations=aggregations, token_metrics=metrics
        )

    def retrieve_by_reranker(
        self,
        query: str,
        k: int,
        combine_config: CombineConfig,
        filter: Dict[str, Any] = {},
        aggregation: bool = False,
        usage: bool = False,
    ) -> RetrievalResult:
        """Two-stage retrieval: keyword search followed by reranking."""
        # First stage: keyword search
        if not self.keyword_search:
            raise ValueError("Keyword search not initialized")
        ks_docs, aggregations = self.keyword_search.retrieve(
            query, combine_config.keyword_top_k, filter=filter, aggregation=aggregation
        )

        # Extract documents for reranking
        docs_to_rerank = [doc for doc, _ in ks_docs]

        metrics = None
        # Second stage: reranking
        if self.reranker and docs_to_rerank:
            reranked_docs = self.reranker.rerank(docs_to_rerank, query)
            # Calculate reranker token usage
            if usage:
                rerank_tokens = ResourceTracker.calculate_reranker_token_count(
                    query, docs_to_rerank
                )
                metrics = TokenMetrics(
                    rerank_tokens=rerank_tokens, total_tokens=rerank_tokens
                )
            return RetrievalResult(
                documents=reranked_docs[:k],
                aggregations=aggregations,
                token_metrics=metrics,
            )

        return RetrievalResult(
            documents=ks_docs[:k], aggregations=aggregations, token_metrics=metrics
        )

    def retrieve_by_fusion(
        self,
        query: str,
        k: int,
        combine_config: CombineConfig,
        filter: Dict[str, Any] = {},
        aggregation: bool = False,
        usage: bool = False,
    ) -> RetrievalResult:
        """Retrieve using logistic regression model for fusion."""
        docs, doc_features, aggregations = self._retrieve_with_features(
            query, combine_config, filter, aggregation
        )

        # Calculate token metrics from all retrieval methods
        metrics = None
        if usage:
            vector_tokens = ResourceTracker.calculate_vector_search_token_count(
                query, docs[: combine_config.vector_top_k]
            )
            rerank_tokens = (
                ResourceTracker.calculate_reranker_token_count(query, docs)
                if self.reranker
                else 0
            )
            metrics = TokenMetrics(
                vector_tokens=vector_tokens,
                rerank_tokens=rerank_tokens,
                total_tokens=vector_tokens + rerank_tokens,
            )

        scores = []
        for feature_list in doc_features:
            if self.lr_model:
                scores.append(self.lr_model.predict(feature_list))
            else:
                scores.append(0)

        # Combine with documents
        scored_docs = list(zip(docs, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        return RetrievalResult(
            documents=scored_docs[:k], aggregations=aggregations, token_metrics=metrics
        )

    def _retrieve_with_features(
        self,
        query: str,
        combine_config: CombineConfig,
        filter: Dict[str, Any] = {},
        aggregation: bool = False,
    ) -> Tuple[List[Document], List[List[str]], Optional[Dict]]:
        ks_docs = []
        aggregations = None

        if self.keyword_search:
            ks_docs, aggregations = self.keyword_search.retrieve(
                query,
                combine_config.keyword_top_k,
                filter=filter,
                aggregation=aggregation,
            )
        vs_docs = []
        if self.vector_db:
            vs_docs = self.vector_db.similarity_search_with_score(
                query, combine_config.vector_top_k, filter=filter
            )

        combined = []
        seen = set()
        for item in ks_docs + vs_docs:
            if item[0].metadata["pid"] not in seen:
                combined.append(item)
                seen.add(item[0].metadata["pid"])

        combined_docs = [doc for doc, _ in combined]

        reranked_docs = []
        if self.reranker:
            reranked_docs = self.reranker.rerank(combined_docs, query)

        _, ks_score_dict, ks_rank_dict = docs_to_dict(ks_docs)
        _, vs_score_dict, vs_rank_dict = docs_to_dict(vs_docs)
        reranked_docs_dict, reranked_score_dict, reranked_rank_dict = docs_to_dict(
            reranked_docs
        )

        docs, doc_features = [], []
        for pid in reranked_docs_dict.keys():
            docs.append(reranked_docs_dict[pid])

            features = []
            features.append(0)  # placeholder
            features.append(ks_rank_dict.get(pid, -1))  # 1. keyword rank
            features.append(ks_score_dict.get(pid, 0))  # 2. keyword score
            miss = 1 if ks_rank_dict.get(pid, -1) == -1 else 0
            features.append(miss)  # 3. keyword miss

            features.append(vs_rank_dict.get(pid, -1))  # 4. vector rank
            features.append(vs_score_dict.get(pid, 0))  # 5. vector score
            miss = 1 if vs_rank_dict.get(pid, -1) == -1 else 0
            features.append(miss)  # 6. vector miss

            assert pid in reranked_rank_dict
            features.append(reranked_rank_dict[pid])  # 7. rerank rank
            features.append(reranked_score_dict[pid])  # 8. rerank score
            features.append(0)  # 9. placeholder
            doc_features.append(features)

        features_to_use = self.lr_features

        non_zero_features = []
        for i, data in enumerate(doc_features):
            features = []
            if features_to_use:
                for f_id in features_to_use:
                    f_value = data[int(f_id)]
                    if f_value != 0.0:
                        features.append(f"{f_id}:{f_value}")

            non_zero_features.append([str(data[0])] + features)

        return docs, non_zero_features, aggregations

    def delete(
        self,
        ids: Optional[List[str]] = None,
        source_id: Optional[str] = None,
        **kwargs: str,
    ):
        """Clear the retriever."""
        if self.vector_db:
            self.vector_db.delete(ids=ids, source_id=source_id, **kwargs)
        if self.keyword_search:
            self.keyword_search.delete(ids=ids, source_id=source_id, **kwargs)

    def delete_all(self, delete_index: bool=True):
        """Clear the retriever."""
        if self.vector_db:
            self.vector_db.delete_all(delete_index=delete_index)
        if self.keyword_search:
            self.keyword_search.delete_all(delete_index=delete_index)

    def get_filter_fields(self):
        """Get the filter fields."""
        if not self.keyword_search:
            raise ValueError("Keyword search not initialized")
        return self.keyword_search.get_index_mappings()

    def get_index_stats(self) -> Dict[str, int]:
        """Get the number of documents in elasticsearch and milvus indices.

        Returns:
            Dict[str, int]: Dictionary containing document counts for each index
                {'es_docs': int, 'vector_docs': int}
        """
        stats = {'es_docs': 0, 'vector_docs': 0}

        if self.keyword_search:
            # Count documents in elasticsearch
            result = self.keyword_search.client.count(index=self.index_name)
            stats['es_docs'] = result['count']

        if self.vector_db:
            # Count documents in milvus
            stats['vector_docs'] = self.vector_db.get_count()
            # col.num_entities is not a reliable way to get the number of documents
            # if hasattr(self.vector_db, 'col') and self.vector_db.col:
            #     stats['vector_docs'] = self.vector_db.col.num_entities

        return stats

    def check_indices(self) -> Dict[str, bool]:
        """Check if elasticsearch and milvus indices exist and are valid.

        Returns:
            Dict containing index status {'es_valid': bool, 'vector_valid': bool}
        """
        status = {'es_valid': False, 'vector_valid': False}

        if self.keyword_search:
            try:
                exists = self.keyword_search.client.indices.exists(index=self.index_name)
                if exists:
                    # Test query to verify index is working
                    self.keyword_search.client.search(index=self.index_name, query={"match_all": {}})
                    status['es_valid'] = True
            except Exception as e:
                logger.error(f"Elasticsearch index check failed: {e}")

        if self.vector_db:
            try:
                # Verify collection exists and can be queried
                count = self.vector_db.get_count()
                status['vector_valid'] = True
            except Exception as e:
                logger.error(f"Vector DB index check failed: {e}")

        return status