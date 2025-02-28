from typing import Any, Dict, List, Optional, Tuple, Callable
import uuid
import logging
import json

from langchain_core.documents import Document

from denser_retriever.core.types import RetrievalResult, TokenMetrics
from denser_retriever.core.utils import docs_to_dict
from denser_retriever.core.resource_tracking import ResourceTracker
from denser_retriever.core.keyword import ESIndexData
from denser_retriever.core.vectordb.milvus import MilvusIndexData
from denser_retriever.core.shared import SharedComponents
from denser_retriever.core.utils import config_to_features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DenserRetriever:
    def __init__(
            self,
            shared: SharedComponents,
            es_data: Optional[ESIndexData] = None,
            milvus_data: Optional[MilvusIndexData] = None
    ):
        self.keyword_search = shared.keyword_search
        self.vector_db = shared.vector_db
        self.reranker = shared.reranker
        self.embeddings = shared.embeddings
        self.lr_model = shared.lr_model if hasattr(shared, 'lr_model') else None
        self.lr_features = shared.lr_features if hasattr(shared, 'lr_features') else None

        self.es_data = es_data
        self.milvus_data = milvus_data
        if es_data and self.keyword_search:
            self.keyword_search.create_index(es_data)
        if milvus_data and self.vector_db:
            self.vector_db.create_index(milvus_data)

    def _ingest_elasticsearch(
            self,
            docs: List[Document],
            progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ) -> int:
        """Process Elasticsearch ingestion and return number of documents processed."""
        if not self.keyword_search:
            return 0

        num_docs = len(docs)
        logger.info(f"Adding {num_docs} documents to keyword search")
        self.keyword_search.add_documents(self.es_data, docs, progress_callback=progress_callback)
        return num_docs

    def _ingest_vector_db(
            self,
            docs: List[Document],
            texts: List[str],
            progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ) -> Tuple[int, int]:
        """Process Vector DB ingestion and return docs processed and tokens used."""
        if not self.vector_db:
            return 0, 0

        num_docs = len(docs)
        token_count = ResourceTracker.calculate_vector_token_count(texts)

        if num_docs > 0:
            self.vector_db.add_documents(
                self.milvus_data,
                docs,
                self.embeddings,
                progress_callback=progress_callback
            )

        return num_docs, token_count

    def ingest(
            self,
            docs: List[Document],
            overwrite_pid: bool = True,
            progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Tuple[List[str], Dict[str, float]]:
        if overwrite_pid:
            for doc in docs:
                doc.metadata["pid"] = uuid.uuid4().hex

        texts = [doc.page_content for doc in docs]
        total_docs = len(docs)

        if progress_callback:
            progress_callback({
                "total_docs": total_docs,
                "es_progress": 0,
                "vector_progress": 0,
                "es_docs_processed": 0,
                "vector_docs_processed": 0,
                "status": "starting"
            })

        es_docs_processed = 0
        try:
            es_docs_processed = self._ingest_elasticsearch(docs, progress_callback)
            if progress_callback:
                progress_callback({
                    "total_docs": total_docs,
                    "es_progress": (es_docs_processed / total_docs) * 100,
                    "es_docs_processed": es_docs_processed,
                    "status": "elasticsearch_ingestion_complete"
                })
        except Exception as e:
            logger.error(f"Elasticsearch ingestion error: {e}")
            if progress_callback:
                progress_callback({
                    "total_docs": total_docs,
                    "es_progress": 0,
                    "es_docs_processed": 0,
                    "status": "elasticsearch_ingestion_failed",
                    "error": str(e)
                })

        vector_docs_processed, token_count = 0, 0
        try:
            vector_docs_processed, token_count = self._ingest_vector_db(docs, texts, progress_callback)
            if progress_callback:
                progress_callback({
                    "total_docs": total_docs,
                    "vector_progress": (vector_docs_processed / total_docs) * 100,
                    "vector_docs_processed": vector_docs_processed,
                    "status": "vector_db_ingestion_complete"
                })
        except Exception as e:
            logger.error(f"Vector DB ingestion error: {e}")
            if progress_callback:
                progress_callback({
                    "total_docs": total_docs,
                    "vector_progress": 0,
                    "vector_docs_processed": 0,
                    "status": "vector_db_ingestion_failed",
                    "error": str(e)
                })

        max_docs_processed = max(es_docs_processed, vector_docs_processed)

        if progress_callback:
            progress_callback({
                "total_docs": total_docs,
                "es_progress": (es_docs_processed / total_docs) * 100,
                "vector_progress": (vector_docs_processed / total_docs) * 100,
                "es_docs_processed": es_docs_processed,
                "vector_docs_processed": vector_docs_processed,
                "status": "ingestion_complete"
            })

        metrics = {
            "es_docs": es_docs_processed,
            "vector_docs": vector_docs_processed,
            "vector_tokens": token_count,
        }
        return [doc.metadata["pid"] for doc in docs[:max_docs_processed]], metrics

    def retrieve(
            self,
            query: str,
            method: str,
            k: int,
            keyword_top_k: int = 100,
            vector_top_k: int = 100,
            filter: Dict[str, Any] = {},
            aggregation: bool = False,
            usage: bool = False
    ) -> RetrievalResult:
        logger.info(f"Retrieve query: {query} top_k: {k} method: {method}")

        if method == "vector":
            return self.retrieve_by_vector(query, k, filter, usage)
        elif method == "hybrid":
            return self.retrieve_by_hybrid(
                query, k, keyword_top_k, vector_top_k, filter, aggregation, usage
            )
        elif method == "reranker":
            return self.retrieve_by_reranker(
                query, k, keyword_top_k, filter, aggregation, usage
            )
        elif method == "fusion":
            return self.retrieve_by_fusion(
                query, k, keyword_top_k, vector_top_k, filter, aggregation, usage
            )
        else:
            raise ValueError(f"Unknown combine method {method}")

    def retrieve_by_vector(
            self,
            query: str,
            k: int,
            filter: Dict[str, Any] = {},
            usage: bool = False
    ) -> RetrievalResult:
        """Vector-only search using the vector database."""
        if not self.vector_db:
            raise ValueError("Vector database not initialized")

        vs_docs = self.vector_db.retrieve(self.milvus_data, query, self.embeddings, k, filter=filter)

        metrics = None
        if usage:
            embedding_tokens = ResourceTracker.calculate_vector_search_token_count(
                query, [doc for doc, _ in vs_docs]
            )
            metrics = TokenMetrics(
                keyword_queries=0,
                vector_tokens=embedding_tokens,
                rerank_tokens=0,
                total_tokens=embedding_tokens
            )

        return RetrievalResult(vs_docs, None, metrics)

    def retrieve_by_hybrid(
            self,
            query: str,
            k: int,
            keyword_top_k: int = 100,
            vector_top_k: int = 100,
            filter: Dict[str, Any] = {},
            aggregation: bool = False,
            usage: bool = False
    ) -> RetrievalResult:
        """Hybrid search using keyword and vector positions."""
        if not self.keyword_search:
            raise ValueError("Keyword search not initialized")
        ks_docs, aggregations = self.keyword_search.retrieve(
            self.es_data,
            query,
            keyword_top_k,
            filter=filter,
            aggregation=aggregation,
            apply_sigmoid=False,
        )

        if not self.vector_db:
            raise ValueError("Vector database not initialized")
        vs_docs = self.vector_db.retrieve(
            self.milvus_data, query, self.embeddings, vector_top_k, filter=filter
        )

        metrics = None
        if usage:
            vector_tokens = ResourceTracker.calculate_vector_search_token_count(
                query, [doc for doc, _ in vs_docs]
            )
            metrics = TokenMetrics(
                keyword_queries = 1,
                vector_tokens=vector_tokens,
                rerank_tokens=0,
                total_tokens=vector_tokens
            )

        _, _, ks_rank_dict = docs_to_dict(ks_docs)
        _, _, vs_rank_dict = docs_to_dict(vs_docs)

        all_docs = {}
        for doc, _ in ks_docs + vs_docs:
            pid = doc.metadata["pid"]
            if pid not in all_docs:
                all_docs[pid] = doc

        hybrid_scores = {}
        max_rank = max(keyword_top_k, vector_top_k)

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
            keyword_top_k: int = 100,
            filter: Dict[str, Any] = {},
            aggregation: bool = False,
            usage: bool = False
    ) -> RetrievalResult:
        """Two-stage retrieval: keyword search followed by reranking."""
        if not self.keyword_search:
            raise ValueError("Keyword search not initialized")
        ks_docs, aggregations = self.keyword_search.retrieve(
            self.es_data, query, keyword_top_k, filter=filter, aggregation=aggregation
        )

        docs_to_rerank = [doc for doc, _ in ks_docs]

        metrics = None
        if self.reranker and docs_to_rerank:
            reranked_docs = self.reranker.rerank(docs_to_rerank, query)
            if usage:
                rerank_tokens = ResourceTracker.calculate_reranker_token_count(
                    query, docs_to_rerank
                )
                metrics = TokenMetrics(
                    keyword_queries=1,
                    vector_tokens=0,
                    rerank_tokens=rerank_tokens,
                    total_tokens=rerank_tokens
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
            keyword_top_k: int = 100,
            vector_top_k: int = 100,
            filter: Dict[str, Any] = {},
            aggregation: bool = False,
            usage: bool = False
    ) -> RetrievalResult:
        """Retrieve using logistic regression model for fusion."""
        docs, doc_features, aggregations = self._retrieve_with_features(query, keyword_top_k, vector_top_k, filter, aggregation)

        metrics = None
        if usage:
            vector_tokens = ResourceTracker.calculate_vector_search_token_count(
                query, docs[: vector_top_k]
            )
            rerank_tokens = (
                ResourceTracker.calculate_reranker_token_count(query, docs)
                if self.reranker
                else 0
            )
            metrics = TokenMetrics(
                keyword_queries=1,
                vector_tokens=vector_tokens,
                rerank_tokens=rerank_tokens,
                total_tokens=vector_tokens + rerank_tokens
            )

        scores = []
        for feature_list in doc_features:
            if self.lr_model:
                scores.append(self.lr_model.predict(feature_list))
            else:
                scores.append(0)

        scored_docs = list(zip(docs, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        return RetrievalResult(
            documents=scored_docs[:k], aggregations=aggregations, token_metrics=metrics
        )

    def _retrieve_with_features(
            self,
            query: str,
            keyword_top_k: int = 100,
            vector_top_k: int = 100,
            filter: Dict[str, Any] = {},
            aggregation: bool = False
    ) -> Tuple[List[Document], List[List[str]], Optional[Dict]]:
        ks_docs = []
        aggregations = None

        if self.keyword_search:
            ks_docs, aggregations = self.keyword_search.retrieve(
                self.es_data,
                query,
                keyword_top_k,
                filter=filter,
                aggregation=aggregation,
            )
        vs_docs = []
        if self.vector_db:
            vs_docs = self.vector_db.retrieve(
                self.milvus_data, query, self.embeddings, vector_top_k, filter=filter
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

        features_to_use = config_to_features[self.lr_features]

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
            self.vector_db.delete(self.milvus_data, ids=ids, source_id=source_id, **kwargs)
        if self.keyword_search:
            self.keyword_search.delete(self.es_data, ids=ids, source_id=source_id, **kwargs)

    def delete_all(self, delete_index: bool = True):
        """Clear the retriever."""
        if self.vector_db:
            self.vector_db.delete_all(self.milvus_data, delete_index=delete_index)
        if self.keyword_search:
            self.keyword_search.delete_all(self.es_data, delete_index=delete_index)

    def check_indices(self, index_name: str) -> Dict[str, bool]:
        status = {'es_valid': self.keyword_search.check_index(index_name),
                  'vector_valid': self.vector_db.check_index(index_name)}

        return status

    def get_count(self) -> Dict[str, int]:
        stats = {}
        assert self.es_data and self.milvus_data

        stats['es_docs'] = self.keyword_search.get_count(self.es_data.index_name)
        stats['vector_docs'] = self.vector_db.get_count(self.milvus_data)
        return stats

    def list_pids(self, batch_size: int = 1000) -> Dict[str, List[str]]:
        es_pids = []
        vector_pids = []

        try:
            if self.keyword_search and self.es_data:
                es_pids = self.keyword_search.list_pids(self.es_data.index_name)
                logger.info(f"Retrieved {len(es_pids)} PIDs from Elasticsearch")
            if self.vector_db and self.milvus_data:
                vector_pids = self.vector_db.list_pids(self.milvus_data, batch_size)
                logger.info(f"Retrieved {len(vector_pids)} PIDs from Milvus")

            # Find common and unique PIDs
            es_set = set(es_pids)
            vector_set = set(vector_pids)
            common_pids = list(es_set & vector_set)
            es_only_pids = list(es_set - vector_set)
            vector_only_pids = list(vector_set - es_set)

            return {
                'es_pids': es_pids,
                'vector_pids': vector_pids,
                'common_pids': common_pids,
                'es_only_pids': es_only_pids,
                'vector_only_pids': vector_only_pids
            }

        except Exception as e:
            logger.error(f"Error listing PIDs: {e}")
            raise

    def list_indices(self) -> Dict[str, List[str]]:
        """List all indices from both Elasticsearch and Milvus.

        Returns:
            Dict[str, List[str]]: Dictionary containing indices from each source:
            {
                'es_indices': List of Elasticsearch indices,
                'vector_indices': List of Milvus collections,
                'common_indices': List of indices present in both,
                'es_only_indices': Indices only in Elasticsearch,
                'vector_only_indices': Indices only in Milvus
            }
        """
        es_indices = []
        vector_indices = []

        try:
            # Get Elasticsearch indices
            if self.keyword_search:
                es_indices = self.keyword_search.list_indices()
                logger.info(f"Retrieved {len(es_indices)} indices from Elasticsearch")

            # Get Milvus collections
            if self.vector_db:
                vector_indices = self.vector_db.list_indices()
                logger.info(f"Retrieved {len(vector_indices)} collections from Milvus")

            # Find common and unique indices
            es_set = set(es_indices)
            vector_set = set(vector_indices)
            common_indices = list(es_set & vector_set)
            es_only_indices = list(es_set - vector_set)
            vector_only_indices = list(vector_set - es_set)

            return {
                'es_indices': es_indices,
                'vector_indices': vector_indices,
                'common_indices': common_indices,
                'es_only_indices': es_only_indices,
                'vector_only_indices': vector_only_indices
            }

        except Exception as e:
            logger.error(f"Error listing indices: {e}")
            raise
