from asyncio.log import logger
from typing import Any, Dict, List, Optional, Tuple
import uuid

from langchain_core.documents import Document

from denser_retriever.core.embeddings import DenserEmbeddings
from denser_retriever.core.keyword import DenserKeywordSearch
from denser_retriever.core.reranker import DenserReranker
from denser_retriever.core.utils import docs_to_dict
from denser_retriever.core.vectordb.base import DenserVectorDB
from denser_retriever.config import FusionConfig
from denser_retriever.core.logistic_regression import LogisticRegression
from denser_retriever.core.utils import config_to_features



class DenserRetriever:
    def __init__(
            self,
            index_name: str,
            keyword_search: Optional[DenserKeywordSearch],
            vector_db: Optional[DenserVectorDB],
            reranker: Optional[DenserReranker],
            embeddings: DenserEmbeddings,
            fusion_config: FusionConfig,
            search_fields: List[str] = [],
            date_fields: List[str] = [],
    ):
        # config parameters
        self.index_name = index_name
        self.fusion_mode = fusion_config.mode
        # models
        self.embeddings = embeddings
        if fusion_config.lr_config:
            self.lr_model = LogisticRegression(fusion_config.lr_config.lr_model)
            self.lr_features = config_to_features[fusion_config.lr_config.lr_features]
        else:
            self.lr_model = None
            self.lr_features = None
        self.keyword_search = keyword_search
        self.vector_db = vector_db
        self.reranker = reranker

        # create index. If exists, remove them first if drop_old is true
        if self.vector_db:
            assert embeddings
            self.vector_db.create_index(index_name, embeddings, search_fields)
        if self.keyword_search:
            self.keyword_search.create_index(index_name, search_fields, date_fields)

    def ingest(self, docs: List[Document], overwrite_pid: bool = True) -> List[str]:
        # add pid into metadata for each document
        if overwrite_pid:
            for _, doc in enumerate(docs):
                doc.metadata["pid"] = uuid.uuid4().hex
        if self.keyword_search:
            logger.info(f"Adding {len(docs)} documents to keyword search")
            self.keyword_search.add_documents(docs)
            logger.info(f"Done adding {len(docs)} documents to keyword search")
        if self.vector_db:
            logger.info(f"Adding {len(docs)} documents to vector db")
            self.vector_db.add_documents(documents=docs)
            logger.info(f"Done adding {len(docs)} documents to vector db")

        return [doc.metadata["pid"] for doc in docs]

    def retrieve(
            self,
            query: str,
            k: int,
            fusion_config: FusionConfig,
            filter: Dict[str, Any] = {},
            aggregation: bool = False
    ):
        """Updated retrieve method to support new fusion modes."""
        logger.info(f"Retrieve query: {query} top_k: {k}")
        if self.fusion_mode == "hybrid":
            return self.retrieve_by_hybrid(query, k, fusion_config, filter, aggregation)
        elif self.fusion_mode == "reranker":
            return self.retrieve_by_reranker(query, k, fusion_config, filter, aggregation)
        elif self.fusion_mode == "model":
            return self.retrieve_by_model(query, k, fusion_config, filter, aggregation)
        else:
            raise ValueError(f"Unknown fusion mode: {self.fusion_mode}")

    def retrieve_by_hybrid(
            self,
            query: str,
            k: int,
            fusion_config: FusionConfig,
            filter: Dict[str, Any] = {},
            aggregation: bool = False
    ) -> List[Tuple[Document, float]]:
        """Hybrid search using keyword and vector positions."""
        # Get keyword search results
        ks_docs, aggregations = self.keyword_search.retrieve(
            query, fusion_config.keyword_top_k, filter=filter, aggregation=aggregation
        )
        # Get vector search results
        vs_docs = self.vector_db.similarity_search_with_score(
            query, fusion_config.vector_top_k, filter=filter
        )

        # Extract position information
        _, _, ks_rank_dict = docs_to_dict(ks_docs)
        _, _, vs_rank_dict = docs_to_dict(vs_docs)

        # Combine all documents
        all_docs = {}
        for doc, _ in ks_docs + vs_docs:
            pid = doc.metadata["pid"]
            if pid not in all_docs:
                all_docs[pid] = doc

        # Calculate hybrid scores
        hybrid_scores = {}
        max_rank = max(fusion_config.keyword_top_k, fusion_config.vector_top_k)

        for pid, doc in all_docs.items():
            # Get ranks (default to max_rank + 1 if not found)
            ks_rank = ks_rank_dict.get(pid, max_rank + 1)
            vs_rank = vs_rank_dict.get(pid, max_rank + 1)

            # Calculate reciprocal rank fusion score
            hybrid_scores[pid] = 0.0
            if ks_rank <= max_rank:
                hybrid_scores[pid] += 1.0 / (ks_rank + 60)  # constant from paper
            if vs_rank <= max_rank:
                hybrid_scores[pid] += 1.0 / (vs_rank + 60)

        # Create final scored list
        scored_docs = [(all_docs[pid], score) for pid, score in hybrid_scores.items()]
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        return scored_docs[:k], aggregations

    def retrieve_by_reranker(
            self,
            query: str,
            k: int,
            fusion_config: FusionConfig,
            filter: Dict[str, Any] = {},
            aggregation: bool = False
    ) -> List[Tuple[Document, float]]:
        """Two-stage retrieval: keyword search followed by reranking."""
        # First stage: keyword search
        ks_docs, aggregations = self.keyword_search.retrieve(
            query, fusion_config.keyword_top_k, filter=filter, aggregation=aggregation
        )

        # Extract documents for reranking
        docs_to_rerank = [doc for doc, _ in ks_docs]

        # Second stage: reranking
        if self.reranker and docs_to_rerank:
            reranked_docs = self.reranker.rerank(docs_to_rerank, query)
            return reranked_docs[:k], aggregations

        return ks_docs[:k], aggregations

    def retrieve_by_model(
            self,
            query: str,
            k: int,
            fusion_config: FusionConfig,
            filter: Dict[str, Any] = {},
            aggregation: bool = False
    ) -> List[Tuple[Document, float]]:
        """Retrieve using logistic regression model for fusion."""
        docs, doc_features, aggregations = self._retrieve_with_features(
            query, fusion_config, filter, aggregation
        )
        scores = []

        for feature_list in doc_features:
            scores.append(self.lr_model.predict(feature_list))

        # Combine with documents
        scored_docs = list(zip(docs, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        return scored_docs[:k], aggregations

    def _retrieve_with_features(
            self,
            query: str,
            fusion_config: FusionConfig,
            filter: Dict[str, Any] = {},
            aggregation: bool = False
    ) -> Tuple[List[Document], List[List[str]]]:
        ks_docs = []
        aggregations = None

        if self.keyword_search:
            ks_docs, aggregations = self.keyword_search.retrieve(
                query, fusion_config.keyword_top_k, filter=filter, aggregation=aggregation
            )
        vs_docs = []
        if self.vector_db:
            vs_docs = self.vector_db.similarity_search_with_score(
                query, fusion_config.vector_top_k, filter=filter
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

    def delete_all(self):
        """Clear the retriever."""
        if self.vector_db:
            self.vector_db.delete_all()
        if self.keyword_search:
            self.keyword_search.delete_all()

    def get_filter_fields(self):
        """Get the filter fields."""
        if not self.keyword_search:
            raise ValueError("Keyword search not initialized")
        return self.keyword_search.get_index_mappings()
