from asyncio.log import logger
from typing import Any, Dict, List, Sequence, Tuple
import uuid

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

from denser_retriever.gradient_boost import DenserGradientBoost
from denser_retriever.keyword import ElasticsearchKeywordSearch
from denser_retriever.reranker import DenserReranker
from denser_retriever.utils import (
    docs_to_dict,
    merge_results,
    min_max_normalize,
    parse_features,
    scale_results,
    standardize_normalize,
)
from denser_retriever.vectordb.base import DenserVectorDB

DEFAULT_EMBEDDINGS = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

config_to_features = {
    "es+vs": ["1,2,3,4,5,6", None],
    "es+rr": ["1,2,3,7,8,9", None],
    "vs+rr": ["4,5,6,7,8,9", None],
    "es+vs+rr": ["1,2,3,4,5,6,7,8,9", None],
    "es+vs_n": ["1,2,3,4,5,6", "2,5"],
    "es+rr_n": ["1,2,3,7,8,9", "2,8"],
    "vs+rr_n": ["4,5,6,7,8,9", "5,8"],
    "es+vs+rr_n": ["1,2,3,4,5,6,7,8,9", "2,5,8"],
}


class DenserRetriever:
    def __init__(
        self,
        index_name: str,
        vector_db: DenserVectorDB,
        keyword_search: ElasticsearchKeywordSearch,
        embeddings: Embeddings,
        gradient_boost: DenserGradientBoost,
        reranker: DenserReranker,
        *,
        embedding_dim: int = 4,
        combine_mode: str = "linear",
        xgb_model_features: str = "es+vs+rr_n",
        filter_fields: List[str] = [],
        keyword_weight: float = 0.5,
        keyword_k: int = 50,
        vector_weight: float = 0.5,
        vector_k: int = 50,
        rerank_weight: float = 0.5,
        rerank_k: int = 50,
    ):
        # config parameters
        self.index_name = index_name
        self.combine_mode = combine_mode
        self.keyword_weight = keyword_weight
        self.keyword_k = keyword_k
        self.vector_weight = vector_weight
        self.vector_k = vector_k
        self.rerank_weight = rerank_weight
        self.rerank_k = rerank_k
        self.xgb_model_features = config_to_features[xgb_model_features]

        # models
        self.embeddings = embeddings
        self.embedding_dim = embedding_dim
        self.gradient_boost = gradient_boost
        self.keyword_search = keyword_search
        self.vector_db = vector_db
        self.reranker = reranker

    # def _init_keyword_store(
    #     self,
    #     filter_fields: List[str] = [],
    # ):
    #     if args["engine"] == "elasticsearch":
    #         self.ks_engine = "elasticsearch"
    #         es_connection = create_elasticsearch_client(
    #             **args["args"],
    #         )
    #         field_types = {}
    #         for field in filter_fields:
    #             comps = field.split(":")
    #             assert len(comps) == 2
    #             field_types[comps[0]] = {"type": comps[-1]}

    #         self.keyword_search = ElasticsearchKeywordSearch(
    #             self.index_name,
    #             es_connection=es_connection,
    #             analysis="default",
    #             field_types=field_types,
    #         )
    #     else:
    #         raise NotImplementedError

    def ingest(self, docs: List[Document]):
        # add pid into metadata for each document
        for _, doc in enumerate(docs):
            doc.metadata["pid"] = uuid.uuid4().hex

        self.keyword_search.add_documents(docs)
        self.vector_db.add_documents(documents=docs)

    def retrieve(
        self, query: str, k: int = 4, filter: Dict[str, Any] = {}, **kwargs: Any
    ):
        if self.combine_mode in ["linear", "rank"]:
            return self._retrieve_by_linear_or_rank(query, k, filter, **kwargs)
        else:
            return self._retrieve_by_model(query, k, filter, **kwargs)

    def _retrieve_by_linear_or_rank(
        self, query: str, k: int = 4, filter: Dict[str, Any] = {}, **kwargs: Any
    ):
        passages = []

        if self.keyword_weight > 0:
            es_docs = self.keyword_search.retrieve(
                query, self.keyword_k, filter=filter, **kwargs
            )
            es_passages = scale_results(es_docs, self.keyword_weight)
            logger.info(f"Keyword search: {len(es_passages)}")
            passages.extend(es_passages)

        if self.vector_weight > 0:
            vector_docs = self.vector_db.similarity_search_with_score(
                query, self.vector_k, filter, **kwargs
            )
            logger.info(f"Vector search: {len(vector_docs)}")

            passages = merge_results(
                passages, vector_docs, 1.0, self.vector_weight, self.combine_mode
            )

        if self.rerank_weight > 0:
            docs = [doc for doc, _ in passages]
            compressed_docs = self.compress_documents(docs, query)
            logger.info(f"Rerank search: {len(compressed_docs)}")
            passages = list(compressed_docs)
        return passages[:k]

    def _retrieve_by_model(
        self, query: str, k: int = 4, filter: Dict[str, Any] = {}, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        docs, doc_features = self._retrieve_with_features(query, filter, **kwargs)

        csr_data = parse_features(doc_features)
        pred = self.gradient_boost.predict(csr_data)

        assert len(pred) == len(docs)
        scores = pred.tolist()

        reranked_docs = list(zip(docs, scores))
        reranked_docs.sort(key=lambda x: x[1], reverse=True)

        return reranked_docs[:k]

    def _retrieve_with_features(
        self, query: str, filter: Dict[str, Any] = {}, **kwargs: Any
    ) -> Tuple[List[Document], List[List[str]]]:
        ks_docs = self.keyword_search.retrieve(query, self.keyword_k, filter, **kwargs)
        vs_docs = self.vector_db.similarity_search_with_score(
            query, k=self.vector_k, filter=filter, **kwargs
        )

        combined = []
        seen = set()
        for item in ks_docs + vs_docs:
            if item[0].page_content not in seen:
                combined.append(item)
                seen.add(item[0].page_content)
        combined_docs = [doc for doc, _ in combined]
        reranked_docs = self.compress_documents(combined_docs, query)

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
            features.append(ks_score_dict.get(pid, -1000))  # 2. keyword score
            miss = 1 if ks_rank_dict.get(pid, -1) == -1 else 0
            features.append(miss)  # 3. keyword miss

            features.append(vs_rank_dict.get(pid, -1))  # 4. vector rank
            features.append(vs_score_dict.get(pid, -1000))  # 5. vector score
            miss = 1 if vs_rank_dict.get(pid, -1) == -1 else 0
            features.append(miss)  # 6. vector miss

            assert pid in reranked_rank_dict
            features.append(reranked_rank_dict[pid])  # 7. rerank rank
            features.append(reranked_score_dict[pid])  # 8. rerank score
            features.append(0)  # 9. placeholder
            doc_features.append(features)

        features_to_use, features_to_normalize = self.xgb_model_features
        features_to_use = features_to_use.split(",")
        features_to_normalize = features_to_normalize.split(",")

        if features_to_normalize:
            features_raw = {f: [] for f in features_to_normalize}
            for data in doc_features:
                for f_name in features_to_normalize:
                    features_raw[f_name].append(float(data[int(f_name)]))

            # normalize features_raw
            standardized_features = {}
            min_max_features = {}
            for f_name in features_to_normalize:
                standardized_features[f_name] = standardize_normalize(
                    features_raw[f_name]
                )
                min_max_features[f_name] = min_max_normalize(features_raw[f_name])

        non_zero_normalized_features = []
        for i, data in enumerate(doc_features):
            features = []
            for f_id in features_to_use:
                f_value = data[int(f_id)]
                if f_value != 0.0:
                    features.append(f"{f_id}:{f_value}")

            if features_to_normalize:
                f_id = len(data[1:]) + 1
                normalized_features = []
                for j, f in enumerate(features_to_normalize):
                    if standardized_features[f][i] != 0.0:
                        normalized_features.append(
                            f"{f_id + 2 * j}:{standardized_features[f][i]}"
                        )
                    if min_max_features[f][i] != 0.0:
                        normalized_features.append(
                            f"{f_id + 2 * j + 1}:{min_max_features[f][i]}"
                        )
                non_zero_normalized_features.append(
                    [data[0]] + features + normalized_features
                )
            else:
                non_zero_normalized_features.append([data[0]] + features)

        return docs, non_zero_normalized_features

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
    ) -> List[Tuple[Document, float]]:
        """
        Rerank documents using CrossEncoder.

        Args:
            documents: A sequence of documents to compress.
            query: The query to use for compressing the documents.
            callbacks: Callbacks to run during the compression process.

        Returns:
            A list of tuples containing the document and its score.
        """
        return self.reranker.compress_documents(documents, query)

    def clear(self):
        """Clear the retriever."""
        self.keyword_search.client.indices.delete(index=self.index_name)

    def get_field_categories(self, field, k: int = 10):
        """
        Get the categories of a field.

        Args:
            field: The field to get the categories of.
            k: The number of categories to return.

        Returns:
            A list of categories.
        """
        return self.keyword_search.get_categories(field, k)

    def get_filter_fields(self):
        """Get the filter fields."""
        return self.keyword_search.get_index_mappings()
