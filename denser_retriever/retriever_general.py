import time

from denser_retriever.reranker import Reranker
from denser_retriever.retriever import Retriever
from denser_retriever.retriever_elasticsearch import RetrieverElasticSearch
from denser_retriever.retriever_milvus import RetrieverMilvus
from denser_retriever.settings import RetrieverSettings
from denser_retriever.utils import (
    aggregate_passages,
    get_logger,
    merge_results,
    scale_results,
)
from denser_retriever.utils import build_dicts, parse_features
from denser_retriever.utils_data import config_to_features
from denser_retriever.utils import standardize_normalize, min_max_normalize
import xgboost as xgb

logger = get_logger(__name__)


class RetrieverGeneral(Retriever):
    """
    General Retriever
    """

    settings: RetrieverSettings

    def __init__(self, index_name: str, config_path: str = "config.yaml"):
        super().__init__(index_name, config_path)
        self.retrieve_type = "general"
        self.retrieverElasticSearch = (
            RetrieverElasticSearch(index_name, config_path)
            if self.settings.keyword_weight > 0
            else None
        )
        self.retrieverMilvus = (
            RetrieverMilvus(index_name, config_path)
            if self.settings.vector_weight > 0
            else None
        )
        self.reranker = (
            Reranker(self.settings.rerank.rerank_model)
            if self.settings.rerank_weight > 0
            else None
        )
        if self.settings.combine == "model":
            self.xgb_model = xgb.Booster()
            self.xgb_model.load_model(self.settings.model)

    def ingest(self, doc_or_passage_file):
        # import pdb; pdb.set_trace()
        if self.settings.keyword_weight > 0:
            self.retrieverElasticSearch.ingest(
                doc_or_passage_file, self.settings.keyword.es_ingest_passage_bs
            )
            logger.info("Done building ES index")
        if self.settings.vector_weight > 0:
            self.retrieverMilvus.ingest(
                doc_or_passage_file, self.settings.vector.vector_ingest_passage_bs
            )
            logger.info("Done building Vector DB index")

    def retrieve(self, query, meta_data, query_id=None):
        if self.settings.combine in ["linear", "rank"]:
            passages = self.retrieve_by_linear_or_rank(query, meta_data, query_id)
        else:  # model-based
            passages = self.retrieve_by_model(query, meta_data, query_id)
        docs = aggregate_passages(passages)
        return passages, docs

    def retrieve_by_linear_or_rank(self, query, meta_data, query_id=None):
        passages = []
        if self.settings.keyword_weight > 0:
            start_time = time.time()
            passages_es = self.retrieverElasticSearch.retrieve(
                query, meta_data, query_id
            )
            passages = scale_results(passages_es, self.settings.keyword_weight)
            retrieve_time_sec = time.time() - start_time
            logger.info(
                f"ElasticSearch passages: {len(passages)} time: {retrieve_time_sec:.3f} sec. "
            )
        if self.settings.vector_weight > 0:
            start_time = time.time()
            passages_vector = self.retrieverMilvus.retrieve(query, meta_data, query_id)
            logger.info(
                f"Vector Search before merging Passages 1: {len(passages)} Passages 2: {len(passages_vector)}"
            )
            passages = merge_results(
                passages,
                passages_vector,
                1.0,
                self.settings.vector_weight,
                self.settings.combine,
            )
            logger.info(f"After merging Passages: {len(passages)}")
            retrieve_time_sec = time.time() - start_time
            logger.info(f"Vector Retrieve time: {retrieve_time_sec:.3f} sec.")
        if self.settings.rerank_weight > 0:
            start_time = time.time()
            passages = passages[: self.settings.rerank.topk]
            passages_rerank = self.reranker.rerank(
                query, passages, self.settings.rerank.rerank_bs, query_id
            )
            passages = merge_results(
                passages,
                passages_rerank,
                1.0,
                self.settings.rerank_weight,
                self.settings.combine,
            )
            rerank_time_sec = time.time() - start_time
            logger.info(f"Rerank time: {rerank_time_sec:.3f} sec.")

            if len(passages) > self.settings.rerank.topk:
                passages = passages[: self.settings.rerank.topk]

        return passages

    # get passages extract features, run model
    def retrieve_by_model(self, query, meta_data, query_id=None):
        passages, passage_features = self.retrieve_and_featurize_passages(
            query, meta_data
        )
        # Convert features to CSR matrix
        csr_data = parse_features(passage_features)
        # Create DMatrix
        test_dmatrix = xgb.DMatrix(csr_data)
        pred = self.xgb_model.predict(test_dmatrix)
        assert len(passages) == len(pred)
        for i, passage in enumerate(passages):
            passage["score"] = pred[i].item()
        passages.sort(key=lambda x: x["score"], reverse=True)
        return passages

    def retrieve_and_featurize_passages(self, query, meta_data):
        passages_keyword = self.retrieverElasticSearch.retrieve(query, meta_data)
        passages_vector = self.retrieverMilvus.retrieve(query, meta_data)
        combined_passages = []
        seen_ids = set()

        # Combine both passages
        for passage in passages_keyword + passages_vector:
            if passage["source"] not in seen_ids:
                combined_passages.append(passage)
                seen_ids.add(passage["source"])

        uid_to_passages_1, uid_to_scores_1, uid_to_ranks_1 = build_dicts(
            passages_keyword
        )
        uid_to_passages_2, uid_to_scores_2, uid_to_ranks_2 = build_dicts(
            passages_vector
        )

        passages_reranked = self.reranker.rerank(
            query, combined_passages, self.settings.rerank.rerank_bs
        )
        uid_to_passages_reranked, uid_to_scores_reranked, uid_to_ranks_reranked = (
            build_dicts(passages_reranked)
        )

        passages, passage_features = [], []
        for pid in uid_to_passages_reranked.keys():
            passages.append(uid_to_passages_reranked[pid])
            features = []
            features.append(0)  # placeholder label
            rank = uid_to_ranks_1.get(pid, -1)
            features.append(rank)  # 1
            features.append(uid_to_scores_1.get(pid, -1000))  # 2
            miss = 1 if rank == -1 else 0
            features.append(miss)  # 3

            rank = uid_to_ranks_2.get(pid, -1)
            features.append(rank)  # 4
            features.append(uid_to_scores_2.get(pid, -1000))  # 5
            miss = 1 if rank == -1 else 0
            features.append(miss)  # 6

            assert pid in uid_to_ranks_reranked
            features.append(uid_to_ranks_reranked[pid])  # 7
            features.append(uid_to_scores_reranked[pid])  # 8
            features.append(0)  # 9
            passage_features.append(features)

        # import pdb; pdb.set_trace()
        retriever_config = self.settings.model_features
        features_to_use, features_to_normalize = config_to_features[retriever_config]
        features_to_use = features_to_use.split(",")
        features_to_normalize = features_to_normalize.split(",")

        if features_to_normalize:
            features_raw = {f: [] for f in features_to_normalize}
            for data in passage_features:
                for f_name in features_to_normalize:
                    features_raw[f_name].append(float(data[int(f_name)]))

            # normalized features_raw
            features_standardize = {}
            features_min_max = {}
            for f in features_to_normalize:
                features_standardize[f] = standardize_normalize(features_raw[f])
                features_min_max[f] = min_max_normalize(features_raw[f])

        # import pdb; pdb.set_trace()
        non_zero_normalized_features = []
        for i, data in enumerate(passage_features):
            # only include nonzero features
            feats = []
            for fid in features_to_use:
                fval = data[int(fid)]
                if fval != 0.0:
                    feats.append(f"{fid}:{fval}")

            if features_to_normalize:
                f_id = len(data[1:]) + 1
                feats_normalized = []
                for j, f in enumerate(features_to_normalize):
                    if features_standardize[f][i] != 0.0:
                        feats_normalized.append(
                            f"{f_id + 2 * j}:{features_standardize[f][i]}"
                        )
                    if features_min_max[f][i] != 0.0:
                        feats_normalized.append(
                            f"{f_id + 2 * j + 1}:{features_min_max[f][i]}"
                        )
                non_zero_normalized_features.append(
                    [data[0]] + feats + feats_normalized
                )
            else:
                non_zero_normalized_features.append([data[0]] + feats)

        return passages, non_zero_normalized_features

    def get_field_categories(self, field, topk):
        return self.retrieverElasticSearch.get_categories(field, topk)
