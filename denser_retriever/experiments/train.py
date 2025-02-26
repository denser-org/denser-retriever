import logging
import os
import sys
import json
import argparse
from typing import Dict

from langchain_core.documents import Document
from sklearn.datasets import load_svmlight_file
from sklearn.linear_model import LogisticRegression
import joblib  # for saving/loading the model

from denser_retriever.core.retriever import DenserRetriever
from denser_retriever.experiments.hf_data_loader import HFDataLoader
from denser_retriever.core.utils import (
    evaluate,
    save_queries,
    save_qrels,
    load_qrels,
    docs_to_dict,
)
from denser_retriever.experiments.utils import prepare_features, save_HF_corpus_as_docs
from denser_retriever.core.utils import config_to_features, load_queries
from denser_retriever.core.keyword import ESIndexData
from denser_retriever.core.vectordb.milvus import MilvusIndexData
from denser_retriever.core.shared import SharedComponents

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DenserData:
    def __init__(self, dir_path):
        self.data_dir = dir_path

    def load_queries(self):
        queries = load_queries(os.path.join(self.data_dir, 'queries.jsonl'))
        return queries

    def load_qrels(self):
        qrels = load_qrels(os.path.join(self.data_dir, 'qrels.jsonl'))
        return qrels


class Experiment:
    def __init__(self, dataset_name, drop_old, config_path):
        data_name = os.path.basename(dataset_name)
        self.output_prefix = os.path.join("exps", f"exp_{data_name}")

        # Read the config file
        with open(config_path) as f:
            config = json.load(f)
        self.ingest_bs = config["ingest_bs"]
        self.max_query_size = config["max_query_size"]
        self.max_query_len = config["max_query_len"]
        self.max_doc_size = config["max_doc_size"]
        self.max_doc_len = config["max_doc_len"]
        self.es_top_k = config["keyword_top_k"]
        self.vector_top_k = config["vector_top_k"]
        self.reranker_top_k = config["reranker_top_k"]

        # Initialize retriever with config
        index_name = data_name.replace("-", "_")
        es_data = ESIndexData(
            index_name=index_name,
            analysis="default",
            drop_old=drop_old
        )
        milvus_data = MilvusIndexData(
            index_name=index_name,
            embedding_size=int(config["embedding"]["size"]),
            drop_old=drop_old
        )
        shared_components = SharedComponents.initialize_from_config(config_path)
        self.retriever = DenserRetriever(
            shared=shared_components,
            es_data=es_data,
            milvus_data=milvus_data
        )

    def ingest(self, dataset_name: str, split: str) -> Dict[str, float]:
        """Ingest dataset and return cost metrics."""
        # Create output directory
        exp_dir = os.path.join(self.output_prefix, split)
        os.makedirs(exp_dir, exist_ok=True)

        # Load and save corpus
        passage_file = os.path.join(exp_dir, "passages.jsonl")
        corpus, _, _ = HFDataLoader(
            hf_repo=dataset_name,
            hf_repo_qrels=None,
            streaming=False,
            keep_in_memory=False
        ).load(split=split)
        save_HF_corpus_as_docs(corpus, passage_file, self.max_doc_size, self.max_doc_len)

        docs = []
        num_docs = 0

        with open(passage_file, "r") as f:
            for line in f:
                docs.append(Document(**json.loads(line)))
                num_docs += 1
                if len(docs) == self.ingest_bs:
                    self.retriever.ingest(docs, overwrite_pid=False)
                    docs = []
                    logger.info(f"Ingested {num_docs} documents.")
            if docs:
                self.retriever.ingest(docs, overwrite_pid=False)

    def generate_feature_data(self, dataset_name, split):
        exp_dir = os.path.join(self.output_prefix, split)
        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)
        query_file = os.path.join(exp_dir, "queries.jsonl")
        qrels_file = os.path.join(exp_dir, "qrels.jsonl")

        _, queries, qrels = HFDataLoader(
            hf_repo=dataset_name,
            hf_repo_qrels=None,
            streaming=False,
            keep_in_memory=False,
        ).load(split=split)
        save_queries(queries, query_file)
        save_qrels(qrels, qrels_file)

        feature_file = os.path.join(exp_dir, "features.svmlight")
        feature_out = open(feature_file, "w")

        for i, q in enumerate(queries):
            if (self.max_query_size > 0 and i >= self.max_query_size):
                break
            logger.info(f"Processing query {i}")
            query_str = q["text"]
            if (self.max_query_len > 0 and len(query_str) > self.max_query_len):
                query_str = query_str[:self.max_query_len]
            qid = q["id"]
            ks_docs, ks_aggregations = self.retriever.keyword_search.retrieve(self.retriever.es_data,
                                                                              query_str, self.es_top_k)
            vs_docs = self.retriever.vector_db.retrieve(self.retriever.milvus_data,
                                                        query_str, self.retriever.embeddings, self.vector_top_k)
            combined = []
            seen = set()

            for item in ks_docs + vs_docs:
                if item[0].metadata["pid"] not in seen:
                    combined.append(item)
                    seen.add(item[0].metadata["pid"])

            combined_docs = [doc for doc, _ in combined]
            reranked_docs = []
            if self.retriever.reranker:
                reranked_docs = self.retriever.reranker.rerank(combined_docs, query_str)

            _, ks_score_dict, ks_rank_dict = docs_to_dict(ks_docs)
            _, vs_score_dict, vs_rank_dict = docs_to_dict(vs_docs)
            reranked_docs_dict, reranked_score_dict, reranked_rank_dict = docs_to_dict(
                reranked_docs
            )

            labels = qrels[qid]
            for pid in reranked_docs_dict.keys():
                features = []
                label = labels.get(pid, 0)
                features.append(str(label))
                features.append(f"qid:{qid}")
                features.append(f"1:{ks_rank_dict.get(pid, -1)}")  # 1. keyword rank
                features.append(f"2:{ks_score_dict.get(pid, 0)}")  # 2. keyword score
                miss = 0 if pid in ks_rank_dict else 1
                features.append(f"3:{miss}")  # 3. keyword miss

                features.append(f"4:{vs_rank_dict.get(pid, -1)}")  # 4. vector rank
                features.append(f"5:{vs_score_dict.get(pid, 0)}")  # 5. vector score
                miss = 0 if pid in vs_rank_dict else 1
                features.append(f"6:{miss}")  # 6. vector miss

                assert pid in reranked_rank_dict
                features.append(f"7:{reranked_rank_dict[pid]}")  # 7. rerank rank
                features.append(f"8:{reranked_score_dict[pid]}")  # 8. rerank score
                features.append("9:0")  # 9. placeholder

                features.append(f"# {pid}")
                feature_out.write(" ".join(map(str, features)) + "\n")

    def generate_score_dict(self, qid, pid, rank_pair, score_pair, score_dict, pred_out):
        if rank_pair.split(":")[1] != "-1":
            score = float(score_pair.split(":")[1])
            if qid not in score_dict:
                score_dict[qid] = {}
            score_dict[qid][pid] = score
            pred_out.write(f"{qid} {pid} {score_dict[qid][pid]}\n")

    # compute elastic search, vector search, and reranker baselines
    def compute_baselines(self, eval_on):
        # evaluate on test split
        output_prefix = os.path.join(self.output_prefix, eval_on)
        feature_file = os.path.join(output_prefix, "features.svmlight")
        scores_keyword = {}
        scores_vector = {}
        scores_reranker = {}
        keyword_out = open(os.path.join(output_prefix, "keyword.pred"), "w")
        vector_out = open(os.path.join(output_prefix, "vector.pred"), "w")
        reranker_out = open(os.path.join(output_prefix, "reranker.pred"), "w")
        for line in open(feature_file, "r"):
            pos = line.index("#")
            assert pos != -1
            pid = line[pos + 1:].strip()
            line = line[:pos]
            comps = line.strip().split(" ")
            qid = comps[1].split(":")[1].strip()
            self.generate_score_dict(qid, pid, comps[2], comps[3], scores_keyword, keyword_out)
            self.generate_score_dict(qid, pid, comps[5], comps[6], scores_vector, vector_out)
            self.generate_score_dict(qid, pid, comps[8], comps[9], scores_reranker, reranker_out)

        qrels_file = os.path.join(output_prefix, "qrels.jsonl")
        qrels = load_qrels(qrels_file)

        logger.info("Evaluate passage results")
        metric_keyword = evaluate(
            qrels, scores_keyword, os.path.join(output_prefix, "metric_keyword.json")
        )
        logger.info(f'Keyword NDCG@10: {metric_keyword[0]["NDCG@10"]}')
        metric_vector = evaluate(
            qrels, scores_vector, os.path.join(output_prefix, "metric_vector.json")
        )
        logger.info(f'Vector NDCG@10: {metric_vector[0]["NDCG@10"]}')
        metric_reranker = evaluate(
            qrels, scores_reranker, os.path.join(output_prefix, "metric_reranker.json")
        )
        logger.info(f'Reranker NDCG@10: {metric_reranker[0]["NDCG@10"]}')

    def train_logistic(self, train_dir, dev_dir, model_dir, retriever_config):
        """Train logistic regression model"""
        # Load training data
        x_train, y_train = load_svmlight_file(os.path.join(train_dir, retriever_config))
        x_valid, y_valid = load_svmlight_file(os.path.join(dev_dir, retriever_config))

        # Train logistic regression
        model = LogisticRegression(max_iter=1000, class_weight='balanced')
        model.fit(x_train.toarray(), y_train)

        # Save feature importance scores
        importance = dict(zip(range(1, x_train.shape[1] + 1),
                              abs(model.coef_[0])))
        print("Feature importance:", importance)

        # Save model weights in readable format
        feature_names = [
            "keyword_rank", "keyword_score", "keyword_miss",
            "vector_rank", "vector_score", "vector_miss",
            "reranker_rank", "reranker_score", "placeholder"
        ]

        weights_dict = {
            'features': feature_names,
            'weights': model.coef_[0].tolist(),
            'intercept': model.intercept_[0],
            'feature_importance': dict(zip(feature_names, abs(model.coef_[0])))
        }

        # Save readable weights
        weights_path = os.path.join(model_dir, f"weights_{retriever_config}.json")
        with open(weights_path, 'w') as f:
            json.dump(weights_dict, f, indent=2)

        # Save model
        model_path = os.path.join(model_dir, f"logistic_{retriever_config}.joblib")
        joblib.dump(model, model_path)
        return model_path

    def test_logistic(self, model_file, test_dir, retriever_config):
        """Test logistic regression model"""
        # Load test data and model
        x_test, y_test = load_svmlight_file(os.path.join(test_dir, retriever_config))
        model = joblib.load(model_file)

        # Get predictions
        pred_proba = model.predict_proba(x_test.toarray())[:, 1]  # Get probability of positive class

        # Save predictions and evaluate
        test_svmlight_file = os.path.join(test_dir, "features.svmlight")
        pred_file = open(os.path.join(test_dir, f"{retriever_config}.pred"), "w")
        res = {}
        id = 0
        for line in open(test_svmlight_file, "r"):
            pos = line.index("#")
            pid = line[pos + 1:].strip()
            line = line[:pos]
            comps = line.strip().split(" ")
            qid = comps[1].split(":")[1].strip()

            if qid not in res:
                res[qid] = {}

            res[qid][pid] = pred_proba[id]
            pred_file.write(f"{qid} {pid} {res[qid][pid]}\n")
            id += 1

        # Evaluate results
        logger.info("Evaluate passage results")
        metric_file = os.path.join(test_dir, f"metric_{retriever_config}.json")
        qrels_file = os.path.join(test_dir, "qrels.jsonl")
        qrels = load_qrels(qrels_file)
        metric = evaluate(qrels, res, metric_file)
        ndcg_passage = metric[0]["NDCG@10"]
        logger.info(f"NDCG@10: {ndcg_passage}")

    def train(self, train_on, eval_on):
        """Train logistic regression models for different feature combinations"""
        for retriever_config in config_to_features.keys():
            logger.info(f"*** Train retrievers: {retriever_config}")
            features_to_use = config_to_features[retriever_config]
            # import pdb; pdb.set_trace()
            # Prepare data for both splits
            for split in [train_on, eval_on]:
                prepare_features(
                    os.path.join(self.output_prefix, split),
                    retriever_config,
                    retriever_config + ".group",
                    features_to_use
                )

            # Train logistic regression
            model_dir = os.path.join(self.output_prefix, "models")
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)

            self.train_logistic(
                os.path.join(self.output_prefix, train_on),
                os.path.join(self.output_prefix, eval_on),
                model_dir,
                retriever_config,
            )
        return model_dir

    def test(self, eval_on, model_dir):
        """Test logistic regression models"""
        for retriever_config in config_to_features.keys():
            logger.info(f"*** Test retrievers: {retriever_config}")
            features_to_use = config_to_features[retriever_config]

            prepare_features(
                os.path.join(self.output_prefix, eval_on),
                retriever_config,
                retriever_config + ".group",
                features_to_use
            )

            self.test_logistic(
                os.path.join(model_dir, f"logistic_{retriever_config}.joblib"),
                os.path.join(self.output_prefix, eval_on),
                retriever_config,
            )

    def report(self, eval_on, metric_str):
        print(f"\n== {metric_str}")
        for metric_file in [
            "metric_keyword.json",
            "metric_vector.json",
            "metric_reranker.json",
            "metric_es+vs.json",
            "metric_es+rr.json",
            "metric_vs+rr.json",
            "metric_es+vs+rr.json"
        ]:
            file = os.path.join(self.output_prefix, eval_on, metric_file)
            for line in open(file, "r"):
                line = line.strip()
                if metric_str in line:
                    print(f"{metric_file}: {line}")
                    break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate retriever methods")
    parser.add_argument("dataset_name", help="Name of the dataset")
    parser.add_argument("train", help="Training split name")
    parser.add_argument("test", help="Test split name")
    parser.add_argument("--config", help="Path to config file")
    parser.add_argument("--ingest-only", action="store_true", help="Only perform data ingestion")

    args = parser.parse_args()

    drop_old = True
    experiment = Experiment(args.dataset_name, drop_old, args.config)
    if drop_old:
        experiment.ingest(args.dataset_name, args.train)

    if args.ingest_only:
        logger.info("Ingest-only mode. Stopping after data ingestion.")
        sys.exit(0)

    # Continue with feature generation and training if not ingest-only
    experiment.generate_feature_data(args.dataset_name, args.train)
    if args.test != args.train:
        experiment.generate_feature_data(args.dataset_name, args.test)
    experiment.compute_baselines(args.test)

    model_dir = experiment.train(args.train, args.test)
    experiment.test(args.test, model_dir)

    logger.info(f"train: {args.train}, eval: {args.test}")
    experiment.report(args.test, "NDCG@10")
