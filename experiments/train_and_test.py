import logging
import os
import sys
import json

from langchain_core.documents import Document
import xgboost as xgb
from sklearn.datasets import load_svmlight_file
import numpy as np
from sklearn.model_selection import GroupKFold

from denser_retriever.keyword import (
    ElasticKeywordSearch,
    create_elasticsearch_client,
)
from denser_retriever.reranker import HFReranker
from denser_retriever.retriever import DenserRetriever
from denser_retriever.vectordb.milvus import MilvusDenserVectorDB
from denser_retriever.embeddings import SentenceTransformerEmbeddings
from experiments.hf_data_loader import HFDataLoader
from denser_retriever.utils import (
    evaluate,
    save_queries,
    save_qrels,
    load_qrels,
    docs_to_dict,
)
from utils import prepare_xgbdata, save_HF_corpus_as_docs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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


class Experiment:
    def __init__(self, dataset_name, drop_old):
        data_name = os.path.basename(dataset_name)
        self.output_prefix = os.path.join("exps", f"exp_{data_name}")
        self.ingest_bs = 2000
        index_name = data_name.replace("-", "_")
        self.retriever = DenserRetriever(
            index_name=index_name,
            keyword_search=ElasticKeywordSearch(
                top_k=100,
                es_connection=create_elasticsearch_client(url="http://localhost:9200"),
                drop_old=drop_old
            ),
            vector_db=MilvusDenserVectorDB(
                top_k=100,
                connection_args={"uri": "http://localhost:19530"},
                auto_id=True,
                drop_old=drop_old
            ),
            reranker=HFReranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2", top_k=100),
            embeddings=SentenceTransformerEmbeddings(
                "Snowflake/snowflake-arctic-embed-m", 768, False
            ),
            gradient_boost=None
        )

        self.max_doc_size = 0
        self.max_query_size = 8000

    def ingest(self, dataset_name, split):
        corpus, queries, qrels = HFDataLoader(
            hf_repo=dataset_name,
            hf_repo_qrels=None,
            streaming=False,
            keep_in_memory=False,
        ).load(split=split)

        exp_dir = os.path.join(self.output_prefix, split)
        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)

        passage_file = os.path.join(exp_dir, "passages.jsonl")
        save_HF_corpus_as_docs(
            corpus, passage_file, self.max_doc_size
        )

        out = open(passage_file, "r")
        docs = []
        num_docs = 0
        for line in out:
            doc_dict = json.loads(line)
            docs.append(Document(**doc_dict))
            if len(docs) == self.ingest_bs:
                self.retriever.ingest(docs, overwrite_pid=False)
                docs = []
                num_docs += self.ingest_bs
                logger.info(f"Ingested {num_docs} documents")
        if len(docs) > 0:
            self.retriever.ingest(docs, overwrite_pid=False)

    def generate_feature_data(self, dataset_name, split):
        _, queries, qrels = HFDataLoader(
            hf_repo=dataset_name,
            hf_repo_qrels=None,
            streaming=False,
            keep_in_memory=False,
        ).load(split=split)

        exp_dir = os.path.join(self.output_prefix, split)
        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)

        query_file = os.path.join(exp_dir, "queries.jsonl")
        save_queries(queries, query_file)
        qrels_file = os.path.join(exp_dir, "qrels.jsonl")
        save_qrels(qrels, qrels_file)
        feature_file = os.path.join(exp_dir, "features.svmlight")
        feature_out = open(feature_file, "w")

        for i, q in enumerate(queries):
            if (self.max_query_size > 0 and i >= self.max_query_size):
                break
            logger.info(f"Processing query {i}")
            qid = q["id"]

            ks_docs = self.retriever.keyword_search.retrieve(
                q["text"], self.retriever.keyword_search.top_k)
            vs_docs = self.retriever.vector_db.similarity_search_with_score(
                q["text"], self.retriever.vector_db.top_k)
            combined = []
            seen = set()

            for item in ks_docs + vs_docs:
                if item[0].metadata["pid"] not in seen:
                    combined.append(item)
                    seen.add(item[0].metadata["pid"])

            combined_docs = [doc for doc, _ in combined]
            reranked_docs = []
            # import pdb; pdb.set_trace()
            if self.retriever.reranker:
                reranked_docs = self.retriever.reranker.rerank(combined_docs, q["text"])


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
                features.append(f"2:{ks_score_dict.get(pid, -1000)}")  # 2. keyword score
                miss = 0 if pid in ks_rank_dict else 1
                features.append(f"3:{miss}")  # 3. keyword miss

                features.append(f"4:{vs_rank_dict.get(pid, -1)}")  # 4. vector rank
                features.append(f"5:{vs_score_dict.get(pid, -1000)}")  # 5. vector score
                miss = 0 if pid in vs_rank_dict else 1
                features.append(f"6:{miss}")  # 6. vector miss

                assert pid in reranked_rank_dict
                features.append(f"7:{reranked_rank_dict[pid]}")  # 7. rerank rank
                features.append(f"8:{reranked_score_dict[pid]}")  # 8. rerank score
                features.append("9:0")  # 9. placeholder

                features.append(f"# {pid}")
                feature_out.write(" ".join(map(str, features)) + "\n")

    def generate_score_dict(self, qid, pid, rank_pair, score_pair, score_dict):
        if rank_pair.split(":")[1] != "-1":
            score = float(score_pair.split(":")[1])
            if qid not in score_dict:
                score_dict[qid] = {}
            score_dict[qid][pid] = score

    # compute elastic search, vector search, and reranker baselines
    def compute_baselines(self, eval_on):
        # evaluate on test split
        output_prefix = os.path.join(self.output_prefix, eval_on)
        feature_file = os.path.join(output_prefix, "features.svmlight")
        scores_keyword = {}
        scores_vector = {}
        scores_reranker = {}
        for line in open(feature_file, "r"):
            pos = line.index("#")
            assert pos != -1
            pid = line[pos + 1:].strip()
            line = line[:pos]
            comps = line.strip().split(" ")
            qid = comps[1].split(":")[1].strip()
            self.generate_score_dict(qid, pid, comps[2], comps[3], scores_keyword)
            self.generate_score_dict(qid, pid, comps[5], comps[6], scores_vector)
            self.generate_score_dict(qid, pid, comps[8], comps[9], scores_reranker)

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

    def read_group(self, dir, retriever_config):
        group = []
        with open(os.path.join(dir, retriever_config + ".group"), "r") as f:
            data = f.readlines()
            for line in data:
                group.append(int(line.split("\n")[0]))
        return group

    def cross_validation_xgb(self, test_dir, retriever_config):
        group_sizes = self.read_group(test_dir, retriever_config)
        groups = []
        for i, size in enumerate(group_sizes):
            groups.extend([i] * size)
        groups = np.array(groups)

        # Prepare GroupKFold cross-validation
        gkf = GroupKFold(n_splits=3)

        # Initialize an array to hold all predictions
        x_data, y_data = load_svmlight_file(os.path.join(test_dir, retriever_config))
        predictions = np.zeros(x_data.shape[0])

        # Perform cross-validation
        for train_index, valid_index in gkf.split(x_data, y_data, groups):
            x_train, x_valid = x_data[train_index], x_data[valid_index]
            y_train, y_valid = y_data[train_index], y_data[valid_index]

            group_train = groups[train_index]
            group_valid = groups[valid_index]

            # Determine group sizes for training and validation sets
            train_group_sizes = np.diff(
                np.where(np.diff(np.concatenate(([-1], group_train, [-1]))))[0]
            )
            valid_group_sizes = np.diff(
                np.where(np.diff(np.concatenate(([-1], group_valid, [-1]))))[0]
            )

            train_dmatrix = xgb.DMatrix(x_train, y_train)
            valid_dmatrix = xgb.DMatrix(x_valid, y_valid)

            train_dmatrix.set_group(train_group_sizes)
            valid_dmatrix.set_group(valid_group_sizes)

            params = {
                "objective": "rank:ndcg",
                "eta": 0.1,
                "gamma": 1.0,
                "min_child_weight": 0.1,
                "max_depth": 6,
                "eval_metric": "ndcg@10",
            }
            xgb_model = xgb.train(
                params,
                train_dmatrix,
                num_boost_round=100,
                evals=[(valid_dmatrix, "validation")],
            )
            print(xgb_model.get_score(importance_type="gain"))

            pred = xgb_model.predict(valid_dmatrix)
            predictions[valid_index] = pred

        svmlight_file = os.path.join(test_dir, "features.svmlight")
        res = {}
        id = 0
        for line in open(svmlight_file, "r"):
            pos = line.index("#")
            assert pos != -1
            pid = line[pos + 1:].strip()
            line = line[:pos]
            comps = line.strip().split(" ")
            qid = comps[1].split(":")[1].strip()
            if qid not in res:
                res[qid] = {}

            res[qid][pid] = predictions[id]
            id += 1
        assert id == len(predictions)
        logger.info("Evaluate passage results")
        metric_file = os.path.join(test_dir, f"metric_{retriever_config}.json")
        qrels_file = os.path.join(test_dir, "qrels.jsonl")
        qrels = load_qrels(qrels_file)
        metric = evaluate(qrels, res, metric_file)
        ndcg_passage = metric[0]["NDCG@10"]
        logger.info(f"NDCG@10: {ndcg_passage}")

    def train_xgb(self, train_dir, dev_dir, mode_dir, retriever_config):
        x_train, y_train = load_svmlight_file(os.path.join(train_dir, retriever_config))
        x_valid, y_valid = load_svmlight_file(os.path.join(dev_dir, retriever_config))

        group_train = self.read_group(train_dir, retriever_config)
        group_valid = self.read_group(dev_dir, retriever_config)
        train_dmatrix = xgb.DMatrix(x_train, y_train)
        valid_dmatrix = xgb.DMatrix(x_valid, y_valid)

        train_dmatrix.set_group(group_train)
        valid_dmatrix.set_group(group_valid)

        params = {
            "objective": "rank:ndcg",
            "eta": 0.1,
            "gamma": 1.0,
            "min_child_weight": 0.1,
            "max_depth": 6,
            "eval_metric": "ndcg@10",
        }
        xgb_model = xgb.train(
            params,
            train_dmatrix,
            num_boost_round=55,
            evals=[(valid_dmatrix, "validation")],
        )
        print(xgb_model.get_score(importance_type="gain"))

        if not os.path.exists(mode_dir):
            os.makedirs(mode_dir)
        model_name = os.path.join(mode_dir, f"xgb_{retriever_config}.json")
        xgb_model.save_model(model_name)

        return model_name

    def test_xgb(self, model_file, test_dir, retriever_config):
        x_test, y_test = load_svmlight_file(os.path.join(test_dir, retriever_config))
        test_dmatrix = xgb.DMatrix(x_test)
        xgb_model = xgb.Booster()
        xgb_model.load_model(model_file)

        print(xgb_model.get_score(importance_type="gain"))
        pred = xgb_model.predict(test_dmatrix)

        test_svmlight_file = os.path.join(test_dir, "features.svmlight")
        res = {}
        id = 0
        for line in open(test_svmlight_file, "r"):
            pos = line.index("#")
            assert pos != -1
            pid = line[pos + 1:].strip()
            line = line[:pos]
            comps = line.strip().split(" ")
            qid = comps[1].split(":")[1].strip()
            if qid not in res:
                res[qid] = {}

            res[qid][pid] = pred[id].item()
            id += 1
        assert id == len(pred)
        logger.info("Evaluate passage results")
        metric_file = os.path.join(test_dir, f"metric_{retriever_config}.json")
        qrels_file = os.path.join(test_dir, "qrels.jsonl")
        qrels = load_qrels(qrels_file)
        metric = evaluate(qrels, res, metric_file)
        ndcg_passage = metric[0]["NDCG@10"]
        logger.info(f"NDCG@10: {ndcg_passage}")

    def cross_validation(self, eval_on):
        for retriever_config in config_to_features.keys():
            logger.info(f"*** Train retrievers: {retriever_config}")
            features_to_use, features_to_normalize = config_to_features[
                retriever_config
            ]

            prepare_xgbdata(
                os.path.join(self.output_prefix, eval_on),
                retriever_config,
                retriever_config + ".group",
                features_to_use,
                features_to_normalize,
            )

            self.cross_validation_xgb(
                os.path.join(self.output_prefix, eval_on),
                retriever_config,
            )

    def train(self, train_on, eval_on):
        splits = [train_on, eval_on]
        for retriever_config in config_to_features.keys():
            logger.info(f"*** Train retrievers: {retriever_config}")
            features_to_use, features_to_normalize = config_to_features[
                retriever_config
            ]

            for split in splits:
                prepare_xgbdata(
                    os.path.join(self.output_prefix, split),
                    retriever_config,
                    retriever_config + ".group",
                    features_to_use,
                    features_to_normalize,
                )

            # run xgboost training and prediction, print each retriever's ndcg@5 and the combined ndcg@5
            model_dir = os.path.join(self.output_prefix, "models")
            self.train_xgb(
                os.path.join(self.output_prefix, train_on),
                os.path.join(self.output_prefix, eval_on),
                model_dir,
                retriever_config,
            )
        return model_dir

    def test(self, eval_on, model_dir):
        for retriever_config in config_to_features.keys():
            logger.info(f"*** Test retrievers: {retriever_config}")
            features_to_use, features_to_normalize = config_to_features[
                retriever_config
            ]

            # for split in splits:
            prepare_xgbdata(
                os.path.join(self.output_prefix, eval_on),
                retriever_config,
                retriever_config + ".group",
                features_to_use,
                features_to_normalize,
            )

            # run xgboost training and prediction, print each retriever's ndcg@5 and the combined ndcg@5
            self.test_xgb(
                os.path.join(model_dir, f"xgb_{retriever_config}.json"),
                os.path.join(self.output_prefix, eval_on),
                retriever_config,
            )

    def report(self, eval_on):
        for metric_file in [
            "metric_keyword.json",
            "metric_vector.json",
            "metric_reranker.json",
            "metric_es+vs.json",
            "metric_es+rr.json",
            "metric_vs+rr.json",
            "metric_es+vs+rr.json",
            "metric_es+vs_n.json",
            "metric_es+rr_n.json",
            "metric_vs+rr_n.json",
            "metric_es+vs+rr_n.json",
        ]:
            file = os.path.join(self.output_prefix, eval_on, metric_file)
            for line in open(file, "r"):
                line = line.strip()
                if "NDCG@10" in line:
                    print(f"{metric_file}: {line}")
                    break


if __name__ == "__main__":
    # dataset = ["mteb/arguana", "test", "test"]
    # dataset = ["mteb/climate-fever", "test", "test"]
    # dataset = ["mteb/cqadupstack-all", "test", "test"]
    # dataset = ["mteb/dbpedia", "dev", "test"]
    # dataset = ["mteb/fever", "train", "test"]
    # dataset = ["mteb/fiqa", "train", "test"]
    # dataset = ["mteb/hotpotqa", "train", "test"]
    # dataset = ["mteb/msmarco", "train", "dev"]
    # dataset = ["mteb/nfcorpus", "train", "test"]
    # dataset = ["mteb/nq", "test", "test"]
    # dataset = ["mteb/quora", "dev", "test"]
    # dataset = ["mteb/scidocs", "test", "test"]
    # dataset = ["mteb/scifact", "train", "test"]
    # dataset = ["mteb/touche2020", "test", "test"]
    # dataset = ["mteb/trec-covid", "test", "test"]
    # dataset_name, train_on, eval_on = dataset
    # model_dir = "/home/ubuntu/denser_output_retriever/exp_msmarco/models/"

    if len(sys.argv) != 4:
        print(
            "Usage: python train_and_test.py [dataset_name] [train] [test]"
        )
        sys.exit(0)

    dataset_name = sys.argv[1]
    train_on = sys.argv[2]
    eval_on = sys.argv[3]
    drop_old = True
    experiment = Experiment(dataset_name, drop_old)
    if drop_old:
        experiment.ingest(dataset_name, train_on)
    # Generate retriever data, this takes time
    experiment.generate_feature_data(dataset_name, train_on)
    if eval_on != train_on:
        experiment.generate_feature_data(dataset_name, eval_on)
    experiment.compute_baselines(eval_on)
    if train_on == eval_on:
        experiment.cross_validation(eval_on)
    else:
        model_dir = experiment.train(train_on, eval_on)
        experiment.test(eval_on, model_dir)
    logger.info(
        f"train: {train_on}, eval: {eval_on}, cross-validation: {train_on == eval_on}"
    )
    experiment.report(eval_on)
