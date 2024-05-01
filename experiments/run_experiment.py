from generate_retriever_data import generate_data
from denser_retriever.utils import load_denser_qrels
from utils_data import prepare_xgbdata
import yaml
from sklearn.datasets import load_svmlight_file
import xgboost as xgb
from xgboost import DMatrix
import os
import logging
from denser_retriever.utils import (
    evaluate,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

config_to_features = {"base": ["1,2,3,4,5,6,7,8,9", None],
                      "vector": ["4,5,6", None],
                      "normalized": ["1,2,3,4,5,6,7,8,9", "2,5,8"]}


class Experiment():

    def __init__(self, dataset_name, retriever_config):
        self.dataset_name = dataset_name
        self.retriever_config = retriever_config
        config = yaml.safe_load(open("experiments/config.yaml"))
        data_dir_name = os.path.basename(self.dataset_name)
        self.output_prefix = os.path.join(config["output_prefix"], f"exp_{data_dir_name}")

    # Generate svmlight features for elasticsearch (100 passages per query), vector search
    # (100 passages per query) and reranker passages (maximum 200 passages per query)
    def generate_retriever_data(self):
        generate_data(self.dataset_name, "train", ingest=True)
        generate_data(self.dataset_name, "dev", ingest=False)
        generate_data(self.dataset_name, "test", ingest=False)

    def update_scores(self, qid, pid, rank_pair, score_pair, scores_keyword):
        if rank_pair.split(":")[1] != "-1":
            score_keyword = float(score_pair.split(":")[1])
            if qid not in scores_keyword:
                scores_keyword[qid] = {}
            scores_keyword[qid][pid] = score_keyword

    # compute elastic search, vector search, and reranker baselines
    def compute_baselines(self):
        # evaluate on test split
        output_prefix = os.path.join(self.output_prefix, "test")
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
            self.update_scores(qid, pid, comps[2], comps[3], scores_keyword)
            self.update_scores(qid, pid, comps[5], comps[6], scores_vector)
            self.update_scores(qid, pid, comps[8], comps[9], scores_reranker)

        qrels_file = os.path.join(output_prefix, "qrels.jsonl")
        qrels = load_denser_qrels(qrels_file)

        logger.info("Evaluate passage results")
        metric_keyword = evaluate(qrels, scores_keyword, os.path.join(output_prefix, "metric_keyword.json"))
        logger.info(f'Keyword NDCG@10: {metric_keyword[0]["NDCG@10"]}')
        metric_vector = evaluate(qrels, scores_vector, os.path.join(output_prefix, "metric_vector.json"))
        logger.info(f'Vector NDCG@10: {metric_vector[0]["NDCG@10"]}')
        metric_reranker = evaluate(qrels, scores_reranker, os.path.join(output_prefix, "metric_reranker.json"))
        logger.info(f'Reranker NDCG@10: {metric_reranker[0]["NDCG@10"]}')

    def read_group(self, dir, retriever_config):
        group = []
        with open(os.path.join(dir, retriever_config + ".group"), "r") as f:
            data = f.readlines()
            for line in data:
                group.append(int(line.split("\n")[0]))
        return group

    def train_and_test(self, train_dir, dev_dir, test_dir, retriever_config):
        x_train, y_train = load_svmlight_file(os.path.join(train_dir, retriever_config))
        x_valid, y_valid = load_svmlight_file(os.path.join(dev_dir, retriever_config))
        x_test, y_test = load_svmlight_file(os.path.join(test_dir, retriever_config))

        group_train = self.read_group(train_dir, retriever_config)
        group_valid = self.read_group(dev_dir, retriever_config)

        train_dmatrix = DMatrix(x_train, y_train)
        valid_dmatrix = DMatrix(x_valid, y_valid)
        test_dmatrix = DMatrix(x_test)

        train_dmatrix.set_group(group_train)
        valid_dmatrix.set_group(group_valid)

        params = {'objective': 'rank:ndcg', 'eta': 0.1, 'gamma': 1.0,
                  'min_child_weight': 0.1, 'max_depth': 6, 'eval_metric': 'ndcg@5'}
        xgb_model = xgb.train(params, train_dmatrix, num_boost_round=200,
                              evals=[(valid_dmatrix, 'validation')])
        print(xgb_model.get_score(importance_type='gain'))
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
        qrels_file = os.path.join(self.output_prefix, "test", "qrels.jsonl")
        qrels = load_denser_qrels(qrels_file)
        metric = evaluate(qrels, res, metric_file)
        ndcg_passage = metric[0]["NDCG@10"]
        logger.info(f"NDCG@10: {ndcg_passage}")

    def fuse_retrievers(self):
        splits = ["train", "dev", "test"]
        retriever_config = self.retriever_config
        features_to_use, features_to_normalize = config_to_features[retriever_config]

        for split in splits:
            prepare_xgbdata(os.path.join(self.output_prefix, split), retriever_config, retriever_config + ".group",
                            features_to_use,
                            features_to_normalize)

        # run xgboost training and prediction, print each retriever's ndcg@5 and the combined ndcg@5
        self.train_and_test(os.path.join(self.output_prefix, "train"), os.path.join(self.output_prefix, "dev"),
                            os.path.join(self.output_prefix, "test"),
                            retriever_config)


if __name__ == "__main__":
    # dataset_name = "mteb/nfcorpus"
    dataset_name = "mteb/msmarco"
    # retriever_config = "base"
    retriever_config = "normalized"
    experiment = Experiment(dataset_name, retriever_config)

    # Generate retriever data, this takes time
    experiment.generate_retriever_data()

    experiment.compute_baselines()
    experiment.fuse_retrievers()
