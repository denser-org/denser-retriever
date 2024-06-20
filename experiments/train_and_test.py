import logging
import os
import sys

import xgboost as xgb
import yaml
from generate_retriever_data import generate_data
from sklearn.datasets import load_svmlight_file
from utils_data import prepare_xgbdata
from xgboost import DMatrix
from denser_retriever.utils_data import config_to_features
import numpy as np
from sklearn.model_selection import GroupKFold

from denser_retriever.utils import (
    evaluate,
    load_denser_qrels,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Experiment:
    def __init__(self, config_file, dataset_name):
        self.dataset_name = dataset_name
        self.config_file = config_file
        config = yaml.safe_load(open(config_file))
        data_dir_name = os.path.basename(self.dataset_name)
        self.output_prefix = os.path.join(
            config["output_prefix"], f"exp_{data_dir_name}"
        )

    # Generate svmlight features for elasticsearch (100 passages per query), vector search
    # (100 passages per query) and reranker passages (maximum 200 passages per query)
    def generate_retriever_data(self, train_on, eval_on):
        generate_data(self.dataset_name, train_on, self.config_file, ingest=True)
        if eval_on != train_on:
            generate_data(self.dataset_name, eval_on, self.config_file, ingest=False)

    def update_scores(self, qid, pid, rank_pair, score_pair, scores_keyword):
        if rank_pair.split(":")[1] != "-1":
            score_keyword = float(score_pair.split(":")[1])
            if qid not in scores_keyword:
                scores_keyword[qid] = {}
            scores_keyword[qid][pid] = score_keyword

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
            pid = line[pos + 1 :].strip()
            line = line[:pos]
            comps = line.strip().split(" ")
            qid = comps[1].split(":")[1].strip()
            self.update_scores(qid, pid, comps[2], comps[3], scores_keyword)
            self.update_scores(qid, pid, comps[5], comps[6], scores_vector)
            self.update_scores(qid, pid, comps[8], comps[9], scores_reranker)

        qrels_file = os.path.join(output_prefix, "qrels.jsonl")
        qrels = load_denser_qrels(qrels_file)

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
            pid = line[pos + 1 :].strip()
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
        qrels = load_denser_qrels(qrels_file)
        metric = evaluate(qrels, res, metric_file)
        ndcg_passage = metric[0]["NDCG@10"]
        logger.info(f"NDCG@10: {ndcg_passage}")

    def train_xgb(self, train_dir, dev_dir, mode_dir, retriever_config):
        x_train, y_train = load_svmlight_file(os.path.join(train_dir, retriever_config))
        x_valid, y_valid = load_svmlight_file(os.path.join(dev_dir, retriever_config))

        group_train = self.read_group(train_dir, retriever_config)
        group_valid = self.read_group(dev_dir, retriever_config)
        train_dmatrix = DMatrix(x_train, y_train)
        valid_dmatrix = DMatrix(x_valid, y_valid)

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
        test_dmatrix = DMatrix(x_test)
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
            pid = line[pos + 1 :].strip()
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
        qrels = load_denser_qrels(qrels_file)
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
    # config_file = "experiments/config_server.yaml"

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

    if len(sys.argv) != 5:
        print(
            "Usage: python train_and_test.py [config_file] [dataset_name] [train] [test]"
        )
        sys.exit(0)

    config_file = sys.argv[1]
    dataset_name = sys.argv[2]
    train_on = sys.argv[3]
    eval_on = sys.argv[4]

    experiment = Experiment(config_file, dataset_name)

    # Generate retriever data, this takes time
    experiment.generate_retriever_data(train_on, eval_on)
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
