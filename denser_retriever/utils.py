import copy
import json
import logging
import sys
from typing import Dict, List, Tuple

import pytrec_eval
import torch
import numpy as np
from scipy.sparse import csr_matrix


def cos_sim(a, b):
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))


def dot_score(a: torch.Tensor, b: torch.Tensor):
    """
    Computes the dot-product dot_prod(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = dot_prod(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    return torch.mm(a, b.transpose(0, 1))


# From https://github.com/beir-cellar/beir/blob/f062f038c4bfd19a8ca942a9910b1e0d218759d4/beir/retrieval/custom_metrics.py#L4
def mrr(
    qrels: Dict[str, Dict[str, int]],
    results: Dict[str, Dict[str, float]],
    k_values: List[int],
) -> Tuple[Dict[str, float]]:
    MRR = {}

    for k in k_values:
        MRR[f"MRR@{k}"] = 0.0

    k_max, top_hits = max(k_values), {}
    logging.info("\n")

    for query_id, doc_scores in results.items():
        top_hits[query_id] = sorted(
            doc_scores.items(), key=lambda item: item[1], reverse=True
        )[0:k_max]

    for query_id in top_hits:
        query_relevant_docs = set(
            [doc_id for doc_id in qrels[query_id] if qrels[query_id][doc_id] > 0]
        )
        for k in k_values:
            for rank, hit in enumerate(top_hits[query_id][0:k]):
                if hit[0] in query_relevant_docs:
                    MRR[f"MRR@{k}"] += 1.0 / (rank + 1)
                    break

    for k in k_values:
        MRR[f"MRR@{k}"] = round(MRR[f"MRR@{k}"] / len(qrels), 5)
        logging.info("MRR@{}: {:.4f}".format(k, MRR[f"MRR@{k}"]))

    return MRR


def recall_cap(
    qrels: Dict[str, Dict[str, int]],
    results: Dict[str, Dict[str, float]],
    k_values: List[int],
) -> Tuple[Dict[str, float]]:
    capped_recall = {}

    for k in k_values:
        capped_recall[f"R_cap@{k}"] = 0.0

    k_max = max(k_values)
    logging.info("\n")

    for query_id, doc_scores in results.items():
        top_hits = sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)[
            0:k_max
        ]
        query_relevant_docs = [
            doc_id for doc_id in qrels[query_id] if qrels[query_id][doc_id] > 0
        ]
        for k in k_values:
            retrieved_docs = [
                row[0] for row in top_hits[0:k] if qrels[query_id].get(row[0], 0) > 0
            ]
            denominator = min(len(query_relevant_docs), k)
            capped_recall[f"R_cap@{k}"] += len(retrieved_docs) / denominator

    for k in k_values:
        capped_recall[f"R_cap@{k}"] = round(capped_recall[f"R_cap@{k}"] / len(qrels), 5)
        logging.info("R_cap@{}: {:.4f}".format(k, capped_recall[f"R_cap@{k}"]))

    return capped_recall


def hole(
    qrels: Dict[str, Dict[str, int]],
    results: Dict[str, Dict[str, float]],
    k_values: List[int],
) -> Tuple[Dict[str, float]]:
    Hole = {}

    for k in k_values:
        Hole[f"Hole@{k}"] = 0.0

    annotated_corpus = set()
    for _, docs in qrels.items():
        for doc_id, score in docs.items():
            annotated_corpus.add(doc_id)

    k_max = max(k_values)
    logging.info("\n")

    for _, scores in results.items():
        top_hits = sorted(scores.items(), key=lambda item: item[1], reverse=True)[
            0:k_max
        ]
        for k in k_values:
            hole_docs = [
                row[0] for row in top_hits[0:k] if row[0] not in annotated_corpus
            ]
            Hole[f"Hole@{k}"] += len(hole_docs) / k

    for k in k_values:
        Hole[f"Hole@{k}"] = round(Hole[f"Hole@{k}"] / len(qrels), 5)
        logging.info("Hole@{}: {:.4f}".format(k, Hole[f"Hole@{k}"]))

    return Hole


def top_k_accuracy(
    qrels: Dict[str, Dict[str, int]],
    results: Dict[str, Dict[str, float]],
    k_values: List[int],
) -> Tuple[Dict[str, float]]:
    top_k_acc = {}

    for k in k_values:
        top_k_acc[f"Accuracy@{k}"] = 0.0

    k_max, top_hits = max(k_values), {}
    logging.info("\n")

    for query_id, doc_scores in results.items():
        top_hits[query_id] = [
            item[0]
            for item in sorted(
                doc_scores.items(), key=lambda item: item[1], reverse=True
            )[0:k_max]
        ]

    for query_id in top_hits:
        query_relevant_docs = set(
            [doc_id for doc_id in qrels[query_id] if qrels[query_id][doc_id] > 0]
        )
        for k in k_values:
            for relevant_doc_id in query_relevant_docs:
                if relevant_doc_id in top_hits[query_id][0:k]:
                    top_k_acc[f"Accuracy@{k}"] += 1.0
                    break

    for k in k_values:
        top_k_acc[f"Accuracy@{k}"] = round(top_k_acc[f"Accuracy@{k}"] / len(qrels), 5)
        logging.info("Accuracy@{}: {:.4f}".format(k, top_k_acc[f"Accuracy@{k}"]))

    return top_k_acc


loggers = {}


def get_logger(name="default"):
    global loggers
    if loggers.get(name):
        return loggers.get(name)
    else:
        logger = logging.getLogger(name)
        logger.propagate = False
        logger.setLevel(logging.INFO)

        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)

        formatter = logging.Formatter(
            fmt="%(asctime)s %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        loggers[name] = logger
        return logger


def get_logger_file(name):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s",
        datefmt="%m-%d %H:%M",
        filename=name,
        filemode="a",
    )
    console = logging.StreamHandler()
    logging.getLogger().addHandler(console)
    return logging


def evaluate(
    qrels: Dict[str, Dict[str, int]],
    results: Dict[str, Dict[str, float]],
    metric_file: str = None,
    k_values: List[int] = [1, 3, 5, 10, 100, 1000],
    ignore_identical_ids: bool = True,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float]]:
    if ignore_identical_ids:
        print(
            "For evaluation, we ignore identical query and document ids (default), please explicitly set ``ignore_identical_ids=False`` to ignore this."  # noqa: E501
        )
        popped = []
        for qid, rels in results.items():
            for pid in list(rels):
                if qid == pid:
                    results[qid].pop(pid)
                    popped.append(pid)

    ndcg = {}
    _map = {}
    recall = {}
    precision = {}

    for k in k_values:
        ndcg[f"NDCG@{k}"] = 0.0
        _map[f"MAP@{k}"] = 0.0
        recall[f"Recall@{k}"] = 0.0
        precision[f"P@{k}"] = 0.0

    map_string = "map_cut." + ",".join([str(k) for k in k_values])
    ndcg_string = "ndcg_cut." + ",".join([str(k) for k in k_values])
    recall_string = "recall." + ",".join([str(k) for k in k_values])
    precision_string = "P." + ",".join([str(k) for k in k_values])
    evaluator = pytrec_eval.RelevanceEvaluator(
        qrels, {map_string, ndcg_string, recall_string, precision_string}
    )
    scores = evaluator.evaluate(results)

    for query_id in scores.keys():
        for k in k_values:
            ndcg[f"NDCG@{k}"] += scores[query_id]["ndcg_cut_" + str(k)]
            _map[f"MAP@{k}"] += scores[query_id]["map_cut_" + str(k)]
            recall[f"Recall@{k}"] += scores[query_id]["recall_" + str(k)]
            precision[f"P@{k}"] += scores[query_id]["P_" + str(k)]

    for k in k_values:
        ndcg[f"NDCG@{k}"] = round(ndcg[f"NDCG@{k}"] / len(scores), 5)
        _map[f"MAP@{k}"] = round(_map[f"MAP@{k}"] / len(scores), 5)
        recall[f"Recall@{k}"] = round(recall[f"Recall@{k}"] / len(scores), 5)
        precision[f"P@{k}"] = round(precision[f"P@{k}"] / len(scores), 5)

    if metric_file:
        out = open(metric_file, "w")
        for eval in [ndcg, _map, recall, precision]:
            json.dump(eval, out, indent=4, ensure_ascii=False)
            out.write("\n")

    return ndcg, _map, recall, precision


def save_HF_corpus_as_denser_passages(corpus, output_file: str, max_doc_size):
    out = open(output_file, "w")
    for i, d in enumerate(corpus):
        if max_doc_size > 0 and i >= max_doc_size:
            break
        data = {"source": d["id"], "title": d["title"], "text": d["text"], "pid": -1}
        json.dump(data, out, ensure_ascii=False)
        out.write("\n")


def save_HF_docs_as_denser_passages(texts, output_file: str, max_doc_size):
    out = open(output_file, "w")
    for i, d in enumerate(texts):
        if max_doc_size > 0 and i >= max_doc_size:
            break
        data = {
            "source": d.metadata["source"],
            "title": "",
            "text": d.page_content,
            "pid": i,
        }
        json.dump(data, out, ensure_ascii=False)
        out.write("\n")


def save_denser_queries(queries, output_file: str):
    out = open(output_file, "w")
    for d in queries:
        data = {"id": d["id"], "text": d["text"]}
        json.dump(data, out, ensure_ascii=False)
        out.write("\n")


def save_denser_qrels(qrels, output_file: str):
    out = open(output_file, "w")
    for q in qrels.keys():
        data = {q: qrels[q]}
        json.dump(data, out, ensure_ascii=False)
        out.write("\n")


def load_denser_qrels(in_file: str):
    res = {}
    for line in open(in_file, "r"):
        obj = json.loads(line)
        assert len(list(obj.keys())) == 1
        query = list(obj.keys())[0]
        res[query] = obj[query]
    return res


def dump_passages(passages: List[Dict[str, str]], output_file: str):
    out = open(output_file, "w")
    for passage in passages:
        json.dump(passage, out, ensure_ascii=False)
        out.write("\n")


def passages_to_dict(passages: List[Dict[str, str]], doc_task, score_name="score"):
    res = {}
    for passage in passages:
        source, pid = passage["source"], passage.get("pid", -1)
        if doc_task or pid == -1:
            uid_str = source
        else:
            uid_str = f"{source}-{pid}"
        assert uid_str not in res
        res[uid_str] = passage[score_name]
    return res


def aggregate_passages(passages: List[Dict[str, str]]):
    """
    Using the highest score among the passages as the document's score. Each passage represents a unique doc.
    """
    # import pdb; pdb.set_trace()
    uid_to_passages, uid_to_scores = {}, {}
    for i, passage in enumerate(passages):
        uid_str = passage["source"]
        if uid_str not in uid_to_passages:
            uid_to_passages[uid_str] = passage
            uid_to_scores[uid_str] = passage["score"]
        elif uid_to_scores[uid_str] < passage["score"]:
            uid_to_passages[uid_str] = passage
            uid_to_scores[uid_str] = passage["score"]

    docs = list(uid_to_passages.values())
    docs.sort(key=lambda x: x["score"], reverse=True)
    return docs


def build_dicts(passages):
    uid_to_passages, uid_to_scores, uid_to_ranks = {}, {}, {}
    for i, passage in enumerate(passages):
        source, id = passage["source"], passage.get("pid", -1)
        if id == -1:
            uid_str = source
        else:
            uid_str = f"{source}-{id}"
        assert uid_str not in uid_to_passages
        assert uid_str not in uid_to_scores
        assert uid_str not in uid_to_ranks
        uid_to_passages[uid_str] = passage
        uid_to_scores[uid_str] = passage["score"]
        uid_to_ranks[uid_str] = i + 1
    return uid_to_passages, uid_to_scores, uid_to_ranks


def merge_score_linear(uid_to_scores_1, uid_to_scores_2, weight_1, weight_2):
    uid_to_score = {}
    all_uids = set().union(*[uid_to_scores_1, uid_to_scores_2])
    for uid in all_uids:
        uid_to_score[uid] = weight_1 * uid_to_scores_1.get(
            uid, 0
        ) + weight_2 * uid_to_scores_2.get(uid, 0)
    return uid_to_score


def merge_score_rank(uid_to_ranks_1, uid_to_ranks_2):
    uid_to_score = {}
    k = 60
    all_uids = set().union(*[uid_to_ranks_1, uid_to_ranks_2])
    for uid in all_uids:
        uid_to_score[uid] = 1 / (k + uid_to_ranks_1.get(uid, 1000)) + 1 / (
            k + uid_to_ranks_2.get(uid, 1000)
        )
    return uid_to_score


def merge_results(passages_1, passages_2, weight_1, weight_2, combine):
    uid_to_passages_1, uid_to_scores_1, uid_to_ranks_1 = build_dicts(
        copy.deepcopy(passages_1)
    )
    uid_to_passages_2, uid_to_scores_2, uid_to_ranks_2 = build_dicts(
        copy.deepcopy(passages_2)
    )
    # import pdb; pdb.set_trace()
    uid_to_passages_1.update(uid_to_passages_2)
    if combine == "linear":
        uid_to_scores = merge_score_linear(
            uid_to_scores_1, uid_to_scores_2, weight_1, weight_2
        )
    else:  # rank
        uid_to_scores = merge_score_rank(uid_to_ranks_1, uid_to_ranks_2)
    assert len(uid_to_passages_1) == len(uid_to_scores)
    sorted_uids = sorted(uid_to_scores.items(), key=lambda x: x[1], reverse=True)
    passages = []
    for uid, _ in sorted_uids:
        passage = uid_to_passages_1[uid]
        passage["score"] = uid_to_scores[uid]
        passages.append(passage)
    return passages


def scale_results(passages, weight):
    res = copy.deepcopy(passages)
    for passage in res:
        passage["score"] *= weight
    return res


def standardize_normalize(data):
    # Convert list to numpy array for easier manipulation
    arr = np.array(data)

    # Calculate mean and standard deviation of the array
    mean = np.mean(arr)
    std_dev = np.std(arr)

    # Standardize the array
    if std_dev != 0:
        standardized_arr = (arr - mean) / std_dev
    else:
        standardized_arr = arr - mean
        # print("Warning: std_dev is zero")

    # Convert back to list if necessary
    return standardized_arr.tolist()


def min_max_normalize(data):
    # Convert list to numpy array for easier manipulation
    arr = np.array(data)

    # Find the minimum and maximum values of the array
    min_val = np.min(arr)
    max_val = np.max(arr)

    # Apply min-max normalization
    if max_val - min_val != 0:
        normalized_arr = (arr - min_val) / (max_val - min_val)
    else:
        normalized_arr = arr - min_val
        # print("Warning: max_val - min_val is zero")

    # Convert back to list if needed
    return normalized_arr.tolist()


# Parse the features
def parse_features(features):
    row = []
    col = []
    value = []

    for i, sample in enumerate(features):
        for feature in sample[1:]:  # Skip the first element as it is the label
            feature_id, feature_value = feature.split(":")
            row.append(i)
            col.append(int(feature_id) - 1)  # We have feature_id starting from 1
            value.append(float(feature_value))

    num_samples = len(features)
    num_features = max(col) + 1  # Determine the number of features

    return csr_matrix((value, (row, col)), shape=(num_samples, num_features))
