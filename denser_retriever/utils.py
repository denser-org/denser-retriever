import json
from typing import Dict, List, Optional, Tuple

import numpy as np
import pytrec_eval

from scipy.sparse import csr_matrix
from collections import defaultdict
from langchain_core.documents import Document

def evaluate(
    qrels: Dict[str, Dict[str, int]],
    results: Dict[str, Dict[str, float]],
    metric_file: Optional[str] = None,
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



def save_queries(queries, output_file: str):
    out = open(output_file, "w")
    for d in queries:
        data = {"id": d["id"], "text": d["text"]}
        json.dump(data, out, ensure_ascii=False)
        out.write("\n")


def save_qrels(qrels, output_file: str):
    out = open(output_file, "w")
    for q in qrels.keys():
        data = {q: qrels[q]}
        json.dump(data, out, ensure_ascii=False)
        out.write("\n")


def load_qrels(in_file: str):
    res = {}
    for line in open(in_file, "r"):
        obj = json.loads(line)
        assert len(list(obj.keys())) == 1
        query = list(obj.keys())[0]
        res[query] = obj[query]
    return res

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


def scale_results(passages: List[Tuple[Document, float]], weight: float):
    return [(doc, score * weight) for doc, score in passages]


def merge_score_linear(
        passages_1: List[Tuple[Document, float]],
        passages_2: List[Tuple[Document, float]],
        weight_1: float,
        weight_2: float,
) -> List[Tuple[Document, float]]:
    # Dictionary to store the combined scores with pid as the key
    combined_scores: Dict[str, float] = defaultdict(float)
    document_map: Dict[str, Document] = {}

    # Add the weighted scores from passages_1
    for doc, score in passages_1:
        pid = doc.metadata["pid"]
        combined_scores[pid] += score * weight_1
        document_map[pid] = doc

    # Add the weighted scores from passages_2
    for doc, score in passages_2:
        pid = doc.metadata["pid"]
        combined_scores[pid] += score * weight_2
        document_map[pid] = doc

    # Create a list of tuples from the combined scores and sort by score in descending order
    merged_passages = [(document_map[pid], score) for pid, score in combined_scores.items()]
    merged_passages.sort(key=lambda x: x[1], reverse=True)

    return merged_passages


# TODO: to fix this function
def merge_score_rank(
    passages_1: List[Tuple[Document, float]], passages_2: List[Tuple[Document, float]]
) -> List[Tuple[Document, float]]:
    rank_dict: Dict[int, float] = {}
    doc_dict: Dict[int, Document] = {}
    k = 60

    for rank, (document, _) in enumerate(passages_1, start=1):
        rank_dict[rank] = 1 / (k + rank)
        doc_dict[rank] = document

    offset = len(passages_1)
    for rank, (document, _) in enumerate(passages_2, start=1):
        rank_dict[offset + rank] = rank_dict.get(offset + rank, 0) + 1 / (k + rank)
        doc_dict[offset + rank] = document

    return [(doc_dict[idx], score) for idx, score in rank_dict.items()]


def merge_results(
    passages_1: List[Tuple[Document, float]],
    passages_2: List[Tuple[Document, float]],
    weight_1: float,
    weight_2: float,
    combine: str,
) -> List[Tuple[Document, float]]:
    if combine == "linear":
        merged_passages = merge_score_linear(passages_1, passages_2, weight_1, weight_2)
    else:  # rank
        merged_passages = merge_score_rank(passages_1, passages_2)

    return merged_passages


def docs_to_dict(
    doc: List[Tuple[Document, float]],
) -> Tuple[Dict[str, Document], Dict[str, float], Dict[str, int]]:
    """Convert a list of documents and scores to dictionaries."""
    doc_dict, score_dict, rank_dict = {}, {}, {}

    for i, (document, score) in enumerate(doc):
        uid_str = document.metadata.get("pid")
        # store the document, score and rank
        doc_dict[uid_str] = document
        score_dict[uid_str] = score
        rank_dict[uid_str] = i + 1

    return doc_dict, score_dict, rank_dict
