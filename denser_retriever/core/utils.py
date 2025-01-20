import json
from typing import Dict, List, Optional, Tuple

import numpy as np
import pytrec_eval

from langchain_core.documents import Document

config_to_features = {
    "es+vs": [1, 2, 3, 4, 5, 6],
    "es+rr": [1, 2, 3, 7, 8, 9],
    "vs+rr": [4, 5, 6, 7, 8, 9],
    "es+vs+rr": [1, 2, 3, 4, 5, 6, 7, 8, 9],
}


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def evaluate(
    qrels: Dict[str, Dict[str, int]],
    results: Dict[str, Dict[str, float]],
    metric_file: Optional[str] = None,
    k_values: List[int] = [1, 3, 5, 10, 20, 100, 1000],
    ignore_identical_ids: bool = True,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float]]:
    if ignore_identical_ids:
        print(
            "For evaluation, we ignore identical query and document ids (default), please explicitly set ``ignore_identical_ids=False`` to ignore this."
            # noqa: E501
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
    # evaluator = None
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


def load_queries(in_file: str):
    res = []
    for line in open(in_file, "r"):
        res.append(json.loads(line))
    return res


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
