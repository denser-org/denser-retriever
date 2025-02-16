import json
from typing import Dict, List, Optional, Tuple
import pytrec_eval


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
            for source_id in list(rels):
                if qid == source_id:
                    results[qid].pop(source_id)
                    popped.append(source_id)
        print(f"Removed {len(popped)} identical query and document ids.")

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
        evals = []
        for eval in [ndcg, _map, recall, precision]:
            evals.append(eval)
        json.dump(evals, out, indent=4, ensure_ascii=False)
    return ndcg, _map, recall, precision
