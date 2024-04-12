from denser.utils_data import HFDataLoader
from denser.utils import save_denser_corpus, save_denser_queries, save_denser_qrels, passages_to_dict, aggregate_passages
from denser.config import topk_passages, output_prefix, max_query_size
from denser.RetrieverGeneral import RetrieverGeneral
from typing import Dict, List, Tuple
import pytrec_eval
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate(qrels: Dict[str, Dict[str, int]],
             results: Dict[str, Dict[str, float]],
             k_values: List[int] = [1, 3, 5, 10, 100, 1000],
             ignore_identical_ids: bool = True) -> Tuple[
    Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float]]:
    if ignore_identical_ids:
        logger.info(
            'For evaluation, we ignore identical query and document ids (default), please explicitly set ``ignore_identical_ids=False`` to ignore this.')
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
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {map_string, ndcg_string, recall_string, precision_string})
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

    for eval in [ndcg, _map, recall, precision]:
        logger.info("\n")
        for k in eval.keys():
            logger.info("{}: {:.4f}".format(k, eval[k]))

    return ndcg, _map, recall, precision


def run_benchmark(dataset_name):
    split = "test"
    corpus, queries, qrels = HFDataLoader(
        hf_repo=dataset_name,
        hf_repo_qrels=None,
        streaming=False,
        keep_in_memory=False,
    ).load(split=split)

    data_dir_name = os.path.basename(dataset_name)
    exp_dir = os.path.join(output_prefix, f"exp_{data_dir_name}")
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    passage_file = os.path.join(exp_dir, "passages.jsonl")
    save_denser_corpus(corpus, passage_file)
    query_file = os.path.join(exp_dir, "queries.jsonl")
    save_denser_queries(queries, query_file)
    qrels_file = os.path.join(exp_dir, "qrels.jsonl")
    save_denser_qrels(qrels, qrels_file)

    index_name = data_dir_name.replace("-", "_")
    retriever = RetrieverGeneral(index_name)
    retriever.ingest(passage_file)

    res = {}
    res_doc = {}
    for i, q in enumerate(queries):
        if max_query_size > 0 and i >= max_query_size:
            break
        logger.info(f"Processing query {i}")
        # import pdb; pdb.set_trace()
        passages, docs = retriever.retrieve(q['text'], topk_passages)
        passage_to_score = passages_to_dict(passages, False)
        res[q['id']] = passage_to_score
        # doc results
        doc_to_score = passages_to_dict(docs, True)
        res_doc[q['id']] = doc_to_score

    res_file = os.path.join(exp_dir, "results.jsonl")
    save_denser_qrels(res, res_file)
    res_doc_file = os.path.join(exp_dir, "results_doc.jsonl")
    save_denser_qrels(res_doc, res_doc_file)
    logger.info("Evaluate passage results")
    ndcg, _map, recall, precision = evaluate(qrels, res)

    logger.info("Evaluate doc results")
    ndcg, _map, recall, precision = evaluate(qrels, res_doc)


if __name__ == "__main__":
    dataset_name = "mteb/nfcorpus"
    # dataset_name = "mteb/trec-covid"
    # dataset_name = "mteb/arguana"
    # dataset_name = "mteb/climate-fever"
    # dataset_name = "mteb/dbpedia"
    # dataset_name = "mteb/fever"
    # dataset_name = "mteb/fiqa"
    # dataset_name = "miracl/hagrid" # not work
    # dataset_name = "mteb/hotpotqa"
    # dataset_name = "mteb/msmarco"
    # dataset_name = "mteb/msmarco-v2"
    # dataset_name = "narrativeqa"
    # dataset_name = "mteb/nq"
    # dataset_name = "mteb/quora"
    # dataset_name = "mteb/scidocs"
    # dataset_name = "mteb/scifact"
    # dataset_name = "mteb/touche2020"
    run_benchmark(dataset_name)
