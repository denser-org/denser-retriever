import logging
import os

import yaml

from denser_retriever.retriever_general import RetrieverGeneral
from denser_retriever.utils import (
    passages_to_dict,
    save_denser_corpus,
    save_denser_qrels,
    save_denser_queries,
    evaluate,
)
from denser_retriever.utils_data import HFDataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_benchmark(dataset_name):
    split = "test"
    corpus, queries, qrels = HFDataLoader(
        hf_repo=dataset_name,
        hf_repo_qrels=None,
        streaming=False,
        keep_in_memory=False,
    ).load(split=split)

    data_dir_name = os.path.basename(dataset_name)
    index_name = data_dir_name.replace("-", "_")
    retriever = RetrieverGeneral(index_name, "experiments/config.yaml")

    output_prefix = retriever.config["output_prefix"]
    exp_dir = os.path.join(output_prefix, f"exp_{data_dir_name}")
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    passage_file = os.path.join(exp_dir, "passages.jsonl")
    save_denser_corpus(corpus, passage_file, retriever.config["max_doc_size"])
    query_file = os.path.join(exp_dir, "queries.jsonl")
    save_denser_queries(queries, query_file)
    qrels_file = os.path.join(exp_dir, "qrels.jsonl")
    save_denser_qrels(qrels, qrels_file)

    retriever.ingest(passage_file)

    res = {}
    res_doc = {}
    for i, q in enumerate(queries):
        if retriever.config["max_query_size"] > 0 and i >= retriever.config["max_query_size"]:
            break
        logger.info(f"Processing query {i}")
        # import pdb; pdb.set_trace()
        passages, docs = retriever.retrieve(q["text"], {}, q["id"])
        passage_to_score = passages_to_dict(passages, False)
        res[q["id"]] = passage_to_score
        # doc results
        doc_to_score = passages_to_dict(docs, True)
        res_doc[q["id"]] = doc_to_score

    config_file = os.path.join(exp_dir, "config.yaml")
    with open(config_file, "w") as file:
        yaml.dump(retriever.config, file, sort_keys=False)
    res_file = os.path.join(exp_dir, "results.jsonl")
    save_denser_qrels(res, res_file)
    res_doc_file = os.path.join(exp_dir, "results_doc.jsonl")
    save_denser_qrels(res_doc, res_doc_file)

    logger.info("Evaluate passage results")
    metric_file = os.path.join(exp_dir, "metric.json")
    metric = evaluate(qrels, res, metric_file)
    ndcg_passage = metric[0]["NDCG@10"]
    logger.info(f"NDCG@10: {ndcg_passage}")



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
