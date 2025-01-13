import argparse
import logging
from typing import Dict

from denser_retriever.core.retriever import DenserRetriever
from denser_retriever.config import RetrieverConfig, load_retriever_config
from denser_retriever.experiments.hf_data_loader import HFDataLoader
from denser_retriever.core.utils import evaluate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_queries(retriever: DenserRetriever, queries: list, fusion_config: Dict,
                    top_k: int, max_query_len: int = 2000, num_queries: int = 0) -> Dict[str, Dict[str, float]]:
    """Process all queries and return qid_pid_score dictionary."""
    qid_to_pid_scores = {}

    for i, query in enumerate(queries):
        if num_queries > 0 and i >= num_queries:
            break
        logger.info(f"Processing query {i + 1}/{len(queries)}")

        # Process query
        query_str = query["text"][:max_query_len] if max_query_len > 0 else query["text"]
        qid = query["id"]

        # Retrieve results
        results, _ = retriever.retrieve(query_str, top_k, fusion_config)

        # Store scores
        if qid not in qid_to_pid_scores:
            qid_to_pid_scores[qid] = {}
        for doc, score in results:
            # import pdb; pdb.set_trace()
            qid_to_pid_scores[qid][doc.metadata["pid"]] = score

    return qid_to_pid_scores


def main():
    parser = argparse.ArgumentParser(description="Evaluate retriever on a dataset")
    parser.add_argument("index_name", help="Name of the index to search")
    parser.add_argument("dataset", help="Dataset name")
    parser.add_argument("--config", required=True, help="Path to config file")
    parser.add_argument("--top-k", type=int, default=10, help="Number of final results to return")
    parser.add_argument("--num-queries", type=int, default=0, help="Number of final results to return")
    parser.add_argument("--output", help="Path to save scores")
    parser.add_argument("--split", default="test", help="Dataset split")
    args = parser.parse_args()

    # Load config and initialize retriever
    config = load_retriever_config(args.config)
    retriever_config = config.get_retriever_config(args.index_name, False)
    retriever = DenserRetriever(**retriever_config)

    # Load test queries and qrels
    _, queries, qrels = HFDataLoader(
        hf_repo=args.dataset,
        hf_repo_qrels=None,
        streaming=False,
        keep_in_memory=False,
    ).load(split=args.split)

    # Process queries
    qid_to_pid_scores = process_queries(
        retriever=retriever,
        queries=queries,
        fusion_config=retriever_config["fusion_config"],
        top_k=args.top_k,
        num_queries=args.num_queries
    )

    # Save scores if requested
    if args.output:
        with open(args.output, 'w') as f:
            for qid, pid_scores in qid_to_pid_scores.items():
                for pid, score in pid_scores.items():
                    f.write(f"{qid} {pid} {score}\n")

    # Evaluate and print results
    # import pdb; pdb.set_trace()
    metrics = evaluate(qrels, qid_to_pid_scores, None)
    print(f"\nResults ({args.split} split):")
    print("-" * 40)
    for metric_name, value in metrics[0].items():
        print(f"{metric_name}: {value:.4f}")


if __name__ == "__main__":
    main()