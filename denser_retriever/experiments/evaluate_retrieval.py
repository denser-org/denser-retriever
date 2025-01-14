import argparse
import logging
from typing import Dict, Optional
import os

from denser_retriever.core.retriever import DenserRetriever
from denser_retriever.config import RetrieverConfig, load_retriever_config, FusionConfig
from denser_retriever.experiments.hf_data_loader import HFDataLoader
from denser_retriever.core.utils import evaluate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_queries_all_methods(
        retriever: DenserRetriever,
        queries: list,
        fusion_config: FusionConfig,
        top_k: int,
        max_query_len: int = 2000,
        num_queries: int = 0
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Process all queries with different retrieval methods.

    Returns:
        Dictionary mapping method name to qid_to_pid_scores for each method
    """
    results = {
        'hybrid': {},
        'reranker': {},
        'model': {}
    }

    for i, query in enumerate(queries):
        if num_queries > 0 and i >= num_queries:
            break
        logger.info(f"Processing query {i + 1}/{len(queries)}")

        query_str = query["text"][:max_query_len] if max_query_len > 0 else query["text"]
        qid = query["id"]

        # Process all three methods
        for method, method_name in [
            (retriever.retrieve_by_hybrid, 'hybrid'),
            (retriever.retrieve_by_reranker, 'reranker'),
            (retriever.retrieve_by_model, 'model')
        ]:
            # Call retrieval method
            results_method, _ = method(
                query=query_str,
                k=top_k,
                fusion_config=fusion_config,
                filter={},
                aggregation=False
            )

            # Store results
            if qid not in results[method_name]:
                results[method_name][qid] = {}

            # Process and store scores
            for doc, score in results_method:
                results[method_name][qid][doc.metadata["pid"]] = score

            logger.info(f"{method_name} returned {len(results_method)} results")


    return results


def evaluate_methods(results: Dict[str, Dict[str, Dict[str, float]]],
                     qrels: Dict,
                     output_prefix: Optional[str] = None) -> None:
    """Evaluate and report results for each method."""
    # Dictionary to store all metrics
    all_metrics = {}

    for method_name, qid_to_pid_scores in results.items():
        print(f"\n=== {method_name.upper()} Results ===")

        # Save predictions if output path provided
        if output_prefix:
            # Create output directory if it doesn't exist
            os.makedirs(output_prefix, exist_ok=True)

            pred_file = f"{output_prefix}/{method_name}.pred"
            with open(pred_file, 'w') as f:
                for qid, pid_scores in qid_to_pid_scores.items():
                    for pid, score in pid_scores.items():
                        if not isinstance(pid, str) or not isinstance(qid, str) or not isinstance(score, (int, float)):
                            import pdb; pdb.set_trace()
                            aa = 33
                        f.write(f"{qid} {pid} {score}\n")

        # Evaluate metrics
        metrics = evaluate(
            qrels,
            qid_to_pid_scores,
            f"{output_prefix}/metric_{method_name}.json" if output_prefix else None
        )

        # Store metrics
        all_metrics[method_name] = metrics[0]

        # Print metrics
        for metric_name, value in metrics[0].items():
            print(f"{metric_name}: {value:.4f}")

    return all_metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate retriever methods on a dataset")
    parser.add_argument("index_name", help="Name of the index to search")
    parser.add_argument("dataset", help="Dataset name")
    parser.add_argument("--config", required=True, help="Path to config file")
    parser.add_argument("--output-dir", help="Directory to save results")
    parser.add_argument("--top-k", type=int, default=100, help="Number of results to return")
    parser.add_argument("--num-queries", type=int, default=0, help="Number of queries to process (0 for all)")
    parser.add_argument("--split", default="test", help="Dataset split")
    args = parser.parse_args()

    # Load config and initialize retriever
    config = load_retriever_config(args.config)
    retriever_config = config.get_retriever_config(args.index_name, False)
    retriever = DenserRetriever(**retriever_config)

    # Load queries and qrels
    _, queries, qrels = HFDataLoader(
        hf_repo=args.dataset,
        hf_repo_qrels=None,
        streaming=False,
        keep_in_memory=False,
    ).load(split=args.split)

    # Process queries with all methods
    results = process_queries_all_methods(
        retriever=retriever,
        queries=queries,
        fusion_config=retriever_config["fusion_config"],
        top_k=args.top_k,
        num_queries=args.num_queries
    )

    # Evaluate and get metrics
    all_metrics = evaluate_methods(
        results,
        qrels,
        args.output_dir
    )

    # Print summary
    print("\n=== Summary (NDCG@10) ===")
    for method, metrics in all_metrics.items():
        print(f"{method}: {metrics['NDCG@10']:.4f}")


if __name__ == "__main__":
    main()