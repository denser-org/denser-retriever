import argparse
import logging
from typing import Dict, Optional, List, Tuple
import os
import json

from denser_retriever.core.retriever import DenserRetriever
from denser_retriever.experiments.hf_data_loader import HFDataLoader
from denser_retriever.core.utils import evaluate
from denser_retriever.core.keyword import ESIndexData
from denser_retriever.core.vectordb.milvus import MilvusIndexData

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_queries_all_methods(
        retriever: DenserRetriever,
        queries: list,
        top_k: int,
        methods: List[str] = [],
        max_query_len: int = 2000,
        num_queries: int = 0,
        usage: bool = True,
) -> Tuple[Dict[str, Dict[str, Dict[str, float]]], Dict[str, Dict[str, int]]]:
    """Process queries with specified retrieval methods and track token usage.

    Returns:
        Tuple containing:
        - Dictionary of results by method and query
        - Dictionary of aggregated token metrics by method
    """
    available_methods = {
        "vector": retriever.retrieve_by_vector,
        "hybrid": retriever.retrieve_by_hybrid,
        "reranker": retriever.retrieve_by_reranker,
        "fusion": retriever.retrieve_by_fusion,
    }

    # If no methods specified, use all available methods
    if methods is None:
        methods = list(available_methods.keys())
    else:
        # Validate methods
        invalid_methods = [m for m in methods if m not in available_methods]
        if invalid_methods:
            raise ValueError(
                f"Invalid methods specified: {invalid_methods}. "
                f"Available methods are: {list(available_methods.keys())}"
            )

    # Initialize results and token tracking
    results = {method: {} for method in methods}
    token_stats = {
        method: {"vector_tokens": 0, "reranker_tokens": 0, "total_queries": 0}
        for method in methods
    }

    for i, query in enumerate(queries):
        if num_queries > 0 and i >= num_queries:
            break
        logger.info(f"Processing query {i + 1}/{len(queries)}")

        query_str = (
            query["text"][:max_query_len] if max_query_len > 0 else query["text"]
        )
        qid = query["id"]

        # Process selected methods
        for method_name in methods:
            method_func = available_methods[method_name]

            # Call retrieval method
            retrieval_result = method_func(
                query=query_str,
                k=top_k,
                filter={},
                aggregation=False,
                usage=usage,
            )

            # Store results
            if qid not in results[method_name]:
                results[method_name][qid] = {}

            # Process and store scores
            for doc, score in retrieval_result.documents:
                results[method_name][qid][doc.metadata["pid"]] = score

            # Aggregate token metrics
            token_metrics = retrieval_result.token_metrics
            if usage and token_metrics:
                token_stats[method_name]["vector_tokens"] += int(token_metrics.vector_tokens)
                token_stats[method_name]["reranker_tokens"] += int(
                    token_metrics.rerank_tokens
                )
                token_stats[method_name]["total_queries"] += 1

            logger.info(
                f"{method_name} returned {len(retrieval_result.documents)} results"
            )

    # Calculate averages and add summary statistics
    if usage:
        for method_name in methods:
            stats = token_stats[method_name]
            num_queries = stats["total_queries"]
            if num_queries > 0:
                stats["avg_vector_tokens"] = int(stats["vector_tokens"] / num_queries)
                stats["avg_reranker_tokens"] = int(stats["reranker_tokens"] / num_queries)

    return results, token_stats


def evaluate_methods(
        results: Dict[str, Dict[str, Dict[str, float]]],
        qrels: Dict,
        output_prefix: Optional[str] = None,
) -> Dict[str, Dict[str, float]]:
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
            with open(pred_file, "w") as f:
                for qid, pid_scores in qid_to_pid_scores.items():
                    for pid, score in pid_scores.items():
                        f.write(f"{qid} {pid} {score}\n")

        # Evaluate metrics
        metrics = evaluate(
            qrels,
            qid_to_pid_scores,
            f"{output_prefix}/metric_{method_name}.json" if output_prefix else None,
        )

        # Store metrics
        all_metrics[method_name] = metrics[0]

        # Print metrics
        for metric_name, value in metrics[0].items():
            print(f"{metric_name}: {value:.4f}")

    return all_metrics


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate retriever methods on a dataset"
    )
    parser.add_argument("index_name", help="Name of the index to search")
    parser.add_argument("dataset", help="Dataset name")
    parser.add_argument("--config", required=True, help="Path to config file")
    parser.add_argument("--output-dir", help="Directory to save results")
    parser.add_argument(
        "--top-k", type=int, default=100, help="Number of results to return"
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        default=0,
        help="Number of queries to process (0 for all)",
    )
    parser.add_argument("--split", default="test", help="Dataset split")
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["vector", "hybrid", "reranker", "fusion"],
        help="Specific methods to evaluate. If not specified, evaluates all methods.",
    )
    parser.add_argument(
        "--no-usage",
        action="store_false",
        dest="usage",
        default=True,
        help="Disable resource usage metrics tracking and reporting",
    )
    args = parser.parse_args()

    # Load config and initialize retriever
    index_name = args.index_name
    config_path = args.config
    with open(config_path) as f:
        config = json.load(f)

    es_data = ESIndexData(
        index_name=index_name,
        analysis="default",
        drop_old=False
    )
    milvus_data = MilvusIndexData(
        index_name=index_name,
        embedding_size=int(config["embedding"]["size"]),  # Match embedding model size
        drop_old=False
    )
    retriever = DenserRetriever(
        config_path=config_path,
        es_data=es_data,
        milvus_data=milvus_data
    )

    # Load queries and qrels
    _, queries, qrels = HFDataLoader(
        hf_repo=args.dataset,
        hf_repo_qrels=None,
        streaming=False,
        keep_in_memory=False,
    ).load(split=args.split)

    # Process queries with specified methods
    results, token_stats = process_queries_all_methods(
        retriever=retriever,
        queries=list(queries),
        top_k=args.top_k,
        methods=args.methods,
        num_queries=args.num_queries,
        usage=args.usage,
    )

    # Evaluate and get metrics
    all_metrics = evaluate_methods(results, qrels, args.output_dir)

    # Print comprehensive summary
    print("\n=== Evaluation Summary ===")
    print("-" * 80)

    for method in token_stats.keys():
        print(f"\nMethod: {method}")
        print("-" * 40)

        # Performance metrics
        metrics = all_metrics[method]
        print(f"NDCG@10: {metrics['NDCG@10']:.4f}")

        # Usage statistics
        stats = token_stats[method]

        print("\nUsage Statistics:")
        print(f"Total Queries: {stats['total_queries']:,}")
        print(
            f"Vector Tokens: {stats['vector_tokens']:,} "
            f"(avg: {stats.get('avg_vector_tokens', 0):,.1f} per query)"
        )
        print(
            f"Reranker Tokens: {stats['reranker_tokens']:,} "
            f"(avg: {stats.get('avg_reranker_tokens', 0):,.1f} per query)"
        )


if __name__ == "__main__":
    main()
