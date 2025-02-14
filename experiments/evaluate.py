import argparse
import logging
from typing import Dict, List, Optional, Tuple
import os
from cohere import Document
from tqdm import tqdm
from denser_retriever.retriever import DenserRetriever
from experiments.hf_data_loader import HFDataLoader
from experiments.utils import evaluate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_queries(
    index_name: str,
    retrievers: Dict[str, DenserRetriever],
    queries: list,
    top_k: int,
    max_query_len: int = 2000,
    num_queries: int = 0,
) -> Tuple[Dict[str, Dict[str, Dict[str, float]]], Dict[str, Dict[str, int]]]:
    """Process queries with specified case.

    Returns:
        Tuple containing:
        - Dictionary of results by case and query
    """

    for i, query in enumerate(queries):
        if num_queries > 0 and i >= num_queries:
            break
        logger.info(f"Processing query {i + 1}/{len(queries)}")

        results = {case_name: {} for case_name in retrievers.keys()}

        query_str = (
            query["text"][:max_query_len] if max_query_len > 0 else query["text"]
        )
        qid = query["id"]

        for case_name in retrievers.keys():
            retirever = retrievers[case_name]

            # Call retrieval method
            retrieval_result = retirever.retrieve(
                query=query_str, limit=top_k, collection_name=index_name
            )

            # Store results
            if qid not in results[case_name]:
                results[case_name][qid] = {}

            # Process and store scores
            for doc, score in retrieval_result:
                if doc.metadata["pid"] not in results[case_name][qid]:
                    results[case_name][qid][doc.metadata["pid"]] = score

            logger.info(f"{case_name} returned {len(retrieval_result)} results")

    return results


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


def convert_to_documents(corpus) -> List[Document]:
    """Convert HuggingFace dataset corpus to list of Documents.

    Args:
        corpus: HuggingFace dataset corpus

    Returns:
        List of Document objects
    """
    documents = []

    for item in tqdm(corpus, desc="Converting documents"):
        # Create Document with text as content and id as pid in metadata
        doc = Document(page_content=item["text"], metadata={"pid": item["id"]})
        documents.append(doc)

    return documents


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate retriever methods on a dataset"
    )
    parser.add_argument("index_name", help="Name of the index to search")
    parser.add_argument("dataset", help="Dataset name")
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
        "--config",
        required=True,
        help="Path to config file",
    )
    args = parser.parse_args()

    retrievers = {}
    method_name = os.path.splitext(os.path.basename(args.config))[0]
    retrievers[method_name] = DenserRetriever.from_config(args.config)

    # Load queries and qrels
    _, queries, qrels = HFDataLoader(
        hf_repo=args.dataset,
        hf_repo_qrels=None,
        streaming=False,
        keep_in_memory=False,
    ).load(split=args.split)

    # Process queries with specified methods
    results = process_queries(
        index_name=args.index_name,
        retrievers=retrievers,
        queries=list(queries),
        top_k=args.top_k,
        num_queries=args.num_queries,
    )

    # Evaluate and get metrics
    all_metrics = evaluate_methods(results, qrels, args.output_dir)

    # Print comprehensive summary
    print("\n=== Evaluation Summary ===")
    print("-" * 80)

    for method in all_metrics.keys():
        print(f"\nMethod: {method}")
        print("-" * 40)

        # Performance metrics
        metrics = all_metrics[method]
        print(f"NDCG@10: {metrics['NDCG@10']:.4f}")


if __name__ == "__main__":
    main()
