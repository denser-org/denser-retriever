import argparse
import json
import logging
from typing import Dict, List, Optional, Tuple
import os
from cohere import Document
from tqdm import tqdm
from denser_retriever.retriever import DenserRetriever
from experiments.hf_data_loader import HFDataLoader
from experiments.llm import LLM
from experiments.utils import evaluate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_queries(
    collection_name: str,
    retrievers: Dict[str, DenserRetriever],
    queries: list,
    top_k: int,
    max_query_length: int,
    num_queries: int = 0,
    summarizer: Optional[LLM] = None,
) -> Tuple[Dict[str, Dict[str, Dict[str, float]]], Dict[str, Dict[str, int]]]:
    """Process queries with specified case.

    Returns:
        Tuple containing:
        - Dictionary of results by case and query
    """
    results = {case_name: {} for case_name in retrievers.keys()}

    for i, query in enumerate(queries):
        if num_queries > 0 and i >= num_queries:
            break
        logger.info(f"Processing query {i + 1}/{len(queries)}")

        query_str = query["text"]

        # Truncate or summarize query if needed
        if max_query_length > 0 and len(query_str) > max_query_length:
            if summarizer:
                # Use LLM to summarize long queries
                summary = summarizer.summarize(query_str, 256)
                query_str = summary + query_str[: (max_query_length - len(summary))]
            else:
                # Fallback to truncation if no summarizer
                query_str = query_str[:max_query_length]

        qid = query["id"]

        for case_name in retrievers.keys():
            retirever = retrievers[case_name]

            logger.info(
                f"Retrieving for case: {case_name}, query length: {len(query_str)}"
            )
            # Call retrieval method
            retrieval_result = retirever.retrieve(
                query=query_str, limit=top_k, collection_name=collection_name
            )

            # Store results
            if qid not in results[case_name]:
                results[case_name][qid] = {}

            # Process and store scores
            for doc, score in retrieval_result:
                if doc.metadata["source_id"] not in results[case_name][qid]:
                    results[case_name][qid][doc.metadata["source_id"]] = score

            logger.info(f"{case_name} returned {len(retrieval_result)} results")

    return results


def evaluate_cases(
    results: Dict[str, Dict[str, Dict[str, float]]],
    qrels: Dict,
    max_query_length: int,
    output_dir: Optional[str] = None,
) -> Dict[str, Dict[str, float]]:
    """Evaluate and report results for each method."""
    # Dictionary to store all metrics
    all_metrics = {}

    for case_name, qid_to_source_id_scores in results.items():
        print(f"\n=== {case_name.upper()} Results ===")

        # Save predictions if output path provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

            pred_file = os.path.join(output_dir, f"{case_name}_{max_query_length}.pred")
            with open(pred_file, "w") as f:
                for qid, source_id_scores in qid_to_source_id_scores.items():
                    for source_id, score in source_id_scores.items():
                        f.write(f"{qid} {source_id} {score}\n")

        metrics_path = (
            os.path.join(output_dir, f"metric_{case_name}_{max_query_length}.json")
            if output_dir
            else None
        )

        # Evaluate metrics
        metrics = evaluate(qrels, qid_to_source_id_scores, metrics_path)

        # Store metrics
        all_metrics[case_name] = metrics[0]

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
        # Create Document with text as content and id as source_id in metadata
        doc = Document(page_content=item["text"], metadata={"source_id": item["id"]})
        documents.append(doc)

    return documents


def main():
    parser = argparse.ArgumentParser(description="Evaluate retriever on a dataset")
    parser.add_argument("collection", help="Name of the collection to search")
    parser.add_argument("dataset", help="Dataset name")
    parser.add_argument(
        "--top-k", type=int, default=100, help="Number of results to return"
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        default=0,
        help="Number of queries to process (0 for all)",
    )
    parser.add_argument(
        "--max-query-length",
        type=int,
        default=0,
        help="Maximum query length (truncation or summarization)",
    )
    parser.add_argument("--split", default="test", help="Dataset split")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to config file",
    )
    parser.add_argument(
        "--summarizer-config",
        help="Path to summarizer config file",
    )
    parser.add_argument("--output-dir", help="Directory to save results")
    args = parser.parse_args()

    # TODO: Support multiple retrievers
    retrievers = {}
    config_name = os.path.splitext(os.path.basename(args.config))[0]
    retrievers[config_name] = DenserRetriever.from_config(args.config)

    # Load queries and qrels
    _, queries, qrels = HFDataLoader(
        hf_repo=args.dataset,
        hf_repo_qrels=None,
        streaming=False,
        keep_in_memory=False,
    ).load(split=args.split)

    # Load summarizer if provided
    summarizer = None
    if args.summarizer_config:
        config = json.load(open(args.summarizer_config))
        summarizer = LLM(
            base_url=config.get("base_url", None),
            api_key=config["api_key"],
            model=config["model"],
        )

    # Process queries with specified methods
    results = process_queries(
        collection_name=args.collection,
        retrievers=retrievers,
        queries=list(queries),
        top_k=args.top_k,
        num_queries=args.num_queries,
        max_query_length=args.max_query_length,
        summarizer=summarizer,
    )

    # Evaluate and get metrics
    all_metrics = evaluate_cases(
        results=results,
        qrels=qrels,
        max_query_length=args.max_query_length,
        output_dir=args.output_dir,
    )

    # Print comprehensive summary
    print("\n=== Evaluation Summary ===")
    print("-" * 80)

    for case_name in all_metrics.keys():
        print(f"Case: {case_name}")
        print("-" * 40)

        # Performance metrics
        metrics = all_metrics[case_name]
        print(f"NDCG@10: {metrics['NDCG@10']:.4f}")


if __name__ == "__main__":
    main()
