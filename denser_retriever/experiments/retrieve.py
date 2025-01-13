import argparse
import json
import logging
from typing import Optional

from denser_retriever.core.retriever import DenserRetriever
from denser_retriever.config import RetrieverConfig, FusionConfig, ESConfig, MilvusConfig, EmbeddingConfig, LinearConfig
from denser_retriever.experiments.hf_data_loader import HFDataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def format_results(results, aggregations):
    """Format retrieval results for output."""
    formatted = {
        "results": [],
        "aggregations": aggregations
    }
    pid_to_score = {}

    for doc, score in results:
        formatted["results"].append({
            "content": doc.page_content,
            "metadata": doc.metadata,
            "score": score
        })
        pid_to_score[doc.metadata["pid"]] = score

    return formatted, pid_to_score


# Read offline scores of the first query from file
def read_offline_scores(file_path: str) -> Optional[dict]:
    out = open(file_path, "r")
    prev_query_id = None
    offline_pid_to_score = {}
    for line in out:
        comps = line.strip().split()
        assert len(comps) == 3
        query_id, doc_id, score = comps
        offline_pid_to_score[doc_id] = float(score)
        if prev_query_id and prev_query_id != query_id:
            break
        prev_query_id = query_id
    return offline_pid_to_score


def create_parser():
    """Create argument parser with all configuration options."""
    parser = argparse.ArgumentParser(description="Retrieve relevant passages for a query")

    # Required arguments
    parser.add_argument("dataset_name", help="Name of the dataset/index to search")
    parser.add_argument("query", help="Search query")

    # Optional config file
    parser.add_argument("--config", help="Path to config JSON file (overrides other arguments)")

    # General settings
    parser.add_argument("--max-query-len", type=int, default=2000, help="Maximum query length")
    parser.add_argument("--top-k", type=int, default=10, help="Number of final results to return")
    parser.add_argument("--aggregation", action="store_true", help="Enable aggregation")
    parser.add_argument("--output", help="Output JSON file path")
    parser.add_argument("--offline_scores", help="Path to offline scores file")

    # Elasticsearch settings
    parser.add_argument("--es-url", default="http://localhost:9200", help="Elasticsearch URL")
    parser.add_argument("--es-analysis", default="default", help="Elasticsearch analysis type")

    # Milvus settings
    parser.add_argument("--milvus-uri", default="http://localhost:19530", help="Milvus URI")

    # Reranker settings
    parser.add_argument("--reranker-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2", help="Reranker model name")

    # Embedding settings
    parser.add_argument("--embedding-type", default="sentence_transformer", help="Embedding type")
    parser.add_argument("--embedding-model", default="Snowflake/snowflake-arctic-embed-m", help="Embedding model name")
    parser.add_argument("--embedding-size", type=int, default=768, help="Embedding size")
    parser.add_argument("--embedding-one-model", action="store_true", help="Use one model for embedding")

    # Fusion settings
    parser.add_argument("--fusion-mode", default="linear", choices=["linear", "rank", "model"], help="Fusion mode")
    parser.add_argument("--vector-top-k", type=int, default=100, help="Vector search top K")
    parser.add_argument("--vector-weight", type=float, default=1.0, help="Vector search weight")
    parser.add_argument("--keyword-top-k", type=int, default=100, help="Keyword search top K")
    parser.add_argument("--keyword-weight", type=float, default=1.0, help="Keyword search weight")
    parser.add_argument("--reranker-top-k", type=int, default=50, help="Reranker top K")
    parser.add_argument("--reranker-weight", type=float, default=1.0, help="Reranker weight")

    return parser


def create_config_from_args(args):
    """Create RetrieverConfig from command line arguments."""
    config = RetrieverConfig(
        max_query_len=args.max_query_len,
        es=ESConfig(
            url=args.es_url,
            analysis=args.es_analysis
        ),
        milvus=MilvusConfig(
            uri=args.milvus_uri
        ),
        reranker_model=args.reranker_model,
        embedding=EmbeddingConfig(
            type=args.embedding_type,
            model=args.embedding_model,
            size=args.embedding_size,
            one_model=args.embedding_one_model
        ),
        fusion_config=FusionConfig(
            mode=args.fusion_mode,
            vector=LinearConfig(
                top_k=args.vector_top_k,
                weight=args.vector_weight
            ),
            keyword=LinearConfig(
                top_k=args.keyword_top_k,
                weight=args.keyword_weight
            ),
            reranker=LinearConfig(
                top_k=args.reranker_top_k,
                weight=args.reranker_weight
            )
        ),
        aggregation=args.aggregation
    )
    return config


def main():
    parser = create_parser()
    args = parser.parse_args()

    # Load config from file if provided, otherwise create from arguments
    if args.config:
        from denser_retriever.config import load_retriever_config
        config = load_retriever_config(args.config)
    else:
        config = create_config_from_args(args)

    # Initialize retriever
    retriever_config = config.get_retriever_config(args.dataset_name, False)
    retriever = DenserRetriever(**retriever_config)

    # Perform retrieval
    results, aggregations = retriever.retrieve(args.query, args.top_k, retriever_config["fusion_config"])
    formatted_results, pid_to_score = format_results(results, aggregations)

    # Output results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(formatted_results, f, indent=2)
    else:
        print("\nTop", len(results), "results for query:", args.query)
        print("-" * 80)
        for i, result in enumerate(formatted_results["results"], 1):
            print(f"\n{i}. Score: {result['score']:.4f}")
            print(f"Content: {result['content'][:200]}...")
            if result['metadata']:
                print("Metadata:", json.dumps(result['metadata'], indent=2, ensure_ascii=False))
            print("-" * 80)

    # Verify online scores against offline scores
    if args.offline_scores:
        offline_pid_to_score = read_offline_scores(args.offline_scores)
        import pdb; pdb.set_trace()
        for pid, score in pid_to_score.items():
            if pid in offline_pid_to_score:
                assert abs(score - offline_pid_to_score[pid]) < 1e-4
        print("Online scores match offline scores.")


if __name__ == "__main__":
    main()
