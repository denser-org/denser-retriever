import argparse
import json
import logging

from denser_retriever import DenserRetriever
from denser_retriever.config import (
    LRConfig,
    RetrieverConfig,
    CombineConfig,
    ESConfig,
    MilvusConfig,
    EmbeddingConfig,
    load_retriever_config,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_parser():
    """Create argument parser with all configuration options."""
    parser = argparse.ArgumentParser(
        description="Retrieve relevant passages for a query"
    )

    # Required arguments
    parser.add_argument("index_name", help="Name of the index to search")
    parser.add_argument("query", help="Search query")

    # Optional config file
    parser.add_argument(
        "--config", help="Path to config JSON file (overrides other arguments)"
    )

    # General settings
    parser.add_argument("--output", help="Output JSON file path")
    parser.add_argument(
        "--top-k", type=int, default=10, help="Number of results to return"
    )
    parser.add_argument(
        "--max-query-len", type=int, default=2000, help="Maximum query length"
    )
    parser.add_argument("--aggregation", action="store_true", help="Enable aggregation")
    parser.add_argument(
        "--no-usage",
        action="store_false",
        dest="usage",
        default=True,
        help="Disable resource usage metrics tracking and reporting",
    )

    # Elasticsearch settings
    parser.add_argument(
        "--es-url", default="http://localhost:9200", help="Elasticsearch URL"
    )
    parser.add_argument(
        "--es-analysis", default="default", help="Elasticsearch analysis type"
    )

    # Milvus settings
    parser.add_argument(
        "--milvus-uri", default="http://localhost:19530", help="Milvus URI"
    )

    # Reranker settings
    parser.add_argument(
        "--reranker-model",
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        help="Reranker model name",
    )

    # Embedding settings
    parser.add_argument(
        "--embedding-type", default="sentence_transformer", help="Embedding type"
    )
    parser.add_argument(
        "--embedding-model",
        default="Snowflake/snowflake-arctic-embed-m",
        help="Embedding model name",
    )
    parser.add_argument(
        "--embedding-size", type=int, default=768, help="Embedding size"
    )
    parser.add_argument(
        "--embedding-one-model", action="store_true", help="Use one model for embedding"
    )

    # Fusion settings
    parser.add_argument(
        "--combine-method", default="fusion", help="Combine method type"
    )
    parser.add_argument(
        "--keyword_top_k", type=int, default=100, help="Keyword search top K"
    )
    parser.add_argument(
        "--vector_top_k", type=int, default=100, help="Vector search top K"
    )
    parser.add_argument(
        "--reranker_top_k", type=int, default=100, help="Reranker top K"
    )
    parser.add_argument(
        "--lr-features", default="es+vs+rr", help="LR features for model fusion"
    )
    parser.add_argument(
        "--lr-model",
        default="denser_retriever/models/weights_es+vs+rr_scifact.json",
        help="Path to LR model weights",
    )

    return parser


def create_lr_config(
    combine_method: str, features: str, model_path: str
) -> LRConfig | None:
    """
    Create learning-to-rank configuration if fusion method is selected.

    Args:
        combine_method: Method used for combining results
        features: Features string (e.g., 'es+vs+rr')
        model_path: Path to the LR model weights file

    Returns:
        LRConfig or None if not using fusion
    """
    if combine_method != "fusion":
        return None

    if not features or not model_path:
        raise ValueError("LR features and model path are required for fusion method")

    return LRConfig(lr_features=features, lr_model=model_path)


def create_config_from_args(args):
    """Create RetrieverConfig from command line arguments."""
    lr_config = create_lr_config(
        combine_method=args.combine_method,
        features=args.lr_features,
        model_path=args.lr_model,
    )

    config = RetrieverConfig(
        max_query_len=args.max_query_len,
        es=ESConfig(url=args.es_url, analysis=args.es_analysis),
        milvus=MilvusConfig(uri=args.milvus_uri),
        reranker_model=args.reranker_model,
        embedding=EmbeddingConfig(
            type=args.embedding_type,
            model=args.embedding_model,
            size=args.embedding_size,
            one_model=args.embedding_one_model,
        ),
        combine_config=CombineConfig(
            method=args.combine_method,
            keyword_top_k=args.keyword_top_k,
            vector_top_k=args.vector_top_k,
            reranker_top_k=args.reranker_top_k,
            lr_config=lr_config,
        ),
        aggregation=args.aggregation,
    )
    return config


def get_retrieval_method(retriever: DenserRetriever, method: str):
    """Get the appropriate retrieval method from the retriever."""
    available_methods = {
        "vector": retriever.retrieve_by_vector,
        "hybrid": retriever.retrieve_by_hybrid,
        "reranker": retriever.retrieve_by_reranker,
        "fusion": retriever.retrieve_by_fusion,
    }

    if method not in available_methods:
        raise ValueError(
            f"Invalid method: {method}. "
            f"Available methods are: {list(available_methods.keys())}"
        )

    return available_methods[method]


def main():
    parser = create_parser()
    args = parser.parse_args()

    # Load config from file if provided, otherwise create from arguments
    if args.config:
        config = load_retriever_config(args.config)
    else:
        config = create_config_from_args(args)

    # Initialize retriever
    retriever_config = config.get_retriever_config(args.index_name, False)
    retriever = DenserRetriever(**retriever_config)

    # Get the specified retrieval method
    retrieve_method = get_retrieval_method(retriever, config.combine_config.method)

    # Perform retrieval using the selected method
    retrieval_result = retrieve_method(
        query=args.query,
        k=args.top_k,
        combine_config=retriever_config["combine_config"],
        filter={},
        aggregation=config.aggregation,
        usage=args.usage,
    )

    formatted_results = retrieval_result.to_json()

    # Output results
    if args.output:
        with open(args.output, "w") as f:
            json.dump(formatted_results, f, indent=2)
    else:
        print(
            f"\nTop {len(retrieval_result.documents)} results for query: {args.query}"
        )
        print(f"Using method: {retriever_config['combine_config'].method}")
        if args.usage and retrieval_result.token_metrics:
            print(f"Usage: {retrieval_result.token_metrics}")
        print("-" * 80)
        for i, (doc, score) in enumerate(retrieval_result.documents, 1):
            print(f"\n{i}. Score: {score:.4f}")
            print(f"Content: {doc.page_content[:200]}...")
            if doc.metadata:
                print(
                    "Metadata:", json.dumps(doc.metadata, indent=2, ensure_ascii=False)
                )
            print("-" * 80)


if __name__ == "__main__":
    main()
