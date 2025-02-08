import argparse
import json
import logging

from denser_retriever import DenserRetriever
from denser_retriever.core.keyword import ESIndexData
from denser_retriever.core.vectordb.milvus import MilvusIndexData

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
    parser.add_argument(
        "--top-k", type=int, default=10, help="Number of results to return"
    )
    args = parser.parse_args()

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

    # Get the specified retrieval method
    retrieve_method = get_retrieval_method(retriever, config["combine_config"]["method"])

    # Perform retrieval using the selected method
    result = retrieve_method(
        query=args.query,
        k=args.top_k,
        filter={},
        aggregation=config["aggregation"],
        usage=True,
    )

    print(
        f"\nTop {len(result.documents)} results for query: {args.query}"
    )
    method = config["combine_config"]["method"]
    print(f"Using method: {method}")
    print(f"Usage: {result.token_metrics}")
    print("-" * 80)
    for i, (doc, score) in enumerate(result.documents, 1):
        print(f"\n{i}. Score: {score:.4f}")
        print(f"Content: {doc.page_content[:200]}...")
        if doc.metadata:
            print(
                "Metadata:", json.dumps(doc.metadata, indent=2, ensure_ascii=False)
            )
        print("-" * 80)


if __name__ == "__main__":
    main()
