import argparse
import logging
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm
from langchain_core.documents import Document
from denser_retriever.retriever import DenserRetriever
from experiments.hf_data_loader import HFDataLoader
from tenacity import retry, stop_after_attempt, wait_exponential

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def convert_to_documents(corpus, max_length: int = 8192) -> List[Document]:
    """Convert HuggingFace dataset corpus to list of Documents.

    Args:
        corpus: HuggingFace dataset corpus

    Returns:
        List of Document objects
    """
    documents = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_length,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
        separators=[
            "\n\n",
            "\n",
            " ",
            ".",
            ",",
            "\u200b",  # Zero-width space
            "\uff0c",  # Fullwidth comma
            "\u3001",  # Ideographic comma
            "\uff0e",  # Fullwidth full stop
            "\u3002",  # Ideographic full stop
            "",
        ],
    )
    for item in tqdm(corpus, desc="Converting documents"):
        # Check if document needs splitting
        if len(item["text"]) > max_length:
            # Split the document and maintain the same pid
            splits = text_splitter.create_documents(
                texts=[item["text"]], metadatas=[{"pid": item["id"]}]
            )
            documents.extend(splits)
        else:
            # Create single Document with qw as content and pid as metadata
            doc = Document(page_content=item["text"], metadata={"pid": item["id"]})
            documents.append(doc)

    return documents


@retry(
    stop=stop_after_attempt(3),  # Retry up to 3 times
    wait=wait_exponential(
        multiplier=1, min=4, max=10
    ),  # Wait between 4-10 seconds, increasing exponentially
    reraise=True,  # Raise the last exception if all retries fail
)
def ingest_batch(
    retriever: DenserRetriever, batch: List[Document], collection_name: str
):
    """Ingest a batch of documents with retry mechanism."""
    retriever.ingest(docs=batch, collection_name=collection_name)


def main():
    parser = argparse.ArgumentParser(description="Ingest Lcorpus")
    parser.add_argument("dataset", help="Dataset name in HuggingFace hub")
    parser.add_argument("--config", required=True, help="Path to retriever config file")
    parser.add_argument(
        "--collection",
        required=True,
        help="Collection name for storage",
    )
    parser.add_argument(
        "--batch-size", type=int, default=1000, help="Batch size for ingestion"
    )
    parser.add_argument(
        "--drop", action="store_true", help="Drop collection before ingesting"
    )
    args = parser.parse_args()

    # Initialize retriever from config
    retriever = DenserRetriever.from_config(args.config)

    if retriever.has_collection(collection_name=args.collection):
        if args.drop:
            logger.info(f"Dropping collection {args.collection}")
            retriever.drop(collection_name=args.collection)
        else:
            logger.info(
                f"Collection {args.collection} already exists, skipping ingestion"
            )
            return

    # Load corpus using HFDataLoader
    logger.info(f"Loading corpus from dataset {args.dataset}")
    data_loader = HFDataLoader(
        hf_repo=args.dataset,
        streaming=False,
        keep_in_memory=False,
    )
    corpus = data_loader.load_corpus()

    # Convert corpus to documents
    documents = convert_to_documents(corpus)
    logger.info(f"Converted {len(documents)} documents")

    # Process in batches
    for i in tqdm(range(0, len(documents), args.batch_size), desc="Ingesting batches"):
        batch = documents[i : i + args.batch_size]
        try:
            ingest_batch(retriever, batch, args.collection)
        except Exception as e:
            logger.error(
                f"Failed to ingest batch {i}-{i+args.batch_size} after all retries: {e}"
            )
            logger.error("Stopping ingestion process")
            break


if __name__ == "__main__":
    main()
