import argparse
import logging
from typing import Dict, List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm
from langchain_core.documents import Document
from denser_retriever.retriever import DenserRetriever
from experiments.hf_data_loader import HFDataLoader
from tenacity import retry, stop_after_attempt, wait_exponential

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def convert_to_documents(
    corpus: Dict[str, Dict[str, str]],
    max_content_length: int = 0,
    split_strategy: str = "split",
    chunk_overlap: int = 0,
) -> List[Document]:
    """Convert HuggingFace dataset corpus to list of Documents.

    Args:
        corpus: HuggingFace dataset corpus

    Returns:
        List of Document objects
    """
    logger.info("Converting corpus to Documents parmeters:")
    logger.info(f"  Max Length: {max_content_length}")
    logger.info(f"  Split Strategy: {split_strategy}")
    logger.info(f"  Chunk Overlap: {chunk_overlap}")

    documents = []
    text_splitter = None

    if max_content_length > 0:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_content_length,
            chunk_overlap=chunk_overlap,
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
        if max_content_length > 0 and len(item["text"]) > max_content_length:
            if split_strategy == "split":
                # Split the document and maintain the same source_id
                splits = text_splitter.create_documents(
                    texts=[item["text"]], metadatas=[{"source_id": item["id"]}]
                )
                documents.extend(splits)
            elif split_strategy == "truncate":
                # Truncate the document and maintain the same source_id
                doc = Document(
                    page_content=item["text"][:max_content_length],
                    metadata={"source_id": item["id"]},
                )
                documents.append(doc)
            elif split_strategy == "summarize":
                # TODO: Implement summarization
                continue
        else:
            # Create Document with text as content and id as source_id in metadata
            doc = Document(
                page_content=item["text"], metadata={"source_id": item["id"]}
            )
            documents.append(doc)

    return documents


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    reraise=True,
)
def ingest(
    retriever: DenserRetriever, batch: List[Document], collection_name: str
):
    """Ingest a batch of documents with retry mechanism."""
    retriever.ingest(docs=batch, collection_name=collection_name)


def main():
    parser = argparse.ArgumentParser(description="Ingest Lcorpus")
    parser.add_argument("collection", help="Name of the collection to search")
    parser.add_argument("dataset", help="Dataset name")
    parser.add_argument(
        "--batch-size", type=int, default=512, help="Batch size for ingestion"
    )
    parser.add_argument(
        "--max-content-length",
        type=int,
        default=0,
        help="Max length of page content",
    )
    parser.add_argument(
        "--split-strategy", help="How to split large documents", default="split"
    )
    parser.add_argument(
        "--chunk-overlap", type=int, default=256, help="Chunk overlap for splitting"
    )
    parser.add_argument(
        "--drop", action="store_true", help="Drop collection before ingesting"
    )
    parser.add_argument("--config", required=True, help="Path to retriever config file")
    args = parser.parse_args()

    # Validate split strategy
    if args.split_strategy not in ["split", "truncate", "summarize"]:
        raise ValueError(f"Invalid split strategy: {args.split_strategy}")

    # Initialize retriever from config
    retriever = DenserRetriever.from_config(args.config)

    logger.info(f"Ingesting documents into collection {args.collection}")

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
    documents = convert_to_documents(
        corpus=corpus,
        max_content_length=args.max_content_length,
        split_strategy=args.split_strategy,
        chunk_overlap=args.chunk_overlap,
    )
    logger.info(f"Converted {len(documents)} documents")

    # Process in batches
    for i in tqdm(range(0, len(documents), args.batch_size), desc="Ingesting batches"):
        batch = documents[i : i + args.batch_size]
        try:
            ingest(
                retriever=retriever, batch=batch, collection_name=args.collection
            )
        except Exception as e:
            logger.error(
                f"Failed to ingest batch {i}-{i+args.batch_size} after all retries: {e}"
            )
            logger.error("Stopping ingestion process")
            break


if __name__ == "__main__":
    main()
