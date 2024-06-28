import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from denser_retriever.utils import save_HF_docs_as_denser_passages
from denser_retriever.retriever_general import RetrieverGeneral
from utils_data import load_document
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Generate text chunks
file_paths = [
    "tests/test_data/state_of_the_union.txt",
    "tests/test_data/dpr.pdf",
    "https://example.com/index.html",
]
documents = []
for file_path in file_paths:
    documents.extend(load_document(file_path))

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
texts = text_splitter.split_documents(documents)
passage_file = "passages.jsonl"
save_HF_docs_as_denser_passages(texts, passage_file, 0)

# Build denser index
retriever_denser = RetrieverGeneral(
    "state_of_the_union", "experiments/config_local.yaml"
)
retriever_denser.ingest(passage_file)

# Query
query = "What did the president say about Ketanji Brown Jackson"
passages, docs = retriever_denser.retrieve(query, {})
logger.info(passages)
os.remove(passage_file)
