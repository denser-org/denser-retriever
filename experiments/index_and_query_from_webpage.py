import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from denser_retriever.utils import save_HF_docs_as_denser_passages
from denser_retriever.retriever_general import RetrieverGeneral

# Load, chunk and index the contents of the blog to create a retriever.
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(docs)
passage_file = "passages.jsonl"
save_HF_docs_as_denser_passages(texts, passage_file, 0)

# Build denser index
retriever_denser = RetrieverGeneral("agent_webpage", "experiments/config_local.yaml")
retriever_denser.ingest(passage_file)

# Query
query = "What is Task Decomposition?"
passages, docs = retriever_denser.retrieve(query, {})
print(passages)
