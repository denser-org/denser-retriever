import textwrap

from langchain_community.document_loaders import WebBaseLoader
from denser_retriever.embeddings import SentenceTransformerEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from denser_retriever.gradient_boost import XGradientBoost
from denser_retriever.keyword import (
    ElasticKeywordSearch,
    create_elasticsearch_client,
)
from denser_retriever.reranker import HFReranker
from denser_retriever.retriever import DenserRetriever
from denser_retriever.vectordb.milvus import MilvusDenserVectorDB

# Specify the website to crawl and index
web_site = "https://denser.ai"
loader = WebBaseLoader(web_site)  # Loader to fetch web content

docs = loader.load()  # Load documents from the website

# Split the loaded documents into smaller chunks for indexing
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(docs)

# Initialize the DenserRetriever with all components
retriever = DenserRetriever(
    index_name="agent_webpage",  # Name for the index
    vector_db=MilvusDenserVectorDB(
        connection_args={"uri": "http://localhost:19530"}
    ),  # Vector database for semantic search
    keyword_search=ElasticKeywordSearch(
        es_connection=create_elasticsearch_client(url="http://localhost:9200"),
        drop_old=False  # Do not drop the old index if it exists
    ),  # Keyword search using Elasticsearch
    embeddings=SentenceTransformerEmbeddings(
        "sentence-transformers/all-MiniLM-L6-v2", 384, True
    ),  # Embedding model for vector search
    reranker=HFReranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2", top_k=10),  # Reranker model
    gradient_boost=XGradientBoost("experiments/models/scifact_xgb_es+vs+rr_n.json"),  # Gradient boosting model
    combine_mode="model",  # How to combine scores from different sources
    xgb_model_features="es+vs+rr_n",  # Features for the gradient boosting model
)

retriever.ingest(texts)  # Ingest the split documents into the retriever

# Define a query to search for
query = "What use cases does Denser AI support?"
res = retriever.retrieve(query, 4)  # Retrieve top 4 relevant results

# Print the results in a readable format
for r in res:
    print(f"{'='*40}")
    print(
        f"Page Content:\n{textwrap.indent(textwrap.fill(r[0].page_content, width=70), '  ')}"
    )
    print(f"\nMetadata: {r[0].metadata}")
    print(f"Score: {r[1]}")
    print(f"{'='*40}\n")
