import textwrap

from langchain_community.document_loaders import WebBaseLoader
from denser_retriever.core.embeddings import SentenceTransformerEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from denser_retriever.core.gradient_boost import XGradientBoost
from denser_retriever.core.keyword import (
    ElasticKeywordSearch,
    create_elasticsearch_client,
)
from denser_retriever.reranker import HFReranker
from denser_retriever.core.retriever import DenserRetriever
from denser_retriever.core.vectordb.milvus import MilvusDenserVectorDB

web_site = "https://denser.ai"
loader = WebBaseLoader(web_site)

docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(docs)

retriever = DenserRetriever(
    index_name="agent_webpage",
    vector_db=MilvusDenserVectorDB(
        connection_args={"uri": "http://localhost:19530"},
        drop_old=True,
    ),
    keyword_search=ElasticKeywordSearch(
        es_connection=create_elasticsearch_client(url="http://localhost:9200"),
        drop_old=True,
    ),
    embeddings=SentenceTransformerEmbeddings(
        "Snowflake/snowflake-arctic-embed-m", 768, False
    ),
    reranker=HFReranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"),
    gradient_boost=XGradientBoost("../models/scifact_xgb_es+vs+rr_n.json"),
    combine_mode="model",
    xgb_model_features="es+vs+rr_n",
)
retriever.ingest(texts)

query = "What use cases does Denser AI support?"
res = retriever.retrieve(query, 4)

for r in res:
    print(f"{'='*40}")
    print(
        f"Page Content:\n{textwrap.indent(textwrap.fill(r[0].page_content, width=70), '  ')}"
    )
    print(f"\nMetadata: {r[0].metadata}")
    print(f"Score: {r[1]}")
    print(f"{'='*40}\n")
