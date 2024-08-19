import textwrap
from langchain_text_splitters import RecursiveCharacterTextSplitter

from denser_retriever.gradient_boost import DenserGradientBoost
from denser_retriever.keyword import (
    DenserKeywordSearch,
    create_elasticsearch_client,
)
from denser_retriever.reranker import DenserReranker
from denser_retriever.retriever import DEFAULT_EMBEDDINGS, DenserRetriever
from denser_retriever.vectordb.milvus import MilvusDenserVectorDB

from langchain_community.document_loaders import WebBaseLoader

web_site = "https://denser.ai"
loader = WebBaseLoader(web_site)

docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(docs)

embeddings = DEFAULT_EMBEDDINGS
retriever = DenserRetriever(
    index_name="agent_webpage",
    vector_db=MilvusDenserVectorDB(
        collection_name="agent_webpage",
        auto_id=True,
        connection_args={"uri": "http://localhost:19530"},
        embedding_function=embeddings,
    ),
    keyword_search=DenserKeywordSearch(
        index_name="agent_webpage",
        field_types={"title": {"type": "keyword"}},
        es_connection=create_elasticsearch_client(url="http://localhost:9200"),
    ),
    embeddings=embeddings,
    reranker=DenserReranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"),
    gradient_boost=DenserGradientBoost(
        "experiments/models/scifact_xgb_es+vs+rr_n.json"
    ),
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

retriever.clear()
