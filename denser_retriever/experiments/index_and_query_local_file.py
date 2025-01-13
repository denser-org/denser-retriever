from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from denser_retriever.core.gradient_boost import XGradientBoost
from denser_retriever.core.keyword import ElasticKeywordSearch, create_elasticsearch_client
from denser_retriever.core.retriever import DenserRetriever
from denser_retriever.core.vectordb.milvus import MilvusDenserVectorDB
from denser_retriever.core.embeddings import SentenceTransformerEmbeddings
from denser_retriever.reranker import HFReranker

docs = TextLoader("../tests/test_data/state_of_the_union.txt").load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
texts = text_splitter.split_documents(docs)

index_name = "state_of_the_union"
retriever = DenserRetriever(
    index_name=index_name,
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

query = "What did the president say about Ketanji Brown Jackson"
res = retriever.retrieve(query, 1)

for r in res:
    print("page_content: " + r[0].page_content)
    print("metadata: " + str(r[0].metadata))
    print("score: " + str(r[1]))

retriever.delete_all()
