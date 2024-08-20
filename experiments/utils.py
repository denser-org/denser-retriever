from langchain_huggingface import HuggingFaceEmbeddings
from denser_retriever.keyword import ElasticKeywordSearch, create_elasticsearch_client
from denser_retriever.reranker import HFReranker
from denser_retriever.vectordb.milvus import MilvusDenserVectorDB

index_name = "unit_test_retriever"

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/msmarco-MiniLM-L-6-v2"
)

milvus = MilvusDenserVectorDB(
    collection_name=index_name,
    embedding_function=embeddings,
    connection_args={"uri": "http://localhost:19530"},
    auto_id=True,
)

elasticsearch = ElasticKeywordSearch(
    index_name=index_name,
    es_connection=create_elasticsearch_client(url="http://localhost:9200"),
)
reranker = HFReranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
