from denser_retriever.keyword import DenserKeywordSearch, create_elasticsearch_client
from denser_retriever.reranker import DenserReranker
from denser_retriever.retriever import DEFAULT_EMBEDDINGS
from denser_retriever.vectordb.milvus import MilvusDenserVectorDB

index_name="unit_test_retriever"

embeddings = DEFAULT_EMBEDDINGS

milvus = MilvusDenserVectorDB(
            collection_name=index_name,
            embedding_function=embeddings,
            connection_args={"uri": "http://localhost:19530"},
            auto_id=True,
        )

elasticsearch = DenserKeywordSearch(
    index_name=index_name,
    field_types={
        "title": {"type": "keyword"},
    },
    es_connection=create_elasticsearch_client(url="http://localhost:9200"),
)
reranker = DenserReranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
