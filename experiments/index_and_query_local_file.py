from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from denser_retriever.gradient_boost import DenserGradientBoost
from denser_retriever.keyword import (
    ElasticsearchKeywordSearch,
    create_elasticsearch_client,
)
from denser_retriever.retriever import DenserRetriever
from denser_retriever.vectordb.milvus import MilvusDenserVectorDB
from experiments.utils import elasticsearch, embeddings, milvus, reranker

docs = TextLoader("tests/test_data/state_of_the_union.txt").load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
texts = text_splitter.split_documents(docs)

index_name = "state_of_the_union"
retriever = DenserRetriever(
    index_name=index_name,
    vector_db=MilvusDenserVectorDB(
        collection_name=index_name,
        auto_id=True,
        connection_args={"uri": "http://localhost:19530"},
        embedding_function=embeddings,
    ),
    keyword_search=ElasticsearchKeywordSearch(
        index_name=index_name,
        field_types={"title": {"type": "keyword"}},
        es_connection=create_elasticsearch_client(url="http://localhost:9200"),
    ),
    reranker=reranker,
    gradient_boost=DenserGradientBoost(
        "experiments/models/msmarco_xgb_es+vs+rr_n.json"
    ),
    embeddings=embeddings,
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

retriever.clear()
