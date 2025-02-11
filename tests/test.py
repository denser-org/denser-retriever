from denser_retriever.keyword_search import ElasticSearch
from denser_retriever.retriever import DenserRetriever
from denser_retriever.vector_store import MilvusVectorStore


if __name__ == "__main__":
    # keword_search = ElasticSearch(
    #     hosts="35.165.196.104", port=9200, username="elastic", password="c+Yi4ubaE5KCjIke"
    # )
    vector_store = MilvusVectorStore(
        host="54.68.68.29", port=19530, user="root", password="Milvus"
    )
    denseRetriever = DenserRetriever(
        # keyword_search=keword_search
        vector_store=vector_store
    )
