import pytest
from langchain_community.document_loaders.csv_loader import CSVLoader
from denser_retriever.embeddings import SentenceTransformerEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from denser_retriever.retriever import DenserRetriever
from tests.utils import elasticsearch, milvus, reranker


# TODO: need rewrite
class TestCPWS:
    def setup_method(self):
        index_name = "unit_test_cpws"

        self.denser_retriever = DenserRetriever(
            index_name=index_name,
            vector_db=milvus,
            keyword_search=elasticsearch,
            reranker=reranker,
            gradient_boost=None,
            embeddings=SentenceTransformerEmbeddings(
                "sentence-transformers/all-MiniLM-L6-v2", 384, True
            ),
            combine_mode="linear",
        )

    @pytest.fixture(autouse=True)
    def titanic_data(self):
        docs = CSVLoader(
            "tests/test_data/cpws_2021_10_top10_en.csv",
        ).load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=100
        )

        texts = text_splitter.split_documents(docs)
        return texts

    def test_ingest(self, titanic_data):
        ids = self.denser_retriever.ingest(titanic_data)
        assert len(ids) == 10

    def test_retrieve(self, titanic_data):
        self.denser_retriever.ingest(titanic_data)
        query = "Cumings"
        filter = {"Sex": "female"}
        k = 2
        results = self.denser_retriever.retrieve(query, k, filter=filter)
        assert len(results) == k
        assert abs(results[0][1] - 3.6725) < 0.01
