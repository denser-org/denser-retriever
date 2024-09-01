import pytest
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from denser_retriever.embeddings import SentenceTransformerEmbeddings
from denser_retriever.retriever import DenserRetriever
from tests.utils import milvus, reranker, elasticsearch


class TestTitanic:
    def setup_method(self):
        index_name = "unit_test_titanic"

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
            search_fields=[
                "Survived:Survived:keyword",
                "Pclass:Pclass:keyword",
                "Sex:Sex:keyword",
                "Age:Age:keyword",
                "SibSp:SibSp:keyword",
                "Parch:Parch:keyword",
                "Embarked:Embarked:keyword",
            ],
        )

    @pytest.fixture(autouse=True, scope="class")
    def titanic_data(self):
        docs = CSVLoader(
            "tests/test_data/titanic_top10.csv",
            metadata_columns=[
                "PassengerId",
                "Survived",
                "Pclass",
                "Sex",
                "Age",
                "SibSp",
                "Parch",
                "Ticket",
                "Fare",
                "Cabin",
                "Embarked",
            ],
        ).load()
        for doc in docs:
            doc.page_content = doc.page_content.split(":")[1]
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=100
        )

        texts = text_splitter.split_documents(docs)
        return texts

    def test_ingest(self, titanic_data):
        ids = self.denser_retriever.ingest(titanic_data)
        # Add assertions to verify the ingestion process
        assert len(ids) == 10

    def test_retrieve(self, titanic_data):
        self.denser_retriever.ingest(titanic_data)
        query = "Cumings"
        k = 2
        filter = {"Sex": "female"}
        results = self.denser_retriever.retrieve(query, k, filter=filter)
        assert abs(results[0][1] - 3.6725) < 0.01
