import pytest
from pymilvus import Collection
from denser_retriever.core.vectordb.milvus import (
    MilvusDenserVectorDB,
    MilvusIndexData,
)
from denser_retriever.core.embeddings import SentenceTransformerEmbeddings
from denser_retriever.core.filter import FieldMapper
from langchain_core.documents import Document

MILVUS_CONNECTION = {
    "uri": "http://localhost:19530",
    "user": "root",
    "password": ""
}


class TestMilvusDenserVectorDB:
    @pytest.fixture
    def embeddings(self):
        return SentenceTransformerEmbeddings(
            model_name="Snowflake/snowflake-arctic-embed-m",
            embedding_size=768,
            one_model=False
        )

    @pytest.fixture
    def index_data(self):
        return MilvusIndexData(
            index_name="unit_test",
            embedding_size=768,
            search_fields=FieldMapper([]),
            drop_old=True
        )

    @pytest.fixture
    def vector_db(self):
        return MilvusDenserVectorDB(connection_args=MILVUS_CONNECTION)

    @pytest.fixture(scope="function", autouse=True)
    def setup_index(self, vector_db, index_data):
        # import pdb; pdb.set_trace()
        vector_db.create_index(index_data)
        yield
        vector_db.delete_all(index_data)  # Cleanup after each test

    def test_create_and_has_index(self, vector_db):
        assert vector_db.check_index("unit_test")
        assert not vector_db.check_index("nonexistent_index")

    def test_add_and_get_count(self, vector_db, index_data, embeddings):
        # Test initial count
        # import pdb; pdb.set_trace()
        count = vector_db.get_count(index_data)
        assert count == 0

        # Add documents
        documents = [
            Document(page_content="content1", metadata={"title": "title1"}),
            Document(page_content="content2", metadata={"title": "title2"}),
        ]
        ids = vector_db.add_documents(index_data, documents, embeddings)

        # Verify addition
        assert len(ids) == 2
        count = vector_db.get_count(index_data)
        assert count == 2

    def test_retrieve(self, vector_db, index_data, embeddings):
        # Add test documents
        documents = [
            Document(page_content="the weather is sunny today", metadata={"title": "weather"}),
            Document(page_content="machine learning models are interesting", metadata={"title": "ML"}),
        ]
        vector_db.add_documents(index_data, documents, embeddings)

        # Search similar documents
        results = vector_db.retrieve(index_data, "weather forecast", embeddings, k=1)
        assert len(results) == 1
        doc, score = results[0]
        assert isinstance(doc, Document)
        assert isinstance(score, float)
        assert "weather" in doc.page_content.lower()