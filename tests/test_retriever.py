from langchain_core.documents import Document
from denser_retriever.core.retriever import DenserRetriever
from denser_retriever.core.keyword import ESIndexData
from denser_retriever.core.vectordb.milvus import MilvusIndexData
from denser_retriever.core.filter import FieldMapper
from denser_retriever.core.shared import SharedComponents


class TestRetriever:
    def setup_method(self):
        # Create index data objects
        es_data = ESIndexData(
            index_name="unit_test_retriever",
            search_fields=FieldMapper(["field1:keyword"]),
            date_fields=[],
            analysis="default",
            drop_old=True
        )

        milvus_data = MilvusIndexData(
            index_name="unit_test_retriever",
            embedding_size=768,  # Matches the Snowflake model size
            search_fields=FieldMapper([]),
            drop_old=True
        )

        # Create retriever with config and index data
        shared_components = SharedComponents.initialize_from_config("denser_retriever/configs/retrieve_msmarco.json")
        self.retriever = DenserRetriever(
            shared=shared_components,
            es_data=es_data,
            milvus_data=milvus_data
        )

        self.test_docs = [
            Document(page_content="content1", metadata={"title": "title1", "source": "source1"}),
            Document(page_content="content2", metadata={"title": "title2", "source": "source2"}),
            Document(page_content="content12", metadata={"title": "title12", "source": "source12"})
        ]

    def test_check_indices(self):
        """Test checking index existence."""
        # Initially indices should exist after setup
        status = self.retriever.check_indices("unit_test_retriever")
        assert status["es_valid"] and status["vector_valid"]

        # After deleting indices, they shouldn't exist
        self.retriever.delete_all(delete_index=True)
        status = self.retriever.check_indices("unit_test_retriever")
        assert not status["es_valid"] and not status["vector_valid"]

        # Check non-existent index
        status = self.retriever.check_indices("nonexistent_index")
        assert not status["es_valid"] and not status["vector_valid"]

    def test_get_count(self):
        """Test document count retrieval."""
        # Initially should be empty
        stats = self.retriever.get_count()
        assert stats["es_docs"] == 0
        assert stats["vector_docs"] == 0

        # Add documents and check count
        self.retriever.ingest(self.test_docs)
        stats = self.retriever.get_count()
        assert stats["es_docs"] == 3
        assert stats["vector_docs"] == 3

        # Delete some documents and check count
        self.retriever.delete(source_id="source1")
        stats = self.retriever.get_count()
        assert stats["es_docs"] == 2
        assert stats["vector_docs"] == 2

        # Delete all documents but keep indices
        self.retriever.delete_all(delete_index=False)
        stats = self.retriever.get_count()
        assert stats["es_docs"] == 0
        assert stats["vector_docs"] == 0

    def test_retrieve(self):
        doc_ids, _ = self.retriever.ingest(self.test_docs)
        results = self.retriever.retrieve("content1", "fusion", 2)
        assert len(results.documents) == 2
        assert results.documents[0][0].page_content == "content1"
        assert abs(results.documents[0][1] - 0.9104) < 0.001

    def test_delete_by_id(self):
        doc_ids, _ = self.retriever.ingest(self.test_docs)
        stats = self.retriever.get_count()
        assert stats["es_docs"] == 3 and stats["vector_docs"] == 3

        self.retriever.delete(ids=[doc_ids[0]])
        stats = self.retriever.get_count()
        assert stats["es_docs"] == 2 and stats["vector_docs"] == 2

    def test_delete_by_source(self):
        self.retriever.ingest(self.test_docs)
        stats = self.retriever.get_count()
        assert stats["es_docs"] == 3 and stats["vector_docs"] == 3

        self.retriever.delete(source_id="source1")
        stats = self.retriever.get_count()
        assert stats["es_docs"] == 2 and stats["vector_docs"] == 2

    def test_delete_by_source_url(self):
        self.retriever.ingest(self.test_docs)
        stats = self.retriever.get_count()
        assert stats["es_docs"] == 3 and stats["vector_docs"] == 3

        self.retriever.delete(source_url="source1")
        stats = self.retriever.get_count()
        assert stats["es_docs"] == 1 and stats["vector_docs"] == 1

    def test_delete_all(self):
        self.retriever.ingest(self.test_docs)
        stats = self.retriever.get_count()
        assert stats["es_docs"] == 3 and stats["vector_docs"] == 3

        # Test delete_all without deleting indices
        self.retriever.delete_all(delete_index=False)
        status = self.retriever.check_indices("unit_test_retriever")
        stats = self.retriever.get_count()
        assert status["es_valid"] and status["vector_valid"]
        assert stats["es_docs"] == 0 and stats["vector_docs"] == 0

        # Test delete_all with deleting indices
        self.retriever.delete_all(delete_index=True)
        status = self.retriever.check_indices("unit_test_retriever")
        assert not status["es_valid"] and not status["vector_valid"]