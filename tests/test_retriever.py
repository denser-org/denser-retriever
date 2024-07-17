from denser_retriever.retriever import DenserRetriever
from langchain_core.documents import Document


class TestRetriever:
    def setup_method(self):
        self.denser_retriever = DenserRetriever.from_milvus(
            index_name="unit_test_retriever",
            milvus_uri="http://localhost:19530",
        )

    def test_ingest(self):
        docs = [
            Document(page_content="content1", metadata={"title": "title1"}),
            Document(page_content="content2", metadata={"title": "title2"}),
        ]
        self.denser_retriever.ingest(docs)
        # Add assertions to verify the ingestion process

    def test_retrieve(self):
        docs = [
            Document(page_content="content1", metadata={"title": "title1"}),
            Document(page_content="content2", metadata={"title": "title2"}),
        ]
        self.denser_retriever.ingest(docs)
        query = "content1"
        k = 2
        result = self.denser_retriever.retrieve(query, k)
        assert len(result) == 1
        assert result[0][0].page_content == "content1"

    def test_clear(self):
        self.denser_retriever.clear()
        # Add assertions to verify the clearing process

    def test_get_field_categories(self):
        docs = [
            Document(page_content="content1", metadata={"title": "title1"}),
            Document(page_content="content2", metadata={"title": "title2"}),
        ]
        self.denser_retriever.ingest(docs)
        field = "category_field"
        k = 10
        categories = self.denser_retriever.get_field_categories(field, k)
        assert isinstance(categories, list)
        assert len(categories) <= k
        for category in categories:
            assert isinstance(category, str)

    def test_get_metadata_fields(self):
        docs = [
            Document(page_content="content1", metadata={"title": "title1"}),
            Document(page_content="content2", metadata={"title": "title2"}),
        ]
        self.denser_retriever.ingest(docs)
        fields = self.denser_retriever.get_filter_fields()
        assert len(fields) == 4
