import pytest
from denser_retriever.keyword import (
    ElasticsearchKeywordSearch,
    create_elasticsearch_client,
)
from langchain_core.documents import Document


class TestElasticsearchKeywordStore:
    @pytest.fixture
    def es_connection(self):
        return create_elasticsearch_client(url="http://localhost:9200")

    @pytest.fixture
    def keyword_store(self, es_connection):
        return ElasticsearchKeywordSearch(
            index_name="unit_test",
            field_types={
                "field1": {"type": "keyword"},
                "field2": {"type": "keyword"},
            },
            es_connection=es_connection,
        )

    def test_create_index(self, keyword_store):
        keyword_store.create_index("unit_test")
        assert keyword_store.client.indices.exists(index="unit_test")

    def test_add_documents(self, keyword_store):
        documents = [
            Document(page_content="content1", metadata={"title": "title1"}),
            Document(page_content="content2", metadata={"title": "title2"}),
        ]
        ids = keyword_store.add_documents(documents)
        assert len(ids) == 2

    def test_retrieve(self, keyword_store: ElasticsearchKeywordSearch):
        documents = [
            Document(page_content="content1", metadata={"title": "title1"}),
            Document(page_content="content2", metadata={"title": "title2"}),
        ]
        keyword_store.add_documents(documents)
        results = keyword_store.retrieve("content1", 1)
        assert len(results) == 1

    def test_get_index_mappings(self, keyword_store):
        mappings = keyword_store.get_index_mappings()
        assert "field1" in mappings
        assert "field2" in mappings

    def test_get_categories(self, keyword_store):
        categories = keyword_store.get_categories("field2")
        assert len(categories) == 0
