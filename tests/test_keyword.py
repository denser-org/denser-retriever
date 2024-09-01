import pytest
from denser_retriever.keyword import (
    DenserKeywordSearch,
    ElasticKeywordSearch,
    create_elasticsearch_client,
)
from langchain_core.documents import Document


class TestElasticsearchKeywordStore:
    @pytest.fixture
    def keyword_search(self):
        return ElasticKeywordSearch(
            es_connection=create_elasticsearch_client(url="http://localhost:9200"),
            drop_old=True
        )

    @pytest.fixture(scope="function", autouse=True)
    def create_index(self, keyword_search):
        keyword_search.create_index(
            "unit_test",
            search_fields=[
                "field1:keyword",
                "field2:keyword",
            ],
        )

    def test_add_documents(self, keyword_search):
        documents = [
            Document(page_content="content1", metadata={"title": "title1"}),
            Document(page_content="content2", metadata={"title": "title2"}),
        ]
        ids = keyword_search.add_documents(documents)
        assert len(ids) == 2

    def test_retrieve(self, keyword_search: DenserKeywordSearch):
        documents = [
            Document(page_content="content1", metadata={"title": "title1"}),
            Document(page_content="content2", metadata={"title": "title2"}),
        ]
        keyword_search.add_documents(documents)
        results = keyword_search.retrieve("content1", 1)
        assert len(results) == 1

    def test_get_index_mappings(self, keyword_search):
        mappings = keyword_search.get_index_mappings()
        assert "field1" in mappings
        assert "field2" in mappings

    def test_get_categories(self, keyword_search):
        categories = keyword_search.get_categories("field2")
        assert len(categories) == 0
