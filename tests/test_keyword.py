import pytest
from denser_retriever.core.keyword import (
    DenserKeywordSearch,
    ElasticKeywordSearch,
    create_elasticsearch_client,
    ESIndexData,
)
from denser_retriever.core.filter import FieldMapper
from langchain_core.documents import Document


class TestElasticsearchKeywordStore:
    @pytest.fixture
    def es_client(self):
        return create_elasticsearch_client(url="http://localhost:9200", username="elastic",
                                              password="")

    @pytest.fixture
    def index_data(self):
        return ESIndexData(
            index_name="unit_test",
            search_fields=FieldMapper([
                "field1:keyword",
                "field2:keyword",
            ]),
            date_fields=[],
            analysis="default",
            drop_old=True
        )

    @pytest.fixture
    def keyword_search(self, es_client):
        return ElasticKeywordSearch(es_connection=es_client)

    @pytest.fixture(scope="function", autouse=True)
    def create_index(self, keyword_search, index_data):
        keyword_search.create_index(index_data)

    def test_add_documents(self, keyword_search, index_data):
        documents = [
            Document(page_content="content1", metadata={"title": "title1"}),
            Document(page_content="content2", metadata={"title": "title2"}),
        ]
        ids = keyword_search.add_documents(index_data, documents)
        assert len(ids) == 2

    def test_retrieve(self, keyword_search: DenserKeywordSearch, index_data):
        documents = [
            Document(page_content="content1", metadata={"title": "title1"}),
            Document(page_content="content2", metadata={"title": "title2"}),
        ]
        keyword_search.add_documents(index_data, documents)
        results, _ = keyword_search.retrieve(index_data, "content1", 1)
        assert len(results) == 1

    def test_get_index_mappings(self, keyword_search, index_data):
        mappings = keyword_search.get_index_mappings(index_data)
        assert "field1" in mappings
        assert "field2" in mappings