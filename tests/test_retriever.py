from langchain_core.documents import Document

from denser_retriever.gradient_boost import XGradientBoost
from denser_retriever.retriever import DenserRetriever

from tests.utils import elasticsearch, milvus, reranker, embeddings


class TestRetriever:
    def setup_method(self):
        index_name = "unit_test_retriever"

        self.denser_retriever = DenserRetriever(
            index_name=index_name,
            vector_db=milvus,
            keyword_search=elasticsearch,
            reranker=reranker,
            gradient_boost=XGradientBoost(
                "experiments/models/scifact_xgb_es+vs+rr_n.json"
            ),
            embeddings=embeddings
        )


    def test_ingest(self):
        docs = [
            Document(page_content="content1", metadata={"title": "title1", "source": "source_test1"}),
            Document(page_content="content2", metadata={"title": "title2", "source": "source_test2"}),
        ]
        self.denser_retriever.ingest(docs)
        # Add assertions to verify the ingestion process

    def test_retrieve(self):
        docs = [
            Document(page_content="content1", metadata={"title": "title1", "source": "source_test1"}),
            Document(page_content="content2", metadata={"title": "title2", "source": "source_test2"}),
        ]
        self.denser_retriever.ingest(docs)
        query = "content1"
        k = 2
        results = self.denser_retriever.retrieve(query, k)
        assert len(results) == k
        assert results[0][0].page_content == "content1"

    # def test_clear(self):
    #     self.denser_retriever.delete_all()
    #     # Add assertions to verify the clearing process
    #     assert True

    def test_get_field_categories(self):
        docs = [
            Document(page_content="content1", metadata={"title": "title1", "source": "source_test1"}),
            Document(page_content="content2", metadata={"title": "title2", "source": "source_test2"}),
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
            Document(page_content="content1", metadata={"title": "title1", "source": "source_test1"}),
            Document(page_content="content2", metadata={"title": "title2", "source": "source_test2"}),
        ]
        self.denser_retriever.ingest(docs)
        fields = self.denser_retriever.get_filter_fields()
        assert len(fields) == 4

    def test_delete_by_source(self):
        docs = [
            Document(page_content="content1", metadata={"title": "title1", "source": "source_test1"}),
            Document(page_content="content2", metadata={"title": "title2", "source": "source_test2"}),
        ]
        self.denser_retriever.ingest(docs)
        self.denser_retriever.delete(source_id="source_test1")
        results = self.denser_retriever.retrieve("content1", k=1)
        # only content2 should be retrieved
        assert len(results) == 1

    def test_delete_by_id(self):
        docs = [
            Document(page_content="content1", metadata={"title": "title1", "source": "source_test1"}),
            Document(page_content="content2", metadata={"title": "title2", "source": "source_test2"}),
        ]
        ids = self.denser_retriever.ingest(docs)
        self.denser_retriever.delete(ids=[ids[0]])
        results = self.denser_retriever.retrieve("content1", k=1)
        assert len(results) == 1

    def test_delete_all(self):
        docs = [
            Document(page_content="content1", metadata={"title": "title1", "source": "source_test1"}),
            Document(page_content="content2", metadata={"title": "title2", "source": "source_test2"}),
        ]
        self.denser_retriever.ingest(docs)
        self.denser_retriever.delete_all()
        assert True
