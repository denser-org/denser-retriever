from langchain_core.documents import Document
from denser_retriever.core.retriever import DenserRetriever
from denser_retriever.config import CombineConfig, LRConfig
from tests.utils import elasticsearch, milvus, reranker, embeddings


class TestRetriever:
    def setup_method(self):
        self.retriever = DenserRetriever(
            index_name="unit_test_retriever",
            keyword_search=elasticsearch,
            vector_db=milvus,
            reranker=reranker,
            embeddings=embeddings,
            combine_config=CombineConfig(
                method="fusion",
                keyword_top_k=100,
                vector_top_k=100,
                reranker_top_k=100,
                lr_config=LRConfig(
                    lr_features="es+vs+rr",
                    lr_model="denser_retriever/models/weights_es+vs+rr_msmarco.json"
                )
            ),
        )
        self.test_docs = [
            Document(page_content="content1", metadata={"title": "title1", "source": "source1"}),
            Document(page_content="content2", metadata={"title": "title2", "source": "source2"}),
            Document(page_content="content12", metadata={"title": "title12", "source": "source12"})
        ]

    def test_retrieve(self):
        doc_ids, _ = self.retriever.ingest(self.test_docs)
        results = self.retriever.retrieve("content1", 2, self.retriever.combine_config)
        assert len(results.documents) == 2
        assert results.documents[0][0].page_content == "content1"

    def test_delete_by_id(self):
        doc_ids, _ = self.retriever.ingest(self.test_docs)
        stats = self.retriever.get_index_stats()
        assert stats["es_docs"] == 3 and stats["vector_docs"] == 3

        self.retriever.delete(ids=[doc_ids[0]])
        stats = self.retriever.get_index_stats()
        assert stats["es_docs"] == 2 and stats["vector_docs"] == 2

    def test_delete_by_source(self):
        self.retriever.ingest(self.test_docs)
        stats = self.retriever.get_index_stats()
        assert stats["es_docs"] == 3 and stats["vector_docs"] == 3

        self.retriever.delete(source_id="source1")
        stats = self.retriever.get_index_stats()
        assert stats["es_docs"] == 2 and stats["vector_docs"] == 2

    def test_delete_by_source_url(self):
        self.retriever.ingest(self.test_docs)
        stats = self.retriever.get_index_stats()
        assert stats["es_docs"] == 3 and stats["vector_docs"] == 3

        self.retriever.delete(source_url="source1")
        stats = self.retriever.get_index_stats()
        assert stats["es_docs"] == 1 and stats["vector_docs"] == 1

    def test_delete_all(self):
        self.retriever.ingest(self.test_docs)
        stats = self.retriever.get_index_stats()
        assert stats["es_docs"] == 3 and stats["vector_docs"] == 3

        # Test delete_all without deleting indices
        self.retriever.delete_all(delete_index=False)
        status = self.retriever.check_indices()
        stats = self.retriever.get_index_stats()
        assert status["es_valid"] and status["vector_valid"]
        assert stats["es_docs"] == 0 and stats["vector_docs"] == 0

        # Test delete_all with deleting indices
        self.retriever.delete_all(delete_index=True)
        status = self.retriever.check_indices()
        assert not status["es_valid"] and not status["vector_valid"]