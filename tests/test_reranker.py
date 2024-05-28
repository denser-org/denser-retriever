from denser_retriever.reranker import Reranker
from denser_retriever.retriever_elasticsearch import RetrieverElasticSearch


class TestReranker:
    def setup_method(self):
        """setup any state tied to the execution of the given method in a
        class.  setup_method is invoked for every tests method of a class.
        """
        self.retriever_elasticsearch = RetrieverElasticSearch(
            "unit_test_denser", "tests/config-denser.yaml"
        )
        rerank_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        self.reranker = Reranker(rerank_model)

    def test_retriever_ingest(self):
        self.retriever_elasticsearch.ingest(
            "tests/test_data/denser_website_passages_top10.jsonl", 10
        )

    def test_reranker(self):
        topk = self.retriever_elasticsearch.settings.keyword.topk
        query = "what is denser ai?"
        passages = self.retriever_elasticsearch.retrieve(query, {})
        assert len(passages) == topk
        assert abs(passages[0]["score"] - 6.171) < 0.01

        passages_reranked = self.reranker.rerank(query, passages, 10)
        assert len(passages_reranked) == topk
        assert abs(passages_reranked[0]["score"] - 3.357) < 0.01
