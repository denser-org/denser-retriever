from denser_retriever.reranker import Reranker
from denser_retriever.retriever_elasticsearch import RetrieverElasticSearch


class TestReranker:
    def setup_method(self):
        """setup any state tied to the execution of the given method in a
        class.  setup_method is invoked for every tests method of a class.
        """
        self.retriever_elasticsearch = RetrieverElasticSearch("unit_test_index", "tests/config-test.yaml")
        self.retriever_elasticsearch_cn = RetrieverElasticSearch("unit_test_cn_index", "tests/config-test-cn.yaml")
        rerank_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        self.reranker = Reranker(rerank_model)

    def test_retriever_ingest(self):
        self.retriever_elasticsearch.ingest("tests/test_data/denser_website_passages_top10.jsonl", 10)
        self.retriever_elasticsearch_cn.ingest("tests/test_data/cpws_passages_top10.jsonl", 10)

    def test_reranker(self):
        topk = 5
        query = "what is denser ai?"
        passages = self.retriever_elasticsearch.retrieve(query, {}, topk)
        assert len(passages) == 5

        passages_reranked = self.reranker.rerank(query, passages, 10)
        assert len(passages_reranked) == 5

    def test_reranker_cn(self):
        topk = 5
        query = "买卖合同纠纷"
        passages = self.retriever_elasticsearch_cn.retrieve(query, {}, topk)
        assert len(passages) == 5

        passages_reranked = self.reranker.rerank(query, passages, 10)
        assert len(passages_reranked) == 5
