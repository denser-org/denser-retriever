from denser_retriever.retriever_general import RetrieverGeneral
from denser_retriever.utils import passages_to_dict


class TestRetrieverGeneral:
    def setup_method(self):
        """setup any state tied to the execution of the given method in a
        class.  setup_method is invoked for every tests method of a class.
        """
        self.retriever = RetrieverGeneral("unit_test_index", "tests/config-test.yaml")
        self.retriever_cn = RetrieverGeneral("unit_test_cn_index", "tests/config-test-cn.yaml")

    def test_retriever_ingest(self):
        self.retriever.ingest("tests/test_data/denser_website_passages_top10.jsonl")
        self.retriever_cn.ingest("tests/test_data/cpws_passages_top10.jsonl")

    def test_retriever_retrieve(self):
        topk = 5
        retrievers = [self.retriever, self.retriever_cn]
        queries = ["what is denser ai?", "买卖合同纠纷"]
        for retriever, query in zip(retrievers, queries):
            passages, docs = retriever.retrieve(query, {}, topk)
            print(passages)
            assert len(passages) == topk
            assert len(docs) > 0
            passage_to_score = passages_to_dict(passages, False)
            print(passage_to_score)
            assert len(passage_to_score) == topk
