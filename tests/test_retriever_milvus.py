from denser_retriever.retriever_milvus import RetrieverMilvus
from datetime import date


class TestRetrieverMilvus:
    def setup_method(self):
        """setup any state tied to the execution of the given method in a
        class.  setup_method is invoked for every tests method of a class.
        """
        self.retriever_milvus = RetrieverMilvus("unit_test_index", "tests/config-test.yaml")
        self.retriever_milvus_cn = RetrieverMilvus("unit_test_cn_index", "tests/config-test-cn.yaml")

    def test_retriever_ingest(self):
        self.retriever_milvus.ingest("tests/test_data/denser_website_passages_top10.jsonl", 10)
        self.retriever_milvus_cn.ingest("tests/test_data/cpws_passages_top10.jsonl", 10)

    def test_retriever_retrieve(self):
        topk = 5
        query = "what is denser ai?"
        passages = self.retriever_milvus.retrieve(query, {}, topk)
        assert len(passages) == 5

    def test_retriever_retrieve_cn(self):
        topk = 10
        queries = ["", "买卖合同纠纷"]
        meta_data = {"所属地区": "新昌县"}
        for query in queries:
            passages = self.retriever_milvus_cn.retrieve(query, meta_data, topk)
            assert len(passages) == 3
            for passage in passages:
                assert passage["所属地区"] == "新昌县"

        queries = ["", "买卖合同纠纷"]
        meta_data = {"裁判日期": (date(2021, 10, 4), date(2021, 10, 5))}
        for query in queries:
            passages = self.retriever_milvus_cn.retrieve(query, meta_data, topk)
            assert len(passages) == 2
            for passage in passages:
                assert passage["裁判日期"] == "2021-10-04" or passage["裁判日期"] == "2021-10-05"

        queries = ["", "买卖合同纠纷"]
        meta_data = {"所属地区": "新昌县", "裁判日期": (date(2021, 10, 4), date(2021, 10, 5))}
        for query in queries:
            passages = self.retriever_milvus_cn.retrieve(query, meta_data, topk)
            assert len(passages) == 1
            for passage in passages:
                assert passage["所属地区"] == "新昌县"
                assert passage["裁判日期"] == "2021-10-04" or passage["裁判日期"] == "2021-10-05"
