from datetime import date

from denser_retriever.retriever_milvus import RetrieverMilvus


class TestRetrieverMilvus:
    def setup_method(self):
        """setup any state tied to the execution of the given method in a
        class.  setup_method is invoked for every tests method of a class.
        """
        self.retriever_milvus_denser = RetrieverMilvus(
            "unit_test_denser", "tests/config-denser.yaml"
        )
        self.retriever_milvus_titanic = RetrieverMilvus(
            "unit_test_titanic",
            "tests/config-titanic.yaml",
        )
        self.retriever_milvus_cpws = RetrieverMilvus(
            "unit_test_cpws", "tests/config-cpws.yaml"
        )

    def test_ingest(self):
        self.retriever_milvus_denser.ingest(
            "tests/test_data/denser_website_passages_top10.jsonl", 10
        )
        self.retriever_milvus_titanic.ingest("tests/test_data/titanic_top10.jsonl", 10)
        self.retriever_milvus_cpws.ingest(
            "tests/test_data/cpws_passages_top10.jsonl", 10
        )

    def test_retrieve_denser(self):
        topk = self.retriever_milvus_denser.config.topk
        query = "what is denser ai?"
        passages = self.retriever_milvus_denser.retrieve(query, {})
        # import pdb; pdb.set_trace()
        assert len(passages) == topk
        assert passages[0]["title"] == "Blog | Denser.ai"
        assert abs(passages[0]["score"] + 1.126) < 0.01

    def test_retrieve_titanic(self):
        topk = self.retriever_milvus_titanic.config.topk
        query = "cumings"
        meta_data = {"Sex": "male"}
        passages = self.retriever_milvus_titanic.retrieve(query, meta_data)
        # import pdb; pdb.set_trace()
        assert len(passages) == topk
        # TODO: why fail?
        # for passage in passages:
        #     assert passage["Sex"] == "female"

    def test_retrieve_cpws(self):
        query = "买卖合同纠纷"
        meta_data = {"所属地区": "新昌县"}
        passages = self.retriever_milvus_cpws.retrieve(query, meta_data)
        assert len(passages) == 3
        for passage in passages:
            assert passage["所属地区"] == "新昌县"

        meta_data = {"裁判日期": (date(2021, 10, 4), date(2021, 10, 5))}
        passages = self.retriever_milvus_cpws.retrieve(query, meta_data)
        assert len(passages) == 2
        for passage in passages:
            assert (
                passage["裁判日期"] == "2021-10-04"
                or passage["裁判日期"] == "2021-10-05"
            )

        meta_data = {
            "所属地区": "新昌县",
            "裁判日期": (date(2021, 10, 4), date(2021, 10, 5)),
        }
        passages = self.retriever_milvus_cpws.retrieve(query, meta_data)
        assert len(passages) == 1
        for passage in passages:
            assert passage["所属地区"] == "新昌县"
            assert (
                passage["裁判日期"] == "2021-10-04"
                or passage["裁判日期"] == "2021-10-05"
            )
