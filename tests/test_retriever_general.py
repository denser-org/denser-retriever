from denser_retriever.retriever_general import RetrieverGeneral
from datetime import date


class TestRetrieverGeneral:
    def setup_method(self):
        """setup any state tied to the execution of the given method in a
        class.  setup_method is invoked for every tests method of a class.
        """
        self.retriever_denser = RetrieverGeneral("unit_test_denser", "tests/config-denser.yaml")
        self.retriever_titanic = RetrieverGeneral("unit_test_titanic", "tests/config-titanic.yaml")
        self.retriever_cpws = RetrieverGeneral("unit_test_cpws", "tests/config-cpws.yaml")

    def test_ingest(self):
        self.retriever_denser.ingest("tests/test_data/denser_website_passages_top10.jsonl")
        self.retriever_titanic.ingest("tests/test_data/titanic_top10.jsonl")
        self.retriever_cpws.ingest("tests/test_data/cpws_passages_top10.jsonl")

    def test_retrieve_denser(self):
        topk = self.retriever_denser.config["rerank"]["topk"]
        query = "what is denser ai?"
        passages, docs = self.retriever_denser.retrieve(query, {})
        assert len(passages) == topk
        assert passages[0]["title"] == 'AI Website Search and Conversational Chatbot | Denser.ai'
        assert abs(passages[0]["score"] - 4.1690) < 0.01

    def test_retrieve_titanic(self):
        query = "cumings"
        meta_data = {"Sex": "female"}
        passages, docs = self.retriever_titanic.retrieve(query, meta_data)
        assert passages[0]["source"] == "2"
        assert abs(passages[0]["score"] - 3.6725) < 0.01


    def test_retrieve_cpws(self):
        query = "买卖合同纠纷"
        meta_data = {"所属地区": "新昌县"}
        passages, docs = self.retriever_cpws.retrieve(query, meta_data)
        assert len(passages) == 3
        for passage in passages:
            assert passage["所属地区"] == "新昌县"

        meta_data = {"裁判日期": (date(2021, 10, 4), date(2021, 10, 5))}
        passages, docs = self.retriever_cpws.retrieve(query, meta_data)
        assert len(passages) == 2
        for passage in passages:
            assert passage["裁判日期"] == "2021-10-04" or passage["裁判日期"] == "2021-10-05"

        meta_data = {"所属地区": "新昌县", "裁判日期": (date(2021, 10, 4), date(2021, 10, 5))}
        passages, docs = self.retriever_cpws.retrieve(query, meta_data)
        assert len(passages) == 1
        for passage in passages:
            assert passage["所属地区"] == "新昌县"
            assert passage["裁判日期"] == "2021-10-04" or passage["裁判日期"] == "2021-10-05"
