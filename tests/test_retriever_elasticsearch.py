from datetime import date, datetime

from denser_retriever.retriever_elasticsearch import RetrieverElasticSearch


class TestRetrieverElasticSearch:
    def setup_method(self):
        """setup any state tied to the execution of the given method in a
        class.  setup_method is invoked for every tests method of a class.
        """

        self.retriever_denser = RetrieverElasticSearch(
            "unit_test_denser", "tests/config-denser.yaml"
        )
        self.retriever_titanic = RetrieverElasticSearch(
            "unit_test_titanic",
            "tests/config-titanic.yaml",
        )
        self.retriever_cpws = RetrieverElasticSearch(
            "unit_test_cpws", "tests/config-cpws.yaml"
        )

    def test_ingest(self):
        self.retriever_denser.ingest(
            "tests/test_data/denser_website_passages_top10.jsonl", 10
        )
        self.retriever_titanic.ingest("tests/test_data/titanic_top10.jsonl", 10)
        self.retriever_cpws.ingest("tests/test_data/cpws_passages_top10.jsonl", 10)

    def test_retrieve_denser(self):
        query = "what is denser ai?"
        passages = self.retriever_denser.retrieve(query, {})
        topk = self.retriever_denser.settings.keyword.topk
        assert len(passages) == topk
        assert (
            passages[0]["title"]
            == "AI Website Search and Conversational Chatbot | Denser.ai"
        )
        assert abs(passages[0]["score"] - 6.171114) < 0.01

    def test_retrieve_titanic(self):
        query = "cumings"
        meta_data = {"Sex": "female"}
        passages = self.retriever_titanic.retrieve(query, meta_data)
        topk = self.retriever_titanic.settings.keyword.topk
        assert len(passages) == topk
        assert passages[0]["source"] == "2"
        abs(passages[0]["score"] - 5.9857755) < 0.01
        for passage in passages:
            assert passage["Sex"] == "female"

        meta_data = {"Birthday": (date(1873, 1, 1), date(1874, 12, 30))}
        passages = self.retriever_titanic.retrieve(query, meta_data)
        for passage in passages:
            d = datetime.strptime(passage["Birthday"], "%Y-%m-%d").date()
            assert d > date(1873, 1, 1) and d < date(1874, 12, 30)

        meta_data = {"Sex": "male"}
        passages = self.retriever_titanic.retrieve(query, meta_data)
        for passage in passages:
            assert passage["Sex"] == "male"

    def test_retrieve_cpws(self):
        query = "买卖合同纠纷"
        meta_data = {"所属地区": "新昌县"}
        passages = self.retriever_cpws.retrieve(query, meta_data)
        assert len(passages) == 3
        for passage in passages:
            assert passage["所属地区"] == "新昌县"

        meta_data = {"裁判日期": (date(2021, 10, 4), date(2021, 10, 5))}
        passages = self.retriever_cpws.retrieve(query, meta_data)
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
        passages = self.retriever_cpws.retrieve(query, meta_data)
        assert len(passages) == 1
        for passage in passages:
            assert passage["所属地区"] == "新昌县"
            assert (
                passage["裁判日期"] == "2021-10-04"
                or passage["裁判日期"] == "2021-10-05"
            )
