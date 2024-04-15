from denser_retriever.retriever_general import RetrieverGeneral
from denser_retriever.utils import passages_to_dict


class TestRetrieverGeneral:
    def setup_method(self):
        """setup any state tied to the execution of the given method in a
        class.  setup_method is invoked for every tests method of a class.
        """

        index_name = "test_index_temp"
        self.retriever = RetrieverGeneral(index_name, "tests/config-test.yaml")

    def test_retriever_ingest(self):
        doc_or_passage_file = "tests/test_data/passages_000000.jsonl"
        self.retriever.ingest(doc_or_passage_file)

    def test_retriever_retrieve(self):
        query = "what is artify4kids"
        topk = 5
        passages, docs = self.retriever.retrieve(query, topk)
        print(passages)
        assert len(passages) == topk
        assert len(docs) > 0
        passage_to_score = passages_to_dict(passages, False)
        print(passage_to_score)
        assert len(passage_to_score) == topk
