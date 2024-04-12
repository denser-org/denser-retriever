from denser.utils_data import HFDataLoader

class TestDataLoader:

    def setup_method(self):
        """setup any state tied to the execution of the given method in a
        class.  setup_method is invoked for every tests method of a class.
        """

        index_name = "test_index_temp"
        # self.retriever = RetrieverGeneral(index_name)

    def test_data_loader(self):
        split = "test"
        corpus, queries, qrels = HFDataLoader(
            hf_repo="mteb/nfcorpus",
            hf_repo_qrels=None,
            streaming=False,
            keep_in_memory=False,
        ).load(split=split)
        # import pdb; pdb.set_trace()
        # print(f"Corpus size: {len(corpus)}")
        assert len(corpus) == 3633
        assert len(queries) == 323
