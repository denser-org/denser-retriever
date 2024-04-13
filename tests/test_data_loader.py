from denser.utils_data import HFDataLoader

class TestDataLoader:
    def test_data_loader(self):
        split = "test"
        corpus, queries, qrels = HFDataLoader(
            hf_repo="mteb/nfcorpus",
            hf_repo_qrels=None,
            streaming=False,
            keep_in_memory=False,
        ).load(split=split)
        # print(f"Corpus size: {len(corpus)}")
        assert len(corpus) == 3633
        assert len(queries) == 323
