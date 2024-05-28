from denser_retriever.retriever_general import RetrieverGeneral
from denser_retriever.utils_data import HFDataLoader
from denser_retriever.utils import evaluate, passages_to_dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

corpus, queries, qrels = HFDataLoader(
    hf_repo="mteb/msmarco",
    hf_repo_qrels=None,
    streaming=False,
    keep_in_memory=False,
).load(split="dev")

index = "msmarco"
retriever_denser = RetrieverGeneral(index, "experiments/config_server.yaml")

res = {}
for i, d in enumerate(queries):
    logger.info(f"Processing query {i}")
    qid, query = d["id"], d["text"]
    passages, docs = retriever_denser.retrieve(query, {})
    passage_to_score = passages_to_dict(passages, False)
    res[qid] = passage_to_score

metric = evaluate(qrels, res)
ndcg_passage = metric[0]["NDCG@10"]
logger.info(f"NDCG@10: {ndcg_passage}")
