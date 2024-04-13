from .Retriever import Retriever
from .RetrieverElasticSearch import RetrieverElasticSearch
from .RetrieverMilvus import RetrieverMilvus
from .Reranker import Reranker
import time
from .utils import get_logger, dump_passages, aggregate_passages
import yaml

logger = get_logger(__name__)

class RetrieverGeneral(Retriever):

    def __init__(self, index_name, config_file):
        config = yaml.safe_load(open(config_file))
        self.config = config
        self.retrieve_type = "general"
        config['index_name'] = index_name
        self.retrieverElasticSearch = RetrieverElasticSearch(index_name, config) \
            if config['keyword_weight'] > 0 else None
        self.retrieverMilvus = RetrieverMilvus(index_name, config) if config['vector_weight'] > 0 else None
        self.reranker = Reranker(config['rerank']['rerank_model']) if config['rerank_weight'] > 0 else None

    def build_dicts(self, passages):
        uid_to_passages, uid_to_scores, uid_to_ranks = {}, {}, {}
        for i, passage in enumerate(passages):
            source, id = passage["source"], passage.get("pid", -1)
            if id == -1:
                uid_str = source
            else:
                uid_str = f"{source}-{id}"
            assert uid_str not in uid_to_passages
            assert uid_str not in uid_to_scores
            assert uid_str not in uid_to_ranks
            uid_to_passages[uid_str] = passage
            uid_to_scores[uid_str] = passage["score"]
            uid_to_ranks[uid_str] = i + 1
        return uid_to_passages, uid_to_scores, uid_to_ranks

    def merge_score_linear(self, uid_to_scores_1, uid_to_scores_2, weight_2):
        uid_to_score = {}
        all_uids = set().union(*[uid_to_scores_1, uid_to_scores_2])
        for uid in all_uids:
            uid_to_score[uid] = uid_to_scores_1.get(uid, 0) + weight_2 * uid_to_scores_2.get(uid, 0)
        return uid_to_score

    def merge_score_rank(self, uid_to_ranks_1, uid_to_ranks_2):
        uid_to_score = {}
        k = 60
        all_uids = set().union(*[uid_to_ranks_1, uid_to_ranks_2])
        for uid in all_uids:
            uid_to_score[uid] = 1 / (k + uid_to_ranks_1.get(uid, 1000)) + 1 / (k + uid_to_ranks_2.get(uid, 1000))
        return uid_to_score

    def merge_results(self, passages_1, passages_2, weight_2):
        if len(passages_1) == 0:
            for passage in passages_2:
                passage['score'] *= weight_2
            return passages_2
        uid_to_passages_1, uid_to_scores_1, uid_to_ranks_1 = self.build_dicts(passages_1)
        uid_to_passages_2, uid_to_scores_2, uid_to_ranks_2 = self.build_dicts(passages_2)
        # import pdb; pdb.set_trace()
        uid_to_passages_1.update(uid_to_passages_2)
        if self.config['merge'] == "linear":
            uid_to_scores = self.merge_score_linear(uid_to_scores_1, uid_to_scores_2, weight_2)
        else: # rank
            uid_to_scores = self.merge_score_rank(uid_to_ranks_1, uid_to_ranks_2)
        assert len(uid_to_passages_1) == len(uid_to_scores)
        sorted_uids = sorted(uid_to_scores.items(), key=lambda x: x[1], reverse=True)
        passages = []
        for (uid, _) in sorted_uids:
            _passage = uid_to_passages_1[uid]
            passage = {'source': _passage['source'],
                       'text': _passage['text'],
                       'title': _passage['title'],
                       'pid': _passage['pid'],
                       'score': uid_to_scores[uid]
                       }
            passages.append(passage)
        return passages

    def ingest(self, doc_or_passage_file):
        # import pdb; pdb.set_trace()
        if self.config['keyword_weight'] > 0:
            self.retrieverElasticSearch.ingest(doc_or_passage_file, self.config['keyword']['es_ingest_passage_bs'])
            logger.info(f"Done building ES index")
        if self.config['vector_weight'] > 0:
            self.retrieverMilvus.ingest(doc_or_passage_file, self.config['vector']['vector_ingest_passage_bs'])
            logger.info(f"Done building Vector DB index")


    def retrieve(self, query, topk):
        passages = []
        # import pdb; pdb.set_trace()
        if self.config['keyword_weight'] > 0:
            start_time = time.time()
            passages_es = self.retrieverElasticSearch.retrieve(query, topk)
            logger.info(f"Keyword Search before merging Passages 1: {len(passages)} Passages 2: {len(passages_es)}")
            passages = self.merge_results(passages, passages_es, self.config['keyword_weight'])
            logger.info(f"After merging Passages: {len(passages)}")
            retrieve_time_sec = time.time() - start_time
            logger.info(f"ElasticSearch Retrieve time: {retrieve_time_sec:.3f} sec.")
            dump_passages(passages, "retriever_keyword.jsonl")
        if self.config['vector_weight'] > 0:
            start_time = time.time()
            passages_vector = self.retrieverMilvus.retrieve(query, topk)
            dump_passages(passages_vector, "retriever_vector.jsonl")
            # import pdb;
            # pdb.set_trace()
            logger.info(f"Vector Search before merging Passages 1: {len(passages)} Passages 2: {len(passages_vector)}")
            passages = self.merge_results(passages, passages_vector, self.config['vector_weight'])
            logger.info(f"After merging Passages: {len(passages)}")
            # import pdb; pdb.set_trace()
            retrieve_time_sec = time.time() - start_time
            logger.info(f"Vector Retrieve time: {retrieve_time_sec:.3f} sec.")
            dump_passages(passages, "retriever_vector_merged.jsonl")
        if self.config['rerank_weight'] > 0:
            start_time = time.time()
            passages = passages[:self.config['rerank']['topk_passages']]
            passages_rerank = self.reranker.rerank(query, passages, self.config['rerank']['rerank_bs'])
            dump_passages(passages_rerank, "retriever_rerank.jsonl")
            passages = self.merge_results(passages, passages_rerank, self.config['rerank_weight'])
            rerank_time_sec = time.time() - start_time
            logger.info(f"Rerank time: {rerank_time_sec:.3f} sec.")
            dump_passages(passages, "retriever_rerank_merged.jsonl")

        if len(passages) > topk:
            passages = passages[:topk]

        docs = aggregate_passages(passages)
        return passages, docs

