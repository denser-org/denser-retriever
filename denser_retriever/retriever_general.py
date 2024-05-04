import time

from denser_retriever.reranker import Reranker
from denser_retriever.retriever import Retriever
from denser_retriever.retriever_elasticsearch import RetrieverElasticSearch
from denser_retriever.retriever_milvus import RetrieverMilvus
from denser_retriever.utils import aggregate_passages, get_logger, merge_results, scale_results

logger = get_logger(__name__)


class RetrieverGeneral(Retriever):
    def __init__(self, index_name, config_file):
        super().__init__(index_name, config_file)
        self.retrieve_type = "general"
        self.retrieverElasticSearch = (
            RetrieverElasticSearch(index_name, config_file) if self.config["keyword_weight"] > 0 else None
        )
        self.retrieverMilvus = RetrieverMilvus(index_name, config_file) if self.config["vector_weight"] > 0 else None
        self.reranker = (
            Reranker(self.config["rerank"]["rerank_model"], self.out_reranker)
            if self.config["rerank_weight"] > 0
            else None
        )

    def ingest(self, doc_or_passage_file):
        # import pdb; pdb.set_trace()
        if self.config["keyword_weight"] > 0:
            self.retrieverElasticSearch.ingest(doc_or_passage_file, self.config["keyword"]["es_ingest_passage_bs"])
            logger.info("Done building ES index")
        if self.config["vector_weight"] > 0:
            self.retrieverMilvus.ingest(doc_or_passage_file, self.config["vector"]["vector_ingest_passage_bs"])
            logger.info("Done building Vector DB index")

    def retrieve(self, query, meta_data, query_id=None):
        passages = []
        if self.config["keyword_weight"] > 0:
            start_time = time.time()
            passages_es = self.retrieverElasticSearch.retrieve(query, meta_data, query_id)
            passages = scale_results(passages_es, self.config["keyword_weight"])
            retrieve_time_sec = time.time() - start_time
            logger.info(f"ElasticSearch passages: {len(passages)} time: {retrieve_time_sec:.3f} sec. ")
        if self.config["vector_weight"] > 0:
            start_time = time.time()
            passages_vector = self.retrieverMilvus.retrieve(query, meta_data, query_id)
            logger.info(f"Vector Search before merging Passages 1: {len(passages)} Passages 2: {len(passages_vector)}")
            passages = merge_results(passages, passages_vector, 1.0, self.config["vector_weight"], self.config["merge"])
            logger.info(f"After merging Passages: {len(passages)}")
            retrieve_time_sec = time.time() - start_time
            logger.info(f"Vector Retrieve time: {retrieve_time_sec:.3f} sec.")
        if self.config["rerank_weight"] > 0:
            start_time = time.time()
            passages = passages[: self.config["rerank"]["topk"]]
            passages_rerank = self.reranker.rerank(query, passages, self.config["rerank"]["rerank_bs"], query_id)
            passages = merge_results(passages, passages_rerank, 1.0, self.config["rerank_weight"], self.config["merge"])
            rerank_time_sec = time.time() - start_time
            logger.info(f"Rerank time: {rerank_time_sec:.3f} sec.")

        if len(passages) > self.config["rerank"]["topk"]:
            passages = passages[: self.config["rerank"]["topk"]]

        docs = aggregate_passages(passages)
        return passages, docs

    def get_field_categories(self, field, topk):
        return self.retrieverElasticSearch.get_categories(field, topk)
