import logging
import os
from enum import Enum

from denser_retriever.reranker import Reranker
from denser_retriever.retriever_elasticsearch import RetrieverElasticSearch
from denser_retriever.retriever_milvus import RetrieverMilvus
from denser_retriever.utils import (
    build_dicts,
    save_HF_corpus_as_denser_passages,
    save_denser_qrels,
    save_denser_queries,
)
from denser_retriever.utils_data import HFDataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RetrieverFeature(Enum):
    KEYWORD_RANK = 1
    KEYWORD_SCORE = 2
    KEYWORD_MISS = 3
    VECTOR_RANK = 4
    VECTOR_SCORE = 5
    VECTOR_MISS = 6
    RERANKER_RANK = 7
    RERANKER_SCORE = 8
    RERANKER_MISS = 9


def generate_data(dataset_name, split, config_file, ingest=False):
    corpus, queries, qrels = HFDataLoader(
        hf_repo=dataset_name,
        hf_repo_qrels=None,
        streaming=False,
        keep_in_memory=False,
    ).load(split=split)

    data_dir_name = os.path.basename(dataset_name)
    index_name = data_dir_name.replace("-", "_")
    retriever_keyword = RetrieverElasticSearch(index_name, config_file)
    retriever_vector = RetrieverMilvus(index_name, config_file)
    reranker = Reranker(retriever_keyword.settings.rerank.rerank_model)

    output_prefix = retriever_keyword.settings.output_prefix
    exp_dir = os.path.join(output_prefix, f"exp_{data_dir_name}", split)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    passage_file = os.path.join(exp_dir, "passages.jsonl")
    save_HF_corpus_as_denser_passages(
        corpus, passage_file, retriever_keyword.settings.max_doc_size
    )
    query_file = os.path.join(exp_dir, "queries.jsonl")
    save_denser_queries(queries, query_file)
    qrels_file = os.path.join(exp_dir, "qrels.jsonl")
    save_denser_qrels(qrels, qrels_file)
    feature_file = os.path.join(exp_dir, "features.svmlight")
    feature_out = open(feature_file, "w")

    if ingest:
        retriever_keyword.ingest(
            passage_file, retriever_keyword.settings.keyword.es_ingest_passage_bs
        )
        retriever_vector.ingest(
            passage_file, retriever_vector.settings.vector.vector_ingest_passage_bs
        )

    for i, q in enumerate(queries):
        if (
            retriever_keyword.settings.max_query_size > 0
            and i >= retriever_keyword.settings.max_query_size
        ):
            break
        logger.info(f"Processing query {i}")
        qid = q["id"]
        passages_keyword = retriever_keyword.retrieve(q["text"], {}, qid)
        passages_vector = retriever_vector.retrieve(q["text"], {}, qid)
        combined_passages = []
        seen_ids = set()

        # Combine both passages
        for passage in passages_keyword + passages_vector:
            if passage["source"] not in seen_ids:
                combined_passages.append(passage)
                seen_ids.add(passage["source"])

        uid_to_passages_1, uid_to_scores_1, uid_to_ranks_1 = build_dicts(
            passages_keyword
        )
        uid_to_passages_2, uid_to_scores_2, uid_to_ranks_2 = build_dicts(
            passages_vector
        )

        labels = qrels[qid]
        passages_reranked = reranker.rerank(
            q["text"],
            combined_passages,
            retriever_keyword.settings.rerank.rerank_bs,
        )
        uid_to_passages_reranked, uid_to_scores_reranked, uid_to_ranks_reranked = (
            build_dicts(passages_reranked)
        )
        feature_str = ""
        for pid in uid_to_passages_reranked.keys():
            label = labels.get(pid, 0)
            feature_str += str(label)
            feature_str += f" qid:{qid}"
            rank = uid_to_ranks_1.get(pid, -1)
            feature_str += f" 1:{rank}"
            feature_str += f" 2:{uid_to_scores_1.get(pid, -1000)}"
            miss = 1 if rank == -1 else 0
            feature_str += f" 3:{miss}"
            rank = uid_to_ranks_2.get(pid, -1)
            feature_str += f" 4:{rank}"
            feature_str += f" 5:{uid_to_scores_2.get(pid, -1000)}"
            miss = 1 if rank == -1 else 0
            feature_str += f" 6:{miss}"
            assert pid in uid_to_ranks_reranked
            feature_str += f" 7:{uid_to_ranks_reranked[pid]}"
            feature_str += f" 8:{uid_to_scores_reranked[pid]}"
            feature_str += " 9:0"
            feature_str += f" # {pid}\n"
        feature_out.write(feature_str)
