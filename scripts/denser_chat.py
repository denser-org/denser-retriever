import openai
import streamlit as st
from elasticsearch import Elasticsearch
from container.denser.config import es_host, es_user, es_passwd, default_openai_model, openai_api_key, topk_passages, \
    use_vector_search, vector_search_engine, keyword_weight, vector_weight, topk_passages_rerank, rerank_bs, emb_dims, \
    emb_model, milvus_host, milvus_port, milvus_user, milvus_passwd
from container.denser.prompt_lib import prompt_default, prompt_legal
import time
import json
import numpy as np
import argparse
import os
import sys
import glob

sys.path.insert(0, 'container/denser')
from container.denser.utils import get_logger_file, postprocess
from container.denser.config import output_prefix, context_window
from container.denser.reranker import Reranker
import tiktoken
from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
)

debug = False
openai.api_key = openai_api_key
es = Elasticsearch(
    hosts=[es_host],
    basic_auth=(es_user, es_passwd),
    request_timeout=600
)
reranker = Reranker()


def search(query_text, passage_index, topk):
    query_dict = {
        "query": {
            "bool": {
                "should": [
                    {
                        "match": {
                            "title": {
                                "query": query_text,
                                "boost": 2.0  # Boost the "title" field with a higher weight
                            }
                        }
                    },
                    {
                        "match": {
                            "content": query_text
                        }
                    }
                ]
            }
        },
        "_source": True,
    }
    res = es.search(index=passage_index, body=query_dict, size=topk)
    topk_used = min(len(res['hits']['hits']), topk)
    passages = []
    for id in range(topk_used):
        _source = res['hits']['hits'][id]['_source']
        passage = {
            'source': _source['_meta']['source'],
            'text': _source['content'],
            'title': _source['_meta']['title'],
            'id': _source['_meta']['id'],
            'score': res['hits']['hits'][id]['_score']
        }
        passages.append(passage)
    return passages


def search_embedding(query_text, topk, faiss_index, metadatas):
    response = openai.Embedding.create(
        input=[query_text],
        engine=emb_model,
        dimensions=emb_dims
    )
    embeddings = [embedding['embedding'] for embedding in response['data']]
    query_embedding = np.reshape(np.array(embeddings).astype('float32'), (1, -1))
    diss, inds = faiss_index.search(query_embedding, topk)
    topk_used = min(len(inds.tolist()[0]), topk)
    passages = []
    inds_list = inds.tolist()[0]
    diss_list = diss.tolist()[0]
    for id in range(topk_used):
        index = inds_list[id]
        passage = {
            'source': metadatas[index]['source'],
            'text': metadatas[index]['text'],
            'title': metadatas[index]['title'],
            'id': metadatas[index]['id'],
            'score': diss_list[id]
        }
        passages.append(passage)
    return passages

def search_embedding_milvus(query_text, topk, milvus_index, metadatas):
    response = openai.Embedding.create(
        input=[query_text],
        engine=emb_model,
        dimensions=emb_dims
    )
    embeddings = [embedding['embedding'] for embedding in response['data']]
    query_embedding = np.array(embeddings)
    search_params = {
        "metric_type": "L2",
        "params": {"nprobe": 10},
    }
    result = milvus_index.search(query_embedding, "embeddings", search_params, limit=topk, output_fields=["pk"])
    topk_used = min(len(result[0]), topk)
    passages = []
    for id in range(topk_used):
        hit = result[0][id]
        index = hit.entity.id
        passage = {
            'source': metadatas[index]['source'],
            'text': metadatas[index]['text'],
            'title': metadatas[index]['title'],
            'id': metadatas[index]['id'],
            'score': hit.entity.distance
        }
        passages.append(passage)
    return passages

def build_dicts(retrieval_res):
    uid_to_passages, uid_to_scores = {}, {}
    for passage in retrieval_res:
        source, id = passage["source"], passage["id"]
        uid_str = f"{source}-{id}"
        uid_to_passages[uid_str] = passage
        uid_to_scores[uid_str] = passage["score"]
    return uid_to_passages, uid_to_scores


def merge_score(uid_to_scores_1, uid_to_scores_2):
    uid_to_score = {}
    all_uids = set().union(*[uid_to_scores_1, uid_to_scores_2])
    for uid in all_uids:
        # Negative distance score as a measure of relevance
        uid_to_score[uid] = keyword_weight * uid_to_scores_1.get(uid, 0) - vector_weight * uid_to_scores_2.get(uid, 0)
    return uid_to_score


def merge_results(es_res, vec_res, topk):
    uid_to_passages, uid_to_scores_es = build_dicts(es_res)
    uid_to_passages_vector, uid_to_scores_vector = build_dicts(vec_res)
    uid_to_passages.update(uid_to_passages_vector)
    uid_to_scores = merge_score(uid_to_scores_es, uid_to_scores_vector)
    assert len(uid_to_passages) == len(uid_to_scores)
    sorted_uids = sorted(uid_to_scores.items(), key=lambda x: x[1], reverse=True)
    if len(sorted_uids) > topk:
        sorted_uids = sorted_uids[:topk]
    passages = []
    for (uid, _) in sorted_uids:
        _passage = uid_to_passages[uid]
        passage = {'source': _passage['source'],
                   'text': _passage['text'],
                   'title': _passage['title'],
                   'id': _passage['id'],
                   'score': uid_to_scores[uid]
                   }
        passages.append(passage)
    return passages



def denser_chat(passage_index, starting_url, optional_str, language="en"):
    log_dir = os.path.join(output_prefix, f"exp_{passage_index}")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logger = get_logger_file(os.path.join(log_dir, f"{passage_index}_log"))
    st.title("Denser")
    st.caption(f"Starting URL: {starting_url}")
    if optional_str:
        st.caption(f"{optional_str}")
    st.divider()
    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = default_openai_model

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    if use_vector_search:
        directory = os.path.join(output_prefix, f"exp_{passage_index}")
        passage_file_pattern = os.path.join(directory, "passages_*.jsonl")
        passage_files = glob.glob(passage_file_pattern)
        metadatas = []
        logger.info(f"Loading vector DB meta data")
        for file in passage_files:
            logger.info(f"Loading {file}")
            for line in open(file, "r"):
                passage = json.loads(line)
                metadatas.append(passage)
        if vector_search_engine == "faiss":
            import faiss
            index_file = os.path.join(output_prefix, f"exp_{passage_index}", "faiss_index")
            faiss_index = faiss.read_index(index_file)
        else: # milvus
            connections.connect("default", host=milvus_host, port=milvus_port, user=milvus_user,
                            password=milvus_passwd)
            has = utility.has_collection(passage_index)
            assert has == True
            fields = [
                FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=False, max_length=100),
                FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=emb_dims)
            ]
            schema = CollectionSchema(fields, "Milvus schema")
            milvus_index = Collection(passage_index, schema, consistency_level="Strong")
            logger.info(f"Loading milvus index")
            milvus_index.load()

    if query := st.chat_input("Input your query here"):
        with st.chat_message("user"):
            st.markdown(query)

        ## modify prompt to include retrieval results
        start_time = time.time()
        if use_vector_search:
            es_res = search(query, passage_index, topk_passages)
            if vector_search_engine == "faiss":
                vec_res = search_embedding(query, topk_passages, faiss_index, metadatas)
                logger.info(f"Faiss top passages {vec_res}")
            else: # milvus
                vec_res = search_embedding_milvus(query, topk_passages, milvus_index, metadatas)
                logger.info(f"Milvus top passages {vec_res}")
            passages = merge_results(es_res, vec_res, topk_passages)
        else:
            passages = search(query, passage_index, topk_passages)

        retrieve_time_sec = time.time() - start_time
        start_time = time.time()
        # passages_original = reranker.rerank(query, passages, rerank_bs)
        # passages = passages_original[:topk_passages_rerank]
        if len(passages) > 0 and language == "en":
            passages = reranker.rerank(query, passages, rerank_bs)
            logger.info(f"Rerank passages")
        rerank_time_sec = time.time() - start_time
        st.write(f"Retrieve time: {retrieve_time_sec:.3f} sec. Rerank time: {rerank_time_sec:.3f} sec")
        passages_original = passages
        passages = passages_original[:topk_passages_rerank]
        logger.info(f"Select top {topk_passages_rerank} passages")

        prompt = prompt_default
        prompt += f"### Query:\n{query}\n"
        if len(passages) > 0:
            prompt += f"\n### Context:\n{passages}\n"
        if language == "en":
            context_limit = 4 * context_window
        else:
            context_limit = context_window
        if len(prompt) > context_limit:
            prompt = prompt[:context_limit]
        prompt += f"### Response:"

        st.session_state.messages.append({"role": "user", "content": prompt})

        enc = tiktoken.encoding_for_model(default_openai_model)
        prompt_length = len(enc.encode(prompt))
        logger.info(f"prompt length:{prompt_length}")

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            for response in openai.ChatCompletion.create(
                    model=st.session_state["openai_model"],
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    stream=True,
                    top_p=0,
                    temperature=0.0
            ):
                full_response += response.choices[0].delta.get("content", "")
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)

        st.session_state.messages.append({"role": "assistant", "content": full_response})
        st.session_state.messages = []
        for i, passage in enumerate(passages):
            score_rerank  = passage['score_rerank'] if 'score_rerank' in passage else 0
            st.write(
                f"[{(i + 1)}]  [{passage['title']}]({passage['source']})  \n{passage['source']}  \n**Score**: {passage['score']} **Score_rerank**: {score_rerank}  \n{passage['text']}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--passage_index', type=str, required=True)
    parser.add_argument('--starting_url', type=str, required=True)
    parser.add_argument('--optional_str', type=str, default="")
    args = parser.parse_args()
    denser_chat(args.passage_index, args.starting_url, args.optional_str)


if __name__ == "__main__":
    main()
