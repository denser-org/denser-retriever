es_weight = 0.5
vector_weight = 0.5
rerank_weight = 0.5
# linear or rank
merge = "linear"

es_user = "elastic"
es_passwd = "WzAkbzjZj9AfNXxzmOmp"
es_host = "http://35.93.131.127:9200"
es_ingest_passage_bs = 5000

milvus_host = "35.93.131.127"
milvus_port = "19530"
milvus_user = "root"
milvus_passwd = "Milvus"
emb_model = "sentence-transformers/all-MiniLM-L6-v2"
emb_dims = 384
vector_ingest_passage_bs = 1000

# https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-12-v2
# cross-encoder/ms-marco-MiniLM-L-2-v2, cross-encoder/ms-marco-MiniLM-L-4-v2, or cross-encoder/ms-marco-MiniLM-L-6-v2
# rerank_model = "cross-encoder/ms-marco-TinyBERT-L-2-v2"
rerank_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"
rerank_bs = 100
topk_passages = 100

output_prefix = "/home/ubuntu/efs/denser_output_retriever/"

## temp parameters
max_doc_size = 10
max_query_size = 2