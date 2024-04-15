import json

import numpy as np
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)
from sentence_transformers import SentenceTransformer

from denser_retriever.retriever import Retriever
from denser_retriever.utils import get_logger

logger = get_logger(__name__)


class RetrieverMilvus(Retriever):
    def __init__(self, index_name, config):
        self.config = config
        self.retrieve_type = "milvus"
        self.index_name = index_name
        self.index = None
        self.source_max_length = 300
        self.title_max_length = 300
        self.text_max_length = 8000

    def create_index(self):
        connections.connect(
            "default",
            host=self.config["vector"]["milvus_host"],
            port=self.config["vector"]["milvus_port"],
            user=self.config["vector"]["milvus_user"],
            password=self.config["vector"]["milvus_passwd"],
        )
        logger.info(f"All Milvus collections: {utility.list_collections()}")
        if utility.has_collection(self.index_name):
            logger.info(f"Remove existing Milvus index {self.index_name}")
            utility.drop_collection(self.index_name)
        fields = [
            FieldSchema(name="uid", dtype=DataType.INT64, is_primary=True, auto_id=False, max_length=100),
            FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=self.source_max_length),
            FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=self.title_max_length),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=self.text_max_length),
            FieldSchema(name="pid", dtype=DataType.INT64),
            FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=self.config["vector"]["emb_dims"]),
        ]
        schema = CollectionSchema(fields, "Milvus schema")
        self.index = Collection(self.index_name, schema, consistency_level="Strong")

    def connect_index(self):
        connections.connect(
            "default",
            host=self.config["vector"]["milvus_host"],
            port=self.config["vector"]["milvus_port"],
            user=self.config["vector"]["milvus_user"],
            password=self.config["vector"]["milvus_passwd"],
        )
        has = utility.has_collection(self.index_name)
        assert has is True
        fields = [
            FieldSchema(name="uid", dtype=DataType.INT64, is_primary=True, auto_id=False, max_length=100),
            FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=self.source_max_length),
            FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=self.title_max_length),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=self.text_max_length),
            FieldSchema(name="pid", dtype=DataType.INT64),
            FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=self.config["vector"]["emb_dims"]),
        ]
        schema = CollectionSchema(fields, "Milvus schema")
        self.index = Collection(self.index_name, schema, consistency_level="Strong")
        logger.info("Loading milvus index")
        self.index.load()

    def generate_embedding(self, passages):
        model = SentenceTransformer(self.config["vector"]["emb_model"])
        embeddings = model.encode(passages)
        return embeddings

    def ingest(self, doc_or_passage_file, batch_size):
        self.create_index()
        batch = []
        uids, sources, titles, texts, pids = [], [], [], [], []
        record_id = 0
        max_retries = 3
        failed_batches = []  # To store information about failed batches

        with open(doc_or_passage_file, "r") as jsonl_file:
            for line in jsonl_file:
                data = json.loads(line)
                batch.append(data["title"] + " " + data["text"])
                uids.append(record_id)
                sources.append(data.get("source", "")[: self.source_max_length])
                titles.append(data.get("title", "")[: self.title_max_length])
                texts.append(data.get("text", "")[: self.text_max_length])
                pids.append(data.get("pid", -1))
                record_id += 1
                if len(batch) == batch_size:
                    success = False
                    for attempt in range(max_retries):
                        try:
                            embeddings = self.generate_embedding(batch)
                            self.index.insert([uids, sources, titles, texts, pids, np.array(embeddings)])
                            self.index.flush()
                            logger.info(f"Milvus vector DB ingesting {doc_or_passage_file} record {record_id}")
                            success = True
                            break  # Exit the retry loop on success
                        except Exception as e:
                            logger.error(f"Attempt {attempt + 1}: Failed to create embeddings - {e}")

                    if not success:
                        failed_batches.append({"sources": sources, "pids": pids, "batch": batch})

                    batch = []
                    uids, sources, titles, texts, pids = [], [], [], [], []

            # import pdb; pdb.set_trace()
            if len(batch) > 0:
                success = False
                for attempt in range(max_retries):
                    try:
                        embeddings = self.generate_embedding(batch)
                        self.index.insert([uids, sources, titles, texts, pids, np.array(embeddings)])
                        self.index.flush()
                        logger.info(f"Milvus vector DB ingesting {doc_or_passage_file} record {record_id}")
                        success = True
                        break  # Exit the retry loop on success
                    except Exception as e:
                        logger.error(f"Attempt {attempt + 1}: Failed to create embeddings for remaining batch - {e}")

                if not success:
                    failed_batches.append({"sources": sources, "pids": pids, "batch": batch})

        # Save failed batches to a JSONL file
        assert ".jsonl" in doc_or_passage_file
        failure_output_file = doc_or_passage_file.replace(".jsonl", ".failed")
        with open(failure_output_file, "w") as fout:
            for failed_batch in failed_batches:
                for source, pid, record in zip(failed_batch["sources"], failed_batch["pids"], failed_batch["batch"]):
                    json.dump({"source": source, "pid": pid, "data": record}, fout)
                    fout.write("\n")

        index = {
            "index_type": "IVF_FLAT",
            "metric_type": "L2",
            "params": {"nlist": 128},
        }

        # import pdb; pdb.set_trace()
        self.index.create_index("embeddings", index)
        self.index.load()

    def retrieve(self, query_text, topk):
        if not self.index:
            self.connect_index()
        embeddings = self.generate_embedding([query_text])
        query_embedding = np.array(embeddings)
        search_params = {
            "metric_type": "L2",
            "params": {"nprobe": 10},
        }
        result = self.index.search(
            query_embedding, "embeddings", search_params, limit=topk, output_fields=["source", "title", "text", "pid"]
        )

        topk_used = min(len(result[0]), topk)
        passages = []
        for id in range(topk_used):
            assert len(result) == 1
            hit = result[0][id]
            passage = {
                "source": hit.entity.source,
                "text": hit.entity.text,
                "title": hit.entity.title,
                "pid": hit.entity.pid,
                "score": -hit.entity.distance,
            }
            passages.append(passage)

        return passages
