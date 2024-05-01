import json
import os
from datetime import datetime

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
    def __init__(self, index_name, config_file):
        super().__init__(index_name, config_file)
        self.retrieve_type = "milvus"
        self.index_name = index_name
        self.index = None
        self.source_max_length = 300
        self.title_max_length = 300
        self.text_max_length = 8000
        self.field_max_length = 300
        self.model = SentenceTransformer(self.config["vector"]["emb_model"], trust_remote_code=True)

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
        for key in self.field_types:
            internal_key = self.field_internal_names[key]
            # both category and date type (unix timestamp) use INT64 type
            fields.append(FieldSchema(name=internal_key, dtype=DataType.INT64, max_length=self.field_max_length),)
            self.field_cat_to_id[key] = {}
            self.field_id_to_cat[key] = []

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

        for key in self.field_types:
            internal_key = self.field_internal_names[key]
            # both category and date type (unix timestamp) use INT64 type
            fields.append(FieldSchema(name=internal_key, dtype=DataType.INT64, max_length=self.field_max_length), )

        schema = CollectionSchema(fields, "Milvus schema")
        self.index = Collection(self.index_name, schema, consistency_level="Strong")
        logger.info("Loading milvus index")
        self.index.load()

        output_prefix = self.config["output_prefix"]
        exp_dir = os.path.join(output_prefix, f"exp_{self.index_name}")
        fields_file = os.path.join(exp_dir, "milvus_fields.json")
        with open(fields_file, 'r') as file:
            self.field_cat_to_id, self.field_id_to_cat = json.load(file)

    def generate_embedding(self, passages):
        embeddings = self.model.encode(passages)
        return embeddings

    def ingest(self, doc_or_passage_file, batch_size):
        self.create_index()
        batch = []
        uids, sources, titles, texts, pids = [], [], [], [], []
        fieldss = [[] for _ in self.field_types.keys()]
        record_id = 0
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

                for i, field in enumerate(self.field_types.keys()):
                    category_or_date_str = data.get(field).strip()
                    if category_or_date_str:
                        type = self.field_types[field]["type"]
                        if type == "date":
                            date_obj = datetime.strptime(category_or_date_str, '%Y-%m-%d')
                            unix_time = int(date_obj.timestamp())
                            fieldss[i].append(unix_time)
                        else: # categorical
                            if category_or_date_str not in self.field_cat_to_id[field]:
                                self.field_cat_to_id[field][category_or_date_str] = len(self.field_cat_to_id[field])
                                self.field_id_to_cat[field].append(category_or_date_str)
                            fieldss[i].append(self.field_cat_to_id[field][category_or_date_str])
                    else: # missing category value
                        fieldss[i].append(-1)
                record_id += 1
                if len(batch) == batch_size:
                    embeddings = self.generate_embedding(batch)
                    record = [uids, sources, titles, texts, pids, np.array(embeddings)]
                    record += fieldss
                    self.index.insert(record)
                    self.index.flush()
                    logger.info(f"Milvus vector DB ingesting {doc_or_passage_file} record {record_id}")

                    batch = []
                    uids, sources, titles, texts, pids = [], [], [], [], []
                    fieldss = [[] for _ in self.field_types.keys()]

            if len(batch) > 0:
                embeddings = self.generate_embedding(batch)
                record = [uids, sources, titles, texts, pids, np.array(embeddings)]
                record += fieldss
                self.index.insert(record)
                self.index.flush()
                logger.info(f"Milvus vector DB ingesting {doc_or_passage_file} record {record_id}")

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
        output_prefix = self.config["output_prefix"]
        exp_dir = os.path.join(output_prefix, f"exp_{self.index_name}")
        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)
        fields_file = os.path.join(exp_dir, "milvus_fields.json")
        with open(fields_file, 'w') as file:
            json.dump([self.field_cat_to_id, self.field_id_to_cat], file, ensure_ascii=False, indent=4)  # 'indent' for pretty printing

    def retrieve(self, query_text, meta_data, query_id=None):
        if not self.index:
            self.connect_index()
        embeddings = self.generate_embedding([query_text])
        query_embedding = np.array(embeddings)

        exprs = []
        for field in meta_data:
            category_or_date_str = meta_data.get(field)
            internal_field = self.field_internal_names.get(field)
            type = self.field_types[field]["type"]
            if type == "date":
                if len(category_or_date_str) == 2:
                    start_unix_time = int(datetime.combine(category_or_date_str[0], datetime.min.time()).timestamp())
                    end_unix_time = int(datetime.combine(category_or_date_str[1], datetime.min.time()).timestamp())
                    exprs.append(f"{internal_field} >= {start_unix_time}")
                    exprs.append(f"{internal_field} <= {end_unix_time}")
                else:
                    unix_time = int(datetime.combine(category_or_date_str[0], datetime.min.time()).timestamp())
                    exprs.append(f"{internal_field} == {unix_time}")
            else:
                category_id = self.field_cat_to_id[field].get(category_or_date_str)
                if category_id:
                    exprs.append(f"{internal_field}=={category_id}")
        expr_str = " and ".join(exprs)
        search_params = {
            "metric_type": "L2",
            "params": {"nprobe": 10},
        }
        result = self.index.search(
            query_embedding, "embeddings", search_params, limit=self.config["vector"]["topk"], expr=expr_str,
            output_fields=["source", "title", "text", "pid"] + list(self.field_internal_names.values())
        )

        topk_used = min(len(result[0]), self.config["vector"]["topk"])
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
            for field in self.field_types.keys():
                internal_field = self.field_internal_names[field]
                cat_id_or_unix_time = hit.entity.__dict__['fields'].get(internal_field)
                type = self.field_types[field]["type"]
                if type == "date":
                    date = datetime.utcfromtimestamp(cat_id_or_unix_time).strftime('%Y-%m-%d')
                    passage[field] = date
                else:
                    passage[field] = self.field_id_to_cat[field][cat_id_or_unix_time]
            passages.append(passage)

        return passages
