from __future__ import annotations

from pydantic import BaseModel
import yaml
import os
from dotenv import load_dotenv


class RetrieverSettings(BaseModel):
    version: str = "1.0"
    combine: str = "model"
    keyword_weight: float = 0.5
    vector_weight: float = 0.5
    rerank_weight: float = 0.5
    model: str = "tests/test_data/scifact_xgb_es+vs+rr_n.json"
    model_features: str = "es+vs+rr_n"
    output_prefix: str = "output"
    max_doc_size: int = 0
    max_query_size: int = 0
    keyword: Keyword
    vector: Vector
    rerank: Rerank
    fields: list = []

    @staticmethod
    def from_yaml(yaml_file: str = "config.yaml") -> RetrieverSettings:
        return RetrieverSettings._from_yaml(yaml_file)

    def _from_yaml(yaml_file: str) -> RetrieverSettings:
        data = yaml.safe_load(open(yaml_file))
        # Load environment variables
        load_dotenv()
        data["keyword"]["es_host"] = os.getenv("ES_HOST")
        data["keyword"]["es_passwd"] = os.getenv("ES_PASSWD")
        data["vector"]["milvus_host"] = os.getenv("MILVUS_HOST")
        data["vector"]["milvus_passwd"] = os.getenv("MILVUS_PASSWD")
        return RetrieverSettings(**data)


class Keyword(BaseModel):
    es_user: str = "elastic"
    es_passwd: str
    es_host: str = "localhost"
    es_ingest_passage_bs: int
    topk: int


class Vector(BaseModel):
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_user: str = "root"
    milvus_passwd: str
    emb_model: str
    emb_dims: int
    one_model: bool = True
    vector_ingest_passage_bs: int
    topk: int


class Rerank(BaseModel):
    rerank_model: str
    rerank_bs: int = 100
    topk: int = 5
