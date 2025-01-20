from typing import Optional, Literal, Dict, Any
from pydantic import BaseModel

from denser_retriever.core.keyword import (
    ElasticKeywordSearch,
    create_elasticsearch_client,
)
from denser_retriever.core.reranker import HFReranker
from denser_retriever.core.vectordb.milvus import MilvusDenserVectorDB
from denser_retriever.core.embeddings import (
    VoyageAPIEmbeddings,
    SentenceTransformerEmbeddings,
    BGEEmbeddings,
)

# Define the combine method type
CombineMethod = Literal["vector", "hybrid", "reranker", "fusion"]


class ESConfig(BaseModel):
    url: str = "http://localhost:9200"
    analysis: str = "default"


class MilvusConfig(BaseModel):
    uri: str = "http://localhost:19530"


class EmbeddingConfig(BaseModel):
    type: str = "sentence_transformer"
    model: str = "Snowflake/snowflake-arctic-embed-m"
    size: int = 768
    one_model: bool = False


class LRConfig(BaseModel):
    lr_features: str
    lr_model: str


class CombineConfig(BaseModel):
    method: CombineMethod = "reranker"
    keyword_top_k: int = 100
    vector_top_k: int = 100
    reranker_top_k: int = 100
    lr_config: Optional[LRConfig] = None


class RetrieverConfig(BaseModel):
    max_query_len: int = 2000
    es: ESConfig = ESConfig()
    milvus: Optional[MilvusConfig] = None
    reranker_model: Optional[str] = None
    embedding: Optional[EmbeddingConfig] = None
    combine_config: CombineConfig = CombineConfig()
    voyage_api_key: Optional[str] = None
    is_retriever: bool = True
    aggregation: bool = False

    def get_retriever_config(
        self, index_name: str, drop_old: bool = False
    ) -> Dict[str, Any]:
        """Generate inference configuration dictionary"""

        # Configure keyword search
        keyword_search = ElasticKeywordSearch(
            es_connection=create_elasticsearch_client(url=self.es.url),
            drop_old=drop_old,
            analysis=self.es.analysis,
        )

        # Configure vector database
        if self.milvus:
            vector_db = MilvusDenserVectorDB(
                connection_args={"uri": self.milvus.uri}, drop_old=drop_old
            )
        else:
            vector_db = None

        # Configure reranker
        if self.reranker_model:
            reranker = HFReranker(model_name=self.reranker_model)
        else:
            reranker = None

        # Configure embeddings based on type
        if self.embedding:
            if self.embedding.type == "sentence_transformer":
                embeddings = SentenceTransformerEmbeddings(
                    self.embedding.model, self.embedding.size, self.embedding.one_model
                )
            elif self.embedding.type == "voyage":
                if not self.voyage_api_key:
                    raise ValueError("Voyage API key is required for Voyage embeddings")
                embeddings = VoyageAPIEmbeddings(
                    api_key=self.voyage_api_key,
                    model_name=self.embedding.model,
                    embedding_size=self.embedding.size,
                )
            elif self.embedding.type == "bge":
                embeddings = BGEEmbeddings(
                    model_name=self.embedding.model, embedding_size=self.embedding.size
                )
            else:
                raise ValueError(f"Unknown embedding type: {self.embedding.type}")
        else:
            embeddings = None

        return {
            "index_name": index_name,
            "keyword_search": keyword_search,
            "vector_db": vector_db,
            "reranker": reranker,
            "embeddings": embeddings,
            "combine_config": self.combine_config,
        }


class TrainConfig(RetrieverConfig):
    # Training specific settings
    ingest_bs: int = 2000
    max_query_size: int = 10
    max_doc_size: int = 100
    max_doc_len: int = 8000

    # Note: max_query_len is already inherited from RetrieverConfig
    # but can be overridden with a different default value if needed
    max_query_len: int = 2000
    is_retriever: bool = False


def load_train_config(config_path: str) -> TrainConfig:
    """Load training configuration from JSON file"""
    import json

    with open(config_path, "r") as f:
        config_dict = json.load(f)
    return TrainConfig(**config_dict)


def load_retriever_config(config_path: str) -> RetrieverConfig:
    """Load configuration from JSON file"""
    import json

    with open(config_path, "r") as f:
        config_dict = json.load(f)
    return RetrieverConfig(**config_dict)


def default_retriever_config() -> RetrieverConfig:
    """Load default configuration"""
    return RetrieverConfig()


def default_train_config() -> TrainConfig:
    """Load default configuration"""
    return TrainConfig()


# Example usage
if __name__ == "__main__":
    retriever_config = "denser_retriever/configs/fusion.json"
    print(f"{retriever_config}\n{load_retriever_config(retriever_config)}")

    train_config = "denser_retriever/configs/default.json"
    print(f"{train_config}\n{load_train_config(train_config)}")

    print(f"Default Retriever Config:\n{default_retriever_config()}")
    print(f"Default Train Config:\n{default_train_config()}")
