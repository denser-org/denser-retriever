from denser_retriever.core.keyword import (
    ElasticKeywordSearch,
    create_elasticsearch_client,
)
from denser_retriever.core.reranker import HFReranker, BGEReranker
from denser_retriever.core.vectordb.milvus import MilvusDenserVectorDB
from denser_retriever.core.embeddings import (
    VoyageAPIEmbeddings,
    SentenceTransformerEmbeddings,
    BGEEmbeddings,
    BGEM3Embeddings,
)
from denser_retriever.core.logistic_regression import LogisticRegression
from denser_retriever.core.utils import load_config_with_env_vars
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SharedComponents:
    """Singleton class to hold shared components."""
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SharedComponents, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self.keyword_search = None
            self.vector_db = None
            self.reranker = None
            self.embeddings = None
            self.lr_model = None
            self._initialized = True

    @classmethod
    def initialize_from_config(cls, config_path: str):
        """Initialize shared components from config file."""
        config = load_config_with_env_vars(config_path)
        logger.info(f"Loading config from {config_path}")
        logger.info(f"Config: {config}")
        instance = cls()

        # Initialize Elasticsearch
        es_config = config.get('es', {})
        instance.keyword_search = ElasticKeywordSearch(
            es_connection=create_elasticsearch_client(
                url=es_config.get('url'),
                username=es_config.get('username'),
                password=es_config.get('password')
            )
        )

        # Initialize Milvus
        milvus_config = config.get('milvus', {})
        if milvus_config:
            instance.vector_db = MilvusDenserVectorDB(
                connection_args={
                    "uri": milvus_config.get('uri'),
                    "user": milvus_config.get('user'),
                    "password": milvus_config.get('password')
                }
            )

        # Initialize embeddings
        embedding_config = config.get('embedding', {})
        if embedding_config:
            if embedding_config['type'] == 'sentence_transformer':
                instance.embeddings = SentenceTransformerEmbeddings(
                    model_name=embedding_config['model'],
                    embedding_size=embedding_config['size'],
                    one_model=embedding_config['one_model']
                )
            elif embedding_config['type'] == 'voyage':
                if not embedding_config.get('api_key'):
                    raise ValueError("Voyage API key is required for Voyage embeddings")
                instance.embeddings = VoyageAPIEmbeddings(
                    api_key=embedding_config['api_key'],
                    model_name=embedding_config['model'],
                    embedding_size=embedding_config['size']
                )
            elif embedding_config['type'] == 'bge':
                instance.embeddings = BGEEmbeddings(
                    model_name=embedding_config['model'],
                    embedding_size=embedding_config['size']
                )
            elif embedding_config['type'] == 'bgem3':
                instance.embeddings = BGEM3Embeddings(
                    model_name=embedding_config['model'],
                    embedding_size=embedding_config['size']
                )
            else:
                raise ValueError(f"Unknown embedding type: {embedding_config['type']}")

        # Initialize reranker
        reranker_model = config.get('reranker_model')
        if reranker_model:
            if reranker_model.startswith('BAAI'):
                instance.reranker = BGEReranker(model_name=reranker_model)
            else:
                instance.reranker = HFReranker(model_name=reranker_model)

        # Initialize LR model
        lr_config = config.get('combine_config', {}).get('lr_config', {})
        if lr_config:
            instance.lr_model = LogisticRegression(lr_config.get('lr_model'))
            instance.lr_features = lr_config.get('lr_features')

        return instance
