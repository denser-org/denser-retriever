from abc import ABC, abstractmethod
from typing import Dict


class DenserEmbeddings(ABC):
    embedding_size: int

    @abstractmethod
    def embed_documents(self, texts) -> list:
        pass

    @abstractmethod
    def embed_query(self, text) -> list:
        pass


class SentenceTransformerEmbeddings(DenserEmbeddings):
    _instances: Dict[str, 'SentenceTransformerEmbeddings'] = {}

    def __new__(cls, model_name: str, embedding_size: int, one_model: bool):
        # Create unique key from all parameters
        key = f"{model_name}_{embedding_size}_{one_model}"
        if key in cls._instances:
            return cls._instances[key]

        instance = super(SentenceTransformerEmbeddings, cls).__new__(cls)
        cls._instances[key] = instance
        instance.__initialized = False
        return instance

    def __init__(self, model_name: str, embedding_size: int, one_model: bool):
        if hasattr(self, '__initialized') and self.__initialized:
            return

        try:
            import sentence_transformers
        except ImportError as exc:
            raise ImportError(
                "Could not import sentence_transformers python package. "
                "Please install it with `pip install sentence-transformers`."
            ) from exc

        self.client = sentence_transformers.SentenceTransformer(
            model_name, trust_remote_code=True
        )
        self.embedding_size = embedding_size
        self.one_model = one_model
        self.__initialized = True

    def embed_documents(self, texts):
        return self.client.encode(texts)

    def embed_query(self, text):
        if self.one_model:
            embeddings = self.client.encode([text])
        else:
            embeddings = self.client.encode([text], prompt_name="query")
        return embeddings


class BGEEmbeddings(DenserEmbeddings):
    _instances: Dict[str, 'BGEEmbeddings'] = {}

    def __new__(cls, model_name: str, embedding_size: int):
        key = f"{model_name}_{embedding_size}"
        if key in cls._instances:
            return cls._instances[key]

        instance = super(BGEEmbeddings, cls).__new__(cls)
        cls._instances[key] = instance
        instance.__initialized = False
        return instance

    def __init__(self, model_name: str, embedding_size: int):
        if hasattr(self, '__initialized') and self.__initialized:
            return

        try:
            from FlagEmbedding import FlagICLModel
        except ImportError as exc:
            raise ImportError(
                "Could not import FlagEmbedding python package."
            ) from exc

        self.client = FlagICLModel(model_name,
                                   query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
                                   examples_for_task=None,  # set `examples_for_task=None` to use model without examples
                                   use_fp16=True)  # Setting use_fp16 to True speeds up computation with a slight performance degradation
        self.embedding_size = embedding_size
        self.__initialized = True

    def embed_documents(self, texts):
        return self.client.encode_corpus(texts)

    def embed_query(self, text):
        return self.client.encode_queries(text)


class VoyageAPIEmbeddings(DenserEmbeddings):
    _instances: Dict[str, 'VoyageAPIEmbeddings'] = {}

    def __new__(cls, api_key: str, model_name: str, embedding_size: int):
        key = f"{api_key}_{model_name}_{embedding_size}"
        if key in cls._instances:
            return cls._instances[key]

        instance = super(VoyageAPIEmbeddings, cls).__new__(cls)
        cls._instances[key] = instance
        instance.__initialized = False
        return instance

    def __init__(self, api_key: str, model_name: str, embedding_size: int):
        if hasattr(self, '__initialized') and self.__initialized:
            return

        try:
            import voyageai
        except ImportError as exc:
            raise ImportError(
                "Could not import voyage python package. "
                "Please install it with `pip install voyageai`."
            ) from exc

        self.client = voyageai.Client(api_key)
        self.model_name = model_name
        self.embedding_size = embedding_size
        self.__initialized = True

    def embed_documents(self, texts):
        """
        Embeds multiple documents using the Voyage API.
        Args:
            texts: A list of document texts.
        Returns:
            A list of document embeddings.
        """
        embeddings = self.client.embed(texts, model=self.model_name).embeddings
        return embeddings

    def embed_query(self, text):
        """
        Embeds a single query using the Voyage API.
        Args:
            text: The query text.
        Returns:
            The query embedding.
        """
        embeddings = self.client.embed([text], model=self.model_name).embeddings
        return embeddings

if __name__ == "__main__":
    # Same parameters = same instance
    emb1 = SentenceTransformerEmbeddings("Snowflake/snowflake-arctic-embed-m", 768, False)
    emb2 = SentenceTransformerEmbeddings("Snowflake/snowflake-arctic-embed-m", 768, False)
    print(emb1 is emb2)  # True

    # Different parameters = different instances
    emb3 = SentenceTransformerEmbeddings("Snowflake/snowflake-arctic-embed-m-long", 768, True)
    print(emb1 is emb3)  # False