from abc import ABC, abstractmethod


class DenserEmbeddings(ABC):
    embedding_size: int
    
    @abstractmethod
    def embed_documents(self, texts) -> list:
        pass

    @abstractmethod
    def embed_query(self, text) -> list:
        pass


class SentenceTransformerEmbeddings(DenserEmbeddings):
    def __init__(self, model_name: str, embedding_size: int, one_model: bool):
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

    def embed_documents(self, texts):
        return self.client.encode(texts)

    def embed_query(self, text):
        if self.one_model:
            embeddings = self.client.encode([text])
        else:
            embeddings = self.client.encode([text], prompt_name="query")
        return embeddings


class BGEEmbeddings(DenserEmbeddings):
    def __init__(self, model_name: str, embedding_size: int):
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

    def embed_documents(self, texts):
        return self.client.encode_corpus(texts)

    def embed_query(self, text):
        return self.client.encode_queries(text)

class VoyageAPIEmbeddings(DenserEmbeddings):
    def __init__(self, api_key: str, model_name: str, embedding_size: int):
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
