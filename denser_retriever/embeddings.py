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
