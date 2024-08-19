from abc import ABC, abstractmethod


class DenserEmbeddings(ABC):
    @abstractmethod
    def embed_documents(self, texts) -> list:
        pass

    @abstractmethod
    def embed_query(self, text) -> list:
        pass


class SentenceTransformerEmbeddings(DenserEmbeddings):
    def __init__(self, model_name: str):
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

    def embed_documents(self, texts):
        embeddings = self.client.encode(texts)
        return embeddings

    def embed_query(self, text):
        return self.client.encode([text])
