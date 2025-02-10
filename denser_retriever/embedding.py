from abc import ABC, abstractmethod
from typing import List


class EmbeddingModel(ABC):
    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents/passages.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors as floats
        """
        pass

    @abstractmethod
    def embed_query(self, text: str) -> List[List[float]]:
        """Embed a single query text.

        Args:
            text: Query text to embed

        Returns:
            List containing single embedding vector as floats
        """
        pass


class SentenceTransformer(EmbeddingModel):
    def __init__(self, model_name: str, one_model: bool):
        try:
            import sentence_transformers
        except ImportError as exc:
            raise ImportError(
                "Could not import sentence_transformers python package. "
            ) from exc

        self.client = sentence_transformers.SentenceTransformer(
            model_name, trust_remote_code=True
        )
        self.one_model = one_model

    def embed_documents(self, texts):
        return self.client.encode(texts)

    def embed_query(self, text):
        if self.one_model:
            return self.client.encode([text])
        else:
            return self.client.encode([text], prompt_name="query")


class BGEEmbedding(EmbeddingModel):
    def __init__(self, model_name: str):
        try:
            from FlagEmbedding import FlagICLModel
        except ImportError as exc:
            raise ImportError("Could not import FlagEmbedding python package.") from exc

        self.client = FlagICLModel(
            model_name,
            query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
            examples_for_task=None,  # set `examples_for_task=None` to use model without examples
            use_fp16=True,
        )  # Setting use_fp16 to True speeds up computation with a slight performance degradation

    def embed_documents(self, texts):
        return self.client.encode_corpus(texts)

    def embed_query(self, text):
        return self.client.encode_queries(text)


class BGEM3Embedding(EmbeddingModel):
    def __init__(self, model_name: str):
        try:
            from FlagEmbedding import BGEM3FlagModel
        except ImportError as exc:
            raise ImportError("Could not import FlagEmbedding python package.") from exc

        self.client = BGEM3FlagModel(
            model_name,
            use_fp16=True,
        )  # Setting use_fp16 to True speeds up computation with a slight performance degradation

    def embed_documents(self, texts):
        return self.client.encode(texts)["dense_vecs"].tolist()

    def embed_query(self, text):
        return self.client.encode([text])["dense_vecs"].tolist()
