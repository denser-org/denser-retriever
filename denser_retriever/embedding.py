from abc import ABC, abstractmethod
from typing import List


class EmbeddingModel(ABC):
    @abstractmethod
    def embed_documents(self, texts: List[str]) -> list:
        """Embed a list of documents/passages.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors as floats
        """
        pass

    @abstractmethod
    def embed_query(self, text: str) -> list:
        """Embed a single query text.

        Args:
            text: Query text to embed

        Returns:
            List containing single embedding vector as floats
        """
        pass


class SentenceTransformerEmbeddings(EmbeddingModel):
    def __init__(self, model_name: str):
        try:
            import sentence_transformers
        except ImportError as exc:
            raise ImportError(
                "Could not import sentence_transformers python package. "
            ) from exc

        self._client = sentence_transformers.SentenceTransformer(
            model_name, trust_remote_code=True
        )

    def embed_documents(self, texts: List[str]) -> list:
        return self._client.encode(texts)

    def embed_query(self, text: str) -> list:
        return self._client.encode([text])


class FlagICLModelEmbeddings(EmbeddingModel):
    def __init__(self, model_name: str):
        try:
            from FlagEmbedding import FlagICLModel
        except ImportError as exc:
            raise ImportError("Could not import FlagICLModel python package.") from exc

        self._client = FlagICLModel(
            model_name,
            query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
            examples_for_task=None,  # set `examples_for_task=None` to use model without examples
            use_fp16=True,
        )  # Setting use_fp16 to True speeds up computation with a slight performance degradation

    def embed_documents(self, texts: List[str]) -> list:
        return self._client.encode_corpus(texts)

    def embed_query(self, text: str) -> list:
        return self._client.encode_queries(text)


class BGEM3FlagModelEmbeddings(EmbeddingModel):
    def __init__(self, model_name: str):
        try:
            from FlagEmbedding import BGEM3FlagModel
        except ImportError as exc:
            raise ImportError(
                "Could not import BGEM3FlagModel python package."
            ) from exc

        self._client = BGEM3FlagModel(
            model_name,
            use_fp16=True,
        )  # Setting use_fp16 to True speeds up computation with a slight performance degradation

    def embed_documents(self, texts: List[str]) -> list:
        return self._client.encode(texts)["dense_vecs"].tolist()

    def embed_query(self, text: str) -> list:
        return self._client.encode([text])["dense_vecs"].tolist()
