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
    def __init__(self, model_name: str, batch_size: int = 256):
        try:
            import sentence_transformers
        except ImportError as exc:
            raise ImportError(
                "Could not import sentence_transformers python package. "
            ) from exc

        self._client = sentence_transformers.SentenceTransformer(
            model_name, trust_remote_code=True
        )
        self._batch_size = batch_size

    def embed_documents(self, texts: List[str]) -> list:
        return self._client.encode(sentences=texts, batch_size=self._batch_size)

    def embed_query(self, text: str) -> list:
        return self._client.encode(sentences=[text], batch_size=self._batch_size)


class FlagICLModelEmbeddings(EmbeddingModel):
    def __init__(
        self,
        model_name: str,
        query_max_length: int = 512,
        passage_max_length: int = 512,
        batch_size: int = 256,
        use_fp16: bool = True,
    ):
        try:
            from FlagEmbedding import FlagICLModel
        except ImportError as exc:
            raise ImportError("Could not import FlagICLModel python package.") from exc

        self._client = FlagICLModel(
            model_name,
            query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
            examples_for_task=None,
            batch_size=batch_size,
            query_max_length=query_max_length,
            passage_max_length=passage_max_length,
            use_fp16=use_fp16,
        )

    def embed_documents(self, texts: List[str]) -> list:
        return self._client.encode_corpus(texts)

    def embed_query(self, text: str) -> list:
        return self._client.encode_queries(text)


class BGEM3FlagModelEmbeddings(EmbeddingModel):
    def __init__(
        self,
        model_name: str,
        query_max_length: int = 512,
        passage_max_length: int = 512,
        batch_size: int = 256,
        use_fp16: bool = True,
    ):
        try:
            from FlagEmbedding import BGEM3FlagModel
        except ImportError as exc:
            raise ImportError(
                "Could not import BGEM3FlagModel python package."
            ) from exc

        self._client = BGEM3FlagModel(
            model_name,
            use_fp16=use_fp16,
            batch_size=batch_size,
            query_max_length=query_max_length,
            passage_max_length=passage_max_length,
        )

    def embed_documents(self, texts: List[str]) -> list:
        return self._client.encode(texts)["dense_vecs"].tolist()

    def embed_query(self, text: str) -> list:
        return self._client.encode([text])["dense_vecs"].tolist()
