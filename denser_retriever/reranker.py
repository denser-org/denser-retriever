from abc import ABC, abstractmethod
import operator
from typing import List, Sequence, Tuple
import time
import logging
import cohere
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

class DenserReranker(ABC):
    def __init__(self, top_k: int = 50, weight: float = 0.5):
        self.top_k = top_k
        self.weight = weight

    @abstractmethod
    def rerank(
        self,
        documents: Sequence[Document],
        query: str,
    ) -> List[Tuple[Document, float]]:
        pass


class HFReranker(DenserReranker):
    """Rerank documents using a HuggingFaceCrossEncoder model."""

    def __init__(self, model_name: str, top_k: int, **kwargs):
        super().__init__(top_k=top_k)
        self.model = CrossEncoder(model_name, **kwargs)

    def rerank(
        self,
        documents: Sequence[Document],
        query: str,
    ) -> List[Tuple[Document, float]]:
        """
        Rerank documents using CrossEncoder.

        Args:
            documents: A sequence of documents to rerank.
            query: The query to use for ranking the documents.

        Returns:
            A list of tuples containing the document and its score.
        """
        if not documents:
            return []
        start_time = time.time()
        scores = self.model.predict([(query, doc.page_content) for doc in documents], convert_to_tensor=True)
        docs_with_scores = list(zip(documents, scores))
        result = sorted(docs_with_scores, key=operator.itemgetter(1), reverse=True)
        rerank_time_sec = time.time() - start_time
        logger.info(f"Rerank time: {rerank_time_sec:.3f} sec.")
        logger.info(f"Reranked {len(result)} documents.")
        return result


class CohereReranker(DenserReranker):
    """Rerank documents using the Cohere API."""

    def __init__(self, api_key: str, model_name: str = "rerank-english-v3.0", **kwargs):
        """
        Initialize Cohere reranker.

        Args:
            api_key: The API key for Cohere.
            model_name: The name of the Cohere model to use for reranking.
        """
        super().__init__()
        self.client = cohere.Client(api_key)
        self.model_name = model_name

    def rerank(
            self,
            documents: Sequence[Document],
            query: str,
    ) -> List[Tuple[Document, float]]:
        """
        Rerank documents using Cohere's reranking model.

        Args:
            documents: A sequence of documents to rerank.
            query: The query to use for ranking the documents.

        Returns:
            A list of tuples containing the document and its score.
        """
        if not documents:
            return []

        start_time = time.time()

        # Prepare documents for reranking
        texts = [doc.page_content for doc in documents]
        response = self.client.rerank(
            model=self.model_name,
            query=query,
            documents=texts
        )
        # Combine documents with scores from the rerank response
        docs_with_scores = [(documents[result.index], result.relevance_score) for result in response.results]

        # Sort the documents by their scores in descending order
        result = sorted(docs_with_scores, key=operator.itemgetter(1), reverse=True)

        rerank_time_sec = time.time() - start_time
        logger.info(f"Cohere Rerank time: {rerank_time_sec:.3f} sec.")
        logger.info(f"Reranked {len(result)} documents.")
        return result
