from abc import ABC, abstractmethod
import operator
from typing import List, Sequence, Tuple
import time
import logging
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
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

    def __init__(self, model_name: str, **kwargs):
        super().__init__()
        self.model = HuggingFaceCrossEncoder(model_name=model_name)

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
        scores = self.model.score([(query, doc.page_content) for doc in documents])
        docs_with_scores = list(zip(documents, scores))
        result = sorted(docs_with_scores, key=operator.itemgetter(1), reverse=True)
        rerank_time_sec = time.time() - start_time
        logger.info(f"Rerank time: {rerank_time_sec:.3f} sec.")
        logger.info(f"Reranked {len(result)} documents.")
        return result

