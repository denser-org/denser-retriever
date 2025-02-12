from abc import ABC, abstractmethod
import operator
from typing import List, Sequence, Tuple
import cohere
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
from denser_retriever.utils import sigmoid


class Reranker(ABC):
    @abstractmethod
    def rerank(
        self,
        documents: Sequence[Document],
        query: str,
    ) -> List[Tuple[Document, float]]:
        pass


class CrossEncoderReranker(Reranker):
    def __init__(self, model_name: str, **kwargs):
        super().__init__()
        self.model = CrossEncoder(
            model_name,
            trust_remote_code=True,
            automodel_args={"torch_dtype": "auto"},
            **kwargs
        )

    def rerank(
        self, documents: Sequence[Document], query: str, apply_sigmoid: bool = False
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

        scores = self.model.predict(
            [(query, doc.page_content) for doc in documents], convert_to_tensor=False
        )

        if apply_sigmoid:
            scores = sigmoid(scores)

        scores = [float(score) for score in scores]
        docs_with_scores = list(zip(documents, scores))

        return sorted(docs_with_scores, key=operator.itemgetter(1), reverse=True)


class CohereReranker(Reranker):
    """Rerank documents using the Cohere API."""

    def __init__(self, api_key: str, model_name: str = "rerank-english-v3.0"):
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
        self, documents: Sequence[Document], query: str, apply_sigmoid: bool = False
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

        # Prepare documents for reranking
        texts = [doc.page_content for doc in documents]
        response = self.client.rerank(
            model=self.model_name, query=query, documents=texts
        )

        # Combine documents with scores from the rerank response
        docs_with_scores = [
            (
                documents[result.index],
                (
                    sigmoid(result.relevance_score)
                    if apply_sigmoid
                    else result.relevance_score
                ),
            )
            for result in response.results
        ]

        # Sort the documents by their scores in descending order
        return sorted(docs_with_scores, key=operator.itemgetter(1), reverse=True)


class BGEReranker(Reranker):
    def __init__(
        self, model_name: str = "BAAI/bge-reranker-v2-m3", use_fp16: bool = True
    ):
        super().__init__()
        from FlagEmbedding import FlagReranker

        self.model_name = model_name
        self.model = FlagReranker(model_name, use_fp16=use_fp16)

    def rerank(
        self, documents: Sequence[Document], query: str, apply_sigmoid: bool = False
    ) -> List[Tuple[Document, float]]:
        if not documents:
            return []

        pairs = [[query, doc.page_content[:4000]] for doc in documents]
        scores = self.model.compute_score(pairs)

        if apply_sigmoid:
            scores = sigmoid(scores)

        docs_with_scores = list(zip(documents, scores))
        return sorted(docs_with_scores, key=operator.itemgetter(1), reverse=True)
