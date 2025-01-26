from abc import ABC, abstractmethod
import operator
from typing import List, Sequence, Tuple, Dict
import time
import logging
import cohere
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document
from denser_retriever.core.utils import sigmoid

logger = logging.getLogger(__name__)


class DenserReranker(ABC):
    @abstractmethod
    def rerank(
            self,
            documents: Sequence[Document],
            query: str,
    ) -> List[Tuple[Document, float]]:
        pass


class HFReranker(DenserReranker):
    """Rerank documents using a HuggingFaceCrossEncoder model with singleton pattern."""

    _instances: Dict[str, 'HFReranker'] = {}  # Store instances by model name

    def __new__(cls, model_name: str, **kwargs):
        # If an instance with this model_name exists, return it
        if model_name in cls._instances:
            return cls._instances[model_name]

        # Create new instance
        instance = super(HFReranker, cls).__new__(cls)
        cls._instances[model_name] = instance

        # Initialize the instance
        instance.__initialized = False
        return instance

    def __init__(self, model_name: str, **kwargs):
        # Skip initialization if already initialized
        if hasattr(self, '__initialized') and self.__initialized:
            return

        super().__init__()
        self.model = CrossEncoder(model_name, trust_remote_code=True, **kwargs)
        self.__initialized = True

    def rerank(
            self,
            documents: Sequence[Document],
            query: str,
            apply_sigmoid: bool = False
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
        scores = self.model.predict(
            [(query, doc.page_content) for doc in documents], convert_to_tensor=False
        )
        if apply_sigmoid:
            scores = sigmoid(scores)
        scores = [float(score) for score in scores]
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
            apply_sigmoid: bool = False
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
            model=self.model_name, query=query, documents=texts
        )
        # Combine documents with scores from the rerank response
        docs_with_scores = [
            (documents[result.index], sigmoid(result.relevance_score) if apply_sigmoid else result.relevance_score)
            for result in response.results
        ]

        # Sort the documents by their scores in descending order
        result = sorted(docs_with_scores, key=operator.itemgetter(1), reverse=True)

        rerank_time_sec = time.time() - start_time
        logger.info(f"Cohere Rerank time: {rerank_time_sec:.3f} sec.")
        logger.info(f"Reranked {len(result)} documents.")
        return result


class BGEReranker(DenserReranker):
    _instances: Dict[str, 'BGEReranker'] = {}

    def __new__(cls, model_name: str = "BAAI/bge-reranker-v2-m3", **kwargs):
        if model_name in cls._instances:
            return cls._instances[model_name]
        instance = super(BGEReranker, cls).__new__(cls)
        cls._instances[model_name] = instance
        instance.__initialized = False
        return instance

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", use_fp16: bool = True):
        if hasattr(self, '__initialized') and self.__initialized:
            return
        super().__init__()
        from FlagEmbedding import FlagReranker
        self.model_name = model_name
        self.model = FlagReranker(model_name, use_fp16=use_fp16)
        self.__initialized = True

    def rerank(
            self,
            documents: Sequence[Document],
            query: str,
            apply_sigmoid: bool = False
    ) -> List[Tuple[Document, float]]:
        if not documents:
            return []
        start_time = time.time()
        pairs = [[query, doc.page_content[:4000]] for doc in documents]
        scores = self.model.compute_score(pairs)
        if apply_sigmoid:
            scores = sigmoid(scores)
        docs_with_scores = list(zip(documents, scores))
        result = sorted(docs_with_scores, key=operator.itemgetter(1), reverse=True)
        rerank_time_sec = time.time() - start_time
        logger.info(f"BGE Rerank time: {rerank_time_sec:.3f} sec.")
        logger.info(f"Reranked {len(result)} documents.")
        return result


if __name__ == '__main__':
    # Test HFReranker singleton
    # reranker1 = HFReranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    # reranker2 = HFReranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    # print("HF singleton test (same model):", reranker1 is reranker2)
    #
    # reranker3 = HFReranker(model_name="cross-encoder/ms-marco-TinyBERT-L-4")
    # print("HF singleton test (different models):", reranker1 is reranker3)

    # Test BGEReranker singleton
    # model_name = "BAAI/bge-reranker-v2-m3"
    model_name = "BAAI/bge-reranker-v2-gemma"
    bge_reranker = BGEReranker(model_name)

    # Test reranking
    test_pairs = [
        ['what is panda?', 'hi'],
        ['what is panda?', 'The giant panda is a bear species endemic to China.']
    ]
    test_docs = [Document(page_content=pair[1]) for pair in test_pairs]
    results = bge_reranker.rerank(test_docs, "what is panda?")
    print(results)
