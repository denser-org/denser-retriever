from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from langchain.docstore.document import Document


@dataclass
class TokenMetrics:
    """Token usage metrics for retrieval operations.

    Attributes:
        vector_tokens: Number of tokens used for vector search
        keyword_tokens: Number of tokens used for keyword search
        rerank_tokens: Number of tokens used for reranking
        total_tokens: Total number of tokens used
    """

    vector_tokens: int = 0
    keyword_tokens: int = 0
    rerank_tokens: int = 0
    total_tokens: int = 0


@dataclass
class RetrievalResult:
    """Result of a retrieval operation.

    Attributes:
        documents: List of (document, score) tuples
        aggregations: Optional aggregation results
        token_metrics: Detailed token usage metrics
    """

    documents: List[Tuple[Document, float]]
    aggregations: Optional[Dict] = None
    token_metrics: Optional[TokenMetrics] = None

    @property
    def scores(self) -> List[float]:
        return [score for _, score in self.documents]

    @property
    def top_document(self) -> Optional[Document]:
        return self.documents[0][0] if self.documents else None

    def get_documents_above_score(self, threshold: float) -> List[Document]:
        return [doc for doc, score in self.documents if score >= threshold]

    def __len__(self) -> int:
        return len(self.documents)

    def __str__(self) -> str:
        return f"RetrievalResult(documents={self.documents}, aggregations={self.aggregations}, token_metrics={self.token_metrics})"

    def to_json(self) -> Dict:
        """Convert retrieval results to JSON format.

        Returns:
            Dict containing JSON-formatted results with documents, scores, metadata,
            aggregations and token metrics.
        """
        json_output = {
            "results": [
                {"content": doc.page_content, "metadata": doc.metadata, "score": score}
                for doc, score in self.documents
            ],
            "aggregations": self.aggregations,
            "token_metrics": None,
        }

        if self.token_metrics:
            json_output["token_metrics"] = {
                "vector_tokens": self.token_metrics.vector_tokens,
                "keyword_tokens": self.token_metrics.keyword_tokens,
                "rerank_tokens": self.token_metrics.rerank_tokens,
                "total_tokens": self.token_metrics.total_tokens,
            }

        return json_output
