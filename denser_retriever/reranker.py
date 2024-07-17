import operator
from typing import List, Sequence, Tuple

from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.documents import Document


class DenserReranker:
    """Rerank documents using a HuggingFaceCrossEncoder model."""

    def __init__(self, model_name: str):
        self.model = HuggingFaceCrossEncoder(model_name=model_name)

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
    ) -> List[Tuple[Document, float]]:
        """
        Rerank documents using CrossEncoder.

        Args:
            documents: A sequence of documents to compress.
            query: The query to use for compressing the documents.
            callbacks: Callbacks to run during the compression process.

        Returns:
            A list of tuples containing the document and its score.
        """
        if not documents:
            return []
        scores = self.model.score([(query, doc.page_content) for doc in documents])
        docs_with_scores = list(zip(documents, scores))
        result = sorted(docs_with_scores, key=operator.itemgetter(1), reverse=True)
        return result
