from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from qdrant_client.http import models

from denser_retriever.vectordb.base import DenserVectorDB


class QdrantDenserVectorDB(DenserVectorDB):
    def __init__(
        self,
        collection_name: str,
        embedding: Embeddings,
        location: Optional[str] = None,
        url: Optional[str] = None,
        **args: Any,
    ):
        try:
            from langchain_qdrant import QdrantVectorStore
        except ImportError:
            raise ImportError(
                "Please install langchain-qdrant to use QdrantDenserVectorDB."
            )
        self.store = QdrantVectorStore.from_documents(
            documents=[],
            collection_name=collection_name,
            embedding=embedding,
            location=location,
            url=url,
            **args,
        )

    def add_documents(
        self,
        documents: List[Document],
        **kwargs: Any,
    ) -> List[str]:
        """Add documents to the vector db.

        Args:
            documents (List[Document]): Documents to add to the vector db.

        Returns:
            List[str]: IDs of the added texts.
        """
        return self.store.add_documents(documents, **kwargs)

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Dict[str, Any] = {},
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Search for similar documents to the query.

        Args:
            query (str): Query text.
            k (int): Number of documents to return.
            param (Optional[dict]): Additional parameters for the search.
            expr (Optional[str]): Expression to filter the search.
            timeout (Optional[float]): Timeout for the search.

        Returns:
            List[Tuple[Document, float]]: List of tuples of documents and their similarity scores.
        """
        docs =  self.store.similarity_search_with_score(
            query, k, filter=self.filter_expression(filter), **kwargs
        )
        # change the distance to similarity measure
        docs = [(doc, -score) for doc, score in docs]
        return docs


    def filter_expression(
        self,
        filter_dict: Dict[str, Any],
    ):
        filter = []
        for key, value in filter_dict.items():
            if value is None:
                continue
            if isinstance(value, tuple) and len(value) == 2:
                start, end = value
                filter.append(
                    models.FieldCondition(
                        key=key, range=models.DatetimeRange(gte=start, lte=end)
                    )
                )
            else:
                filter.append(
                    models.FieldCondition(
                        key=key, match=models.MatchValue(value=str(value))
                    )
                )
        return models.Filter(must=filter)
