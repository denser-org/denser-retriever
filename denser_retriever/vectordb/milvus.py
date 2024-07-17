from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from denser_retriever.vectordb.base import DenserVectorDB


class MilvusDenserVectorDB(DenserVectorDB):
    def __init__(
        self,
        embedding_function: Embeddings,
        collection_name: str = "DenserCollection",
        collection_description: str = "",
        drop_old: Optional[bool] = False,
        auto_id: bool = False,
        connection_args: Optional[dict] = None,
        **args: Any,
    ):
        try:
            from langchain_milvus import Milvus
        except ImportError:
            raise ImportError(
                "Please install langchain-milvus to use MilvusDenserVectorDB."
            )
        self.store = Milvus(
            embedding_function=embedding_function,
            collection_name=collection_name,
            collection_description=collection_description,
            drop_old=drop_old,
            auto_id=auto_id,
            connection_args=connection_args,
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
        return self.store.similarity_search_with_score(
            query, k, expr=self.filter_expression(filter), **kwargs
        )

    def filter_expression(
        self,
        filter_dict: Dict[str, Any],
    ) -> Any:
        """Generate a Milvus expression from a filter dictionary."""
        expressions = []
        for key, value in filter_dict.items():
            if value is None:
                continue
            if isinstance(value, tuple) and len(value) == 2:
                start, end = value
                expressions.append(f"{key} >= '{start}' and {key} <= '{end}'")
            else:
                expressions.append(f"{key} == '{value}'")
        return " and ".join(expressions)
