from abc import ABC
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from denser_retriever.embedings import DenserEmbeddings


class DenserVectorDB(ABC):
    """
    Interface for a denser vector database.
    """

    def __init__(self, top_k: int = 4, weight: float = 0.5):
        self.top_k = top_k
        self.weight = weight

    def create_index(
        self,
        index_name: str,
        embeddings: DenserEmbeddings,
        search_fields: List[str],
        **args: Any,
    ):
        raise NotImplementedError(
            f"create_index has not been implemented for {self.__class__.__name__}"
        )

    def add_documents(
        self,
        documents: List[Document],
        **kwargs: Any,
    ) -> List[str]:
        raise NotImplementedError(
            f"upsert has not been implemented for {self.__class__.__name__}"
        )

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Dict[str, Any] = {},
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        raise NotImplementedError(
            f"similarity_search_with_score has not been implemented for {self.__class__.__name__}"
        )

    def filter_expression(
        self,
        filter_dict: Dict[str, Any],
    ) -> Any:
        raise NotImplementedError(
            f"filter_expression has not been implemented for {self.__class__.__name__}"
        )

    def delete(
        self, ids: Optional[List[str]] = None, expr: Optional[str] = None, **kwargs: str
    ):
        raise NotImplementedError(
            f"clear has not been implemented for {self.__class__.__name__}"
        )

    def delete_all(self):
        raise NotImplementedError(
            f"clear has not been implemented for {self.__class__.__name__}"
        )
