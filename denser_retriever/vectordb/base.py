from abc import ABC
from typing import Any, Dict, List, Tuple

from langchain_core.documents import Document


class DenserVectorDB(ABC):
    """
    Interface for a denser vector database.
    """

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
