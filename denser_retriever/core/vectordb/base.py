from abc import ABC
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document
from denser_retriever.core.embeddings import DenserEmbeddings


class DenserVectorDB(ABC):
    """
    Interface for a denser vector database.
    """

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

    def retrieve(
        self,
        query: str,
        k: int = 100,
        filter: Dict[str, Any] = {},
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        raise NotImplementedError(
            f"retrieve has not been implemented for {self.__class__.__name__}"
        )

    def delete(
        self,
        ids: Optional[List[str]] = None,
        source_id: Optional[str] = None,
        source_url: Optional[str] = None,
        **kwargs: str,
    ):
        raise NotImplementedError(
            f"clear has not been implemented for {self.__class__.__name__}"
        )

    def delete_all(self, delete_index: bool = True):
        raise NotImplementedError(
            f"clear has not been implemented for {self.__class__.__name__}"
        )

    def list_indices(self) -> List[str]:
        raise NotImplementedError