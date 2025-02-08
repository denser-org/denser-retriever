# type: ignore[attr-defined]
"""Enterprise-grade AI retriever solution that seamlessly integrates to enhance your AI applications."""

import sys
from denser_retriever.core.embeddings import (
    DenserEmbeddings,
    SentenceTransformerEmbeddings,
    VoyageAPIEmbeddings,
)
from denser_retriever.core.keyword import DenserKeywordSearch, ElasticKeywordSearch
from denser_retriever.core.reranker import DenserReranker, HFReranker, CohereReranker, BGEReranker
from denser_retriever.core.retriever import DenserRetriever
from denser_retriever.core.vectordb.base import DenserVectorDB
from denser_retriever.core.vectordb import MilvusDenserVectorDB

if sys.version_info >= (3, 8):
    from importlib import metadata as importlib_metadata
else:
    import importlib_metadata


def get_version() -> str:
    try:
        return importlib_metadata.version(__name__)
    except importlib_metadata.PackageNotFoundError:  # pragma: no cover
        return "unknown"


version: str = get_version()

__all__ = [
    "DenserEmbeddings",
    "SentenceTransformerEmbeddings",
    "VoyageAPIEmbeddings",
    "DenserKeywordSearch",
    "ElasticKeywordSearch",
    "DenserReranker",
    "HFReranker",
    "CohereReranker",
    "BGEReranker",
    "DenserRetriever",
    "DenserVectorDB",
    "MilvusDenserVectorDB",
]
