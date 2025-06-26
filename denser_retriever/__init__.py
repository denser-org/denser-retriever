# type: ignore[attr-defined]
"""Enterprise-grade AI retriever solution that seamlessly integrates to enhance your AI applications."""

import sys
from .embeddings import (
    DenserEmbeddings,
    SentenceTransformerEmbeddings,
    VoyageAPIEmbeddings,
)
from .keyword import DenserKeywordSearch, ElasticKeywordSearch
from .reranker import DenserReranker, HFReranker, CohereReranker
from .retriever import DenserRetriever
from .vectordb.base import DenserVectorDB
from .vectordb.milvus import MilvusDenserVectorDB

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
    "DenserRetriever",
    "DenserVectorDB",
    "MilvusDenserVectorDB",
]
