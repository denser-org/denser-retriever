import logging
from typing import Dict, List
from transformers import AutoTokenizer
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class ResourceTracker:
    # Initialize tokenizers as class variables
    vector_tokenizer = AutoTokenizer.from_pretrained("Snowflake/snowflake-arctic-embed-m")
    reranker_tokenizer = AutoTokenizer.from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2")

    @classmethod
    def estimate_es_storage_gb(cls, texts: list, metadata_fields: Dict[str, int] = None) -> float:
        """
        Estimate Elasticsearch storage size in GB
        """
        # Estimate text storage (assume average 4 bytes per character)
        total_chars = sum(len(text) for text in texts)
        text_storage_bytes = total_chars * 4

        # Estimate metadata storage
        metadata_storage_bytes = 0
        if metadata_fields:
            for field, size in metadata_fields.items():
                metadata_storage_bytes += size * len(texts)

        # Add 20% overhead for indexing
        total_storage_bytes = (text_storage_bytes + metadata_storage_bytes) * 1.2

        # Convert to GB
        return total_storage_bytes / (1024 ** 3)

    @classmethod
    def estimate_vector_storage_gb(cls, texts: list) -> float:
        """
        Estimate vector storage size in GB
        """
        # Vector storage (768-dimensional float vectors = 768 * 4 bytes per vector)
        vector_storage_bytes = len(texts) * 768 * 4

        # Convert to GB
        return vector_storage_bytes / (1024 ** 3)

    @classmethod
    def calculate_vector_token_count(cls, texts: List[str]) -> int:
        """Calculate token count for vector operations during ingestion"""
        return sum(len(cls.vector_tokenizer.encode(text)) for text in texts)

    @classmethod
    def calculate_vector_search_token_count(cls, query: str, results: List[Document]) -> int:
        """Calculate token count for vector search operation"""
        # Count query tokens
        query_tokens = len(cls.vector_tokenizer.encode(query))

        # Count result tokens
        result_tokens = sum(len(cls.vector_tokenizer.encode(doc.page_content))
                            for doc in results)

        return query_tokens + result_tokens

    @classmethod
    def calculate_reranker_token_count(cls, query: str, results: List[Document]) -> int:
        """Calculate token count for reranker"""
        # For each passage, we need query + passage tokens since they're concatenated
        query_tokens = len(cls.reranker_tokenizer.encode(query))
        total_query_tokens = query_tokens * len(results)

        # Add tokens for each passage
        passage_tokens = sum(len(cls.reranker_tokenizer.encode(doc.page_content))
                             for doc in results)

        return total_query_tokens + passage_tokens
