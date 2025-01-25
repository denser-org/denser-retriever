"""Integration test for CrossEncoderReranker."""

from typing import List

from langchain_core.documents import Document

from denser_retriever.core.reranker import HFReranker


def test_rerank() -> None:
    texts = [
        "aaa1",
        "bbb1",
        "aaa2",
        "bbb2",
        "aaa3",
        "bbb3",
    ]
    docs = list(map(lambda text: Document(page_content=text), texts))
    reranker = HFReranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    actual_docs = reranker.rerank(docs, "bbb2")
    actual = list(map(lambda doc: doc[0].page_content, actual_docs))[0:3]
    assert actual[0] == "bbb2"


def test_rerank_empty() -> None:
    docs: List[Document] = []
    reranker = HFReranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    actual_docs = reranker.rerank(docs, "query")
    assert len(actual_docs) == 0
