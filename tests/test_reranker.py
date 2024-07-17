"""Integration test for CrossEncoderReranker."""

from typing import List

from langchain_core.documents import Document

from denser_retriever.reranker import DenserReranker


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
    compressor = DenserReranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    actual_docs = compressor.compress_documents(docs, "bbb2")
    actual = list(map(lambda doc: doc[0].page_content, actual_docs))[0:3]
    expected_returned = ["bbb2", "bbb1", "bbb3"]
    expected_not_returned = ["aaa1", "aaa2", "aaa3"]
    assert all([text in actual for text in expected_returned])
    assert all([text not in actual for text in expected_not_returned])
    assert actual[0] == "bbb2"


def test_rerank_empty() -> None:
    docs: List[Document] = []
    compressor = DenserReranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    actual_docs = compressor.compress_documents(docs, "query")
    assert len(actual_docs) == 0
