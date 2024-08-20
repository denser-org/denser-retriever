import pytest
from langchain_huggingface import HuggingFaceEmbeddings

from denser_retriever.gradient_boost import XGradientBoost
from denser_retriever.retriever import DenserRetriever
from tests.utils import elasticsearch, milvus, reranker
import json

from langchain.docstore.document import Document


def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return data


def jsonl_to_documents(jsonl_data):
    documents = []
    for item in jsonl_data:
        doc = Document(page_content=item.get('text'),
                       metadata={"source": item.get('source'),
                                 "title": item.get('title'),
                                 "pid": item.get('pid')})
        documents.append(doc)
    return documents


class TestDenser:
    def setup_method(self):
        index_name = "unit_test_denser"

        self.denser_retriever = DenserRetriever(
            index_name=index_name,
            vector_db=milvus,
            keyword_search=elasticsearch,
            reranker=reranker,
            gradient_boost=XGradientBoost(
                "experiments/models/scifact_xgb_es+vs+rr_n.json"
            ),
            embeddings=HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            ),
            combine_mode="model",
        )

    @pytest.fixture(autouse=True)
    def denser_data(self):
        file_path = "tests/test_data/denser_website_passages_top10.jsonl"

        # Load and convert to LangChain Document objects
        jsonl_data = load_jsonl(file_path)
        docs = jsonl_to_documents(jsonl_data)
        for doc in docs:
            doc.page_content = doc.metadata["title"] + " " + doc.page_content
        return docs

    def test_ingest(self, denser_data):
        ids = self.denser_retriever.ingest(denser_data)
        assert len(ids) == 10

    def test_retrieve(self, denser_data):
        self.denser_retriever.ingest(denser_data)
        query = "what is denser ai?"
        k = 2
        results = self.denser_retriever.retrieve(query, k)
        print(results)
        assert len(results) == k
        assert abs(results[0][1] - 1.4220) < 0.01
