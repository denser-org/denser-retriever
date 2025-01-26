from langchain_core.documents import Document
from denser_retriever.core.retriever import DenserRetriever
from denser_retriever.config import load_retriever_config

# Create sample documents
texts = [
    Document(page_content="Python is a high-level programming language known for its simplicity and readability."),
    Document(page_content="Machine learning is a subset of AI that enables systems to learn from data."),
    Document(page_content="Natural Language Processing (NLP) helps computers understand human language."),
    Document(page_content="Deep learning models use neural networks with multiple layers.")
]

# Initialize retriever
index_name = "tech_docs"
config = load_retriever_config("denser_retriever/configs/fusion_msmarco.json")
retriever_config = config.get_retriever_config(index_name, True)
retriever = DenserRetriever(**retriever_config)

# Ingest documents
retriever.ingest(texts)

# Test queries
queries = [
    "What is Python?",
    "Explain machine learning",
]

# Retrieve results
for query in queries:
    result = retriever.retrieve(query=query, k=5, usage=True)
    print(f"\nQuery: {query}")
    print(result.to_json())

# Cleanup
retriever.delete_all(delete_index=True)