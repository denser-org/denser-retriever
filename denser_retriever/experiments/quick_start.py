from langchain_core.documents import Document
from denser_retriever.core.retriever import DenserRetriever
from denser_retriever.core.keyword import ESIndexData
from denser_retriever.core.vectordb.milvus import MilvusIndexData

# Create sample documents
texts = [
    Document(page_content="Python is a high-level programming language known for its simplicity and readability."),
    Document(page_content="Machine learning is a subset of AI that enables systems to learn from data."),
    Document(page_content="Natural Language Processing (NLP) helps computers understand human language."),
    Document(page_content="Deep learning models use neural networks with multiple layers.")
]

# Initialize retriever
index_name = "tech_docs"
# Create retriever with config and index data
es_data = ESIndexData(
    index_name=index_name,
    analysis="default",
    drop_old=True
)
milvus_data = MilvusIndexData(
    index_name=index_name,
    embedding_size=768,  # Match embedding model size
    drop_old=True
)
retriever = DenserRetriever(
    config_path="denser_retriever/configs/fusion_msmarco.json",
    es_data=es_data,
    milvus_data=milvus_data
)

# Ingest documents
retriever.ingest(texts)

result = retriever.retrieve(query="Explain machine learning", k=5, usage=True)
print(result.to_json())

# Cleanup
retriever.delete_all(delete_index=True)