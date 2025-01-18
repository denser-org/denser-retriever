from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from denser_retriever.core.retriever import DenserRetriever
from denser_retriever.config import load_retriever_config

docs = TextLoader("tests/test_data/state_of_the_union.txt").load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
texts = text_splitter.split_documents(docs)

index_name = "state_of_the_union"
config = load_retriever_config("denser_retriever/configs/fusion.json")
retriever_config = config.get_retriever_config(index_name, True)
retriever = DenserRetriever(**retriever_config)
retriever.ingest(texts)

query = "What did the president say about Ketanji Brown Jackson"
results, _, token_metrics = retriever.retrieve(
    query=query,
    k=5,
    combine_config=retriever_config["combine_config"],
    usage=True
)
print(results)
print(token_metrics)

retriever.delete_all()
