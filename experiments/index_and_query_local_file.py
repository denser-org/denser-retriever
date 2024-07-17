from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from denser_retriever.retriever import DenserRetriever

docs = TextLoader("tests/test_data/state_of_the_union.txt").load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
texts = text_splitter.split_documents(docs)

retriever = DenserRetriever.from_qdrant(
    index_name="state_of_the_union",
    combine_mode="model",
    xgb_model_path="./experiments/models/msmarco_xgb_es+vs+rr_n.json",
    xgb_model_features="es+vs+rr_n",
    location=":memory:",
)

retriever.ingest(texts)

query = "What did the president say about Ketanji Brown Jackson"
res = retriever.retrieve(query, 1)

for r in res:
    print("page_content: " + r[0].page_content)
    print("metadata: " + str(r[0].metadata))
    print("score: " + str(r[1]))

retriever.clear()
