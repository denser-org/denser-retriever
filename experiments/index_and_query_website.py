from langchain_text_splitters import RecursiveCharacterTextSplitter

from denser_retriever.retriever import DenserRetriever
from experiments.utils_data import CustomWebBaseLoader

web_site = "https://denser.ai"
loader = CustomWebBaseLoader(web_site)

docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(docs)

retriever = DenserRetriever.from_milvus(
    index_name="agent_webpage",
    milvus_uri="http://localhost:19530",
    combine_mode="model",
    xgb_model_path="./experiments/models/msmarco_xgb_es+vs+rr_n.json",
    xgb_model_features="es+vs+rr_n",
)
retriever.ingest(texts)

query = "What use cases does Denser AI support?"
res = retriever.retrieve(query, 1)

for r in res:
    print("page_content: " + r[0].page_content)
    print("metadata: " + str(r[0].metadata))
    print("score: " + str(r[1]))

retriever.clear()
