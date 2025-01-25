from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from denser_retriever.core.retriever import DenserRetriever
from denser_retriever.config import CombineConfig, LRConfig, load_retriever_config

docs = TextLoader("tests/test_data/state_of_the_union.txt").load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
texts = text_splitter.split_documents(docs)

index_name = "state_of_the_union"
config = load_retriever_config("denser_retriever/configs/fusion_msmarco.json")
retriever_config = config.get_retriever_config(index_name, True)
retriever = DenserRetriever(**retriever_config)

## Ingest the documents
retriever.ingest(texts)

# Create the CombineConfig object to specify the retrieval method
combine_config = CombineConfig(
    method="fusion",
    keyword_top_k=100,
    vector_top_k=100,
    reranker_top_k=100,
    lr_config=LRConfig(
        lr_features="es+vs+rr",
        lr_model="denser_retriever/models/weights_es+vs+rr_msmarco.json"
    )
)

query = "What did the president say about Ketanji Brown Jackson"
retrieval_result = retriever.retrieve(
    query=query,
    k=5,
    combine_config=combine_config,
    usage=True
)
print(retrieval_result.to_json())
retriever.delete_all(delete_index=True)
