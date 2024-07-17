import logging
import time
from datetime import date

import streamlit as st
from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from denser_retriever.retriever import DenserRetriever

logger = logging.getLogger(__name__)

filter_fields = [
    "case_id:keyword",
    "court:keyword",
    "location:keyword",
    "case_type:keyword",
    "trial_procedure:keyword",
    "trial_date:date",
    "publication_date:date",
    "cause:keyword",
    "legal_basis:keyword",
]

index_name = "unit_test_cpws"
retriever = DenserRetriever.from_milvus(
    index_name,
    milvus_uri="http://localhost:19530",
    combine_mode="linear",
    filter_fields=filter_fields,
)

docs = CSVLoader(
    "tests/test_data/cpws_2021_10_top10_en.csv",
    metadata_columns=[
        "case_id",
        "court",
        "location",
        "case_type",
        "trial_procedure",
        "trial_date",
        "publication_date",
        "cause",
        "legal_basis",
    ],
).load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
texts = text_splitter.split_documents(docs)

retriever.ingest(texts)

starting_url = "https://wenshu.court.gov.cn/"
optional_str = 'Try questions such as "买卖合同纠纷"'


def denser_search():
    st.title("Denser Search Demo")
    st.caption(f"Starting URL: {starting_url}")
    if optional_str:
        st.caption(f"{optional_str}")
    st.divider()

    fields_and_types = retriever.get_filter_fields()

    meta_data = {}
    for field, type in fields_and_types.items():
        if field in ["content", "title", "source", "pid"]:
            continue
        if type == "date":
            option = st.sidebar.date_input(
                field,
                (date(1985, 1, 1), date(2021, 12, 31)),
                date(1985, 1, 1),
                date(2021, 12, 31),
                format="MM.DD.YYYY",
            )
        else:
            categories = retriever.get_field_categories(field, 10)
            option = st.sidebar.selectbox(
                field,
                tuple(categories),
                index=None,
                placeholder="Select ...",
            )
        meta_data[field] = option

    if query := st.text_input("Input your query here", value=""):
        st.write(f"Query: {query}")
        st.write(f"Metadata: {meta_data}")

        start_time = time.time()
        res = retriever.retrieve(
            query,
            filter=meta_data,
        )
        docs = [doc for doc, _ in res]
        retrieve_time_sec = time.time() - start_time
        st.write(f"Retrieve time: {retrieve_time_sec:.3f} sec.")

        N_cards_per_row = 3
        chars_to_show = 80
        if docs:
            for n_row, row in enumerate(docs):
                i = n_row % N_cards_per_row
                if i == 0:
                    st.write("---")
                    cols = st.columns(N_cards_per_row, gap="large")
                # draw the card
                with cols[n_row % N_cards_per_row]:
                    st.markdown(f"*{row.page_content[:chars_to_show].strip()}*")
                    for field in meta_data:
                        st.markdown(f"*{field}: {row.metadata.get(field)}*")
                    st.markdown(f"**{row.metadata['source']}**")


if __name__ == "__main__":
    denser_search()
