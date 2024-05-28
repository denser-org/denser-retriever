import logging
import time
from datetime import date

import streamlit as st

from denser_retriever.retriever_general import RetrieverGeneral

logger = logging.getLogger(__name__)

index_name = "unit_test_titanic"
retriever = RetrieverGeneral(index_name, "tests/config-titanic.yaml")
retriever.ingest("tests/test_data/titanic_top10.jsonl")

starting_url = "https://github.com/datasciencedojo/datasets/blob/master/titanic.csv"
optional_str = 'Try questions such as "cumings"'


def denser_search():
    st.title("Denser Search Demo")
    st.caption(f"Starting URL: {starting_url}")
    if optional_str:
        st.caption(f"{optional_str}")
    st.divider()

    fields_and_types = retriever.retrieverElasticSearch.get_index_mappings()

    meta_data = {}
    for field, type in fields_and_types.items():
        if field in ["content", "title", "source", "pid"]:
            continue
        if type == "date":
            option = st.sidebar.date_input(
                field,
                (date(1858, 1, 1), date(1910, 12, 31)),
                date(1858, 1, 1),
                date(1910, 12, 31),
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
        passages, docs = retriever.retrieve(query, meta_data)
        retrieve_time_sec = time.time() - start_time
        st.write(f"Retrieve time: {retrieve_time_sec:.3f} sec.")

        N_cards_per_row = 3
        chars_to_show = 80
        if passages:
            for n_row, row in enumerate(docs):
                i = n_row % N_cards_per_row
                if i == 0:
                    st.write("---")
                    cols = st.columns(N_cards_per_row, gap="large")
                # draw the card
                with cols[n_row % N_cards_per_row]:
                    st.caption(f"{row['title'].strip()}")
                    st.markdown(f"**{row['score']}**")
                    st.markdown(f"*{row['text'][:chars_to_show].strip()}*")
                    for field in meta_data:
                        st.markdown(f"*{field}: {row.get(field)}*")
                    st.markdown(f"**{row['source']}**")


if __name__ == "__main__":
    denser_search()
