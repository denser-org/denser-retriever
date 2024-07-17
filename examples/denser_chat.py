import logging
import os
import time

import openai
import streamlit as st

from denser_retriever.retriever import DenserRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

logger = logging.getLogger(__name__)

index_name = "unit_test_denser"

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

openai.api_key = os.getenv("OPENAI_API_KEY")
default_openai_model = "gpt-4o"
starting_url = "https://denser.ai"
optional_str = 'Try to ask "What did the president say about Ketanji Brown Jackson?" '


def denser_chat():
    st.title("Denser Chat Demo")
    st.caption(f"Starting URL: {starting_url}")
    if optional_str:
        st.caption(f"{optional_str}")
    st.divider()
    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = default_openai_model

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if query := st.chat_input("Input your query here"):
        with st.chat_message("user"):
            st.markdown(query)

        start_time = time.time()
        passages = retriever.retrieve(query)
        # 取出document
        docs = [passage[0] for passage in passages]
        retrieve_time_sec = time.time() - start_time
        st.write(f"Retrieve time: {retrieve_time_sec:.3f} sec.")

        prompt = (
            "### Instructions:\n"
            "The following context consists of an ordered list of sources. If you can find answers from the context, use the context to provide a long response. You MUST cite the context titles and source URls strictly in Markdown format in your response. If you cannot find the answer from the sources, use your knowledge to come up a reasonable answer and do not cite any sources. If the query asks to summarize the file or uploaded file, provide a summarization based on the provided sources. If the conversation involves casual talk or greetings, rely on your knowledge for an appropriate response."  # noqa: E501
        )

        prompt += f"### Query:\n{query}\n"
        if len(docs) > 0:
            prompt += f"\n### Context:\n{docs}\n"
        prompt += "### Response:"

        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            for response in openai.ChatCompletion.create(
                model=st.session_state["openai_model"],
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                stream=True,
                top_p=0,
                temperature=0.0,
            ):
                full_response += response.choices[0].delta.get("content", "") # type: ignore
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)

        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )
        st.session_state.messages = []
        st.caption("Sources")
        for i, passage in enumerate(passages):
            doc = passage[0]
            st.write(
                f"[{(i + 1)}]  [{doc.metadata['source']}]({doc.metadata['source']})  \n{doc.metadata['source']}  \n**Score**: {passage[1]}"  # noqa: E501
            )


if __name__ == "__main__":
    denser_chat()
