import logging
import os
import time
import openai
import streamlit as st
from langchain_core.documents import Document
from denser_retriever.core.retriever import DenserRetriever
from denser_retriever.config import load_retriever_config

logger = logging.getLogger(__name__)

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
retriever.ingest(texts)

openai.api_key = os.getenv("OPENAI_API_KEY")
default_openai_model = "gpt-4o"


def denser_chat():
    st.title("Denser Chat Demo")
    st.caption("Try asking about Python, machine learning, or NLP!")
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
        retrieval_result = retriever.retrieve(query=query, k=5, usage=True)
        retrieve_time_sec = time.time() - start_time
        st.write(f"Retrieve time: {retrieve_time_sec:.3f} sec.")

        docs = [doc for doc, _ in retrieval_result.documents]
        prompt = (
            "### Instructions:\n"
            "Use the provided context to answer the query. If no relevant context is found, use your knowledge.\n"
            f"### Query:\n{query}\n"
        )
        if docs:
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
                    temperature=0.0,
            ):
                full_response += response.choices[0].delta.get("content", "")
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)

        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )
        st.session_state.messages = []

        st.caption("Retrieved Documents")
        for i, (doc, score) in enumerate(retrieval_result.documents):
            st.write(f"[{i + 1}] Score: {score:.3f}\n{doc.page_content}\n")


if __name__ == "__main__":
    denser_chat()