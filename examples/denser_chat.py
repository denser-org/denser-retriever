import logging
import os
import time

import openai
import streamlit as st
import tiktoken

from denser_retriever.retriever_general import RetrieverGeneral

logger = logging.getLogger(__name__)

index_name = "unit_test_denser"
retriever = RetrieverGeneral(index_name, "tests/config-denser.yaml")

openai.api_key = os.getenv("OPENAI_API_KEY")
default_openai_model = "gpt-3.5-turbo-0125"
starting_url = "https://denser.ai"
optional_str = 'Try questions such as "what is denser ai?" '


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
        passages, docs = retriever.retrieve(query, {})
        retrieve_time_sec = time.time() - start_time
        st.write(f"Retrieve time: {retrieve_time_sec:.3f} sec.")

        prompt = (
            "### Instructions:\n"
            "The following context consists of an ordered list of sources. If you can find answers from the context, use the context to provide a long response. You MUST cite the context titles and source URls strictly in Markdown format in your response. If you cannot find the answer from the sources, use your knowledge to come up a reasonable answer and do not cite any sources. If the query asks to summarize the file or uploaded file, provide a summarization based on the provided sources. If the conversation involves casual talk or greetings, rely on your knowledge for an appropriate response."  # noqa: E501
        )

        prompt += f"### Query:\n{query}\n"
        if len(passages) > 0:
            prompt += f"\n### Context:\n{passages}\n"
        prompt += "### Response:"

        st.session_state.messages.append({"role": "user", "content": prompt})

        enc = tiktoken.encoding_for_model(default_openai_model)
        prompt_length = len(enc.encode(prompt))
        logger.info(f"prompt length:{prompt_length}")

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
                full_response += response.choices[0].delta.get("content", "")
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)

        st.session_state.messages.append({"role": "assistant", "content": full_response})
        st.session_state.messages = []
        st.caption("Sources")
        for i, passage in enumerate(passages):
            st.write(
                f"[{(i + 1)}]  [{passage['title']}]({passage['source']})  \n{passage['source']}  \n**Score**: {passage['score']}"  # noqa: E501
            )


if __name__ == "__main__":
    denser_chat()
