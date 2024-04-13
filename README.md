# 🎙️ Introduction

The Denser Retriever project is set to establish a unified and extensive Retriever hub. This hub will not only incorporate the vector database, which is optimized for recall, but also integrate traditional keyword-based search methods, optimized for precision, along with alternatives like ML rerankers. Our goal is to deliver an Enterprise-grade AI retriever solution that seamlessly integrates to enhance your AI applications.

## Denser Retriever Features

The initial release of Denser Retriever provides the following features.

- Including various retrievers such as keyword search, vector search, hybrid, and optionally with a ML reranker
- Providing step-by-step guidance to setup keyword and vector search services (elasticsearch and milvus), which are utilized to support various retrievers
- Benchmarking on MTEB datasets with different retrievers to assess the quality of various retrievers
- A unified framework for both passage and long documents retrieval tasks
- Demonstrating how to use Denser retriever to power an end-to-end AI chat application

## Install Denser Retriever

To get started, install Denser Retriever using `pip` with the following command:

```bash
pip install denser-retriever
```

You can also clone the code and do an editable installation

```bash
git clone https://github.com/denser-org/denser-retriever.git
cd denser-retriever
pip install -e .
```

## Documentation

The documentation can be found at here, including unit tests, retriever experiments on MTEB datasets, and end-to-end chat application. 