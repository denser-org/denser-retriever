# <img src="assets/images/logo.png" alt="denser logo" width="40"/> Denser Retriever

<div align="center">

<!-- [![Build status](https://github.com/denser-org/denser-retriever/workflows/build/badge.svg?branch=main&event=push)](https://github.com/denser-org/denser-retriever/actions?query=workflow%3Abuild) -->

[![Python Version](https://img.shields.io/pypi/pyversions/denser-retriever.svg)](https://pypi.org/project/denser-retriever/)
[![Dependencies Status](https://img.shields.io/badge/dependencies-up%20to%20date-brightgreen.svg)](https://github.com/denser-org/denser-retriever/pulls?utf8=%E2%9C%93&q=is%3Apr%20author%3Aapp%2Fdependabot)

[![Code style: ruff](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/astral-sh/ruff)
[![Security: bandit](https://img.shields.io/badge/security-bandit-green.svg)](https://github.com/PyCQA/bandit)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/denser-org/denser-retriever/blob/main/.pre-commit-config.yaml)
[![Semantic Versions](https://img.shields.io/badge/%20%20%F0%9F%93%A6%F0%9F%9A%80-semantic--versions-e10079.svg)](https://github.com/denser-org/denser-retriever/releases)
[![License](https://img.shields.io/github/license/denser-org/denser-retriever)](https://github.com/denser-org/denser-retriever/blob/main/LICENSE)
![Coverage Report](assets/images/coverage.svg)

Enterprise-grade AI retriever solution that seamlessly integrates to enhance your AI applications.

</div>

## 📝 Description

The Denser Retriever project is set to establish a unified and extensive Retriever hub. This hub will not only incorporate the vector database, which is optimized for recall, but also integrate traditional keyword-based search methods, optimized for precision, along with alternatives like ML rerankers. Our goal is to deliver an Enterprise-grade AI retriever solution that seamlessly integrates to enhance your AI applications.

## 🚀 Features

The initial release of Denser Retriever provides the following features.

- Including various retrievers such as keyword search, vector search, hybrid, and optionally with a ML reranker
- Providing step-by-step guidance to setup keyword and vector search services (elasticsearch and milvus), which are utilized to support various retrievers
- Benchmarking on MTEB datasets with different retrievers to assess the quality of various retrievers
- A unified framework for both passage and long documents retrieval tasks
- Demonstrating how to use Denser retriever to power an end-to-end AI chat application

## 📦 Installation

You can install the latest version of Denser Retriever from PyPI with the following command:

```bash
pip install denser-retriever
```

## 📃 Documentation

The official documentation is hosted on [retriever.denser.ai](https://retriever.denser.ai).

To get started, run the unit tests below under directory `denser-retriever`.

```shell
python -m pytest tests/test_data_loader.py
python -m pytest tests/test_retriever_general.py
```

After that, you can launch a simple end-to-end [streamlit](https://streamlit.io/) app with the retriever indices built from the above unit tests.

```shell
streamlit run scripts/denser_chat.py
streamlit run scripts/denser_search.py
```

To evaluate retrievers' accuracy on [mteb](https://github.com/embeddings-benchmark/mteb) benchmark datasets, you can run the following command:

```shell
python examples/run_benchmark.py
```

## 👨🏼‍💻 Development

You can start developing Denser Retriever on your local machine.

See [DEVELOPMENT.md](DEVELOPMENT.md) for more details.

## 🛡 License

[![License](https://img.shields.io/github/license/denser-org/denser-retriever)](https://github.com/denser-org/denser-retriever/blob/main/LICENSE)

This project is licensed under the terms of the `MIT` license. See [LICENSE](https://github.com/denser-org/denser-retriever/blob/main/LICENSE) for more details.

## 📃 Citation

```bibtex
@misc{denser-retriever,
  author = {denser-org},
  title = {Enterprise-grade AI retriever solution that seamlessly integrates to enhance your AI applications.},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/denser-org/denser-retriever}}
}
```
