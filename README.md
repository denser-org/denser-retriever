# <img src="https://github.com/denser-org/denser-retriever/blob/main/www/public/icon-192.png?raw=true" alt="denser logo" width="40"/> Denser Retriever

<div align="center">

<!-- [![Build status](https://github.com/denser-org/denser-retriever/workflows/build/badge.svg?branch=main&event=push)](https://github.com/denser-org/denser-retriever/actions?query=workflow%3Abuild) -->

[![Python Version](https://img.shields.io/pypi/pyversions/denser-retriever.svg)](https://pypi.org/project/denser-retriever/)
[![Dependencies Status](https://img.shields.io/badge/dependencies-up%20to%20date-brightgreen.svg)](https://github.com/denser-org/denser-retriever/pulls?utf8=%E2%9C%93&q=is%3Apr%20author%3Aapp%2Fdependabot)

[![Code style: ruff](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/astral-sh/ruff)
[![Security: bandit](https://img.shields.io/badge/security-bandit-green.svg)](https://github.com/PyCQA/bandit)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/denser-org/denser-retriever/blob/main/.pre-commit-config.yaml)
[![Semantic Versions](https://img.shields.io/badge/%20%20%F0%9F%93%A6%F0%9F%9A%80-semantic--versions-e10079.svg)](https://github.com/denser-org/denser-retriever/releases)
[![License](https://img.shields.io/github/license/denser-org/denser-retriever)](https://github.com/denser-org/denser-retriever/blob/main/LICENSE)

An enterprise-grade AI retriever designed to streamline AI integration into your applications, ensuring cutting-edge accuracy.

</div>

## 🚀 Features

- Supporting heterogeneous retrievers such as **keyword search**, **vector search**, and **ML model reranking**
- Leveraging ML technique to effectively combine heterogeneous retrievers
- **Comprehensive benchmark** on [MTEB](https://github.com/embeddings-benchmark/mteb) Retrieval dataset
- Demonstrating how to use Denser retriever to power an **end-to-end applications** such as chatbot and semantic search
![mteb_ndcg_plot](https://github.com/denser-org/denser-retriever/blob/main/mteb_ndcg_plot.png?raw=true)

## 📦 Installation

Download the source code with the following command:

```bash
git clone git@github.com:denser-org/denser-retriever.git
cd denser-retriever
```

To install Denser Retriever, you can run:

```bash
pip install denser-retriever
```
or
```bash
poetry add denser-retriever
```

## Quick Start

Go to directory `experiments` and run the following experiments.

### Build a retriever on a file

Run the following command to build a retriever on a file `tests/tests_data/state_of_the_union.txt`.

```bash
python index_and_query_local_file.py
```

### Build a retriever on a webpage

Run the following command to build a retriever on webpage `https://denser.ai`.

```bash
python index_and_query_webiste.py
```

### Build a retriever on `mteb/scifact` dataset

Run the following command to build a retriever on `mteb/scifact` dataset.

```bash
python train_and_test.py mteb/scifact train test configs/default.json
```
which trains a retriever on `mteb/scifact` dataset and tests the retriever on the test set. After training, we can run the following command to test the retriever with a query.

```bash
python retrieve.py scifact "1/2000 in UK have abnormal PrP positivity." --config configs/default.json
```

### Build a retriever on `mteb/lecardv2` dataset

Run the following command to build a retriever on `mteb/lecardv2` dataset.

```bash
python train_and_test.py mteb/lecardv2 test test configs/lecardv2.json
```
which trains a retriever on `mteb/lecardv2` dataset and tests the retriever on the test set. After training, we can run the following command to test the retriever with a query.

```bash
python retrieve.py lecardv2 "黎红春抢劫、潘某1窝藏一审刑事判决书上海市黄浦区人民法院刑事判决书（2019）沪0101刑初808号：上海市黄浦区人民检察院以沪黄检一>部刑诉〔2019〕1589号起诉书指控被告人黎红春犯抢劫罪、被告人潘某1犯窝藏罪，于2019年8月23日向本院提起公诉。" --config configs/lecardv2.json
```

## 📝 Experiments

### [MTEB Retrieval experiment](https://retriever-docs.denser.ai/docs/core/experiments/mteb_retrieval)

## 📃 Documentation

The official documentation is hosted on [retriever.denser.ai](https://retriever.denser.ai).
Click [here](https://retriever.denser.ai/docs/quick-start) to get started.

## 👨🏼‍💻 Development

You can start developing Denser Retriever on your local machine.

See [DEVELOPMENT.md](DEVELOPMENT.md) for more details.

## 🛡 License

[![License](https://img.shields.io/github/license/denser-org/denser-retriever)](https://github.com/denser-org/denser-retriever/blob/main/LICENSE)

This project is licensed under the terms of the `MIT` license.
See [LICENSE](https://github.com/denser-org/denser-retriever/blob/main/LICENSE) for more details.

## 📃 Citation

```bibtex
@misc{denser-retriever,
  author = {denser-org},
  title = {An enterprise-grade AI retriever designed to streamline AI integration into your applications, ensuring cutting-edge accuracy.},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/denser-org/denser-retriever}}
}
```
