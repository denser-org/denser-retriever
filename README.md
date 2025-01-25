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

## Key Features:

- Optimizes retrieval by combining keyword search, vector search, and reranking
- Simple example for quick start
- Built-in support for [MTEB](https://github.com/embeddings-benchmark/mteb) scifact and MsMarco datasets
- Full MTEB Retrieval benchmark experiments
- Ready-to-use components for chatbots and semantic search applications

## Installation

Download the source code and install the package with poetry.

```bash
git clone git@github.com:denser-org/denser-retriever.git
cd denser-retriever
poetry install
```

## Quick Start

We need to start the elasticsearch and Milvus services before running the experiments.

```bash
docker compose up -d
```

After starting the services, we run the following command to use `"tests/test_data/state_of_the_union.txt"` file to build a retriever and run a
query `"What did the president say about Ketanji Brown Jackson"` to retrieve the top 10 passages.

```commandline
python -m denser_retriever.experiments.toy_example
```

## Ingetstion

We use a [MTEB dataset](https://github.com/embeddings-benchmark/mteb) scifact to illustrate the ingestion. To ingest a
dataset, run the following command. The first argument is the dataset name, the second and third arguments
are the dataset splits. The fourth argument is the path to the configuration file. The fifth argument is
the `--ingest-only` flag, which is used to ingest the dataset without training.

```bash
python -m denser_retriever.experiments.train mteb/scifact train test --config denser_retriever/configs/train_default.json --ingest-only
```

We note that the ingestion stops when it exceeds the following quotas specified
in `denser_retriever/configs/cost_config.json`.

```json
{
  "es_storage_quota_gb": 2.0,
  "vector_storage_quota_gb": 2.0,
  "vector_token_quota_million": 2.0
}
```

The ingestion storage and vector tokens and their costs are reported after the ingestion. If you run a large experiment,
make sure to set the quotas to `inf` so you have infinite budget to run experiments. As a reference, the ingestion of
scifact dataset (5183 documents) has the
following ingestion stats:

```json
{
  "num_docs": 5183,
  "es_storage_gb": 0.03670822083950042,
  "vector_storage_gb": 0.014828681945800781,
  "vector_tokens": 1635249
}
```

## Retrieval

Denser retriever consists of three components: keyword search, vector search, and reranker. We provide the following
methods to combine these components:

| Method   | Description                                              | Components Used                           |
|----------|----------------------------------------------------------|-------------------------------------------|
| vector   | Direct vector database search results                    | Vector Search                             |
| hybrid   | Combines rank positions from keyword and vector searches | Keyword Search + Vector Search            |
| reranker | Keyword search followed by reranking                     | Keyword Search + Reranker                 |
| fusion   | Uses logistic regression to combine all components       | Keyword Search + Vector Search + Reranker |

To retrieve passages for a given question with the `fusion` method, use the following command. The first argument is the
retriever index name, the second argument is the question, and the third argument is the path to the configuration file.
The retriever index is generated in the ingestion section above.
Replace `fusion.json` with `vector.json`, `hybrid.json`, `reranker.json` to use the corresponding method.

```bash
python -m denser_retriever.experiments.retrieve scifact "0-dimensional biomaterials show inductive properties." --config denser_retriever/configs/fusion_scifact.json
```

Alternatively, you can also use command-line arguments. Replace `fusion`
with `vector`, `hybrid`, `reranker` to use the corresponding method.

```bash
python -m denser_retriever.experiments.retrieve scifact "0-dimensional biomaterials show inductive properties." --combine-method fusion
```

Either of the above commands leads to the following results:

```text
Top 10 results for query: 0-dimensional biomaterials show inductive properties.
Using method: fusion
Usage: {'vector_tokens': 38522, 'reranker_tokens': 68623}
--------------------------------------------------------------------------------

1. Score: 0.5843
Content: Life forms that have low body mass can hunt for food on the undersurface of branches or along shear cliff faces quite unperturbed by gravity. For larger animals, the hunt for dinner and the struggle t...
Metadata: {
  "title": "Periosteal bone formation--a neglected determinant of bone strength.",
  "source": null,
  "pid": "40212412"
}
--------------------------------------------------------------------------------

2. Score: 0.5039
Content: BACKGROUND Carbon nanotubes (CNT) hold great promise to create new and better products for commercial and biomedical applications, but their long-term adverse health effects are a major concern. The o...
Metadata: {
  "title": "Induction of stem-like cells with malignant properties by chronic exposure of human lung epithelial cells to single-walled carbon nanotubes",
  "source": null,
  "pid": "16532419"
}
--------------------------------------------------------------------------------
```

## Scifact Dataset Experiment

### Training

Training refers to train a logistic regression model to fuse keyword search, vector search and rerank. The trained
logistic regression model is used in the `fusion` combine method. Without training, we can still use `vector`, `hybrid`,
and `reranker` methods to retrieve passages.

The following command shows a training example, where `mteb/scifact` is the dataset name, `train` and `test` are the
splits to use, and `train_default.json` is the training configuration file. `train_default.json` specifies the
elasticsearch and vector search setup, as well as the embedding and reranker models to use.

```bash
python -m denser_retriever.experiments.train mteb/scifact train test --config denser_retriever/configs/train_default.json
```

The training process:

1. Ingests documents from the dataset
2. Generates features for each split
3. Computes baselines (keyword search, vector search and reranker) performance metrics
4. Trains the model using train split
5. Reports final performance metrics on test split

After training, we get the following accuracy report.

```text
== NDCG@10
metric_keyword.json: "NDCG@10": 0.58425,
metric_vector.json: "NDCG@10": 0.73167,
metric_reranker.json: "NDCG@10": 0.67021,
metric_es+vs.json: "NDCG@10": 0.73476,
metric_es+rr.json: "NDCG@10": 0.68317,
metric_vs+rr.json: "NDCG@10": 0.73946,
metric_es+vs+rr.json: "NDCG@10": 0.74344,
```

`keyword` and `vector` are the keyword search and vector search
respectively. `reranker` is the reranking of the joint set of keyword search and vector search
results. `es+vs`, `es+rr`, `vs+rr`, and `es+vs+rr` are the features used to train logistic models. For
example, `es+vs+rr` is a logistic regression model which was trained using keyword (es), vector search (vs) and
reranker (rr) features, while `es+vs` is a logistic regression model which was trained using keyword and vector search
features. Four models `weights_es+vs.json`, `weights_es+rr.json`, `weights_vs+rr.json` and `weights_es+vs+rr.json` are
saved to the `exps/exp_scifact/models` directory. We use the `weights_es+vs+rr.json` model in the `fusion` combine
method (see Retrieving and Evaluation and Retrieving Sections) to retrieve passages, as it outperforms other methods.

### Evaluation

Run the following command to evaluate the retrieval performance of different methods: `vector`, `hybrid`, `reranker`,
and `fusion`. The first argument is the index name (generated from the training above), the second argument is the
dataset name, the third argument is the
path to the `fusion` configuration file, the fourth argument is the output directory, the fifth argument is the number
of top-k passages to retrieve, and the sixth argument is the number of queries to evaluate.

```bash
python -m denser_retriever.experiments.evaluate \
    scifact \
    mteb/scifact \
    --config denser_retriever/configs/fusion_scifact.json \
    --output-dir exps/exp_scifact/pred \
    --top-k 100 \
    --num-queries 0
```

With the following cost configuration in `denser_retriever/configs/cost_config.json`,

```json
{
  "storage_cost_per_gb": 0.4,
  "es_query_cost": 0.001,
  "vector_token_cost_per_million": 0.08,
  "reranker_token_cost_per_million": 0.05
}
```

the evaluation accuracy and cost are listed below. As expected, the `fusion` method which uses
the `weights_es+vs+rr.json` model from the training, obtained identical ndcg@10 number as the `es+vs+rr` from training.

| Method   | NDCG@10 | COST |
|----------|---------|------|
| vector   | 0.7317  | 0.83 |
| hybrid   | 0.6832  | 1.13 |
| reranker | 0.6759  | 0.83 |
| fusion   | 0.7434  | 2.10 |

## MsMarco Dataset Experiment

### Training

Scifact dataset is a small dataset, and the training process is fast. For a large dataset like MsMarco, the training
takes two days to complete. We provide a script to train the MsMarco dataset with the following command.

```bash
python -m denser_retriever.experiments.train mteb/msmarco train dev --config denser_retriever/configs/train_default.json
```

After training, we get the following accuracy report. We note that the keyword search and vector search lead to the
NDCG@10 scores of 0.21841 and 0.41771 respectively. The fusion method which uses the `weights_es+vs+rr.json` model,
obtained the NDCG@10 score of 0.47066.

```text
== NDCG@10
metric_keyword.json: "NDCG@10": 0.21841,
metric_vector.json: "NDCG@10": 0.41771,
metric_reranker.json: "NDCG@10": 0.47103,
metric_es+vs.json: "NDCG@10": 0.40606,
metric_es+rr.json: "NDCG@10": 0.46554,
metric_vs+rr.json: "NDCG@10": 0.47467,
metric_es+vs+rr.json: "NDCG@10": 0.47066,
```

### Evaluation

Run the following command to evaluate 6980 queries in MsMarco dev dataset.

```bash
python -m denser_retriever.experiments.evaluate \
    msmarco \
    mteb/msmarco \
    --split dev \
    --config denser_retriever/configs/fusion_msmarco.json \
    --output-dir exps/exp_msmarco/pred \
    --top-k 100 \
    --num-queries 0
```

With the following cost configuration in `denser_retriever/configs/cost_config.json`,

```json
{
  "storage_cost_per_gb": 0.4,
  "es_query_cost": 0.001,
  "vector_token_cost_per_million": 0.08,
  "reranker_token_cost_per_million": 0.05
}
```

the evaluation accuracy and cost are listed below. For MsMarco dataset, the vector and reranker methods lead to similar
NDCG@10 scores. The fusion method outperforms the other methods with a higher cost.

| Method   | NDCG@10 | COST  |
|----------|---------|-------|
| vector   | 0.4147  | 4.30  |
| hybrid   | 0.3416  | 11.28 |
| reranker | 0.4013  | 10.02 |
| fusion   | 0.4707  | 16.65 |

## Unit Tests

Run the following command to run all unit tests.

```bash
pytest tests
```

If you want to run a specific test, for example the `test_retrieve` method in `test_retriever.py`, you can use the following command.

```bash
pytest tests/test_retriever.py::TestRetriever::test_retrieve
```

## 📃 Documentation (slightly outdated)

The official documentation is hosted on [retriever.denser.ai](https://retriever.denser.ai). The complete MTEB retrieval experiment is available at [retriever-docs.denser.ai](https://retriever-docs.denser.ai/docs/core/experiments/mteb_retrieval).

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
