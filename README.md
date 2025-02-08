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
- Built-in experiments for [MTEB](https://github.com/embeddings-benchmark/mteb) scifact, MsMarco and LeCaRDv2 datasets
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

After starting the services, here is the code to build a retriever and run a query. The retriever building and query are governed by the configuration file `denser_retriever/configs/fusion_msmarco.json`, which specifies the embedding model, reranker model and top-k arguments in retrieval.

```python
from langchain_core.documents import Document
from denser_retriever.core.retriever import DenserRetriever
from denser_retriever.core.keyword import ESIndexData
from denser_retriever.core.vectordb.milvus import MilvusIndexData

# Create sample documents
texts = [
    Document(page_content="Python is a high-level programming language known for its simplicity and readability."),
    Document(page_content="Machine learning is a subset of AI that enables systems to learn from data."),
    Document(page_content="Natural Language Processing (NLP) helps computers understand human language."),
    Document(page_content="Deep learning models use neural networks with multiple layers.")
]

# Initialize retriever
index_name = "tech_docs"
# Create retriever with config and index data
es_data = ESIndexData(
    index_name=index_name,
    analysis="default",
    drop_old=True
)
milvus_data = MilvusIndexData(
    index_name=index_name,
    embedding_size=768,  # Match embedding model size
    drop_old=True
)
retriever = DenserRetriever(
    config_path="denser_retriever/configs/fusion_msmarco.json",
    es_data=es_data,
    milvus_data=milvus_data
)

# Ingest documents
retriever.ingest(texts)

result = retriever.retrieve(query="Explain machine learning", k=5, usage=True)
print(result.to_json())

# Cleanup
retriever.delete_all(delete_index=True)
```

You can also find the code at `experiments/quick_start.py`. Run the following command to execute the code.

```bash
python -m denser_retriever.experiments.quick_start
```

You can easily build a RAG-based chatbot with the following command. It 1) retrieves search results, and 2) passes the results to a LLM (gpt-4o) to generate final results. 

```bash
export OPENAI_API_KEY="your_api_key_here"
streamlit run denser_retriever/examples/chat.py
``` 

## Ingetstion

We use a [MTEB dataset](https://github.com/embeddings-benchmark/mteb) scifact to illustrate the ingestion. A dataset consists of a corpus, a query set and annotations: the ground truth docs for each query. To ingest a
dataset, run the following command. The first argument is the dataset name, the second and third arguments
are the dataset splits. The fourth argument is the path to the configuration file. The fifth argument is
the `--ingest-only` flag, which is used to ingest the dataset without training.

```bash
python -m denser_retriever.experiments.train mteb/scifact train test \
 --config denser_retriever/configs/train_default.json --ingest-only
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
The retriever index is generated in the ingestion section above. We can change the method `fusion` in the configure file to use other methods:`vector`, `hybrid`, or `reranker`.

```bash
python -m denser_retriever.experiments.retrieve scifact \
"0-dimensional biomaterials show inductive properties." \
--config denser_retriever/configs/fusion_scifact.json
```
The above command leads to the following results:

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


## Mteb Experiments

<details>
<summary>
Scifact Dataset Experiment
</summary>

 ### Training 

Training refers to train a logistic regression model to fuse keyword search, vector search and rerank. The trained
logistic regression model is used in the `fusion` combine method. Without training, we can still use `vector`, `hybrid`,
and `reranker` methods to retrieve passages.

The following command shows a training example, where `mteb/scifact` is the dataset name, `train` and `test` are the
splits to use, and `train_default.json` is the training configuration file. `train_default.json` specifies the
elasticsearch, vector search, embedding and reranker models to use.

```bash
python -m denser_retriever.experiments.train mteb/scifact train test \
--config denser_retriever/configs/train_default.json
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

Where
* `keyword`: keyword search
* `vector`: vector search
* `reranker`: reranking of the joint set of keyword search and vector search
results
* `es+vs`: use elasticsearch and vector search features to train a logistic regression model to combine elastic search and vector search
* `es+rr`: similar to above but use elasticsearch and reranker features
* `vs+rr`: similar to above but use vector search and reranker features
* `es+vs+rr`: similar to above but use elasticsearch, vector search and reranker features

Four models `weights_es+vs.json`, `weights_es+rr.json`, `weights_vs+rr.json` and `weights_es+vs+rr.json` are
saved to the `exps/exp_scifact/models` directory. We use the `weights_es+vs+rr.json` to power`fusion` method.

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

the evaluation accuracy is listed below. As expected, the `fusion` method which uses
the `weights_es+vs+rr.json` model from the training, obtained identical ndcg@10 number as the `es+vs+rr` from training.

| Method   | NDCG@10 |
|----------|---------|
| vector   | 0.7317  |
| hybrid   | 0.6832  |
| reranker | 0.6759  |
| fusion   | 0.7434  |

</details>

<details>
<summary>
MsMarco Dataset Experiment
</summary>

### Training

Scifact dataset is a small dataset, and the training process is fast. For a large dataset like MsMarco, the training
takes two days to complete. We provide a script to train the MsMarco dataset with the following command.

```bash
python -m denser_retriever.experiments.train mteb/msmarco train dev \
--config denser_retriever/configs/train_default.json
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

the evaluation accuracy is listed below. For MsMarco dataset, the vector and reranker methods lead to similar
NDCG@10 scores. The fusion method outperforms the other methods with a higher cost.

| Method   | NDCG@10 |
|----------|---------|
| vector   | 0.4147  |
| hybrid   | 0.3416  |
| reranker | 0.4013  |
| fusion   | 0.4707  |

</details>

<details>
<summary>
LeCaRDv2 Dataset Experiment
</summary>

LeCaRDv2 dataset involves identifying and retrieving the case document that best matches or is most relevant to the scenario described in each of the provided queries. The query set contains 159 queries, each outlining a distinct situation. The corpus set includes 3795 candidate case documents. The original data link is at https://github.com/THUIR/LeCaRDv2


### Evaluation

LeCaRDv2 dataset only has 159 test queries and does not have a training dataset. While we cannot train a Logistic Regression model on this data, we use the model trained on MsMarco dataset to evaluate. The embedding model of `BAAI/bge-m3` and reranker model `BAAI/bge-reranker-v2-m3` are used in the evaluation. All models are specified in the `fusion_lecardv2.json`.

```bash
python -m denser_retriever.experiments.evaluate \
    lecardv2 \
    mteb/lecardv2 \
    --split test \
    --config denser_retriever/configs/fusion_lecardv2.json \
    --output-dir exps/exp_lecardv2/pred \
    --top-k 100
```

The evaluation accuracy is listed below. The hybrid method achieves the highest NDCG@10 score (0.7510), which is strong when compared to the [Huggingface Mteb leaderboard](https://huggingface.co/spaces/mteb/leaderboard). The fusion method does not perform well mainly due to 1) the reranker model does not perform well on this dataset, and 2) the fusion logistic regression model is trained on MsMarco dataset.

| Method   | NDCG@10 | 
|----------|---------|
| vector   | 0.7034  |
| hybrid   | 0.7510  |
| reranker | 0.6022  |
| fusion   | 0.7054  |

</details>

## Unit Tests

Run the following command to run all unit tests.

```bash
pytest tests
```

If you want to run a specific test, for example the `test_retrieve` method in `test_retriever.py`, you can use the following command.

```bash
pytest tests/test_retriever.py::TestRetriever::test_retrieve
```

## 📃 Documentation (outdated)

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
