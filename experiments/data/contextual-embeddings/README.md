# Anthropic Contextual Retrieval Dataset

In Anthropic's recent blog post [Introducing Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval),
they proposed a method that improves the retrieval step in RAG. The method is called “Contextual Retrieval”, which leads
to significant improvements in retrieval accuracy and translates to better performance in downstream tasks.
In this experiment, we benchmark the Anthropic Retrieval dataset in Denser Retriever. Our key
findings are as follows:

1. The Anthropic cookbook demonstrates a prototype of Retrieval-Augmented Generation (RAG) but lacks scalability for
   large systems. For instance, it loads all document embeddings into memory, which becomes impractical for a large
   retrieval corpus. We first reproduce the experiments using
   the [Denser Retriever](https://github.com/denser-org/denser-retriever) codebase. With built-in support for
   Elasticsearch and vector search, our implementation is prepared for deployment in large-scale industrial
   applications.

2. Denser Retriever offers various configuration options for building retrievers. Users can choose between paid API
   services or open-source (free) models to balance accuracy and cost. In our experiments on the Anthropic contextual
   retrieval dataset, we demonstrate that comparable accuracy can be achieved by substituting paid model APIs with
   open-source models. This flexibility is crucial for production deployments where managing costs is a priority.

## Dataset

The dataset referenced in the blog
post [Introducing Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) and the
Anthropic [Cookbook](https://github.com/anthropics/anthropic-cookbook/tree/main/skills/contextual-embeddings) is located
in the `original_dataset` directory. It contains a total of 248 queries and 737 documents.

In order to utilize the dataset in Denser Retriever experiments, we run the following command to generate two
datasets, `data_base` and `data_context`.

```bash
python experiments/data/contextual-embeddings/create_data.py
```

The `data_base` dataset is the original dataset, and the `data_context`
dataset is augmented with contextual text as proposed in the Anthropic blog post. Each dataset consists of query
file `queries.json`,
document file `passages.jsonl`, and relevance file `qrels.jsonl`. We note that the difference between the two datasets
is that the `data_context` document file `passages.jsonl` contains the augmented document contexts from the Anthropic
API (see the blog post for more details). We include this file so that
users can directly use them without calling the Anthropic API.

## Baseline Experiment

We run the Denser Retriever experiments on the `data_base` dataset with the following command.

```bash
python experiments/train_and_test.py anthropic_base test test
```

`anthropic_base` is the experiment dataset name, two `test` are the splits for training and test respectively. They are
identical in our case. Under the hood, we apply 3-fold cross-validation to the whole dataset (248 queries). Three train
and test experiments are conducted. In each run, 2/3 of the queries are used to train Denser Retriever and 1/3 of the
remainder queries are used to test. The final test accuracy is averaged over these three experiments. Following the
Anthropic blog setting, we use Voyage `voyage-2` API model for vector embedding and Cohere `rerank-english-v3.0` API
model for re-ranking. Interested users can refer to the `experiments/train_and_test.py` script for more details.

The following shows the results of the Denser Retriever on the `data_base` dataset. `keyword` is the BM25
method, `vector` is the Voyage-2 vector search, `reranker` is Cohere `rerank-english-v3.0` rerank accuracy based
on `keyword` and `vector` searches. All remaining methods are the combinations of keyword search, vector search and
reranker, which are proposed and implemented in Denser Retriever. For example, `es+vs` is the combination
of keyword search and vector search. `es+vs_n` is a variant of `es+vs` with additional normalization in combining
keyword search and vector search. While Anthropic blog only reported Recall@20, we additionally report NDCG@20, as the
latter factors in the ground truth position when evaluating the search results.

```
== NDCG@20
metric_keyword.json: "NDCG@20": 0.47541,
metric_vector.json: "NDCG@20": 0.73526,
metric_reranker.json: "NDCG@20": 0.81858,
metric_es+vs.json: "NDCG@20": 0.74581,
metric_es+rr.json: "NDCG@20": 0.81903,
metric_vs+rr.json: "NDCG@20": 0.81635,
metric_es+vs+rr.json: "NDCG@20": 0.82218,
metric_es+vs_n.json: "NDCG@20": 0.75084,
metric_es+rr_n.json: "NDCG@20": 0.80381,
metric_vs+rr_n.json: "NDCG@20": 0.81515,
metric_es+vs+rr_n.json: "NDCG@20": 0.81169,

== Recall@20
metric_keyword.json: "Recall@20": 0.70488,
metric_vector.json: "Recall@20": 0.90063,
metric_reranker.json: "Recall@20": 0.94158,
metric_es+vs.json: "Recall@20": 0.90711,
metric_es+rr.json: "Recall@20": 0.94081,
metric_vs+rr.json: "Recall@20": 0.93568,
metric_es+vs+rr.json: "Recall@20": 0.94249,
metric_es+vs_n.json: "Recall@20": 0.90903,
metric_es+rr_n.json: "Recall@20": 0.92276,
metric_vs+rr_n.json: "Recall@20": 0.94105,
metric_es+vs+rr_n.json: "Recall@20": 0.93232,
```

The baseline experiment result is consistent with the Anthropic cookbook run. Specifically, the
Recall@20 of 0.90063 confirms the Anthropic cookbook result of 0.9006. The keyword and vector search lead to the
Recall@20 of 0.704 and 0.900 respectively. The reranker further improves the Recall@20 to 0.94158. The same trend is
observed in the NDCG@20 metric. We note that the method es+vs+rr_n does not offer better accuracy than
reranker, partially due to the strong performance of the reranker. This is different from
the [MTEB benchmarks](https://retriever-docs.denser.ai/docs/core/experiments/mteb_retrieval) where es+vs+rr_n offers the
best accuracy.

## Contextual Experiment

We now run the Denser Retriever experiments on the `data_context` dataset with the following command. The difference
between the `data_base` and `data_context` datasets is that the `data_context` dataset contains the augmented
document contexts from the Anthropic API.

```bash
python experiments/train_and_test.py anthropic_context test test
```

Upon running the above command, the Denser Retriever is trained and tested on the `data_context` dataset. The results
are shown below.

```
== NDCG@20
metric_keyword.json: "NDCG@20": 0.7041,
metric_vector.json: "NDCG@20": 0.75732,
metric_reranker.json: "NDCG@20": 0.8393,
metric_es+vs.json: "NDCG@20": 0.76807,
metric_es+rr.json: "NDCG@20": 0.83337,
metric_vs+rr.json: "NDCG@20": 0.8298,
metric_es+vs+rr.json: "NDCG@20": 0.83267,
metric_es+vs_n.json: "NDCG@20": 0.76734,
metric_es+rr_n.json: "NDCG@20": 0.83657,
metric_vs+rr_n.json: "NDCG@20": 0.82849,
metric_es+vs+rr_n.json: "NDCG@20": 0.83085,

== Recall@20
metric_keyword.json: "Recall@20": 0.89267,
metric_vector.json: "Recall@20": 0.94489,
metric_reranker.json: "Recall@20": 0.96102,
metric_es+vs.json: "Recall@20": 0.9543,
metric_es+rr.json: "Recall@20": 0.958,
metric_vs+rr.json: "Recall@20": 0.95195,
metric_es+vs+rr.json: "Recall@20": 0.95699,
metric_es+vs_n.json: "Recall@20": 0.94153,
metric_es+rr_n.json: "Recall@20": 0.95766,
metric_vs+rr_n.json: "Recall@20": 0.95228,
metric_es+vs+rr_n.json: "Recall@20": 0.95699,
```

The results of the contextual experiments are consistent with the Anthropic cookbook run. Specifically, the
Recall@20 of 94.48 is similar to Anthropic cookbook result of 94.08. The keyword and vector search lead to the Recall@20
of 89.26 and 94.48 respectively. The reranker further improves the Recall@20 to 96.10. The same trend is observed in the
NDCG@20 metric. Compared to the baseline experiment, the contextual augmentation improves the keyword search more than
that of the vector search. Specifically, it boosts the keyword Recall@20 from 70.48 to 89.26 (18.78 points increase),
while the vector Recall@20 only improves from 90.06 to 94.48 (4.42 points increase). The reranker further improves the
Recall@20 from 94.15 to 96.10. The same trend is observed in the NDCG@20 metric.

## Contextual Experiments with Open Source (Free) Models

In above experiments (either baseline or contextual), we used the paid API models from Voyage and Cohere. It costs **a
few cents** and around **$1** from Voyage and Cohere respectively. In practice, a production retriever system can
consist of many more documents than the 737 documents used in this experiment. Therefore, it is crucial to have
different model choices to reduce the cost. In [Denser Retriever](https://github.com/denser-org/denser-retriever), users
can optionally opt out vector search or reranker models if they are costly. In addition, users can utilize different
models including open source free models from HuggingFACE to build their retriever to meet the use cases.

### Using bge-reranker-base Rerank Model

If we replace Cohere `rerank-english-v3.0` with `BAAI/bge-reranker-base` model as following

```python
reranker = HFReranker(model_name="BAAI/bge-reranker-base", top_k=100),
```

we get the following results

```
== NDCG@20
metric_keyword.json: "NDCG@20": 0.7041,
metric_vector.json: "NDCG@20": 0.75732,
metric_reranker.json: "NDCG@20": 0.74044,
metric_es+vs.json: "NDCG@20": 0.77171,
metric_es+rr.json: "NDCG@20": 0.76259,
metric_vs+rr.json: "NDCG@20": 0.76896,
metric_es+vs+rr.json: "NDCG@20": 0.7829,
metric_es+vs_n.json: "NDCG@20": 0.77052,
metric_es+rr_n.json: "NDCG@20": 0.76974,
metric_vs+rr_n.json: "NDCG@20": 0.76036,
metric_es+vs+rr_n.json: "NDCG@20": 0.77677,

== Recall@20
metric_keyword.json: "Recall@20": 0.89267,
metric_vector.json: "Recall@20": 0.94489,
metric_reranker.json: "Recall@20": 0.91969,
metric_es+vs.json: "Recall@20": 0.95027,
metric_es+rr.json: "Recall@20": 0.92113,
metric_vs+rr.json: "Recall@20": 0.93212,
metric_es+vs+rr.json: "Recall@20": 0.94724,
metric_es+vs_n.json: "Recall@20": 0.93817,
metric_es+rr_n.json: "Recall@20": 0.91599,
metric_vs+rr_n.json: "Recall@20": 0.93212,
metric_es+vs+rr_n.json: "Recall@20": 0.93817,
```

In terms of Recall@20 metric, the open source model `BAAI/bge-reranker-base` is worse than the paid
model `rerank-english-v3.0`. The Recall@20 of the open source model is 91.97, while the paid model is 96.10. However,
the
Denser Retriever method `es+vs_rr_n` boosts the Recall@20 to 93.81, significantly reducing the gap. The NDCG@20
metric shows a similar trend. The NDCG@20 of the open source model is 74.04, the paid model is 83.93, and
the `es+vs+rr_n` is
77.67.

### Using jina-reranker-v2-base-multilingual Rerank Model

If we use `jinaai/jina-reranker-v2-base-multilingual` model as following

```python
reranker = HFReranker(model_name="jinaai/jina-reranker-v2-base-multilingual", top_k=100,
                      automodel_args={"torch_dtype": "float32"}, trust_remote_code=True),
```

we get the following results

```
== NDCG@20
metric_keyword.json: "NDCG@20": 0.7041,
metric_vector.json: "NDCG@20": 0.75732,
metric_reranker.json: "NDCG@20": 0.79981,
metric_es+vs.json: "NDCG@20": 0.77244,
metric_es+rr.json: "NDCG@20": 0.80677,
metric_vs+rr.json: "NDCG@20": 0.80539,
metric_es+vs+rr.json: "NDCG@20": 0.81169,
metric_es+vs_n.json: "NDCG@20": 0.77717,
metric_es+rr_n.json: "NDCG@20": 0.79943,
metric_vs+rr_n.json: "NDCG@20": 0.80551,
metric_es+vs+rr_n.json: "NDCG@20": 0.80659,

== Recall@20
metric_keyword.json: "Recall@20": 0.89267,
metric_vector.json: "Recall@20": 0.94489,
metric_reranker.json: "Recall@20": 0.96304,
metric_es+vs.json: "Recall@20": 0.94825,
metric_es+rr.json: "Recall@20": 0.96035,
metric_vs+rr.json: "Recall@20": 0.96169,
metric_es+vs+rr.json: "Recall@20": 0.95128,
metric_es+vs_n.json: "Recall@20": 0.93548,
metric_es+rr_n.json: "Recall@20": 0.95766,
metric_vs+rr_n.json: "Recall@20": 0.95195,
metric_es+vs+rr_n.json: "Recall@20": 0.94926,
```

The model `jina-reranker-v2-base-multilingual` outperforms `rerank-english-v3.0` (96.30 vs 96.10) in terms of Recall@20
metric. The NDCG@20 of `jina-reranker-v2-base-multilingual` model is 79.98 while `rerank-english-v3.0` model is 83.93.
The `es+vs+rr` is 81.16 which helps close the accuracy gap.