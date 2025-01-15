# Denser Retriever CLI

A command-line interface for training evaluation, and retrieving.

## Training

Training refers to train a logistic regression model to fuse keyword search, vector search and rerank. The trained
logistic regression model is used in the `fusion` combine method. Without training, we can still use `vector`, `hybrid`,
and `reranker` methods to retrieve passages.

The following command shows a training example, where `mteb/scifact` is the dataset name, `train` and `test` are the
splits to use, and `train_default.json` is the training configuration file. `train_default.json` specifies the
elasticsearch and vector search setup, as well as the embedding and reranker models to use.

```bash
python -m denser_retriever.experiments.train mteb/scifact train test denser_retriever/configs/train_default.json
```

The training process:

1. Ingests documents from the dataset
2. Generates features for each split
3. Computes baselines (keyword search, vector search and reranker) performance metrics
4. Trains the model using train split
5. Reports final performance metrics on test split

After training, we get the following accuracy report. `keyword` and `vector` are the keyword search and vector search
respectively. `reranker` is the reranking of the joint set of keyword search and vector search
results. `es+vs`, `es+rr`, `vs+rr`, and `es+vs+rr` are the features used to train logistic models. For
example, `es+vs+rr` is a logistic regression model which was trained using keyword (es), vector search (vs) and
reranker (rr) features, while `es+vs` is a logistic regression model which was trained using keyword and vector search
features. Four models `weights_es+vs.json`, `weights_es+rr.json`, `weights_vs+rr.json` and `weights_es+vs+rr.json` are
saved to the `exps/exp_scifact/models` directory. We use the `weights_es+vs+rr.json` model in the `fusion` combine
method (see Evaluation and Retrieving Sections below) to retrieve passages, as it outperforms other methods.

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

## Evaluation

Denser retriever consists of three components: keyword search, vector search, and reranker. We provide the following
methods to combine these components:

1. `vector`: Use the vector db search results directly.
2. `hybrid`: Use the rank positions of keyword search and vector db search results to combine the final results.
3. `reranker`: Use keyword search followed by a reranker to get the final results.
4. `fusion`: Use a logistic regression model to fuse keyword search, vector search and reranker.

Run the following command to evaluate the retrieval performance of different methods: `vector`, `hybrid`, `reranker`,
and `fusion`. The first argument is the index name (generated from the training above), the second argument is the dataset name, the third argument is the
path to the `fusion` configuration file, the fourth argument is the output directory, the fifth argument is the number
of top-k passages to retrieve, and the sixth argument is the number of queries to evaluate.

```bash
python -m denser_retriever.experiments.evaluate \
    scifact \
    mteb/scifact \
    --config denser_retriever/configs/fusion.json \
    --output-dir exps/exp_scifact/pred \
    --top-k 100 \
    --num-queries 0
```

After running the evaluation, we get the following results. As expected, the `fusion` method which uses the `weights_es+vs+rr.json` model from the training, obtained identical ndcg@10 number as the `es+vs+rr` from training. In addition, fusion method outperforms the other methods.

```text
vector: 0.7317
hybrid: 0.6832
reranker: 0.6759
fusion: 0.7434
```

## Retrieving



To retrieve passages for a given question with the `fusion` method, use the following command. The first argument is the
retriever index name, the second argument is the question, and the third argument is the path to the configuration file.
The retriever index is generated in the Training section above.
Replace `fusion.json` with `vector.json`, `hybrid.json`, `reranker.json` to use the corresponding method.

```bash
python -m denser_retriever.experiments.retrieve scifact "0-dimensional biomaterials show inductive properties." --config denser_retriever/configs/fusion.json
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


