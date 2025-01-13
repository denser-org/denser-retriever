# Denser Retriever CLI

A command-line interface for retrieving and training.

## Retrieving

There are three ways to use the retriever:

1. Using a configuration file:

```bash
python -m denser_retriever.experiments.retrieve scifact "What causes cancer?" --config denser_retriever/configs/tri-force-linear.json
```

The config files `tri-force-linear.json`, `tri-force-rank.json`, `tri-force-model` are provided in the `configs` directory, which uses a `linear`, `rank` and `model` fusion mode respectively to retrieve passages. 

To use a trained xgboost model to fuse passages we use `tri-force-model` config file

```bash
python -m denser_retriever.experiments.retrieve scifact "What causes cancer?" --config denser_retriever/configs/tri-force-model.json
```

`tri-force-model.json` includes the path to a xgboost model: 
```json
"xgb_model": "/home/ubuntu/denser-retriever/exps/exp_scifact/models/xgb_es+vs+rr_n.json"
```

See Training section below to see how to train a xgboost model.

2. Using command-line arguments:

```bash
python -m denser_retriever.experiments.retrieve scifact "What causes cancer?" \
    --fusion-mode linear \
    --vector-top-k 100 \
    --keyword-top-k 100 \
    --reranker-top-k 50
    ...
```

3. Using a config file with command line overrides:

```bash
python -m denser_retriever.experiments.retrieve scifact "What causes cancer?" \
    --config denser_retriever/configs/tri-force-linear.json \
    --vector-top-k 150  # Override just this parameter
   ```

## Training (Optional)

Train a denser retriever is optional. The purpose of training is to train a xgboost model which intelligently fuses keyword search, vector search and reranker model to improve retrieve accuracy. Without training, we can still use `linear` or `rank` fusion mode to fuse keyword search, vector search and reranker. 

The following command shows a training example, where `mteb/scifact` is the dataset name, `train` and `test` are the splits to use, and `default.json` is the training configuration file.

```bash
python -m denser_retriever.experiments.train mteb/scifact train test denser_retriever/configs/default.json
```

The training process:

1. Ingests documents from the dataset
2. Generates training features for each split
3. Computes baseline performance metrics
4. Trains the model using cross-validation or train/test splits
5. Reports final performance metrics

After training, the following models are saved to the `exps/exp_scifact/models` directory: xgb_es+rr.json  xgb_es+rr_n.json  xgb_es+vs+rr.json  xgb_es+vs+rr_n.json  xgb_es+vs.json  xgb_es+vs_n.json  xgb_vs+rr.json  xgb_vs+rr_n.json. We can use one of the trained models to retrieve passages.


