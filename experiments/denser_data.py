from denser_retriever.utils import (
    load_qrels,
    load_queries,
)
import os


class DenserData:
    def __init__(self, dir_path):
        self.data_dir = dir_path

    def load_queries(self):
        queries = load_queries(os.path.join(self.data_dir, 'queries.jsonl'))
        return queries

    def load_qrels(self):
        qrels = load_qrels(os.path.join(self.data_dir, 'qrels.jsonl'))
        return qrels
