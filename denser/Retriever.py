import random
from abc import ABC, abstractmethod

import numpy as np
import torch


class Retriever(ABC):
    """
    Base class for all Retriever
    Extend this class and implement __call__ for custom retrievers.
    """

    def __init__(self, retriever_config, seed=42, **kwargs):
        # self.retriever_config = retriever_config
        # self.embedding_model = None
        # self.keyword_model = None
        # self.reranker = None
        self.retrieve_type = None
        # self.seed = seed
        # random.seed(self.seed)
        # np.random.seed(self.seed)
        # torch.manual_seed(self.seed)
        # torch.cuda.manual_seed_all(self.seed)

    @abstractmethod
    def ingest(self, data):
        pass

    @abstractmethod
    def retrieve(self, query, topk):
        return None