from abc import ABC, abstractmethod


class Retriever(ABC):
    """
    Base class for all Retriever
    """

    def __init__(self):
        self.retrieve_type = None

    @abstractmethod
    def ingest(self, data):
        pass

    @abstractmethod
    def retrieve(self, query, topk):
        return None
