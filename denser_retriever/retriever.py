from abc import ABC, abstractmethod


class Retriever(ABC):
    """
    Base class for all Retriever
    """

    def __init__(self, index_name, config):
        self.retrieve_type = None
        self.filter_types = {}
        fields = config.get("filter_fields")
        if fields:
            for f in config.get("filter_fields"):
                comps = f.split(":")
                assert len(comps) == 2
                self.filter_types[comps[0]] = {"type": comps[1]}

    @abstractmethod
    def ingest(self, data):
        pass

    @abstractmethod
    def retrieve(self, query, topk):
        return None
