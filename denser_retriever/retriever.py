from abc import ABC, abstractmethod
import yaml

class Retriever(ABC):
    """
    Base class for all Retriever
    """

    def __init__(self, index_name, config_file):
        config = yaml.safe_load(open(config_file))
        self.config = config
        self.retrieve_type = None
        self.field_types = {}
        self.field_internal_names = {}
        self.field_cat_to_id = {}
        self.field_id_to_cat = {}
        fields = config.get("fields")
        if fields:
            for f in config.get("fields"):
                comps = f.split(":")
                assert len(comps) == 2 or len(comps) == 3
                self.field_types[comps[0]] = {"type": comps[-1]}
                if len(comps) == 3:
                    self.field_internal_names[comps[0]] = comps[1]
                self.field_cat_to_id[comps[0]] = {}
                self.field_id_to_cat[comps[0]] = []

    @abstractmethod
    def ingest(self, data):
        pass

    @abstractmethod
    def retrieve(self, query, topk):
        return None
