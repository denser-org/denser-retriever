import json
import uuid

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

from denser_retriever.retriever import Retriever
from denser_retriever.utils import get_logger

logger = get_logger(__name__)


class RetrieverElasticSearch(Retriever):
    """
    Elasticsearch Retriever
    """

    def __init__(self, index_name, config_file):
        super().__init__(index_name, config_file)
        self.retrieve_type = "elasticsearch"
        self.es = Elasticsearch(
            hosts=[self.config["keyword"]["es_host"]],
            basic_auth=(self.config["keyword"]["es_user"], self.config["keyword"]["es_passwd"]),
            request_timeout=600,
        )

    def create_index(self, index_name):
        # Define the index settings and mappings
        settings = {
            "analysis": {"analyzer": {"default": {"type": "standard"}}},
            "similarity": {
                "custom_bm25": {
                    "type": "BM25",
                    "k1": 1.2,
                    "b": 0.75,
                }
            },
        }
        mappings = {
            "properties": {
                "content": {
                    "type": "text",
                    "similarity": "custom_bm25",  # Use the custom BM25 similarity
                },
                "title": {
                    "type": "text",
                },
                "source": {
                    "type": "text",
                },
                "pid": {
                    "type": "text",
                },
            }
        }

        for key in self.field_types:
            mappings["properties"][key] = self.field_types[key]

        # Create the index with the specified settings and mappings
        if self.es.indices.exists(index=index_name):
            self.es.indices.delete(index=index_name)
        self.es.indices.create(index=index_name, mappings=mappings, settings=settings)

    def ingest(self, doc_or_passage_file, batch_size, refresh_indices=True):
        self.create_index(self.index_name)
        requests = []
        ids = []
        batch_count = 0
        record_id = 0
        with open(doc_or_passage_file, "r") as jsonl_file:
            for line in jsonl_file:
                data = json.loads(line)
                _id = str(uuid.uuid4())
                request = {
                    "_op_type": "index",
                    "_index": self.index_name,
                    "content": data.pop("text"),
                    "title": data.get("title"),  # Index the title
                    "_id": _id,
                    "source": data.pop("source"),
                    "pid": data.pop("pid"),
                }
                for filter in self.field_types.keys():
                    v = data.get(filter).strip()
                    if v:
                        request[filter] = v
                ids.append(_id)
                requests.append(request)

                batch_count += 1
                record_id += 1
                if batch_count >= batch_size:
                    # Index the batch
                    bulk(self.es, requests)
                    logger.info(f"ES ingesting {doc_or_passage_file} record {record_id}")
                    batch_count = 0
                    requests = []

        # Index any remaining documents
        if requests:
            bulk(self.es, requests)
            logger.info(f"ES ingesting {doc_or_passage_file} record {record_id}")

        if refresh_indices:
            self.es.indices.refresh(index=self.index_name)

        return ids

    def retrieve(self, query_text, meta_data, query_id=None):
        assert self.es.indices.exists(index=self.index_name)

        query_dict = {
            "query": {
                "bool": {
                    "should": [
                        {
                            "match": {
                                "title": {
                                    "query": query_text,
                                    "boost": 2.0,  # Boost the "title" field with a higher weight
                                }
                            }
                        },
                        {"match": {"content": query_text}},
                    ],
                    "must": [],
                }
            },
            "_source": True,
        }

        for field in meta_data:
            category_or_date = meta_data.get(field)
            if category_or_date:
                if isinstance(category_or_date, tuple):
                    query_dict["query"]["bool"]["must"].append(
                        {
                            "range": {
                                field: {
                                    "gte": category_or_date[0],
                                    "lte": category_or_date[1] if len(category_or_date) > 1 else category_or_date[0],
                                }
                            }
                        }
                    )
                else:
                    query_dict["query"]["bool"]["must"].append({"term": {field: category_or_date}})

        res = self.es.search(index=self.index_name, body=query_dict, size=self.config["keyword"]["topk"])
        topk_used = min(len(res["hits"]["hits"]), self.config["keyword"]["topk"])
        passages = []
        for id in range(topk_used):
            _source = res["hits"]["hits"][id]["_source"]
            passage = {
                "source": _source["source"],
                "text": _source["content"],
                "title": _source["title"],
                "pid": _source["pid"],
                "score": res["hits"]["hits"][id]["_score"],
            }
            for field in meta_data:
                if _source.get(field):
                    passage[field] = _source.get(field)
            passages.append(passage)
        return passages

    def get_index_mappings(self):
        mapping = self.es.indices.get_mapping(index=self.index_name)

        # The mapping response structure can be quite nested, focusing on the 'properties' section
        properties = mapping[self.index_name]["mappings"]["properties"]

        # Function to recursively extract fields and types
        def extract_fields(fields_dict, parent_name=""):
            fields = {}
            for field_name, details in fields_dict.items():
                full_field_name = f"{parent_name}.{field_name}" if parent_name else field_name
                if "properties" in details:
                    fields.update(extract_fields(details["properties"], full_field_name))
                else:
                    fields[full_field_name] = details.get("type", "notype")  # Default 'notype' if no type is found
            return fields

        # Extract fields and types
        all_fields = extract_fields(properties)
        return all_fields

    def get_categories(self, field, topk):
        query = {
            "size": 0,  # No actual documents are needed, just the aggregation results
            "aggs": {
                "all_categories": {
                    "terms": {
                        "field": field,
                        "size": 1000,  # Adjust this value based on the expected number of unique categories
                    }
                }
            },
        }
        response = self.es.search(index=self.index_name, body=query)
        # Extract the aggregation results
        categories = response["aggregations"]["all_categories"]["buckets"]
        if topk > 0:
            categories = categories[:topk]
        res = [category["key"] for category in categories]
        return res
