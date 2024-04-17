from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import json
import uuid

from .Retriever import Retriever
from .utils import get_logger
logger = get_logger(__name__)


class RetrieverElasticSearch(Retriever):
    """
    Elasticsearch Retriever
    """

    def __init__(self, index_name, config):
        self.retrieve_type = "elasticsearch"
        self.index_name = index_name
        self.es = Elasticsearch(
            hosts=[config['keyword']['es_host']],
            basic_auth=(config['keyword']['es_user'], config['keyword']['es_passwd']),
            request_timeout=600
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
            }
        }
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
        with open(doc_or_passage_file, 'r', encoding='utf-8') as jsonl_file:
            for line in jsonl_file:
                data = json.loads(line)
                _id = str(uuid.uuid4())
                request = {
                    "_op_type": "index",
                    "_index": self.index_name,
                    "content": data.pop("text"),
                    "title": data.get("title"),  # Index the title
                    "_id": _id,
                    "_meta": data
                }
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

    def retrieve(self, query_text, topk):
        assert self.es.indices.exists(index=self.index_name)

        query_dict = {
            "query": {
                "bool": {
                    "should": [
                        {
                            "match": {
                                "title": {
                                    "query": query_text,
                                    "boost": 2.0  # Boost the "title" field with a higher weight
                                }
                            }
                        },
                        {
                            "match": {
                                "content": query_text
                            }
                        }
                    ]
                }
            },
            "_source": True,
        }
        res = self.es.search(index=self.index_name, body=query_dict, size=topk)
        topk_used = min(len(res['hits']['hits']), topk)
        passages = []
        for id in range(topk_used):
            _source = res['hits']['hits'][id]['_source']
            passage = {
                'source': _source['_meta']['source'],
                'text': _source['content'],
                'title': _source['_meta']['title'],
                'pid': _source['_meta']['pid'],
                'score': res['hits']['hits'][id]['_score']
            }
            passages.append(passage)
        return passages