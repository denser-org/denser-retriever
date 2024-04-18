# https://milvus.io/docs/authenticate.md
# reset milvus password

from pymilvus import connections
from pymilvus import Collection
from pymilvus import utility
connections.connect("default", host="54.68.68.29", port="19530", user='root',
    password='Milvus',)
print(f"All Milvus collections: {utility.list_collections()}")

index = "test_index_temp"
collection = Collection(index)  # Get an existing collection.
print(f"{index} collection.num_entities: {collection.num_entities}")