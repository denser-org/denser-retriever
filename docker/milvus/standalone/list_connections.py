# https://milvus.io/docs/authenticate.md
# reset milvus password

from pymilvus import connections
from pymilvus import Collection
from pymilvus import utility
connections.connect("default", host="35.93.131.127", port="19530", user='root',
    password='!RedComputer7961',)
print(f"All Milvus collections: {utility.list_collections()}")

index = "test_index_temp"
collection = Collection(index)  # Get an existing collection.
print(f"{index} collection.num_entities: {collection.num_entities}")