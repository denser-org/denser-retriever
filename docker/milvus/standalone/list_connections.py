# https://milvus.io/docs/authenticate.md
# reset milvus password

from pymilvus import Collection
from pymilvus import connections, utility

connections.connect(
    "default",
    host="localhost",
    port="19530",
    user="root",
    password="Milvus",
)
print(f"All Milvus collections: {utility.list_collections()}")

index = "nq"
collection = Collection(index)  # Get an existing collection.
print(f"{index} collection.num_entities: {collection.num_entities}")
