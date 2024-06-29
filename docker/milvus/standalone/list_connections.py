# https://milvus.io/docs/authenticate.md
# reset milvus password

from pymilvus import Collection
from pymilvus import connections, utility
import os
from dotenv import load_dotenv

load_dotenv()

connections.connect(
    "default",
    host=os.getenv('MILVUS_HOST'),
    port="19530",
    user="root",
    password=os.getenv('MILVUS_PASSWD'),
)
print(f"All Milvus collections: {utility.list_collections()}")

index = "scidocs"
collection = Collection(index)  # Get an existing collection.
print(f"{index} collection.num_entities: {collection.num_entities}")
