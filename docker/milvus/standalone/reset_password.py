# https://milvus.io/docs/authenticate.md
# reset milvus password

from pymilvus import connections, utility

connections.connect(
    alias="default",
    host="localhost",
    port="19530",
    user="root",
    password="Milvus",
)
utility.reset_password("root", "Milvus", "YOUR_PASSWORD", using="default")
