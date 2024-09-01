import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import uuid4

import numpy as np
from langchain_core.documents import Document
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    MilvusException,
    connections,
    utility,
)

from denser_retriever.embeddings import DenserEmbeddings
from denser_retriever.filter import FieldMapper
from denser_retriever.vectordb.base import DenserVectorDB

logger = logging.getLogger(__name__)

DEFAULT_MILVUS_CONNECTION = {
    "uri": "http://localhost:19530",
}


class MilvusDenserVectorDB(DenserVectorDB):
    def __init__(
        self,
        drop_old: Optional[bool] = False,
        auto_id: bool = False,
        connection_args: Optional[dict] = None,
        **args: Any,
    ):
        super().__init__(**args)
        self.drop_old = drop_old
        self.auto_id = auto_id
        self.connection_args = connection_args

    def _create_connection_alias(self, connection_args: dict) -> str:
        """Create the connection to the Milvus server."""

        # Grab the connection arguments that are used for checking existing connection
        host: str = connection_args.get("host", None)
        port: Union[str, int] = connection_args.get("port", None)
        address: str = connection_args.get("address", None)
        uri: str = connection_args.get("uri", None)
        user = connection_args.get("user", None)

        # Order of use is host/port, uri, address
        if host is not None and port is not None:
            given_address = str(host) + ":" + str(port)
        elif uri is not None:
            if uri.startswith("https://"):
                given_address = uri.split("https://")[1]
            elif uri.startswith("http://"):
                given_address = uri.split("http://")[1]
            else:
                given_address = uri  # Milvus lite
        elif address is not None:
            given_address = address
        else:
            given_address = None
            logger.debug("Missing standard address type for reuse attempt")

        # User defaults to empty string when getting connection info
        if user is not None:
            tmp_user = user
        else:
            tmp_user = ""

        # If a valid address was given, then check if a connection exists
        if given_address is not None:
            for con in connections.list_connections():
                addr = connections.get_connection_addr(con[0])
                if (
                    con[1]
                    and ("address" in addr)
                    and (addr["address"] == given_address)
                    and ("user" in addr)
                    and (addr["user"] == tmp_user)
                ):
                    logger.debug("Using previous connection: %s", con[0])
                    return con[0]

        # Generate a new connection if one doesn't exist
        alias = uuid4().hex
        try:
            connections.connect(alias=alias, **connection_args)
            logger.debug("Created new connection using: %s", alias)
            return alias
        except MilvusException as e:
            logger.error("Failed to create new connection using: %s", alias)
            raise e

    def create_index(
        self,
        index_name: str,
        embedding_function: DenserEmbeddings,
        search_fields: List[str],
        embedding_size: int,
        **kwargs,
    ):
        """Create the index for the vector db."""
        self.index_name = index_name
        self.search_fields = FieldMapper(search_fields)
        self.embeddings = embedding_function
        self.embedding_size = embedding_size
        self.source_max_length = 500
        self.title_max_length = 500
        self.text_max_length = 8000
        self.field_max_length = 500

        self.connection_args = self.connection_args or DEFAULT_MILVUS_CONNECTION
        self.alias = self._create_connection_alias(self.connection_args)
        self.col: Optional[Collection] = None

        # Grab the existing collection if it exists
        if utility.has_collection(self.index_name, using=self.alias):
            self.col = Collection(
                self.index_name,
                using=self.alias,
            )
            if self.drop_old:
                self.col.drop()
                self.col = None

        # Either creates or references a collection. It does not remove records.
        self._create_collection()

    def _create_collection(self):
        fields = [
            FieldSchema(
                name="uid",
                dtype=DataType.VARCHAR,
                is_primary=True,
                auto_id=False,
                max_length=100,
            ),
            FieldSchema(
                name="source", dtype=DataType.VARCHAR, max_length=self.source_max_length
            ),
            FieldSchema(
                name="title", dtype=DataType.VARCHAR, max_length=self.title_max_length
            ),
            FieldSchema(
                name="text", dtype=DataType.VARCHAR, max_length=self.text_max_length
            ),
            FieldSchema(name="pid", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(
                name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_size
            ),
        ]
        for key in self.search_fields.get_keys():
            # both category and date type (unix timestamp) use INT64 type
            fields.append(
                FieldSchema(
                    name=key,
                    dtype=DataType.INT64,
                    max_length=self.field_max_length,
                ),
            )
        schema = CollectionSchema(fields=fields, description="Denser Vector DB")
        try:
            self.col = Collection(
                self.index_name,
                schema,
                consistency_level="Strong",
                using=self.alias,
            )
        except MilvusException as e:
            logger.error(
                "Failed to create collection: %s error: %s", self.index_name, e
            )
            raise e

    def add_documents(
        self,
        documents: List[Document],
        batch_size: int = 1000,
        **kwargs: Any,
    ) -> List[str]:
        """Add documents to the vector db.

        Args:
            documents (List[Document]): Documents to add to the vector db.

        Returns:
            List[str]: IDs of the added texts.
        """
        batch = []
        ids = [str(uuid4()) for _ in range(len(documents))]
        uid_list, sources, titles, texts, pid_list = [], [], [], [], []
        fields_list = [[] for _ in range(len(self.search_fields.get_keys()))]
        failed_batches = []  # To store information about failed batches
        for doc, id in zip(documents, ids):
            batch.append(
                (doc.metadata.get("title", "")[: self.title_max_length - 10]
                + " "
                + doc.page_content[:2000]).strip()
            )
            uid_list.append(id)
            sources.append(
                doc.metadata.get("source", "")[: self.source_max_length - 10]
            )
            titles.append(doc.metadata.get("title", "")[: self.title_max_length - 10])
            texts.append(doc.page_content[: self.text_max_length - 1000]
            )  # buffer
            pid_list.append(doc.metadata.get("pid", "-1"))

            for i, field_original_key in enumerate(self.search_fields.get_original_keys()):
                data = doc.metadata.get(field_original_key, -1)
                converted_data = self.search_fields.convert_for_storage(
                    {field_original_key: data}
                )
                fields_list[i].append(converted_data)

            if len(batch) == batch_size:
                embeddings = self.embeddings.embed_documents(batch)
                record = [
                    uid_list,
                    sources,
                    titles,
                    texts,
                    pid_list,
                    np.array(embeddings),
                ]
                record += fields_list

                try:
                    self.col.insert(record)
                except Exception as e:
                    logger.error(
                        f'Milvus index insert error at record {doc.metadata["pid"]} - {e}'
                    )

                self.col.flush()
                logger.info(f"Milvus vector DB ingesting {id}")

                batch = []
                uid_list, sources, titles, texts, pid_list = [], [], [], [], []
                fields_list = []

        if len(batch) > 0:
            embeddings = self.embeddings.embed_documents(batch)
            record = [
                uid_list,
                sources,
                titles,
                texts,
                pid_list,
                np.array(embeddings),
            ]
            record += fields_list
            try:
                self.col.insert(record)
            except Exception as e:
                logger.error(f"Milvus index insert error at record {id} - {e}")
                failed_batches.append(
                    {
                        "sources": sources,
                        "pids": pid_list,
                        "batch": batch,
                    }
                )
            self.col.flush()
            logger.info(f"Milvus vector DB ingesting {id}")

        index = {
            "index_type": "FLAT",
            "metric_type": "L2",
        }

        self.col.create_index("embeddings", index)
        self.col.load()
        return ids

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 100,
        filter: Dict[str, Any] = {},
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Search for similar documents to the query.

        Args:
            query (str): Query text.
            k (int): Number of documents to return.
            param (Optional[dict]): Additional parameters for the search.
            expr (Optional[str]): Expression to filter the search.
            timeout (Optional[float]): Timeout for the search.

        Returns:
            List[Tuple[Document, float]]: List of tuples of documents and their similarity scores.
        """
        start_time = time.time()
        embeddings = self.embeddings.embed_query(query)
        query_embeddings = np.array(embeddings)
        embedding_time_sec = time.time() - start_time
        logger.info(f"Query embedding time: {embedding_time_sec:.3f} sec.")

        exprs = []
        for field in filter:
            original_key = filter.get(field)
            key = self.search_fields.get_key(original_key)
            type = self.search_fields.get_field_type(key)

            assert (
                original_key is not None
            ), f"Field {field} not found in the search fields."
            if type == "date":
                if len(original_key) == 2:
                    start_unix_time = int(
                        datetime.combine(
                            original_key[0], datetime.min.time()
                        ).timestamp()
                    )
                    end_unix_time = int(
                        datetime.combine(
                            original_key[1], datetime.min.time()
                        ).timestamp()
                    )
                    exprs.append(f"{key} >= {start_unix_time}")
                    exprs.append(f"{key} <= {end_unix_time}")
                else:
                    unix_time = int(
                        datetime.combine(
                            original_key[0], datetime.min.time()
                        ).timestamp()
                    )
                    exprs.append(f"{key} == {unix_time}")
            else:
                category_id = self.search_fields.get_key(original_key)
                if category_id is not None:
                    exprs.append(f"{key}=={category_id}")
        expr_str = " and ".join(exprs)
        search_params = {
            "metric_type": "L2",
            "params": {"nprobe": 10},
        }
        output_fields = ["source", "title", "text", "pid", "uid"] + self.search_fields.get_keys()

        start_time = time.time()
        result = self.col.search(
            data=query_embeddings,
            anns_field="embeddings",
            param=search_params,
            limit=k,
            expr=expr_str,
            output_fields=output_fields,
        )
        retrieve_time_sec = time.time() - start_time
        logger.info(f"Vector DB retrieve time: {retrieve_time_sec:.3f} sec.")
        logger.info(f"Retrieved {len(result[0])} documents.")

        top_k_used = min(len(result[0]), self.top_k)  # type: ignore

        ret = []
        for id in range(top_k_used):
            assert len(result) == 1  # type: ignore
            hit = result[0][id]  # type: ignore
            doc = Document(page_content=hit.entity.text, metadata={})
            doc.metadata = {
                "id": hit.entity.uid,
                "source": hit.entity.source,
                "text": hit.entity.text,
                "title": hit.entity.title,
                "pid": hit.entity.pid,
            }
            score = (-hit.entity.distance)

            for field in self.search_fields.get_keys():
                original_value = self.search_fields.convert_to_original({
                    field: hit.entity.get(field)
                })
                doc.metadata[field] = original_value
            #     cat_id_or_unix_time = hit.entity.get(key)
            #     type = self.search_fields.get_field_type(field)
            #     if type == "date":
            #         date = datetime.utcfromtimestamp(cat_id_or_unix_time).strftime(
            #             "%Y-%m-%d"
            #         )
            #         doc.metadata[field] = date
            #     else:
            #         doc.metadata[field] = cat_id_or_unix_time
            pair = (doc, score)
            ret.append(pair)
        return ret

    def filter_expression(
        self,
        filter_dict: Dict[str, Any],
    ) -> Any:
        """Generate a Milvus expression from a filter dictionary."""
        expressions = []
        for key, value in filter_dict.items():
            if value is None:
                continue
            if isinstance(value, tuple) and len(value) == 2:
                start, end = value
                expressions.append(f"{key} >= '{start}' and {key} <= '{end}'")
            else:
                expressions.append(f"{key} == '{value}'")
        return " and ".join(expressions)

    def delete(self, ids: Optional[List[str]] = None, **kwargs: str):
        """Delete documents from the vector db.

        Args:
            ids (Optional[List[str]]): IDs of the documents to delete.
            expr (Optional[str]): Expression to filter the deletion.
        """
        # self.col.delete(ids=ids, **kwargs)

    def delete_all(self):
        """Delete all documents from the vector db."""
        self.store.delete(expr="pk > -1")
