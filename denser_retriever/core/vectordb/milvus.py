from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import logging
import time
from datetime import datetime
from uuid import uuid4

import numpy as np
from langchain_core.documents import Document
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    MilvusException,
    SearchResult,
    connections,
    utility,
)

from denser_retriever.core.embeddings import DenserEmbeddings
from denser_retriever.core.filter import FieldMapper
from denser_retriever.core.vectordb.base import DenserVectorDB
from denser_retriever.core.utils import sigmoid

logger = logging.getLogger(__name__)

DEFAULT_MILVUS_CONNECTION = {
    "uri": "http://localhost:19530",
}


def _create_connection_alias(connection_args: dict) -> str:
    """Create the connection to the Milvus server."""
    host: str = connection_args.get("host", None)
    port: Union[str, int] = connection_args.get("port", None)
    address: str = connection_args.get("address", None)
    uri: str = connection_args.get("uri", None)
    user = connection_args.get("user", None)

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

    tmp_user = user if user is not None else ""

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

    alias = uuid4().hex
    try:
        connections.connect(alias=alias, **connection_args)
        logger.debug("Created new connection using: %s", alias)
        return alias
    except MilvusException as e:
        logger.error("Failed to create new connection using: %s", alias)
        raise e


@dataclass
class MilvusIndexData:
    """Data class containing Milvus index configuration."""
    index_name: str
    embedding_size: int = 768
    search_fields: FieldMapper = FieldMapper()
    drop_old: bool = False
    source_max_length: int = 500
    title_max_length: int = 500
    text_max_length: int = 30000
    field_max_length: int = 500
    collection: Optional[Collection] = None
    alias: Optional[str] = None

    def init_collection(self, connection_args: Dict[str, Any]):
        """Initialize collection with connection and schema."""
        self.alias = _create_connection_alias(connection_args)

        if utility.has_collection(self.index_name, using=self.alias):
            self.collection = Collection(
                self.index_name,
                using=self.alias,
            )
            if self.drop_old:
                self.collection.drop()
                self.collection = None

        if not self.collection:
            self.collection = self._create_collection()

        # Create index before loading
        index = {
            "index_type": "FLAT",
            "metric_type": "L2",
        }
        self.collection.create_index("embeddings", index)
        self.collection.load()  # Load after creating index

    def _create_collection(self):
        """Create a new collection with the configured schema."""
        fields = [
            FieldSchema(
                name="pid",
                dtype=DataType.VARCHAR,
                is_primary=True,
                auto_id=False,
                max_length=100,
            ),
            FieldSchema(
                name="source",
                dtype=DataType.VARCHAR,
                max_length=self.source_max_length
            ),
            FieldSchema(
                name="title",
                dtype=DataType.VARCHAR,
                max_length=self.title_max_length
            ),
            FieldSchema(
                name="text",
                dtype=DataType.VARCHAR,
                max_length=self.text_max_length
            ),
            FieldSchema(
                name="embeddings",
                dtype=DataType.FLOAT_VECTOR,
                dim=self.embedding_size
            ),
        ]

        for key in self.search_fields.get_keys():
            fields.append(
                FieldSchema(
                    name=key,
                    dtype=DataType.INT64,
                    max_length=self.field_max_length,
                ),
            )

        schema = CollectionSchema(fields=fields, description="Denser Vector DB")
        try:
            return Collection(
                self.index_name,
                schema,
                consistency_level="Strong",
                using=self.alias,
            )
        except MilvusException as e:
            logger.error(
                "Failed to create collection: %s error: %s",
                self.index_name,
                e
            )
            raise e


class MilvusDenserVectorDB(DenserVectorDB):
    """Milvus vector database implementation."""

    def __init__(
            self,
            connection_args: Optional[dict] = None,
            **args: Any,
    ):
        super().__init__(**args)
        self.connection_args = connection_args or DEFAULT_MILVUS_CONNECTION

    def create_index(
            self,
            index_data: MilvusIndexData,
            **kwargs,
    ):
        """Create the index for the vector db."""
        index_data.init_collection(self.connection_args)

    def add_documents(
            self,
            index_data: MilvusIndexData,
            documents: List[Document],
            embedding_model: DenserEmbeddings,
            batch_size: int = 200,
            progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
            **kwargs: Any,
    ) -> List[str]:
        if not index_data.collection:
            raise ValueError("Collection not initialized. Call create_index() first.")

        batch = []
        pid_list, sources, titles, texts = [], [], [], []
        seen_pids = set()
        fields_list = [[] for _ in range(len(index_data.search_fields.get_keys()))]
        failed_batches = []
        total_docs = len(documents)
        docs_processed = 0

        for i, doc in enumerate(documents):
            batch.append(
                (
                        doc.metadata.get("title", "")[: index_data.title_max_length - 10]
                        + " "
                        + doc.page_content[:2000]
                ).strip()
            )
            pid = doc.metadata.get("pid", "-1")
            if pid in seen_pids:
                logger.warning(f"Duplicate pid found in vector ingestion: {pid}")
                pid = str(uuid4())
            seen_pids.add(pid)
            pid_list.append(pid)
            sources.append(doc.metadata.get("source", "")[: index_data.source_max_length - 10])
            titles.append(doc.metadata.get("title", "")[: index_data.title_max_length - 10])
            texts.append(doc.page_content[:10000])

            for j, field_original_key in enumerate(index_data.search_fields.get_original_keys()):
                data = doc.metadata.get(field_original_key, -1)
                converted_data = index_data.search_fields.convert_for_storage({field_original_key: data})
                fields_list[j].append(converted_data)

            docs_processed += 1

            if progress_callback and (len(batch) == batch_size or i == len(documents) - 1):
                progress_callback({
                    "total_docs": total_docs,
                    "es_progress": 100,  # ES is already complete at this point
                    "vector_progress": (docs_processed / total_docs) * 100,
                    "es_docs_processed": total_docs,
                    "vector_docs_processed": docs_processed,
                    "status": "vector_db_ingesting"
                })

                embeddings = embedding_model.embed_documents(batch)
                record = [pid_list, sources, titles, texts, np.array(embeddings)] + fields_list

                try:
                    index_data.collection.insert(record)
                    index_data.collection.flush()
                except Exception as e:
                    logger.error(f'Milvus index insert error at record {doc.metadata["pid"]} - {e}')
                logger.info(f"Processed {len(batch)} documents, {docs_processed}/{total_docs} total.")

                batch = []
                pid_list, sources, titles, texts = [], [], [], []
                fields_list = [[] for _ in range(len(index_data.search_fields.get_keys()))]

        if len(batch) > 0:
            embeddings = embedding_model.embed_documents(batch)
            record = [pid_list, sources, titles, texts, np.array(embeddings)] + fields_list

            try:
                index_data.collection.insert(record)
                index_data.collection.flush()
            except Exception as e:
                logger.error(f"Milvus index insert error - {e}")
                failed_batches.append({
                    "sources": sources,
                    "pids": pid_list,
                    "batch": batch,
                })
        logger.info(f"Processed {len(batch)} documents, {docs_processed}/{total_docs} total.")

        index_data.collection.load()

        if progress_callback:
            progress_callback({
                "total_docs": total_docs,
                "es_progress": 100,
                "vector_progress": 100,
                "es_docs_processed": total_docs,
                "vector_docs_processed": total_docs,
                "status": "vector_db_complete"
            })

        return list(seen_pids)

    def retrieve(
            self,
            index_data: MilvusIndexData,
            query: str,
            embedding_model: DenserEmbeddings,
            k: int = 100,
            filter: Dict[str, Any] = {},
            apply_sigmoid: bool = False
    ) -> List[Tuple[Document, float]]:
        """Search for similar documents to the query."""
        if not index_data.collection:
            raise ValueError("Collection not initialized. Call create_index() first.")

        start_time = time.time()
        embeddings = embedding_model.embed_query(query)
        query_embeddings = np.array(embeddings)
        embedding_time_sec = time.time() - start_time
        logger.info(f"Query embedding time: {embedding_time_sec:.3f} sec.")

        exprs = []
        for field in filter:
            original_key = filter.get(field)
            key = index_data.search_fields.get_key(original_key)
            type = index_data.search_fields.get_field_type(key)

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
                category_id = index_data.search_fields.get_key(original_key)
                if category_id is not None:
                    exprs.append(f"{key}=={category_id}")

        expr_str = " and ".join(exprs)
        search_params = {
            "metric_type": "L2",
            "params": {"nprobe": 10},
        }
        output_fields = [
                            "pid",
                            "source",
                            "title",
                            "text"
                        ] + index_data.search_fields.get_keys()

        start_time = time.time()
        result = index_data.collection.search(
            data=query_embeddings,
            anns_field="embeddings",
            param=search_params,
            limit=k,
            expr=expr_str,
            output_fields=output_fields,
        )
        assert isinstance(result, SearchResult)
        retrieve_time_sec = time.time() - start_time
        logger.info(f"Vector DB retrieve time: {retrieve_time_sec:.3f} sec.")
        logger.info(f"Retrieved {len(result[0])} documents.")

        top_k_used = min(len(result[0]), k)

        ret = []
        for id in range(top_k_used):
            assert len(result) == 1
            hit = result[0][id]
            doc = Document(page_content=hit.entity.text, metadata={})
            doc.metadata = {
                "pid": hit.entity.pid,
                "source": hit.entity.source,
                "text": hit.entity.text,
                "title": hit.entity.title,
            }
            score = -hit.entity.distance

            for field in index_data.search_fields.get_keys():
                original_value = index_data.search_fields.convert_to_original(
                    {field: hit.entity.get(field)}
                )
                doc.metadata[field] = original_value
            pair = (doc, sigmoid(score) if apply_sigmoid else score)
            ret.append(pair)
        return ret

    def delete(
            self,
            index_data: MilvusIndexData,
            ids: Optional[List[str]] = None,
            source_id: Optional[str] = None,
            source_url: Optional[str] = None,
    ):
        """Delete documents from the vector db."""
        if not index_data.collection:
            raise ValueError("Collection not initialized. Call create_index() first.")

        if isinstance(ids, list) and len(ids) > 0:
            expr = f"pid in {ids}"
            index_data.collection.delete(expr=expr)
        elif source_id:
            index_data.collection.delete(expr=f"source == '{source_id}'")
        elif source_url:
            index_data.collection.delete(expr=f"source like '{source_url}%'")
        else:
            raise ValueError("No ids or source_id provided")

        index_data.collection.flush()

    def delete_all(
            self,
            index_data: MilvusIndexData,
            delete_index: bool = True
    ):
        """Delete all documents from the vector db."""
        if not index_data.collection:
            raise ValueError("Collection not initialized. Call create_index() first.")

        if delete_index:
            index_data.collection.drop()
            index_data.collection = None  # Reset the collection reference after dropping
        else:
            index_data.collection.delete(expr="pid != ''")
            index_data.collection.flush()  # Ensure changes are persisted


    def check_index(self, index_name: str) -> bool:
        alias = _create_connection_alias(self.connection_args)
        return utility.has_collection(index_name, using=alias)

    def get_count(self, index_data: MilvusIndexData) -> Dict[str, Any]:
        if not index_data.collection:
            raise ValueError("Collection not initialized. Call create_index() first.")

        return len(index_data.collection.query(expr="pid != ''", output_fields=["pid"]))

    def list_pids(self, index_data: MilvusIndexData, batch_size: int = 1000) -> List[str]:
        if not index_data.collection:
            raise ValueError("Collection not initialized. Call create_index() first.")
        try:
            # Get total count
            total_count = len(index_data.collection.query(
                expr="pid != ''",
                output_fields=["pid"]
            ))
            logger.info(f"Found {total_count} total documents in collection {index_data.index_name}")
            pids = []
            offset = 0
            while offset < total_count:
                # Get batch of PIDs
                batch = index_data.collection.query(
                    expr="pid != ''",
                    output_fields=["pid"],
                    limit=batch_size,
                    offset=offset
                )
                # Extract PIDs from batch
                batch_pids = [doc['pid'] for doc in batch]
                pids.extend(batch_pids)
                offset += batch_size
                logger.debug(f"Retrieved {len(pids)}/{total_count} PIDs")
            logger.info(f"Retrieved all {len(pids)} PIDs from collection {index_data.index_name}")
            return pids
        except Exception as e:
            logger.error(f"Error retrieving PIDs from collection {index_data.index_name}: {e}")
            raise

    def list_indices(self) -> List[str]:
        try:
            # Create connection
            alias = _create_connection_alias(self.connection_args)

            # Get all collections
            collection_names = utility.list_collections(using=alias)

            logger.info(f"Retrieved {len(collection_names)} collections")
            return collection_names

        except Exception as e:
            logger.error(f"Error retrieving collections: {e}")
            raise