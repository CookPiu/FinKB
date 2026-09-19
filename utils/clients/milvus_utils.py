"""Milvus 集合 fin_chunks：定义、写入、删除与稠密/稀疏分别检索。

文本、表格、摘要、术语、图片描述都在同一集合，用 kind 区分。
稠密与稀疏分开检索、在代码中融合（见 query/tools/semantic_search.py），不用内置 hybrid_search。
"""

from __future__ import annotations

import threading
from typing import Any

from pymilvus import DataType, MilvusClient

from common.config.settings import get_settings

DENSE_DIM = 1024
TEXT_MAX_LEN = 65535
SECTION_MAX_LEN = 1024

OUTPUT_FIELDS = [
    "chunk_id",
    "doc_id",
    "version",
    "kind",
    "content_type",
    "entity_ids",
    "section_path",
    "page_start",
    "page_end",
    "publish_date",
    "derived",
    "text",
]

_client: MilvusClient | None = None
_lock = threading.Lock()


def get_client() -> MilvusClient:
    global _client
    if _client is None:
        with _lock:
            if _client is None:
                _client = MilvusClient(uri=get_settings().milvus_uri, timeout=30)
    return _client


def collection_name() -> str:
    return get_settings().milvus_collection


def ensure_collection() -> None:
    client = get_client()
    name = collection_name()
    if client.has_collection(name):
        client.load_collection(name)
        return

    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("chunk_id", DataType.VARCHAR, is_primary=True, max_length=128)
    schema.add_field("doc_id", DataType.VARCHAR, max_length=64)
    schema.add_field("version", DataType.INT32)
    schema.add_field("kind", DataType.VARCHAR, max_length=32)
    schema.add_field("content_type", DataType.VARCHAR, max_length=64)
    schema.add_field(
        "entity_ids", DataType.ARRAY, element_type=DataType.VARCHAR, max_capacity=64, max_length=64
    )
    schema.add_field("section_path", DataType.VARCHAR, max_length=SECTION_MAX_LEN)
    schema.add_field("page_start", DataType.INT16)
    schema.add_field("page_end", DataType.INT16)
    schema.add_field("publish_date", DataType.INT64)
    schema.add_field("derived", DataType.BOOL)
    schema.add_field("text", DataType.VARCHAR, max_length=TEXT_MAX_LEN)
    schema.add_field("dense", DataType.FLOAT_VECTOR, dim=DENSE_DIM)
    schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="dense", index_type="HNSW", metric_type="COSINE", params={"M": 16, "efConstruction": 200}
    )
    index_params.add_index(field_name="sparse", index_type="SPARSE_INVERTED_INDEX", metric_type="IP")
    for f in ("doc_id", "kind", "content_type"):
        index_params.add_index(field_name=f, index_type="INVERTED")

    client.create_collection(name, schema=schema, index_params=index_params, consistency_level="Strong")
    client.load_collection(name)


def upsert(rows: list[dict[str, Any]], batch_size: int = 64) -> None:
    client = get_client()
    for i in range(0, len(rows), batch_size):
        client.upsert(collection_name(), rows[i : i + batch_size])


def delete(filter_expr: str) -> None:
    get_client().delete(collection_name(), filter=filter_expr)


def count(filter_expr: str = "") -> int:
    res = get_client().query(collection_name(), filter=filter_expr, output_fields=["count(*)"])
    return int(res[0]["count(*)"]) if res else 0


def search_dense(vec: list[float], limit: int, filter_expr: str = "") -> list[dict[str, Any]]:
    res = get_client().search(
        collection_name(),
        data=[vec],
        anns_field="dense",
        limit=limit,
        filter=filter_expr,
        output_fields=OUTPUT_FIELDS,
        search_params={"metric_type": "COSINE", "params": {"ef": max(64, limit * 2)}},
    )
    return [{"score": hit["distance"], **hit["entity"]} for hit in res[0]]


def search_sparse(vec: dict[int, float], limit: int, filter_expr: str = "") -> list[dict[str, Any]]:
    res = get_client().search(
        collection_name(),
        data=[vec],
        anns_field="sparse",
        limit=limit,
        filter=filter_expr,
        output_fields=OUTPUT_FIELDS,
        search_params={"metric_type": "IP", "params": {"drop_ratio_search": 0.0}},
    )
    return [{"score": hit["distance"], **hit["entity"]} for hit in res[0]]


def quote_str(value: str) -> str:
    """把字符串安全地放进 Milvus 过滤表达式。"""
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
