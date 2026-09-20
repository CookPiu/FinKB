"""
Milvus 集合 fin_chunks：建表、写入、删除，以及稠密 / 稀疏分别检索。
文本、表格、摘要、图片描述都在同一集合，用 kind 区分。
稠密与稀疏分开检索、在代码里做 RRF 融合（见 utils/search_utils.py），不用内置 hybrid_search：
内置融合只返回融合分，拿不到稠密余弦原始分，而拒答判断要用它。
"""
from pymilvus import DataType, MilvusClient

from common.config.milvus_config import milvus_config

DENSE_DIM = 1024  # BGE-M3 稠密向量维度
TEXT_MAX_LEN = 65535  # VARCHAR 上限（按 UTF-8 字节计）
SECTION_MAX_LEN = 1024

# 检索时返回的字段（不含向量）
OUTPUT_FIELDS = [
    "chunk_id", "doc_id", "version", "kind", "content_type", "entity_ids",
    "section_path", "page_start", "page_end", "publish_date", "derived", "text",
]

# 全局 Milvus 客户端单例
_milvus_client = None


def get_milvus_client():
    """获取 Milvus 客户端单例"""
    global _milvus_client
    if _milvus_client is None:
        _milvus_client = MilvusClient(uri=milvus_config.milvus_uri, timeout=30)
    return _milvus_client


def ensure_collection():
    """集合不存在时按固定 schema 创建；存在时确保已加载到内存"""
    client = get_milvus_client()
    name = milvus_config.chunks_collection
    if client.has_collection(name):
        client.load_collection(name)
        return

    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
    # chunk_id = {doc_id}-{version}-{seq:04d}，重跑时 upsert 覆盖同一条，保证幂等
    schema.add_field("chunk_id", DataType.VARCHAR, is_primary=True, max_length=128)
    schema.add_field("doc_id", DataType.VARCHAR, max_length=64)
    schema.add_field("version", DataType.INT32)
    schema.add_field("kind", DataType.VARCHAR, max_length=32)
    schema.add_field("content_type", DataType.VARCHAR, max_length=64)
    schema.add_field("entity_ids", DataType.ARRAY, element_type=DataType.VARCHAR, max_capacity=64, max_length=64)
    schema.add_field("section_path", DataType.VARCHAR, max_length=SECTION_MAX_LEN)
    schema.add_field("page_start", DataType.INT16)
    schema.add_field("page_end", DataType.INT16)
    schema.add_field("publish_date", DataType.INT64)
    schema.add_field("derived", DataType.BOOL)
    schema.add_field("text", DataType.VARCHAR, max_length=TEXT_MAX_LEN)
    schema.add_field("dense", DataType.FLOAT_VECTOR, dim=DENSE_DIM)
    schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)

    index_params = client.prepare_index_params()
    index_params.add_index(field_name="dense", index_type="HNSW", metric_type="COSINE",
                           params={"M": 16, "efConstruction": 200})
    index_params.add_index(field_name="sparse", index_type="SPARSE_INVERTED_INDEX", metric_type="IP")
    # 标量倒排索引：按文档、类型过滤时用
    for field_name in ["doc_id", "kind", "content_type"]:
        index_params.add_index(field_name=field_name, index_type="INVERTED")

    client.create_collection(name, schema=schema, index_params=index_params, consistency_level="Strong")
    client.load_collection(name)


def upsert_rows(rows, batch_size=64):
    """分批 upsert，避免单次请求过大"""
    client = get_milvus_client()
    for start in range(0, len(rows), batch_size):
        client.upsert(milvus_config.chunks_collection, rows[start:start + batch_size])


def delete_rows(filter_expr: str):
    get_milvus_client().delete(milvus_config.chunks_collection, filter=filter_expr)


def count_rows(filter_expr: str = "") -> int:
    result = get_milvus_client().query(milvus_config.chunks_collection, filter=filter_expr,
                                       output_fields=["count(*)"])
    if not result:
        return 0
    return int(result[0]["count(*)"])


def _hits_to_rows(hits):
    """把 Milvus 的命中结果展平成字典：业务字段 + score（相似度）"""
    rows = []
    for hit in hits:
        row = dict(hit["entity"])
        row["score"] = hit["distance"]
        rows.append(row)
    return rows


def search_dense(vector, limit: int, filter_expr: str = ""):
    """
    稠密向量检索（余弦相似度）
    :return: [{"score", "chunk_id", "doc_id", ...OUTPUT_FIELDS}]，按相似度降序
    """
    result = get_milvus_client().search(
        milvus_config.chunks_collection,
        data=[vector],
        anns_field="dense",
        limit=limit,
        filter=filter_expr,
        output_fields=OUTPUT_FIELDS,
        # HNSW 的 ef 必须不小于 limit，取 2 倍留余量
        search_params={"metric_type": "COSINE", "params": {"ef": max(64, limit * 2)}},
    )
    return _hits_to_rows(result[0])


def search_sparse(vector, limit: int, filter_expr: str = ""):
    """
    稀疏向量检索（内积）
    :param vector: {维度: 权重}
    :return: 同 search_dense
    """
    result = get_milvus_client().search(
        milvus_config.chunks_collection,
        data=[vector],
        anns_field="sparse",
        limit=limit,
        filter=filter_expr,
        output_fields=OUTPUT_FIELDS,
        search_params={"metric_type": "IP", "params": {"drop_ratio_search": 0.0}},
    )
    return _hits_to_rows(result[0])


def quote_str(value: str) -> str:
    """把字符串安全地放进 Milvus 过滤表达式：转义反斜杠和双引号后加上双引号"""
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return '"' + escaped + '"'
