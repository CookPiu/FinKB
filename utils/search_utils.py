"""
语义检索：稠密 + 稀疏混合检索（Milvus 内置 hybrid_search，服务端 RRF 融合），再补上来源元数据。
命中 hit 为 dict：chunk_id, doc_id, kind, content_type, section_path, page_start, page_end, derived, text, score，
以及来源元数据 file_name, document_title, source_path（来自 documents，全程随证据携带）。
"""
from utils.clients.milvus_utils import hybrid_search, quote_str
from utils.clients.mongo_utils import get_db
from utils.lm.embedding_utils import generate_query_embedding

# 文档元数据缓存：doc_id → documents 里的 file_name / document_title / source_path
_doc_cache = {}


def join_quoted(values: list) -> str:
    """每个值加引号转义后用逗号连接"""
    return ", ".join([quote_str(v) for v in values])


def build_filter(kinds=None, content_types=None, doc_ids=None) -> str:
    """
    拼 Milvus 过滤表达式，各条件之间为 and；参数为空表示该项不限
    :return: 表达式字符串，没有任何条件时为空串
    """
    parts = []
    if kinds:
        parts.append(f"kind in [{join_quoted(kinds)}]")
    if content_types:
        parts.append(f"content_type in [{join_quoted(content_types)}]")
    if doc_ids:
        parts.append(f"doc_id in [{join_quoted(doc_ids)}]")
    return " and ".join(parts)


def new_hit(row: dict) -> dict:
    """由 Milvus 返回的一行建立命中记录；来源元数据随后填写"""
    return {
        "chunk_id": row["chunk_id"],
        "doc_id": row["doc_id"],
        "kind": row["kind"],
        "content_type": row["content_type"],
        "section_path": row["section_path"],
        "page_start": row["page_start"],
        "page_end": row["page_end"],
        "derived": row["derived"],
        "text": row["text"],
        "score": round(float(row["score"]), 5),
        "file_name": "",
        "document_title": "",
        "source_path": "",
    }


def get_doc_meta(doc_ids: set) -> dict:
    """
    读取文档元数据，同一文档在进程内只查一次 Mongo
    :return: {doc_id: 文档元数据}；documents 里查不到的 doc_id 不在结果中
    """
    missing = [doc_id for doc_id in doc_ids if doc_id not in _doc_cache]
    if missing:
        fields = {"file_name": 1, "document_title": 1, "source_path": 1}
        for doc in get_db().documents.find({"_id": {"$in": missing}}, fields):
            _doc_cache[doc["_id"]] = doc
    result = {}
    for doc_id in doc_ids:
        if doc_id in _doc_cache:
            result[doc_id] = _doc_cache[doc_id]
    return result


def attach_doc_meta(hits: list) -> list:
    """
    给命中补上来源元数据
    文档记录已被删除的切片直接丢掉（正常情况下切片会与记录一起删除）
    """
    meta = get_doc_meta({hit["doc_id"] for hit in hits})
    results = []
    for hit in hits:
        doc = meta.get(hit["doc_id"])
        if doc is None:
            continue
        hit["file_name"] = doc.get("file_name", "")
        hit["document_title"] = doc.get("document_title", "")
        hit["source_path"] = doc.get("source_path", "")
        results.append(hit)
    return results


def semantic_search(query: str, top_k: int = 5, candidates: int = 30, kinds=None, content_types=None,
                    doc_ids=None) -> list:
    """
    稠密 + 稀疏混合检索
    :param query: 问题
    :param top_k: 最多返回几条
    :param candidates: 每一路先取多少条候选
    :param kinds / content_types / doc_ids: 过滤条件，见 build_filter
    :return: 命中 dict 列表，按融合分降序
    """
    embedding = generate_query_embedding(query)
    expr = build_filter(kinds, content_types, doc_ids)
    rows = hybrid_search(embedding["dense"], embedding["sparse"], candidates, top_k, expr)
    return attach_doc_meta([new_hit(row) for row in rows])


if __name__ == "__main__":
    # 运行：uv run python -m utils.search_utils
    # 依赖：Milvus、本地 BGE-M3（CPU 首次加载约 20 秒）
    for test_hit in semantic_search("贵州茅台2026年第一季度营业收入是多少", top_k=5):
        print(f"{test_hit['score']:.5f} {test_hit['file_name']} p{test_hit['page_start']} "
              f"{test_hit['text'][:50]}")
