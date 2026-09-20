"""
语义检索：稠密、稀疏分别检索，在代码中做 RRF 融合并保留两路原始分
不用 Milvus 内置 hybrid_search：内置融合只返回融合分，拿不到稠密余弦原始分，
评测要看稠密分分布，逐路分析召回效果也需要两路的原始名次。
命中 hit 为 dict：chunk_id, doc_id, version, kind, content_type, section_path, page_start, page_end, derived, text,
score_dense, score_sparse, rank_dense, rank_sparse, score_rrf,
以及来源元数据 file_name, document_title, source_path（来自 documents，全程随证据携带）。
"""
from utils.clients.milvus_utils import quote_str, search_dense, search_sparse
from utils.clients.mongo_utils import get_db
from utils.lm.embedding_utils import generate_query_embedding

RRF_K = 60

# 文档元数据缓存：doc_id → documents 里的 file_name / document_title / source_path / version / status
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
    """由 Milvus 返回的一行建立命中记录；分数、名次与来源元数据随后填写"""
    return {
        "chunk_id": row["chunk_id"],
        "doc_id": row["doc_id"],
        "version": row["version"],
        "kind": row["kind"],
        "content_type": row["content_type"],
        "section_path": row["section_path"],
        "page_start": row["page_start"],
        "page_end": row["page_end"],
        "derived": row["derived"],
        "text": row["text"],
        "score_dense": None,
        "score_sparse": None,
        "rank_dense": None,
        "rank_sparse": None,
        "score_rrf": 0.0,
        "file_name": "",
        "document_title": "",
        "source_path": "",
    }


def get_or_add_hit(hits: dict, row: dict) -> dict:
    """同一切片在两路都出现时只保留一条命中记录"""
    chunk_id = row["chunk_id"]
    if chunk_id not in hits:
        hits[chunk_id] = new_hit(row)
    return hits[chunk_id]


def rrf_fuse(dense: list, sparse: list, k: int = RRF_K) -> list:
    """
    倒数名次融合：每一路贡献 1 / (k + 名次)，两路都召回的切片分数累加
    :param dense: search_dense 的结果（按相似度降序）
    :param sparse: search_sparse 的结果（按内积降序）
    :param k: RRF 平滑常数
    :return: 去重后的命中列表，按融合分降序；两路原始分保留 4 位小数，未召回的一路为 None
    """
    hits = {}
    for rank, row in enumerate(dense, 1):
        hit = get_or_add_hit(hits, row)
        hit["score_dense"] = round(float(row["score"]), 4)
        hit["rank_dense"] = rank
        hit["score_rrf"] += 1.0 / (k + rank)
    for rank, row in enumerate(sparse, 1):
        hit = get_or_add_hit(hits, row)
        hit["score_sparse"] = round(float(row["score"]), 4)
        hit["rank_sparse"] = rank
        hit["score_rrf"] += 1.0 / (k + rank)
    return sorted(hits.values(), key=lambda h: h["score_rrf"], reverse=True)


def get_doc_meta(doc_ids: set) -> dict:
    """
    读取文档元数据，同一文档在进程内只查一次 Mongo
    :return: {doc_id: 文档元数据}；documents 里查不到的 doc_id 不在结果中
    """
    missing = [doc_id for doc_id in doc_ids if doc_id not in _doc_cache]
    if missing:
        fields = {"file_name": 1, "document_title": 1, "source_path": 1, "version": 1, "status": 1}
        for doc in get_db().documents.find({"_id": {"$in": missing}}, fields):
            _doc_cache[doc["_id"]] = doc
    result = {}
    for doc_id in doc_ids:
        if doc_id in _doc_cache:
            result[doc_id] = _doc_cache[doc_id]
    return result


def attach_doc_meta(hits: list, top_k: int) -> list:
    """
    给命中补上来源元数据并取前 top_k 条
    只保留文档当前版本的切片：新旧版本切换的瞬间 Milvus 里可能两版并存
    """
    meta = get_doc_meta({hit["doc_id"] for hit in hits})
    results = []
    for hit in hits:
        doc = meta.get(hit["doc_id"])
        if doc is None or doc.get("version") != hit["version"] or doc.get("status") == "superseded":
            continue
        hit["file_name"] = doc.get("file_name", "")
        hit["document_title"] = doc.get("document_title", "")
        hit["source_path"] = doc.get("source_path", "")
        results.append(hit)
        if len(results) >= top_k:
            break
    return results


def semantic_search(query: str, top_k: int = 5, candidates: int = 30, kinds=None, content_types=None,
                    doc_ids=None) -> list:
    """
    稠密 + 稀疏检索并融合
    :param query: 问题
    :param top_k: 最多返回几条
    :param candidates: 每一路先取多少条候选
    :param kinds / content_types / doc_ids: 过滤条件，见 build_filter
    :return: 命中 dict 列表，按融合分降序
    """
    embedding = generate_query_embedding(query)
    expr = build_filter(kinds, content_types, doc_ids)
    dense = search_dense(embedding["dense"], candidates, expr)
    sparse = search_sparse(embedding["sparse"], candidates, expr)
    fused = rrf_fuse(dense, sparse)
    return attach_doc_meta(fused, top_k)
