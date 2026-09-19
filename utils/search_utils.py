"""semantic_search：稠密、稀疏分别检索，在代码中做 RRF 融合并保留两路原始分。

不用 Milvus 内置 hybrid_search：内置融合只返回融合分，拿不到稠密余弦原始分，
而充分性判断（M2）依赖稠密原始分，且需要逐路分析召回效果。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from utils.lm import embedding_utils as embedding
from utils.clients import milvus_utils as milvus
from utils.clients import mongo_utils as mongo

RRF_K = 60


@dataclass
class SearchHit:
    chunk_id: str
    doc_id: str
    version: int
    kind: str
    content_type: str
    section_path: str
    page_start: int
    page_end: int
    derived: bool
    text: str
    entity_ids: list[str] = field(default_factory=list)
    score_dense: float | None = None
    score_sparse: float | None = None
    rank_dense: int | None = None
    rank_sparse: int | None = None
    score_rrf: float = 0.0
    # 来源元数据（来自 documents），全程随证据携带
    file_name: str = ""
    document_title: str = ""
    source_path: str = ""
    publish_date: int | None = None


def build_filter(
    kinds: list[str] | None = None,
    content_types: list[str] | None = None,
    doc_ids: list[str] | None = None,
    entity_ids: list[str] | None = None,
) -> str:
    parts = []
    for field_name, values in (("kind", kinds), ("content_type", content_types), ("doc_id", doc_ids)):
        if values:
            parts.append(f"{field_name} in [{', '.join(milvus.quote_str(v) for v in values)}]")
    if entity_ids:
        parts.append(f"ARRAY_CONTAINS_ANY(entity_ids, [{', '.join(milvus.quote_str(v) for v in entity_ids)}])")
    return " and ".join(parts)


def rrf_fuse(dense: list[dict[str, Any]], sparse: list[dict[str, Any]], k: int = RRF_K) -> list[SearchHit]:
    hits: dict[str, SearchHit] = {}

    def get(row: dict[str, Any]) -> SearchHit:
        cid = row["chunk_id"]
        if cid not in hits:
            hits[cid] = SearchHit(
                chunk_id=cid,
                doc_id=row["doc_id"],
                version=row["version"],
                kind=row["kind"],
                content_type=row["content_type"],
                section_path=row["section_path"],
                page_start=row["page_start"],
                page_end=row["page_end"],
                derived=row["derived"],
                text=row["text"],
                entity_ids=list(row.get("entity_ids") or []),
            )
        return hits[cid]

    for rank, row in enumerate(dense, 1):
        h = get(row)
        h.score_dense, h.rank_dense = round(float(row["score"]), 4), rank
        h.score_rrf += 1.0 / (k + rank)
    for rank, row in enumerate(sparse, 1):
        h = get(row)
        h.score_sparse, h.rank_sparse = round(float(row["score"]), 4), rank
        h.score_rrf += 1.0 / (k + rank)
    return sorted(hits.values(), key=lambda h: h.score_rrf, reverse=True)


_doc_cache: dict[str, dict] = {}


def _doc_meta(doc_ids: set[str]) -> dict[str, dict]:
    missing = [d for d in doc_ids if d not in _doc_cache]
    if missing:
        for d in mongo.get_db().documents.find(
            {"_id": {"$in": missing}},
            {"file_name": 1, "document_title": 1, "source_path": 1, "publish_date": 1, "version": 1, "status": 1},
        ):
            _doc_cache[d["_id"]] = d
    return {d: _doc_cache[d] for d in doc_ids if d in _doc_cache}


def semantic_search(
    query: str,
    *,
    top_k: int = 5,
    candidates: int = 30,
    kinds: list[str] | None = None,
    content_types: list[str] | None = None,
    doc_ids: list[str] | None = None,
    entity_ids: list[str] | None = None,
) -> list[SearchHit]:
    enc = embedding.encode_query(query)
    expr = build_filter(kinds, content_types, doc_ids, entity_ids)
    dense = milvus.search_dense(enc.dense, candidates, expr)
    sparse = milvus.search_sparse(enc.sparse, candidates, expr)
    fused = rrf_fuse(dense, sparse)

    meta = _doc_meta({h.doc_id for h in fused})
    out: list[SearchHit] = []
    for h in fused:
        m = meta.get(h.doc_id)
        # 只保留文档当前版本的切片（新旧版本切换的瞬间可能两版并存）
        if m is None or m.get("version") != h.version or m.get("status") == "superseded":
            continue
        h.file_name = m.get("file_name", "")
        h.document_title = m.get("document_title", "")
        h.source_path = m.get("source_path", "")
        h.publish_date = m.get("publish_date")
        out.append(h)
        if len(out) >= top_k:
            break
    return out
