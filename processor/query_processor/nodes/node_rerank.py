"""节点：精排与证据组装。

- 语义检索结果交给云端 qwen3-rerank 精排取前 TOP_CHUNKS（失败时退回 RRF 排序）；
- 证据顺序：财务事实 → 文档摘要 → 文本切片，统一编号 E1..En，全程携带来源元数据；
- 充分性：命中事实或摘要、或已确认实体（按文档过滤）即交给生成，由模型的【依据不足】兜底；
  无实体的全库检索还要求稠密最高分 ≥ TAU。不充分时直接写入固定拒答话术。
"""

from __future__ import annotations

from typing import Any

from common import answer_templates as templates
from common.logging.logger import logger, node_log
from common.models.evidence import Evidence
from processor.query_processor.state import QueryGraphState, QueryPlan
from utils.clients import mongo_utils as mongo
from utils.entity_utils import registry
from utils.lm import reranker_utils as reranker

TAU = 0.58  # 无实体过滤时，稠密最高分低于此值判为证据不足（由 M1 基线的正负例分布定出）
TOP_CHUNKS = 6


@node_log("node_rerank")
def node_rerank(state: QueryGraphState) -> dict:
    plan = QueryPlan.model_validate(state["plan"])
    reg = registry()
    entities = [reg.by_id[i] for i in state.get("entity_ids", [])]
    docs = mongo.documents_by_file()
    doc_meta = {d["_id"]: d for d in docs.values()}
    entity_of_doc = {docs[f]["_id"]: e for e in entities for f in e.files if f in docs}

    def make(kind: str, text: str, doc_id: str, **kw: Any) -> Evidence:
        d, e = doc_meta[doc_id], entity_of_doc.get(doc_id)
        return Evidence(
            eid=0,
            kind=kind,
            text=text,
            doc_id=doc_id,
            file_name=d["file_name"],
            document_title=d["document_title"],
            content_type=d["content_type"],
            source_path=d.get("source_path", ""),
            entity_name=e.name if e and e.type != "document" else "",
            entity_codes=e.codes if e and e.type != "document" else [],
            **kw,
        )

    evidence: list[Evidence] = []
    for f in state.get("facts") or []:
        evidence.append(make("fact", f["text"], f["doc_id"], page_start=f["page_start"], page_end=f["page_end"]))
    for s in state.get("summaries") or []:
        evidence.append(make("summary", "（文档摘要）" + s["text"], s["doc_id"]))

    # 充分性不依赖精排分，先判断：不充分时直接拒答，省掉一次精排调用
    hits = state.get("embedding_chunks") or []
    top_dense = state.get("top_dense") or 0.0
    sufficient = bool(evidence or hits) and (bool(evidence) or bool(state.get("doc_ids")) or top_dense >= TAU)
    logger.info("facts/summaries=%d hits=%d top_dense=%.3f sufficient=%s", len(evidence), len(hits), top_dense, sufficient)
    if not sufficient:
        return {"evidence": [], "kind": "refuse", "answer": templates.REFUSE}

    if hits:
        try:
            order = reranker.rerank(plan.standalone_query, [h["text"][:2000] for h in hits], TOP_CHUNKS)
        except Exception as e:  # noqa: BLE001 - 精排失败时退回融合排序
            logger.warning("精排失败，使用 RRF 排序：%s", e)
            order = list(range(min(TOP_CHUNKS, len(hits))))
        for i in order:
            h = hits[i]
            evidence.append(
                make("chunk", h["text"], h["doc_id"], page_start=h["page_start"], page_end=h["page_end"],
                     score_dense=h["score_dense"], derived=h["derived"])
            )
    for i, e in enumerate(evidence, 1):
        e.eid = i
    return {"evidence": [vars(e) for e in evidence]}
