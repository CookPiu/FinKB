"""
节点：精排与证据组装
- 语义检索结果交给云端 qwen3-rerank 精排取前 TOP_CHUNKS（失败时退回 RRF 排序）；
- 证据顺序：财务事实 → 文档摘要 → 文本切片，统一编号 E1..En，全程携带来源元数据；
- 充分性：命中事实或摘要、或已确认实体（按文档过滤）即交给生成，由模型的【依据不足】兜底；
  无实体的全库检索还要求稠密最高分 ≥ TAU。不充分时直接写入固定拒答话术。
"""
from common.answer_templates import REFUSE
from common.logging.logger import logger, node_log, step_log
from processor.query_processor.state import QueryGraphState, create_query_default_state
from utils.clients.mongo_utils import get_documents_by_file
from utils.entity_utils import get_entity_map
from utils.lm.reranker_utils import rerank

TAU = 0.58  # 无实体过滤时，稠密最高分低于此值判为证据不足（由 M1 基线的正负例分布定出）
TOP_CHUNKS = 6


@step_log("validate_and_get_data")
def validate_and_get_data(state: QueryGraphState):
    """
    取出并校验精排与证据组装所需的入参
    :return: 元组 (独立问题, 已确认实体 ID, 检索命中, 稠密最高分)
    :raise ValueError: 计划里没有独立问题
    """
    plan = state.get("plan") or {}
    standalone_query = plan.get("standalone_query", "")
    if not standalone_query:
        logger.error("no standalone_query found in plan")
        raise ValueError("no standalone_query found in plan")
    hits = state.get("embedding_chunks") or []
    top_dense = state.get("top_dense") or 0.0
    return standalone_query, state.get("entity_ids", []), hits, top_dense


@node_log("node_rerank")
def node_rerank(state: QueryGraphState):
    """
    节点功能：把三路取证结果组装成带编号的证据列表。
    先放财务事实与文档摘要，再判断充分性（不依赖精排分：不充分时直接拒答，省掉一次精排调用），
    最后精排语义检索结果并追加文本切片。
    上游：node_fact_lookup / node_summary_fetch / node_search_embedding；下游：node_answer_output。
    """
    standalone_query, entity_ids, hits, top_dense = validate_and_get_data(state)
    docs = get_documents_by_file()
    doc_meta = get_doc_meta_by_id(docs)
    entity_of_doc = get_entity_of_doc(entity_ids, docs)
    evidence = build_fact_and_summary_evidence(state, doc_meta, entity_of_doc)

    sufficient = is_sufficient(evidence, hits, state.get("doc_ids"), top_dense)
    logger.info(f"facts/summaries={len(evidence)} hits={len(hits)} top_dense={top_dense:.3f} sufficient={sufficient}")
    if not sufficient:
        state["evidence"] = []
        state["kind"] = "refuse"
        state["answer"] = REFUSE
        return state

    for i in rerank_hits(standalone_query, hits):
        hit = hits[i]
        item = build_evidence("chunk", hit["text"], doc_meta[hit["doc_id"]], entity_of_doc.get(hit["doc_id"]))
        item["page_start"] = hit["page_start"]
        item["page_end"] = hit["page_end"]
        item["score_dense"] = hit["score_dense"]
        item["derived"] = hit["derived"]
        evidence.append(item)
    for eid, item in enumerate(evidence, 1):
        item["eid"] = eid
    state["evidence"] = evidence
    return state


def get_doc_meta_by_id(docs: dict) -> dict:
    """文件名索引的文档元数据改为 doc_id 索引"""
    doc_meta = {}
    for doc in docs.values():
        doc_meta[doc["_id"]] = doc
    return doc_meta


def get_entity_of_doc(entity_ids: list, docs: dict) -> dict:
    """doc_id → 该文档所属的实体（只含本轮已确认的实体）"""
    entity_map = get_entity_map()
    entity_of_doc = {}
    for entity_id in entity_ids:
        entity = entity_map[entity_id]
        for file_name in entity["files"]:
            if file_name in docs:
                entity_of_doc[docs[file_name]["_id"]] = entity
    return entity_of_doc


def build_evidence(kind: str, text: str, doc: dict, entity) -> dict:
    """
    组装一条证据（键见 utils/citation_utils.py），页码、稠密分等由调用方按需补上
    :param kind: fact / summary / chunk
    :param doc: 文档元数据（get_documents_by_file 的值）
    :param entity: 文档所属实体，没有则为 None；document 类实体（政策报告等）不写实体名与代码
    """
    show_entity = entity is not None and entity["type"] != "document"
    entity_name = ""
    entity_codes = []
    if show_entity:
        entity_name = entity["name"]
        entity_codes = entity["codes"]
    return {
        "eid": 0,
        "kind": kind,
        "text": text,
        "doc_id": doc["_id"],
        "file_name": doc["file_name"],
        "document_title": doc["document_title"],
        "content_type": doc["content_type"],
        "page_start": None,
        "page_end": None,
        "source_path": doc.get("source_path", ""),
        "entity_name": entity_name,
        "entity_codes": entity_codes,
        "score_dense": None,
        "derived": False,
    }


@step_log("build_fact_and_summary_evidence")
def build_fact_and_summary_evidence(state: QueryGraphState, doc_meta: dict, entity_of_doc: dict) -> list:
    """财务事实在前、文档摘要在后"""
    evidence = []
    for fact in state.get("facts") or []:
        item = build_evidence("fact", fact["text"], doc_meta[fact["doc_id"]], entity_of_doc.get(fact["doc_id"]))
        item["page_start"] = fact["page_start"]
        item["page_end"] = fact["page_end"]
        evidence.append(item)
    for summary in state.get("summaries") or []:
        text = "（文档摘要）" + summary["text"]
        evidence.append(build_evidence("summary", text, doc_meta[summary["doc_id"]], entity_of_doc.get(summary["doc_id"])))
    return evidence


@step_log("is_sufficient")
def is_sufficient(evidence: list, hits: list, doc_ids, top_dense: float) -> bool:
    """
    证据是否足以交给模型生成
    有事实或摘要、或检索限定在已确认实体的文档内，即可交给生成；全库检索还要求稠密最高分 ≥ TAU
    """
    if not evidence and not hits:
        return False
    return bool(evidence) or bool(doc_ids) or top_dense >= TAU


@step_log("rerank_hits")
def rerank_hits(query: str, hits: list) -> list:
    """
    云端精排
    :return: 选中命中的下标，按相关度降序，最多 TOP_CHUNKS 个；精排失败时退回 RRF 排序的前 TOP_CHUNKS 个
    """
    if not hits:
        return []
    fallback = list(range(min(TOP_CHUNKS, len(hits))))
    try:
        order = rerank(query, [hit["text"][:2000] for hit in hits], TOP_CHUNKS)
    except Exception as e:
        logger.warning(f"精排失败，使用 RRF 排序：{e}")
        return fallback
    if not order:  # 精排返回空结果时同样退回 RRF 排序，保证证据不为空
        logger.warning("精排返回空结果，使用 RRF 排序")
        return fallback
    return order


if __name__ == "__main__":
    # 运行：uv run python -m processor.query_processor.nodes.node_rerank
    # 依赖：Mongo、Milvus、BGE-M3、百炼精排（先跑语义检索得到候选）
    from processor.query_processor.nodes.node_search_embedding import node_search_embedding
    demo_plan = {"standalone_query": "贵州茅台一季度营业收入是多少？", "intent": "announcement",
                 "entity_mentions": [], "metrics": [], "wants_summary": False}
    demo_state = create_query_default_state(session_id="demo-rerank", plan=demo_plan)
    demo_state.update(node_search_embedding(demo_state))
    result = node_rerank(demo_state)
    print(result["kind"] or "sufficient", [(e["eid"], e["file_name"], e["page_start"]) for e in result["evidence"]])
