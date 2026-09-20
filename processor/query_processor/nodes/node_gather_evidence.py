"""
节点：取证
解析问题里提到的对象 → 混合检索（有对象则限定在其文档内）→ 精排 → 统一编号 E1..En。
证据只有一种来源：切片。财务指标事实与文档摘要在导入时就做成了切片（kind=fact / summary），
所以数值题、概览题和正文题走同一条检索路径，这里不必分三路取证。
这里不做"该不该回答"的判断（交给 node_answer_output 里的模型），只在一条证据都没有时短路成固定拒答。
"""
from common.answer_templates import REFUSE
from common.logging.logger import logger, node_log, step_log
from processor.query_processor.state import QueryGraphState, create_query_default_state
from utils.clients.mongo_utils import get_documents_by_file
from utils.entity_utils import get_entities, get_entity_map, resolve_mentions
from utils.lm.reranker_utils import rerank
from utils.search_utils import semantic_search

CANDIDATES = 30  # 稠密、稀疏各取多少条候选
TOP_K = 20  # 融合后保留多少条进入精排
TOP_CHUNKS = 8  # 精排后取多少条切片作为证据


@step_log("validate_and_get_data")
def validate_and_get_data(state: QueryGraphState):
    """
    取出并校验取证所需的入参
    :return: 元组 (独立问题, 提及的对象)
    :raise ValueError: 计划里没有独立问题（node_query_plan 未执行）
    """
    plan = state.get("plan") or {}
    standalone_query = plan.get("standalone_query", "")
    if not standalone_query:
        logger.error("no standalone_query found in plan")
        raise ValueError("no standalone_query found in plan")
    return standalone_query, plan.get("entity_mentions", [])


@node_log("node_gather_evidence")
def node_gather_evidence(state: QueryGraphState):
    """
    节点功能：解析对象、检索、精排并编号。
    上游 node_query_plan；下游 node_answer_output。
    澄清候选与库外对象只记录下来交给回答节点参考，不在这里决定走向。
    """
    standalone_query, mentions = validate_and_get_data(state)
    if state.get("clarified"):
        # 上一轮澄清已由 node_query_plan 用代码确定对象
        entity_ids = state.get("entity_ids", [])
        candidates = []
        unknown_mentions = []
    else:
        resolution = resolve_mentions(mentions, get_entities())
        entity_ids = [entity["id"] for entity in resolution["entities"]]
        candidates = resolution["candidates"]
        unknown_mentions = resolution["unknown_mentions"]
    doc_ids = get_doc_ids(entity_ids)
    logger.info(f"entities={entity_ids} candidates={[e['id'] for e in candidates]} unknown={unknown_mentions}")

    evidence = gather_evidence(standalone_query, entity_ids, doc_ids)
    state["entity_ids"] = entity_ids
    state["candidate_ids"] = [entity["id"] for entity in candidates]
    state["candidate_names"] = [entity["name"] for entity in candidates]
    state["unknown_mentions"] = unknown_mentions
    state["doc_ids"] = doc_ids
    state["evidence"] = evidence
    if not evidence:
        # 一条证据都没有：不必调用模型，直接用固定拒答话术
        logger.info("没有任何证据，直接拒答")
        state["kind"] = "refuse"
        state["answer"] = REFUSE
    return state


@step_log("gather_evidence")
def gather_evidence(standalone_query: str, entity_ids: list, doc_ids: list) -> list:
    """
    检索、精排并编号
    :return: 证据列表（键见 utils/citation_utils.py）
    """
    docs = get_documents_by_file()
    doc_meta = get_doc_meta_by_id(docs)
    entity_of_doc = get_entity_of_doc(entity_ids, docs)

    hits = semantic_search(standalone_query, top_k=TOP_K, candidates=CANDIDATES, doc_ids=doc_ids or None)
    evidence = []
    for index in rerank_hits(standalone_query, hits):
        hit = hits[index]
        evidence.append(build_evidence(hit, doc_meta[hit["doc_id"]], entity_of_doc.get(hit["doc_id"])))
    for eid, item in enumerate(evidence, 1):
        item["eid"] = eid
    return evidence


# ---------- 对象与文档 ----------

def get_doc_ids(entity_ids: list) -> list:
    """实体对应的文档 ID（按实体表里的文件名到 documents 查找，未导入的文件跳过）"""
    docs = get_documents_by_file()
    entity_map = get_entity_map()
    doc_ids = []
    for entity_id in entity_ids:
        for file_name in entity_map[entity_id]["files"]:
            if file_name in docs:
                doc_ids.append(docs[file_name]["_id"])
    return doc_ids


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


# ---------- 证据 ----------

def build_evidence(hit: dict, doc: dict, entity) -> dict:
    """
    检索命中 → 证据（键见 utils/citation_utils.py）
    :param hit: 命中 dict（见 utils/search_utils.py）
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
        "kind": hit["kind"],
        "text": hit["text"],
        "doc_id": hit["doc_id"],
        "file_name": doc["file_name"],
        "document_title": doc["document_title"],
        "content_type": doc["content_type"],
        "page_start": hit["page_start"],
        "page_end": hit["page_end"],
        "source_path": doc.get("source_path", ""),
        "entity_name": entity_name,
        "entity_codes": entity_codes,
        "derived": hit["derived"],
    }


# ---------- 精排 ----------

@step_log("rerank_hits")
def rerank_hits(query: str, hits: list) -> list:
    """
    云端精排
    :return: 选中命中的下标，按相关度降序，最多 TOP_CHUNKS 个；精排失败或返回空结果时退回融合排序
    """
    if not hits:
        return []
    fallback = list(range(min(TOP_CHUNKS, len(hits))))
    try:
        order = rerank(query, [hit["text"][:2000] for hit in hits], TOP_CHUNKS)
    except Exception as e:
        logger.warning(f"精排失败，使用融合排序：{e}")
        return fallback
    if not order:
        logger.warning("精排返回空结果，使用融合排序")
        return fallback
    return order


if __name__ == "__main__":
    # 运行：uv run python -m processor.query_processor.nodes.node_gather_evidence
    # 依赖：Mongo、Milvus、BGE-M3（CPU 首次加载约 20 秒）、百炼精排
    demo_plan = {"standalone_query": "贵州茅台一季度营业收入是多少？", "entity_mentions": ["贵州茅台"]}
    demo_state = create_query_default_state(session_id="demo-gather", plan=demo_plan)
    result = node_gather_evidence(demo_state)
    for item in result["evidence"]:
        print(item["eid"], item["kind"], item["file_name"], item["page_start"], item["text"][:40])
