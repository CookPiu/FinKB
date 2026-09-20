"""
节点：取证
把原来的实体确认、财务事实、文档摘要、语义检索、精排合并成一个节点：
解析问题里的对象 → 按对象取财务事实与文档摘要 → 语义检索（有对象则限定在其文档内）→ 精排 → 统一编号 E1..En。
这里不做“该不该回答”的判断（交给 node_answer_output 里的模型），只在一条证据都没有时短路成固定拒答。
"""
import re

from common.answer_templates import REFUSE
from common.logging.logger import logger, node_log, step_log
from processor.query_processor.state import QueryGraphState, create_query_default_state
from utils.clients.mongo_utils import get_db, get_documents_by_file
from utils.entity_utils import get_entities, get_entity_map, resolve_mentions
from utils.lm.reranker_utils import rerank
from utils.search_utils import semantic_search

CANDIDATES = 30  # 稠密、稀疏各取多少条候选
TOP_K = 20  # 融合后保留多少条进入精排
TOP_CHUNKS = 6  # 精排后取多少条切片作为证据
MAX_FACT_GROUPS = 8  # 财务事实最多取几组
# 比较行名与指标名时忽略空白、括号与百分号（“营业收入（元）”与“营业收入”视为同一指标）
ITEM_NOISE = re.compile(r"[\s()（）%％]")


@step_log("validate_and_get_data")
def validate_and_get_data(state: QueryGraphState):
    """
    取出并校验取证所需的入参
    :return: 元组 (独立问题, 提及的对象, 财务指标)
    :raise ValueError: 计划里没有独立问题（node_query_plan 未执行）
    """
    plan = state.get("plan") or {}
    standalone_query = plan.get("standalone_query", "")
    if not standalone_query:
        logger.error("no standalone_query found in plan")
        raise ValueError("no standalone_query found in plan")
    return standalone_query, plan.get("entity_mentions", []), plan.get("metrics", [])


@node_log("node_gather_evidence")
def node_gather_evidence(state: QueryGraphState):
    """
    节点功能：解析对象、取三类证据、精排并编号。
    上游 node_query_plan；下游 node_answer_output。
    澄清候选与库外对象只记录下来交给回答节点参考，不在这里决定走向。
    """
    standalone_query, mentions, metrics = validate_and_get_data(state)
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

    evidence = gather_evidence(standalone_query, entity_ids, doc_ids, metrics)
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
def gather_evidence(standalone_query: str, entity_ids: list, doc_ids: list, metrics: list) -> list:
    """
    取三类证据并编号：财务事实 → 文档摘要 → 文本切片
    :return: 证据列表（键见 utils/citation_utils.py）
    """
    docs = get_documents_by_file()
    doc_meta = get_doc_meta_by_id(docs)
    entity_of_doc = get_entity_of_doc(entity_ids, docs)

    evidence = []
    for fact in fact_lookup(get_company_ids(entity_ids), metrics):
        item = build_evidence("fact", fact["text"], doc_meta[fact["doc_id"]], entity_of_doc.get(fact["doc_id"]))
        item["page_start"] = fact["page_start"]
        item["page_end"] = fact["page_end"]
        evidence.append(item)
    for summary in fetch_summaries(doc_ids):
        text = "（文档摘要）" + summary["text"]
        evidence.append(build_evidence("summary", text, doc_meta[summary["doc_id"]],
                                       entity_of_doc.get(summary["doc_id"])))

    hits = semantic_search(standalone_query, top_k=TOP_K, candidates=CANDIDATES, doc_ids=doc_ids or None)
    for index in rerank_hits(standalone_query, hits):
        hit = hits[index]
        item = build_evidence("chunk", hit["text"], doc_meta[hit["doc_id"]], entity_of_doc.get(hit["doc_id"]))
        item["page_start"] = hit["page_start"]
        item["page_end"] = hit["page_end"]
        item["derived"] = hit["derived"]
        evidence.append(item)

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


def get_company_ids(entity_ids: list) -> list:
    """只保留上市公司（财务事实只从公司定期报告中抽取）"""
    entity_map = get_entity_map()
    return [i for i in entity_ids if entity_map[i]["type"] == "company"]


# ---------- 证据 ----------

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
        "derived": False,
    }


# ---------- 财务事实 ----------

def normalize_item(text: str) -> str:
    """去掉空白、括号与百分号，用于行名与指标名的比较"""
    return ITEM_NOISE.sub("", text)


def is_wanted_item(item: str, wanted: list) -> bool:
    """行名与任一指标名互相包含（都已规范化；空指标名跳过）"""
    for name in wanted:
        if name and (name in item or item in name):
            return True
    return False


def get_name_distance(item: str, wanted: list) -> int:
    """行名与指标名的长度差（取最小），越小说明越接近指标本身"""
    item_len = len(normalize_item(item))
    return min([abs(item_len - len(name)) for name in wanted])


def group_fact_rows(entity_ids: list, wanted: list) -> list:
    """
    查出匹配指标的事实行，按 (文档, 章节, 起始页, 行名) 分组：同一表格同一行的各列归为一组
    :return: [{"doc_id", "section", "page_start", "item", "rows"}]，按首次出现顺序
    """
    groups = {}
    for row in get_db().financial_facts.find({"entity_id": {"$in": entity_ids}}):
        if not is_wanted_item(normalize_item(row["item"]), wanted):
            continue
        key = (row["doc_id"], row["section"], row["page_start"], row["item"])
        if key not in groups:
            groups[key] = {
                "doc_id": row["doc_id"],
                "section": row["section"],
                "page_start": row["page_start"],
                "item": row["item"],
                "rows": [],
            }
        groups[key]["rows"].append(row)
    return list(groups.values())


def build_fact(group: dict) -> dict:
    """一组事实行合并成一条证据文本：行名：列=值；…（单位）〔章节〕"""
    rows = group["rows"]
    cells = []
    for row in rows:
        if row["column"]:
            cells.append(f"{row['column']}={row['value']}")
        else:
            cells.append(row["value"])
    text = f"{group['item']}：{'；'.join(cells)}"
    unit = rows[0]["unit"]
    if unit:
        text += f"（{unit}）"
    text += f"〔{group['section']}〕"
    return {
        "doc_id": group["doc_id"],
        "item": group["item"],
        "text": text,
        "page_start": group["page_start"],
        "page_end": rows[0]["page_end"],
    }


@step_log("fact_lookup")
def fact_lookup(entity_ids: list, metrics: list) -> list:
    """
    查找财务事实
    :param entity_ids: 上市公司实体 ID
    :param metrics: 计划里的指标名
    :return: [{doc_id, item, text, page_start, page_end}]，最多 MAX_FACT_GROUPS 条
    """
    if not entity_ids or not metrics:
        return []
    wanted = [normalize_item(m) for m in metrics if m.strip()]
    groups = group_fact_rows(entity_ids, wanted)
    # 行名越接近指标名越靠前（“营业收入”优先于“营业收入增长率”之类）；sorted 稳定，同分保持原顺序
    groups = sorted(groups, key=lambda group: get_name_distance(group["item"], wanted))
    facts = []
    for group in groups[:MAX_FACT_GROUPS]:
        facts.append(build_fact(group))
    return facts


# ---------- 文档摘要 ----------

@step_log("fetch_summaries")
def fetch_summaries(doc_ids: list) -> list:
    """
    读取文档摘要（导入时 node_enrich 生成；没有摘要的文档跳过）
    :return: [{"doc_id", "text"}]
    """
    if not doc_ids:
        return []
    docs = get_db().documents.find({"_id": {"$in": doc_ids}, "summary": {"$ne": None}}, {"summary": 1})
    return [{"doc_id": doc["_id"], "text": doc["summary"]} for doc in docs]


# ---------- 精排 ----------

@step_log("rerank_hits")
def rerank_hits(query: str, hits: list) -> list:
    """
    云端精排
    :return: 选中命中的下标，按相关度降序，最多 TOP_CHUNKS 个；精排失败或返回空结果时退回 RRF 排序
    """
    if not hits:
        return []
    fallback = list(range(min(TOP_CHUNKS, len(hits))))
    try:
        order = rerank(query, [hit["text"][:2000] for hit in hits], TOP_CHUNKS)
    except Exception as e:
        logger.warning(f"精排失败，使用 RRF 排序：{e}")
        return fallback
    if not order:
        logger.warning("精排返回空结果，使用 RRF 排序")
        return fallback
    return order


if __name__ == "__main__":
    # 运行：uv run python -m processor.query_processor.nodes.node_gather_evidence
    # 依赖：Mongo、Milvus、BGE-M3（CPU 首次加载约 20 秒）、百炼精排
    demo_plan = {"standalone_query": "贵州茅台一季度营业收入是多少？", "entity_mentions": ["贵州茅台"],
                 "metrics": ["营业收入"]}
    demo_state = create_query_default_state(session_id="demo-gather", plan=demo_plan)
    result = node_gather_evidence(demo_state)
    for item in result["evidence"]:
        print(item["eid"], item["kind"], item["file_name"], item["page_start"], item["text"][:40])
