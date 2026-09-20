"""
节点：财务事实查找
按实体与指标名匹配 financial_facts，同一表格同一行的各列合并为一条证据。
"""
import re

from common.logging.logger import logger, node_log, step_log
from processor.query_processor.state import QueryGraphState, create_query_default_state
from utils.clients.mongo_utils import get_db
from utils.entity_utils import get_entity_map

MAX_FACT_GROUPS = 8
# 比较行名与指标名时忽略空白、括号与百分号（"营业收入（元）"与"营业收入"视为同一指标）
ITEM_NOISE = re.compile(r"[\s()（）%％]")


@step_log("validate_and_get_data")
def validate_and_get_data(state: QueryGraphState):
    """
    取出并校验事实查找所需的入参
    :return: 元组 (已确认实体 ID, 计划里的财务指标)
    :raise ValueError: 计划缺失（正常路由下本节点只在计划有 metrics 时触发）
    """
    plan = state.get("plan") or {}
    if not plan:
        logger.error("no plan found in state")
        raise ValueError("no plan found in state")
    return state.get("entity_ids", []), plan.get("metrics", [])


@node_log("node_fact_lookup")
def node_fact_lookup(state: QueryGraphState):
    """
    节点功能：按计划里的财务指标查找上市公司的财务事实。
    计划有 metrics 且涉及上市公司时由路由触发，与 node_search_embedding / node_summary_fetch 并行。
    下游：node_rerank（事实排在证据最前面）。
    """
    entity_ids, metrics = validate_and_get_data(state)
    facts = fact_lookup(get_company_ids(entity_ids), metrics)
    # 并行节点只返回自己写的键：整状态返回会与同一超步的其他节点冲突（InvalidUpdateError）
    return {"facts": facts}


def get_company_ids(entity_ids: list) -> list:
    """只保留上市公司（财务事实只从公司定期报告中抽取）"""
    entity_map = get_entity_map()
    return [i for i in entity_ids if entity_map[i]["type"] == "company"]


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
    # 行名越接近指标名越靠前（"营业收入"优先于"营业收入增长率"之类）；sorted 稳定，同分保持原顺序
    groups = sorted(groups, key=lambda group: get_name_distance(group["item"], wanted))
    facts = []
    for group in groups[:MAX_FACT_GROUPS]:
        facts.append(build_fact(group))
    return facts


if __name__ == "__main__":
    # 运行：uv run python -m processor.query_processor.nodes.node_fact_lookup
    # 依赖：Mongo（financial_facts）
    demo_plan = {"standalone_query": "贵州茅台一季度营业收入是多少？", "intent": "announcement",
                 "entity_mentions": ["贵州茅台"], "metrics": ["营业收入"], "wants_summary": False}
    demo_state = create_query_default_state(session_id="demo-fact-lookup", plan=demo_plan, entity_ids=["maotai"])
    for fact in node_fact_lookup(demo_state)["facts"]:
        print(fact["text"])
