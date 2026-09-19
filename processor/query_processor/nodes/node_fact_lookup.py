"""节点：财务事实查找。按实体与指标名匹配 financial_facts，同一表格同一行的各列合并为一条证据。"""

from __future__ import annotations

import re
from collections import defaultdict

from common.logging.logger import node_log
from processor.query_processor.state import QueryGraphState, QueryPlan
from utils.clients import mongo_utils as mongo
from utils.entity_utils import registry

MAX_FACT_GROUPS = 8
_NORM = re.compile(r"[\s()（）%％]")


def _norm(s: str) -> str:
    return _NORM.sub("", s)


def fact_lookup(entity_ids: list[str], metrics: list[str]) -> list[dict]:
    """返回 [{doc_id, item, text, page_start, page_end}]。"""
    if not entity_ids or not metrics:
        return []
    wanted = [_norm(m) for m in metrics if m.strip()]
    rows: dict[tuple, list[dict]] = defaultdict(list)
    for f in mongo.get_db().financial_facts.find({"entity_id": {"$in": entity_ids}}):
        item = _norm(f["item"])
        if any(w in item or item in w for w in wanted if w):
            rows[(f["doc_id"], f["section"], f["page_start"], f["item"])].append(f)
    # 行名越接近指标名越靠前（“营业收入”优先于“营业收入增长率”之类）
    ordered = sorted(rows.items(), key=lambda kv: min(abs(len(_norm(kv[0][3])) - len(w)) for w in wanted))
    out = []
    for (doc_id, section, page, item), fs in ordered[:MAX_FACT_GROUPS]:
        cells = "；".join(f"{f['column']}={f['value']}" if f["column"] else f["value"] for f in fs)
        unit = fs[0]["unit"]
        out.append(
            {
                "doc_id": doc_id,
                "item": item,
                "text": f"{item}：{cells}" + (f"（{unit}）" if unit else "") + f"〔{section}〕",
                "page_start": page,
                "page_end": fs[0]["page_end"],
            }
        )
    return out


@node_log("node_fact_lookup")
def node_fact_lookup(state: QueryGraphState) -> dict:
    plan = QueryPlan.model_validate(state["plan"])
    reg = registry()
    companies = [i for i in state.get("entity_ids", []) if reg.by_id[i].type == "company"]
    return {"facts": fact_lookup(companies, plan.metrics)}
