"""节点：实体确认与路由。

按规划结果解析实体（代码精确 → 最长别名），决定走向：
- 投资建议 / 实时行情 / 寒暄 / 库外问题 / 有歧义需澄清 / 提到库外公司或产品：直接写入固定话术（answer），
  后续跳到 node_answer_output，不检索也不生成；
- 其余：记下实体对应的文档，交给取证节点。
"""

from __future__ import annotations

from common import answer_templates as templates
from common.logging.logger import logger, node_log
from processor.query_processor.state import QueryGraphState, QueryPlan
from utils.clients import mongo_utils as mongo
from utils.entity_utils import registry

FOCUS_INTENTS = {"product_info", "risk", "announcement", "investment_advice"}
ENTITY_REQUIRED_INTENTS = {"product_info", "risk", "announcement"}


def _direct_answer(route: str, candidate_names: list[str]) -> tuple[str, str]:
    """固定话术路由 → (kind, answer)。"""
    if route == "clarify":
        return "clarify", templates.clarify(candidate_names)
    text = {
        "decline_advice": templates.DECLINE_ADVICE,
        "realtime_notice": templates.REALTIME,
        "chitchat": templates.CHITCHAT,
        "out_of_scope": templates.OUT_OF_SCOPE,
        "refuse": templates.REFUSE,
    }[route]
    return ("refuse" if route == "out_of_scope" else route), text


@node_log("node_entity_confirm")
def node_entity_confirm(state: QueryGraphState) -> dict:
    reg = registry()
    plan = QueryPlan.model_validate(state["plan"])
    route, candidates = "retrieve", []

    if state.get("clarified"):
        entity_ids = state["entity_ids"]
    else:
        res = reg.resolve(plan.entity_mentions)
        entity_ids = [e.id for e in res.entities]
        candidates = res.candidates
        if res.status == "none" and state.get("focus_entity_ids") and plan.intent in FOCUS_INTENTS:
            entity_ids = [i for i in state["focus_entity_ids"] if i in reg.by_id]  # 省略主语的追问沿用焦点对象
        if plan.intent == "investment_advice":
            route = "decline_advice"
        elif plan.intent == "realtime":
            route = "realtime_notice"
        elif plan.intent == "chitchat":
            route = "chitchat"
        elif plan.intent == "out_of_scope":
            route = "out_of_scope"
        elif res.status == "ambiguous":
            route = "clarify"
        elif res.status == "unknown" and plan.intent in ENTITY_REQUIRED_INTENTS:
            route = "refuse"  # 问的是库外公司/产品
        logger.info("entities=%s route=%s", entity_ids or res.unknown_mentions, route)

    docs = mongo.documents_by_file()
    doc_ids = [docs[f]["_id"] for i in entity_ids for f in reg.by_id[i].files if f in docs]
    update = {"entity_ids": entity_ids, "candidate_ids": [e.id for e in candidates], "doc_ids": doc_ids}
    if route != "retrieve":
        kind, answer = _direct_answer(route, [e.name for e in candidates])
        update |= {"kind": kind, "answer": answer}
    return update
