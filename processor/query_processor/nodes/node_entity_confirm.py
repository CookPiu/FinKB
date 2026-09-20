"""
节点：实体确认与路由
按规划结果解析实体（代码精确 → 最长别名），决定走向：
- 投资建议 / 实时行情 / 寒暄 / 库外问题 / 有歧义需澄清 / 提到库外公司或产品：直接写入固定话术（answer），
  后续跳到 node_answer_output，不检索也不生成；
- 其余：记下实体对应的文档，交给取证节点。
"""
from common.answer_templates import CHITCHAT, DECLINE_ADVICE, OUT_OF_SCOPE, REALTIME, REFUSE, clarify
from common.logging.logger import logger, node_log
from processor.query_processor.state import QueryGraphState, create_query_default_state
from utils.clients.mongo_utils import get_documents_by_file
from utils.entity_utils import get_entities, get_entity_map, resolve_mentions

# 省略主语的追问（"它的管理费呢"）沿用上一轮焦点对象的意图
FOCUS_INTENTS = {"product_info", "risk", "announcement", "investment_advice"}
# 必须落到库内具体对象才能回答的意图：提到的对象不在库里时直接拒答
ENTITY_REQUIRED_INTENTS = {"product_info", "risk", "announcement"}
# 固定话术路由 → 话术（clarify 需要候选名称，单独生成）
DIRECT_TEXTS = {
    "decline_advice": DECLINE_ADVICE,
    "realtime_notice": REALTIME,
    "chitchat": CHITCHAT,
    "out_of_scope": OUT_OF_SCOPE,
    "refuse": REFUSE,
}


@node_log("node_entity_confirm")
def node_entity_confirm(state: QueryGraphState):
    """
    节点功能：确认问题涉及的实体并决定走向。
    - 上一轮澄清已由 node_query_plan 确定对象（clarified）：直接沿用，不再路由到固定话术；
    - 否则解析计划里的实体提及，按意图与解析结果选择路由；非检索路由写入 kind / answer。
    下游：main_graph.route_after_entity_confirm（answer 非空 → node_answer_output，否则并行取证）。
    """
    plan = state["plan"]
    route = "retrieve"
    candidates = []
    if state.get("clarified"):
        entity_ids = state["entity_ids"]
    else:
        resolution = resolve_mentions(plan["entity_mentions"], get_entities())
        entity_ids = get_resolved_ids(resolution, state.get("focus_entity_ids"), plan["intent"])
        candidates = resolution["candidates"]
        route = choose_route(plan["intent"], resolution["status"])
        logger.info(f"entities={entity_ids or resolution['unknown_mentions']} route={route}")

    state["entity_ids"] = entity_ids
    state["candidate_ids"] = [entity["id"] for entity in candidates]
    state["doc_ids"] = get_doc_ids(entity_ids)
    if route != "retrieve":
        kind, answer = build_direct_answer(route, [entity["name"] for entity in candidates])
        state["kind"] = kind
        state["answer"] = answer
    return state


def get_resolved_ids(resolution: dict, focus_entity_ids, intent: str) -> list:
    """已确认的实体 ID；问题没提到任何实体时，符合条件的追问沿用上一轮的焦点对象"""
    entity_ids = [entity["id"] for entity in resolution["entities"]]
    if resolution["status"] == "none" and focus_entity_ids and intent in FOCUS_INTENTS:
        entity_map = get_entity_map()
        entity_ids = [i for i in focus_entity_ids if i in entity_map]
    return entity_ids


def choose_route(intent: str, status: str) -> str:
    """
    按意图与实体解析状态选择路由
    :return: retrieve（检索作答）或固定话术路由名
    """
    if intent == "investment_advice":
        return "decline_advice"
    if intent == "realtime":
        return "realtime_notice"
    if intent == "chitchat":
        return "chitchat"
    if intent == "out_of_scope":
        return "out_of_scope"
    if status == "ambiguous":
        return "clarify"
    if status == "unknown" and intent in ENTITY_REQUIRED_INTENTS:
        # 问的是库外公司/产品
        return "refuse"
    return "retrieve"


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


def build_direct_answer(route: str, candidate_names: list):
    """
    固定话术
    :return: 元组 (kind, answer)；库外问题的 kind 记为 refuse
    """
    if route == "clarify":
        return "clarify", clarify(candidate_names)
    text = DIRECT_TEXTS[route]
    if route == "out_of_scope":
        return "refuse", text
    return route, text


if __name__ == "__main__":
    # 运行：uv run python -m processor.query_processor.nodes.node_entity_confirm
    # 依赖：Mongo（读取 documents）
    demo_plan = {"standalone_query": "华夏的基金管理费多少？", "intent": "product_info",
                 "entity_mentions": ["华夏"], "metrics": [], "wants_summary": False}
    demo_state = create_query_default_state(session_id="demo-entity-confirm", plan=demo_plan)
    result = node_entity_confirm(demo_state)
    print(result["kind"], result["candidate_ids"])
    print(result["answer"])
