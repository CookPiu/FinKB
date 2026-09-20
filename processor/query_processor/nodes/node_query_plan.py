"""
节点：查询规划
读会话；上一轮在等澄清且本轮回复能匹配候选时，由代码直接确定对象（不调用 LLM）；
否则一次 LLM 调用输出结构化的查询计划（JSON，代码补默认值并校验）。
"""
import json
import re

from common.logging.logger import logger, node_log, step_log
from processor.query_processor.state import QueryGraphState, create_query_default_state
from utils.clients.mongo_history_utils import get_history, load_session
from utils.entity_utils import get_entity_map, pick_option
from utils.lm.lm_utils import chat
from utils.load_prompt import load_prompt

INTENTS = [
    "product_info",
    "risk",
    "knowledge",
    "announcement",
    "market_policy",
    "process",
    "investment_advice",
    "realtime",
    "out_of_scope",
    "chitchat",
]
# wants_summary 可接受的"是 / 否"写法（与原 pydantic 布尔字段的宽松解析一致，不区分大小写）
TRUE_TEXTS = ("1", "on", "t", "true", "y", "yes")
FALSE_TEXTS = ("0", "off", "f", "false", "n", "no")


@step_log("validate_and_get_data")
def validate_and_get_data(state: QueryGraphState):
    """
    取出并校验规划所需的入参
    :return: 元组 (会话标识, 用户原问题)
    :raise ValueError: 会话标识或问题为空
    """
    session_id = state.get("session_id", "")
    question = state.get("original_query", "")
    if not session_id or not question:
        logger.error("no session_id or original_query found in state")
        raise ValueError("no session_id or original_query found in state")
    return session_id, question


@node_log("node_query_plan")
def node_query_plan(state: QueryGraphState):
    """
    节点功能：理解问题，产出查询计划 plan。
    - 上一轮在等澄清且本轮回复能对上候选：沿用上一轮的计划，把选中对象写进问题，标记 clarified，不调用 LLM；
    - 否则读取最近几轮历史与焦点对象，调用 LLM 规划（输出无法解析时退回知识类检索）。
    下游：node_entity_confirm。
    """
    session_id, question = validate_and_get_data(state)
    session = load_session(session_id)
    chosen = pick_pending_option(question, session["pending"])
    if chosen is not None:
        state["plan"] = build_clarified_plan(session["pending"]["plan"], chosen)
        state["entity_ids"] = [chosen["id"]]
        state["clarified"] = True
        state["focus_entity_ids"] = session["focus_entity_ids"]
        return state

    history = get_history(session_id)
    plan = plan_query(question, history, get_focus_names(session["focus_entity_ids"]))
    logger.info(f"plan={plan}")
    state["plan"] = plan
    state["history"] = history
    state["focus_entity_ids"] = session["focus_entity_ids"]
    return state


@step_log("pick_pending_option")
def pick_pending_option(question: str, pending):
    """
    上一轮在等澄清时，把本轮回复与候选对象匹配
    :param pending: 会话里的待澄清记录 {"options": 候选实体 ID, "plan": 上一轮计划}，没有则为 None
    :return: 选中的实体；没有待澄清或对不上时返回 None
    """
    if not pending:
        return None
    entity_map = get_entity_map()
    options = [entity_map[i] for i in pending["options"] if i in entity_map]
    return pick_option(question, options)


def build_clarified_plan(pending_plan: dict, chosen: dict) -> dict:
    """沿用上一轮的计划，把澄清选中的对象名写进问题"""
    plan = dict(pending_plan)
    plan["standalone_query"] = f"{chosen['name']}：{plan['standalone_query']}"
    return plan


def get_focus_names(focus_entity_ids: list) -> list:
    """焦点对象的名称（实体表里已不存在的 ID 跳过）"""
    entity_map = get_entity_map()
    return [entity_map[i]["name"] for i in focus_entity_ids if i in entity_map]


def format_history(history: list) -> str:
    """历史问答压成逐行文本，每条最多 200 字"""
    lines = []
    for message in history:
        if message["role"] == "user":
            who = "用户"
        else:
            who = "助手"
        lines.append(f"{who}：{message['text'][:200]}")
    if not lines:
        return "（无）"
    return "\n".join(lines)


def build_plan_messages(question: str, history: list, focus_names: list) -> list:
    """规划提示词：系统提示 + 历史、焦点对象与当前问题"""
    focus_text = "、".join(focus_names)
    if not focus_text:
        focus_text = "（无）"
    user = (
        f"对话历史：\n{format_history(history)}\n\n"
        f"当前焦点对象：{focus_text}\n\n"
        f"当前问题：{question}"
    )
    return [{"role": "system", "content": load_prompt("query_plan")}, {"role": "user", "content": user}]


def build_default_plan(standalone_query: str) -> dict:
    """默认计划：知识类问题、无实体、无指标、不要摘要"""
    return {
        "standalone_query": standalone_query,
        "intent": "knowledge",
        "entity_mentions": [],
        "metrics": [],
        "wants_summary": False,
    }


def extract_json_text(text: str) -> str:
    """取模型输出中第一个 { 到最后一个 } 之间的内容（模型偶尔在 JSON 前后加说明）"""
    match = re.search(r"\{.*\}", text, re.S)
    if match:
        return match.group(0)
    return text


def parse_str_list(value, name: str) -> list:
    """校验字符串列表字段"""
    if not isinstance(value, list):
        raise ValueError(f"{name} 不是列表：{value}")
    for item in value:
        if not isinstance(item, str):
            raise ValueError(f"{name} 含非字符串元素：{item}")
    return value


def parse_bool(value, name: str) -> bool:
    """校验布尔字段：接受 true/false、0/1 与常见的是否字符串"""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return value == 1
    if isinstance(value, str) and value.lower() in TRUE_TEXTS:
        return True
    if isinstance(value, str) and value.lower() in FALSE_TEXTS:
        return False
    raise ValueError(f"{name} 不是布尔值：{value}")


@step_log("parse_plan")
def parse_plan(text: str) -> dict:
    """
    把 LLM 输出解析成查询计划：缺省字段补默认值，取值不合法时报错
    :param text: 模型输出
    :return: 计划 dict（键见 state.py 模块说明）
    :raise ValueError: 不是 JSON 对象、standalone_query 缺失或不是字符串、intent 不在取值范围、列表或布尔字段类型不对
    """
    data = json.loads(extract_json_text(text))
    if not isinstance(data, dict):
        raise ValueError("规划结果不是 JSON 对象")
    standalone_query = data.get("standalone_query")
    if not isinstance(standalone_query, str):
        raise ValueError(f"standalone_query 缺失或不是字符串：{standalone_query}")
    intent = data.get("intent", "knowledge")
    if intent not in INTENTS:
        raise ValueError(f"intent 取值不合法：{intent}")
    plan = build_default_plan(standalone_query)
    plan["intent"] = intent
    plan["entity_mentions"] = parse_str_list(data.get("entity_mentions", []), "entity_mentions")
    plan["metrics"] = parse_str_list(data.get("metrics", []), "metrics")
    plan["wants_summary"] = parse_bool(data.get("wants_summary", False), "wants_summary")
    return plan


@step_log("plan_query")
def plan_query(question: str, history: list, focus_names: list) -> dict:
    """
    调用 LLM 生成查询计划
    :return: 计划 dict；模型输出无法解析或字段不合法时退回知识类检索（standalone_query 取原问题）
    """
    messages = build_plan_messages(question, history, focus_names)
    try:
        raw = chat(messages, json_mode=True)
        return parse_plan(raw)
    except ValueError as e:
        logger.warning(f"规划结果无法解析，退回为知识类检索：{e}")
        return build_default_plan(question)


if __name__ == "__main__":
    # 运行：uv run python -m processor.query_processor.nodes.node_query_plan
    # 依赖：Mongo（读会话）、百炼 LLM
    demo_state = create_query_default_state(session_id="demo-query-plan", original_query="茅台一季度营收多少？")
    result = node_query_plan(demo_state)
    print(result["plan"])
