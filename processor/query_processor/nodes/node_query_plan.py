"""节点：查询规划。读会话；上一轮在等澄清且本轮回复能匹配候选时，由代码直接确定对象（不调用 LLM）；
否则一次 LLM 调用输出结构化的 QueryPlan（pydantic 校验）。"""

from __future__ import annotations

import json
import re

from pydantic import ValidationError

from common.logging.logger import logger, node_log
from processor.query_processor.state import QueryGraphState, QueryPlan
from utils.clients import mongo_history_utils as session
from utils.entity_utils import registry
from utils.lm import lm_utils as llm
from utils.load_prompt import load_prompt


def _format_history(history: list[dict]) -> str:
    lines = []
    for m in history:
        who = "用户" if m["role"] == "user" else "助手"
        lines.append(f"{who}：{m['text'][:200]}")
    return "\n".join(lines) or "（无）"


def _parse(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    return json.loads(m.group(0) if m else text)


def plan_query(question: str, history: list[dict], focus_names: list[str]) -> QueryPlan:
    user = (
        f"对话历史：\n{_format_history(history)}\n\n"
        f"当前焦点对象：{'、'.join(focus_names) or '（无）'}\n\n"
        f"当前问题：{question}"
    )
    messages = [{"role": "system", "content": load_prompt("query_plan")}, {"role": "user", "content": user}]
    try:
        raw = llm.chat(messages, response_format={"type": "json_object"})
        return QueryPlan.model_validate(_parse(raw))
    except (ValidationError, ValueError) as e:
        logger.warning("规划结果无法解析，退回为知识类检索：%s", e)
        return QueryPlan(standalone_query=question)


@node_log("node_query_plan")
def node_query_plan(state: QueryGraphState) -> dict:
    reg = registry()
    sid, question = state["session_id"], state["original_query"]
    sess = session.load(sid)
    pending = sess["pending"]
    if pending:
        options = [reg.by_id[i] for i in pending["options"] if i in reg.by_id]
        chosen = reg.pick_option(question, options)
        if chosen is not None:
            plan = dict(pending["plan"])
            plan["standalone_query"] = f"{chosen.name}：{plan['standalone_query']}"  # 把澄清结果写进问题
            return {"plan": plan, "entity_ids": [chosen.id], "clarified": True,
                    "focus_entity_ids": sess["focus_entity_ids"]}

    focus = [reg.by_id[i] for i in sess["focus_entity_ids"] if i in reg.by_id]
    history = session.history(sid)
    plan = plan_query(question, history, [e.name for e in focus])
    logger.info("plan=%s", plan.model_dump())
    return {"plan": plan.model_dump(), "history": history, "focus_entity_ids": sess["focus_entity_ids"]}
