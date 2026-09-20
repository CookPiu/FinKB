"""
节点：答案输出
- 前序节点已写入固定话术（澄清 / 拒答 / 投资建议 / 实时 / 寒暄）时直接输出；
- 否则按证据流式生成，按句过合规守卫；模型输出【依据不足】时改为固定拒答话术；
- 代码渲染引用、追加风险提示与时效提示，保存会话（焦点对象、待澄清候选、问答记录）。
"""
import re

from common.answer_templates import CHITCHAT, REALTIME_NOTICE, REFUSE, RISK_NOTICE
from common.logging.logger import logger, node_log, step_log
from processor.query_processor.state import QueryGraphState, create_query_default_state
from utils.citation_utils import build_prompt_block, build_source, finalize_citations, render_sources
from utils.clients.mongo_history_utils import delete_sessions, save_turn
from utils.entity_utils import get_entity_map
from utils.guard_utils import StreamGuard
from utils.lm.lm_utils import chat_stream
from utils.load_prompt import load_prompt
from utils.stream_utils import emit

PRODUCT_TYPES = {"fund", "wealth_product"}
REALTIME_WORDS = re.compile(r"最新|今天|今日|实时|现在|目前|当前")
INSUFFICIENT = "【依据不足】"
# 附加提示只加在这两类回答后面
NOTICE_KINDS = ("answer", "decline_advice")
RISK_INTENTS = ("product_info", "risk", "investment_advice")


@step_log("validate_and_get_data")
def validate_and_get_data(state: QueryGraphState):
    """
    取出并校验输出所需的入参
    :return: 元组 (用户原问题, 查询计划, 证据列表)
    :raise ValueError: 问题为空，或既没有现成话术也没有证据
    """
    question = state.get("original_query", "")
    plan = state.get("plan") or {}
    evidence = state.get("evidence") or []
    if not question:
        logger.error("no original_query found in state")
        raise ValueError("no original_query found in state")
    # 两条进入路径：上游写好了固定话术，或 node_rerank 给出了证据
    if not state.get("answer") and not evidence:
        logger.error("no answer and no evidence found in state, cannot perform answer output.")
        raise ValueError("no answer and no evidence found in state, cannot perform answer output.")
    return question, plan, evidence


@node_log("node_answer_output")
def node_answer_output(state: QueryGraphState):
    """
    节点功能：输出最终回答并保存会话。
    有两条进入路径：上游已写入固定话术（node_entity_confirm / node_rerank）时直接推送；
    否则（node_rerank 给出了证据）流式生成。随后校正引用、追加提示、推送来源，保存本轮问答，最后推送 final 事件。
    """
    question, plan, evidence = validate_and_get_data(state)
    guard_hits = []
    if state.get("answer"):
        # 固定话术
        kind = state["kind"]
        answer = state["answer"]
        emit({"type": "delta", "text": answer})
    else:
        kind, answer, guard_hits = generate_answer(question, plan["standalone_query"], evidence)

    sources = []
    if kind == "answer":
        answer, sources = finalize_citations(answer, evidence)
    notices = build_notices(kind, question, plan, state.get("entity_ids", []))
    if notices:
        emit({"type": "delta", "text": "\n\n" + "\n".join(notices)})
        answer = answer + "\n\n" + "\n".join(notices)
    source_dicts = [build_source(item) for item in sources]
    if sources:
        emit({"type": "sources", "sources": source_dicts, "text": render_sources(sources)})

    save_session_turn(state, kind, answer, plan, source_dicts)
    emit({"type": "final", "kind": kind})
    state["kind"] = kind
    state["answer"] = answer
    state["sources"] = source_dicts
    state["guard_hits"] = guard_hits
    # 只有模型生成的回答才保留证据（评测用来判断证据是否命中）
    if kind != "answer":
        state["evidence"] = []
    return state


def build_answer_messages(standalone_query: str, evidence: list) -> list:
    """回答提示词：系统规则 + 编号证据 + 问题"""
    blocks = [build_prompt_block(item) for item in evidence]
    context = "\n\n".join(blocks)
    return [
        {"role": "system", "content": load_prompt("answer_system")},
        {"role": "user", "content": f"证据：\n{context}\n\n问题：{standalone_query}"},
    ]


def stream_answer(messages: list, guard: StreamGuard) -> str:
    """
    流式调用模型，过守卫后逐句推送 delta 事件
    开头先攒几个字，确认不是【依据不足】再开始推送；是【依据不足】则停止接收
    :return: 模型原始输出（未经守卫替换）
    """
    raw = ""
    held = True
    for delta in chat_stream(messages):
        raw += delta
        if held:
            if len(raw) < 6 and INSUFFICIENT.startswith(raw.strip()):
                continue
            if raw.strip().startswith("【依据不足"):
                break
            held = False
            delta = raw
        out = guard.feed(delta).replace(INSUFFICIENT, "")
        if out:
            emit({"type": "delta", "text": out})
    return raw


def apply_guard(raw: str, question: str) -> str:
    """对完整文本重新过一遍守卫，得到与推送给用户一致的文本"""
    guard = StreamGuard(context=question)
    return guard.feed(raw) + guard.flush()


@step_log("generate_answer")
def generate_answer(question: str, standalone_query: str, evidence: list):
    """
    按证据流式生成回答并推送
    :param question: 用户原问题（守卫判断语境用）
    :param standalone_query: 补全指代后的问题（交给模型）
    :param evidence: 证据列表
    :return: 元组 (kind, answer, guard_hits)
    """
    guard = StreamGuard(context=question)
    raw = stream_answer(build_answer_messages(standalone_query, evidence), guard)
    if raw.strip().startswith("【依据不足"):
        emit({"type": "delta", "text": REFUSE})
        return "refuse", REFUSE, []
    tail = guard.flush().replace(INSUFFICIENT, "")
    if tail:
        emit({"type": "delta", "text": tail})
    if guard.hits:
        logger.warning(f"合规守卫拦截：{guard.hits}")
    # 模型在部分回答末尾追加的【依据不足】标记去掉
    answer = apply_guard(raw, question).replace(INSUFFICIENT, "").rstrip()
    return "answer", answer, guard.hits


@step_log("build_notices")
def build_notices(kind: str, question: str, plan: dict, entity_ids: list) -> list:
    """
    回答末尾的附加提示
    - 风险提示：产品类 / 风险类 / 投资建议类问题，或涉及基金、理财产品；
    - 时效提示：问题里有"最新""今天"等时效词。
    """
    notices = []
    if kind not in NOTICE_KINDS:
        return notices
    if plan.get("intent") in RISK_INTENTS or has_product_entity(entity_ids):
        notices.append(RISK_NOTICE)
    if REALTIME_WORDS.search(question):
        notices.append(REALTIME_NOTICE)
    return notices


def has_product_entity(entity_ids: list) -> bool:
    """是否涉及基金或理财产品（实体表里已不存在的 ID 跳过）"""
    entity_map = get_entity_map()
    for entity_id in entity_ids:
        if entity_id in entity_map and entity_map[entity_id]["type"] in PRODUCT_TYPES:
            return True
    return False


@step_log("save_session_turn")
def save_session_turn(state: QueryGraphState, kind: str, answer: str, plan: dict, sources: list):
    """
    保存本轮问答与会话状态
    澄清时保留原焦点并记下候选（下一轮由 node_query_plan 匹配）；其他情况焦点更新为本轮对象（没有则沿用）
    """
    focus = state.get("focus_entity_ids") or []
    if kind != "clarify" and state.get("entity_ids"):
        focus = state["entity_ids"]
    pending = None
    if kind == "clarify":
        pending = {"options": state["candidate_ids"], "plan": plan}
    save_turn(state["session_id"], state["original_query"], answer,
              kind=kind, plan=plan, sources=sources, focus_entity_ids=focus, pending=pending)


if __name__ == "__main__":
    # 运行：uv run python -m processor.query_processor.nodes.node_answer_output
    # 依赖：Mongo（写入会话后立即删除）；走固定话术路径，不调用 LLM
    demo_state = create_query_default_state(session_id="demo-answer-output", original_query="你好",
                                            plan={"intent": "chitchat"}, kind="chitchat", answer=CHITCHAT)
    result = node_answer_output(demo_state)
    print(result["kind"], result["answer"])
    delete_sessions(["demo-answer-output"])
