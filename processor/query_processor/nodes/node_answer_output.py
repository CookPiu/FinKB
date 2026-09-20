"""
节点：判定与输出
一次 LLM 调用完成两件事：判断这个问题该怎么处理（kind），并按证据写出正文；元信息（kind、引用编号、提示语标记）
写在正文之后的 <<<META>>> 行里。代码只做三件事：禁语兜底、引用校验、按标记追加固定提示语。
上游已经短路成固定拒答（一条证据都没有）时，本节点不调用模型。
"""
from common.answer_templates import (
    CHITCHAT,
    DECLINE_ADVICE,
    REALTIME,
    REALTIME_NOTICE,
    REFUSE,
    RISK_NOTICE,
    clarify,
)
from common.logging.logger import logger, node_log, step_log
from processor.query_processor.state import QueryGraphState, create_query_default_state
from utils.citation_utils import build_prompt_block, build_source, finalize_citations, render_sources
from utils.clients.mongo_history_utils import delete_sessions, save_turn
from utils.guard_utils import StreamGuard
from utils.lm.lm_utils import chat_stream
from utils.load_prompt import load_prompt
from utils.model_output_utils import META_MARK, parse_meta, split_body_and_meta
from utils.stream_utils import emit

# 提示语标记 → 固定话术
NOTICE_TEXTS = {"risk": RISK_NOTICE, "realtime": REALTIME_NOTICE}
# 澄清格式示例，写进提示词让模型照着写
CLARIFY_EXAMPLE = clarify(["对象甲", "对象乙"])


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
    # 两条进入路径：上游短路成固定拒答，或 node_gather_evidence 给出了证据
    if not state.get("answer") and not evidence:
        logger.error("no answer and no evidence found in state, cannot perform answer output.")
        raise ValueError("no answer and no evidence found in state, cannot perform answer output.")
    return question, plan, evidence


@node_log("node_answer_output")
def node_answer_output(state: QueryGraphState):
    """
    节点功能：判定 + 生成 + 兜底 + 落库。
    上游 node_gather_evidence；查询图的最后一个节点。
    """
    question, plan, evidence = validate_and_get_data(state)
    guard_hits = []
    notices = []
    if state.get("answer"):
        # 上游已短路（没有任何证据）：直接推送固定话术
        kind = state["kind"]
        answer = state["answer"]
        emit({"type": "delta", "text": answer})
    else:
        kind, answer, notices, guard_hits = generate_answer(question, plan, evidence, state)

    sources = []
    if kind == "answer":
        answer, sources = finalize_citations(answer, evidence)
    answer = append_notices(answer, notices)
    source_dicts = [build_source(item) for item in sources]
    if sources:
        emit({"type": "sources", "sources": source_dicts, "text": render_sources(sources)})

    save_session_turn(state, kind, answer, plan, source_dicts)
    emit({"type": "final", "kind": kind})
    state["kind"] = kind
    state["answer"] = answer
    state["sources"] = source_dicts
    state["notices"] = notices
    state["guard_hits"] = guard_hits
    # 只有依据证据作答时才保留证据（评测用来判断证据是否命中）
    if kind != "answer":
        state["evidence"] = []
    return state


def build_answer_messages(question: str, plan: dict, evidence: list, state: QueryGraphState) -> list:
    """
    回答提示词：规则手册（system）+ 证据、对象情况、问题（user）
    对象情况由代码解析后给出：候选对象供模型决定是否澄清，库外对象供模型决定是否拒答
    """
    system = load_prompt(
        "answer_system",
        refuse=REFUSE,
        decline_advice=DECLINE_ADVICE,
        realtime=REALTIME,
        chitchat=CHITCHAT,
        clarify_example=CLARIFY_EXAMPLE,
    )
    blocks = [build_prompt_block(item) for item in evidence]
    context = "\n\n".join(blocks) or "（没有检索到任何证据）"
    standalone_query = plan.get("standalone_query") or question
    parts = [f"证据：\n{context}", build_entity_note(state), f"问题：{standalone_query}"]
    user = "\n\n".join([part for part in parts if part])
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_entity_note(state: QueryGraphState) -> str:
    """把代码解析出的对象情况告诉模型"""
    lines = []
    if state.get("candidate_names"):
        lines.append("候选对象（问题指向不明确，需要请用户确认）：" + "、".join(state["candidate_names"]))
    if state.get("unknown_mentions"):
        lines.append("以下对象不在知识库中：" + "、".join(state["unknown_mentions"]))
    return "\n".join(lines)


def stream_body(messages: list, guard: StreamGuard) -> str:
    """
    流式调用模型：<<<META>>> 之前的正文过守卫后逐段推送，之后的元信息只收集不推送
    :return: 模型原始输出（未经守卫替换）
    """
    raw = ""
    fed_len = 0  # 已交给守卫处理的正文长度
    in_meta = False
    for delta in chat_stream(messages):
        raw += delta
        if in_meta:
            continue
        out = ""
        if META_MARK in raw:
            in_meta = True
            body = raw.split(META_MARK, 1)[0]
            out = guard.feed(body[fed_len:]) + guard.flush()
            fed_len = len(body)
        else:
            # 分隔符可能被拆进两个分片，末尾留出一段先不处理，等下一片拼上再判断
            safe_end = max(len(raw) - len(META_MARK), fed_len)
            out = guard.feed(raw[fed_len:safe_end])
            fed_len = safe_end
        if out:
            emit({"type": "delta", "text": out})
    if not in_meta:
        out = guard.feed(raw[fed_len:]) + guard.flush()
        if out:
            emit({"type": "delta", "text": out})
    return raw


def apply_guard(text: str, question: str):
    """对完整正文重新过一遍守卫，得到与推送给用户一致的文本"""
    guard = StreamGuard(context=question)
    return guard.feed(text) + guard.flush(), guard.hits


@step_log("generate_answer")
def generate_answer(question: str, plan: dict, evidence: list, state: QueryGraphState):
    """
    一次调用完成判定与生成
    :return: 元组 (kind, answer, notices, guard_hits)
    """
    guard = StreamGuard(context=question)
    raw = stream_body(build_answer_messages(question, plan, evidence, state), guard)
    body, meta_text = split_body_and_meta(raw)
    meta = parse_meta(meta_text, body)
    answer, guard_hits = apply_guard(body, question)
    if guard_hits:
        logger.warning(f"合规守卫拦截：{guard_hits}")
    if not answer.strip():
        # 模型只输出了元信息：退回固定拒答
        logger.warning("模型没有输出正文，改为固定拒答")
        return "refuse", REFUSE, [], guard_hits
    logger.info(f"kind={meta['kind']} cited={meta['cited']} notices={meta['notices']} meta_ok={meta['meta_ok']}")
    return meta["kind"], answer, meta["notices"], guard_hits


@step_log("append_notices")
def append_notices(answer: str, notices: list) -> str:
    """按模型给出的标记追加固定提示语：加不加由模型判断，话术由代码给"""
    texts = []
    for name in notices:
        if name in NOTICE_TEXTS and NOTICE_TEXTS[name] not in answer:
            texts.append(NOTICE_TEXTS[name])
    if not texts:
        return answer
    emit({"type": "delta", "text": "\n\n" + "\n".join(texts)})
    return answer + "\n\n" + "\n".join(texts)


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
    if kind == "clarify" and state.get("candidate_ids"):
        pending = {"options": state["candidate_ids"], "plan": plan}
    save_turn(state["session_id"], state["original_query"], answer,
              kind=kind, plan=plan, sources=sources, focus_entity_ids=focus, pending=pending)


if __name__ == "__main__":
    # 运行：uv run python -m processor.query_processor.nodes.node_answer_output
    # 依赖：Mongo（写入会话后立即删除）；走无证据短路路径，不调用 LLM
    demo_state = create_query_default_state(session_id="demo-answer-output", original_query="你好",
                                            plan={"standalone_query": "你好"}, kind="refuse", answer=REFUSE)
    result = node_answer_output(demo_state)
    print(result["kind"], result["answer"][:40])
    delete_sessions(["demo-answer-output"])
