"""节点：答案输出。

- 前序节点已写入固定话术（澄清 / 拒答 / 投资建议 / 实时 / 寒暄）时直接输出；
- 否则按证据流式生成，按句过合规守卫；模型输出【依据不足】时改为固定拒答话术；
- 代码渲染引用、追加风险提示与时效提示，保存会话（焦点对象、待澄清候选、问答记录）。
"""

from __future__ import annotations

import re

from common import answer_templates as templates
from common.config.settings import get_settings
from common.logging.logger import logger, node_log
from common.models.evidence import Evidence
from processor.query_processor.state import QueryGraphState, QueryPlan
from utils.citation_utils import finalize_citations, render_sources
from utils.clients import mongo_history_utils as session
from utils.entity_utils import registry
from utils.guard_utils import StreamGuard
from utils.lm import lm_utils as llm
from utils.load_prompt import load_prompt
from utils.stream_utils import emit

PRODUCT_TYPES = {"fund", "wealth_product"}
REALTIME_WORDS = re.compile(r"最新|今天|今日|实时|现在|目前|当前")
INSUFFICIENT = "【依据不足】"


def _guarded(raw: str, question: str) -> str:
    g = StreamGuard(context=question)
    return g.feed(raw) + g.flush()


def generate_answer(question: str, plan: QueryPlan, evidence: list[Evidence]) -> tuple[str, str, list[str]]:
    """流式生成并推送；返回 (kind, answer, guard_hits)。"""
    s = get_settings()
    context = "\n\n".join(e.prompt_block() for e in evidence)
    messages = [
        {"role": "system", "content": load_prompt("answer_system")},
        {"role": "user", "content": f"证据：\n{context}\n\n问题：{plan.standalone_query}"},
    ]
    stream = llm.get_client().chat.completions.create(
        model=s.llm_model, messages=messages, temperature=s.llm_temperature, stream=True, extra_body=llm.NO_THINKING
    )
    guard, raw, held = StreamGuard(context=question), "", True
    for chunk in stream:
        delta = chunk.choices[0].delta.content if chunk.choices else None
        if not delta:
            continue
        raw += delta
        if held:  # 先攒几个字，确认不是【依据不足】再开始推送
            if len(raw) < 6 and INSUFFICIENT.startswith(raw.strip()):
                continue
            if raw.strip().startswith("【依据不足"):
                break
            held = False
            delta = raw
        out = guard.feed(delta).replace(INSUFFICIENT, "")
        if out:
            emit({"type": "delta", "text": out})
    if raw.strip().startswith("【依据不足"):
        emit({"type": "delta", "text": templates.REFUSE})
        return "refuse", templates.REFUSE, []
    tail = guard.flush().replace(INSUFFICIENT, "")
    if tail:
        emit({"type": "delta", "text": tail})
    if guard.hits:
        logger.warning("合规守卫拦截：%s", guard.hits)
    # 守卫替换后的文本（与推送给用户的一致）；模型在部分回答末尾追加的【依据不足】标记去掉
    return "answer", _guarded(raw, question).replace(INSUFFICIENT, "").rstrip(), guard.hits


@node_log("node_answer_output")
def node_answer_output(state: QueryGraphState) -> dict:
    reg = registry()
    question = state["original_query"]
    plan = state.get("plan") or {}
    evidence = [Evidence.from_dict(d) for d in state.get("evidence") or []]
    guard_hits: list[str] = []

    if state.get("answer"):  # 固定话术
        kind, answer = state["kind"], state["answer"]
        emit({"type": "delta", "text": answer})
    else:
        kind, answer, guard_hits = generate_answer(question, QueryPlan.model_validate(plan), evidence)

    sources: list[Evidence] = []
    if kind == "answer":
        answer, sources = finalize_citations(answer, evidence)

    notices = []
    entities = [reg.by_id[i] for i in state.get("entity_ids", []) if i in reg.by_id]
    if kind in ("answer", "decline_advice") and (
        plan.get("intent") in ("product_info", "risk", "investment_advice") or any(e.type in PRODUCT_TYPES for e in entities)
    ):
        notices.append(templates.RISK_NOTICE)
    if kind in ("answer", "decline_advice") and REALTIME_WORDS.search(question):
        notices.append(templates.REALTIME_NOTICE)
    if notices:
        emit({"type": "delta", "text": "\n\n" + "\n".join(notices)})
        answer = answer + "\n\n" + "\n".join(notices)
    source_dicts = [e.source() for e in sources]
    if sources:
        emit({"type": "sources", "sources": source_dicts, "text": render_sources(sources)})

    # 澄清时保留原焦点并记下候选；其他情况焦点更新为本轮对象（没有则沿用）
    focus = state.get("focus_entity_ids") or []
    if kind != "clarify" and state.get("entity_ids"):
        focus = state["entity_ids"]
    pending = {"options": state["candidate_ids"], "plan": plan} if kind == "clarify" else None
    session.save_turn(
        state["session_id"], question, answer,
        kind=kind, plan=plan, sources=source_dicts, focus_entity_ids=focus, pending=pending,
    )
    emit({"type": "final", "kind": kind})
    return {"kind": kind, "answer": answer, "sources": source_dicts, "guard_hits": guard_hits,
            "evidence": [vars(e) for e in evidence] if kind == "answer" else []}
