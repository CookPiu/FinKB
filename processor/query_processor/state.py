"""查询图的状态与查询计划。"""

from __future__ import annotations

import copy
from typing import Literal, TypedDict

from pydantic import BaseModel, Field

Intent = Literal[
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


class QueryPlan(BaseModel):
    """规划节点的输出：一次 LLM 调用得到，pydantic 校验；执行由后续节点按计划进行。"""

    standalone_query: str
    intent: Intent = "knowledge"
    entity_mentions: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)
    wants_summary: bool = False


class QueryGraphState(TypedDict, total=False):
    session_id: str
    original_query: str

    # 会话（node_query_plan 读取）
    history: list  # 最近几轮 [{role, text}]
    focus_entity_ids: list  # 上一轮在谈的对象
    clarified: bool  # 本轮是对澄清问题的回答，实体已由代码确定

    # 规划与实体确认
    plan: dict  # QueryPlan.model_dump()
    entity_ids: list
    candidate_ids: list  # 有歧义时的候选
    doc_ids: list  # 实体对应的文档，用于检索过滤

    # 取证（三个节点并行写入各自字段）
    facts: list
    summaries: list
    embedding_chunks: list  # SearchHit 字典
    top_dense: float

    # 精排与回答
    evidence: list  # Evidence 字典
    kind: str  # answer / refuse / clarify / decline_advice / realtime_notice / chitchat
    answer: str
    sources: list
    guard_hits: list


query_graph_default_state: QueryGraphState = {
    "session_id": "",
    "original_query": "",
    "history": [],
    "focus_entity_ids": [],
    "clarified": False,
    "plan": {},
    "entity_ids": [],
    "candidate_ids": [],
    "doc_ids": [],
    "facts": [],
    "summaries": [],
    "embedding_chunks": [],
    "top_dense": 0.0,
    "evidence": [],
    "kind": "",
    "answer": "",
    "sources": [],
    "guard_hits": [],
}


def create_query_default_state(**overrides) -> QueryGraphState:
    state = copy.deepcopy(query_graph_default_state)
    state.update(overrides)
    return state
