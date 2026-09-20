"""
查询图的状态
plan（查询计划，node_query_plan 产出）为 dict：standalone_query 补全指代后的独立问题、intent 意图、
entity_mentions 问题里提到的实体、metrics 财务指标名、wants_summary 是否在问资料的主要内容。
"""
import copy
from typing import TypedDict


class QueryGraphState(TypedDict):
    """查询流程中流转的数据，节点用 state["键"] / state.get("键") 读写"""
    session_id: str
    original_query: str

    # 会话（node_query_plan 读取）
    history: list  # 最近几轮 [{role, text}]
    focus_entity_ids: list  # 上一轮在谈的对象
    clarified: bool  # 本轮是对澄清问题的回答，实体已由代码确定

    # 规划与实体确认
    plan: dict  # 查询计划，键见模块说明
    entity_ids: list
    candidate_ids: list  # 有歧义时的候选
    doc_ids: list  # 实体对应的文档，用于检索过滤

    # 取证（三个节点并行写入各自字段）
    facts: list
    summaries: list
    embedding_chunks: list  # 语义检索命中 dict（见 utils/search_utils.py）
    top_dense: float

    # 精排与回答
    evidence: list  # 证据 dict（见 utils/citation_utils.py）
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
    """创建查询图的初始状态（深拷贝默认值），可覆盖任意字段"""
    state = copy.deepcopy(query_graph_default_state)
    state.update(overrides)
    return state
