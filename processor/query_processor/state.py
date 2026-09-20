"""
查询图的状态
plan（查询计划，node_query_plan 产出）为 dict：standalone_query 补全指代后的独立问题、
entity_mentions 问题里提到的对象。
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
    clarified: bool  # 本轮是对澄清问题的回答，对象已由代码确定

    # 规划
    plan: dict  # 查询计划，键见模块说明

    # 取证（node_gather_evidence 写入）
    entity_ids: list  # 已确认的对象
    candidate_ids: list  # 有歧义时的候选对象 ID
    candidate_names: list  # 候选对象名称，交给模型决定要不要澄清
    unknown_mentions: list  # 问题提到但不在知识库里的对象
    doc_ids: list  # 对象对应的文档，用于检索过滤
    evidence: list  # 证据 dict（见 utils/citation_utils.py）

    # 回答（node_answer_output 写入）
    kind: str  # answer / refuse / clarify / decline_advice / realtime_notice / chitchat
    answer: str
    sources: list
    notices: list  # 模型给出的提示语标记：risk / realtime
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
    "candidate_names": [],
    "unknown_mentions": [],
    "doc_ids": [],
    "evidence": [],
    "kind": "",
    "answer": "",
    "sources": [],
    "notices": [],
    "guard_hits": [],
}


def create_query_default_state(**overrides) -> QueryGraphState:
    """创建查询图的初始状态（深拷贝默认值），可覆盖任意字段"""
    state = copy.deepcopy(query_graph_default_state)
    state.update(overrides)
    return state
