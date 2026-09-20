"""
查询图（LangGraph）
node_query_plan → node_entity_confirm ─┬→ node_answer_output（固定话术：澄清 / 拒答 / 投资建议 / 实时 / 寒暄）
                                       └→ node_fact_lookup ‖ node_summary_fetch ‖ node_search_embedding（按需并行）
                                             → node_rerank → node_answer_output

流式事件通过 LangGraph custom stream 发出：{"type": "delta"|"sources"|"final", ...}。
"""
import uuid

from langgraph.graph import END, StateGraph

from processor.query_processor.nodes.node_answer_output import node_answer_output
from processor.query_processor.nodes.node_entity_confirm import node_entity_confirm
from processor.query_processor.nodes.node_fact_lookup import node_fact_lookup
from processor.query_processor.nodes.node_query_plan import node_query_plan
from processor.query_processor.nodes.node_rerank import node_rerank
from processor.query_processor.nodes.node_search_embedding import node_search_embedding
from processor.query_processor.nodes.node_summary_fetch import node_summary_fetch
from processor.query_processor.state import QueryGraphState, create_query_default_state
from utils.entity_utils import get_entity_map

builder = StateGraph(QueryGraphState)
builder.add_node("node_query_plan", node_query_plan)
builder.add_node("node_entity_confirm", node_entity_confirm)
builder.add_node("node_fact_lookup", node_fact_lookup)
builder.add_node("node_summary_fetch", node_summary_fetch)
builder.add_node("node_search_embedding", node_search_embedding)
builder.add_node("node_rerank", node_rerank)
builder.add_node("node_answer_output", node_answer_output)
builder.set_entry_point("node_query_plan")
builder.add_edge("node_query_plan", "node_entity_confirm")


def route_after_entity_confirm(state: QueryGraphState):
    """已有固定话术则直接输出；否则并行取证：语义检索总是执行，财务事实与摘要按计划决定"""
    if state.get("answer"):
        return "node_answer_output"
    plan = state.get("plan") or {}
    entity_map = get_entity_map()
    targets = ["node_search_embedding"]
    if plan.get("metrics") and any(entity_map[i]["type"] == "company" for i in state.get("entity_ids", [])):
        targets.append("node_fact_lookup")
    if plan.get("wants_summary") and state.get("doc_ids"):
        targets.append("node_summary_fetch")
    return targets


builder.add_conditional_edges(
    "node_entity_confirm",
    route_after_entity_confirm,
    {
        "node_answer_output": "node_answer_output",
        "node_search_embedding": "node_search_embedding",
        "node_fact_lookup": "node_fact_lookup",
        "node_summary_fetch": "node_summary_fetch",
    },
)
builder.add_edge("node_fact_lookup", "node_rerank")
builder.add_edge("node_summary_fetch", "node_rerank")
builder.add_edge("node_search_embedding", "node_rerank")
builder.add_edge("node_rerank", "node_answer_output")
builder.add_edge("node_answer_output", END)

query_app = builder.compile()


def ask(question: str, session_id=None, on_event=None) -> dict:
    """
    回答一个问题
    :param question: 用户问题
    :param session_id: 沿用已有会话（多轮追问）；为空时新建
    :param on_event: 接收流式事件的回调（delta / sources / final），可为空
    :return: 最终状态（kind、answer、sources、plan、evidence 等），含 session_id
    """
    if session_id:
        sid = session_id
    else:
        sid = uuid.uuid4().hex[:12]
    state = create_query_default_state(session_id=sid, original_query=question)
    final = {}
    for mode, chunk in query_app.stream(state, stream_mode=["custom", "values"]):
        if mode == "custom" and on_event:
            on_event(chunk)
        elif mode == "values":
            final = chunk
    final["session_id"] = sid
    return final


if __name__ == "__main__":
    # 运行：uv run python -m processor.query_processor.main_graph
    # 依赖：Mongo、Milvus、BGE-M3（CPU 首次加载约 20 秒）、百炼 LLM 与精排；演示会话跑完即删除
    from utils.clients.mongo_history_utils import delete_sessions
    result = ask("贵州茅台2025年第一季度营业收入是多少？", session_id="demo-main-graph")
    print(f"[kind={result['kind']}]")
    print(result["answer"])
    delete_sessions(["demo-main-graph"])
