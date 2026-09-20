"""
查询图（LangGraph）：三个节点顺序执行。
node_query_plan（改写问题、抽出对象）→ node_gather_evidence（取证 + 精排 + 编号）→ node_answer_output（判定 + 生成）

回答该怎么处理（回答 / 拒答 / 澄清 / 不提供建议 / 时效提示 / 寒暄）由 node_answer_output 里的模型按提示词判断，
代码只保留三道闸门：禁语兜底、引用校验、没有任何证据时直接拒答。
流式事件通过 LangGraph custom stream 发出：{"type": "delta"|"sources"|"final", ...}。
"""
import uuid

from langgraph.graph import END, StateGraph

from processor.query_processor.nodes.node_answer_output import node_answer_output
from processor.query_processor.nodes.node_gather_evidence import node_gather_evidence
from processor.query_processor.nodes.node_query_plan import node_query_plan
from processor.query_processor.state import QueryGraphState, create_query_default_state

# 1. 注册节点
builder = StateGraph(QueryGraphState)
builder.add_node("node_query_plan", node_query_plan)
builder.add_node("node_gather_evidence", node_gather_evidence)
builder.add_node("node_answer_output", node_answer_output)
builder.set_entry_point("node_query_plan")

# 2. 顺序执行，没有分支：取证节点在没有证据时会写好固定拒答，输出节点据此跳过模型调用
builder.add_edge("node_query_plan", "node_gather_evidence")
builder.add_edge("node_gather_evidence", "node_answer_output")
builder.add_edge("node_answer_output", END)

# 3. 编译成可执行的图
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
    result = ask("贵州茅台2026年第一季度营业收入是多少？", session_id="demo-main-graph")
    print(f"[kind={result['kind']}]")
    print(result["answer"])
    delete_sessions(["demo-main-graph"])
