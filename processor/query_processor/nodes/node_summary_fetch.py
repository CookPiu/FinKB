"""
节点：文档摘要
问"这份资料主要讲了什么"时，取实体对应文档在导入时生成的摘要作为证据。
"""
from common.logging.logger import logger, node_log, step_log
from processor.query_processor.state import QueryGraphState, create_query_default_state
from utils.clients.mongo_utils import get_db


@step_log("validate_and_get_data")
def validate_and_get_data(state: QueryGraphState):
    """
    取出并校验摘要读取所需的入参
    :return: 实体对应的文档 ID 列表
    :raise ValueError: 没有文档 ID（正常路由下本节点只在已确定文档时触发）
    """
    doc_ids = state.get("doc_ids") or []
    if not doc_ids:
        logger.error("no doc_ids found in state")
        raise ValueError("no doc_ids found in state")
    return doc_ids


@node_log("node_summary_fetch")
def node_summary_fetch(state: QueryGraphState):
    """
    节点功能：读取实体对应文档的摘要。
    计划 wants_summary 且已确定文档时由路由触发，与 node_search_embedding / node_fact_lookup 并行。
    下游：node_rerank（摘要排在财务事实之后、文本切片之前）。
    """
    doc_ids = validate_and_get_data(state)
    summaries = fetch_summaries(doc_ids)
    # 并行节点只返回自己写的键：整状态返回会与同一超步的其他节点冲突（InvalidUpdateError）
    return {"summaries": summaries}


@step_log("fetch_summaries")
def fetch_summaries(doc_ids: list) -> list:
    """
    读取文档摘要（导入阶段 node_enrich 生成；没有摘要的文档跳过）
    :return: [{"doc_id", "text"}]
    """
    docs = get_db().documents.find({"_id": {"$in": doc_ids}, "summary": {"$ne": None}}, {"summary": 1})
    return [{"doc_id": doc["_id"], "text": doc["summary"]} for doc in docs]


if __name__ == "__main__":
    # 运行：uv run python -m processor.query_processor.nodes.node_summary_fetch
    # 依赖：Mongo（documents）；doc_ids 换成 cli.py status 里列出的真实文档 ID
    demo_state = create_query_default_state(session_id="demo-summary-fetch", doc_ids=["<doc_id>"])
    for summary in node_summary_fetch(demo_state)["summaries"]:
        print(summary["doc_id"], summary["text"][:200])
