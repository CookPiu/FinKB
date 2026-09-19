"""节点：文档摘要。问“这份资料主要讲了什么”时，取实体对应文档在导入时生成的摘要作为证据。"""

from __future__ import annotations

from common.logging.logger import node_log
from processor.query_processor.state import QueryGraphState
from utils.clients import mongo_utils as mongo


@node_log("node_summary_fetch")
def node_summary_fetch(state: QueryGraphState) -> dict:
    doc_ids = state.get("doc_ids") or []
    docs = mongo.get_db().documents.find({"_id": {"$in": doc_ids}, "summary": {"$ne": None}}, {"summary": 1})
    return {"summaries": [{"doc_id": d["_id"], "text": d["summary"]} for d in docs]}
