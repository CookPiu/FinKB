"""节点：语义检索。稠密、稀疏分别检索并在代码中做 RRF 融合（utils/search_utils.py）；有实体时按文档过滤。"""

from __future__ import annotations

from dataclasses import asdict

from common.logging.logger import node_log
from processor.query_processor.state import QueryGraphState, QueryPlan
from utils.search_utils import semantic_search

CANDIDATES = 30
TOP_K = 20


@node_log("node_search_embedding")
def node_search_embedding(state: QueryGraphState) -> dict:
    plan = QueryPlan.model_validate(state["plan"])
    hits = semantic_search(plan.standalone_query, top_k=TOP_K, candidates=CANDIDATES, doc_ids=state.get("doc_ids") or None)
    top_dense = max((h.score_dense for h in hits if h.score_dense is not None), default=0.0)
    return {"embedding_chunks": [asdict(h) for h in hits], "top_dense": top_dense}
