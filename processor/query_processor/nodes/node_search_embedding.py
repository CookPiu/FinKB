"""
节点：语义检索
稠密、稀疏分别检索并在代码中做 RRF 融合（utils/search_utils.py）；有实体时按文档过滤。
"""
from common.logging.logger import node_log
from processor.query_processor.state import QueryGraphState, create_query_default_state
from utils.search_utils import semantic_search

CANDIDATES = 30
TOP_K = 20


@node_log("node_search_embedding")
def node_search_embedding(state: QueryGraphState):
    """
    节点功能：用补全后的独立问题做语义检索。
    非固定话术路由时总会执行，与 node_fact_lookup / node_summary_fetch 并行；
    已确认实体时只在其文档内检索（doc_ids），否则全库检索。
    下游：node_rerank（精排、充分性判断用到稠密最高分 top_dense）。
    """
    doc_ids = state.get("doc_ids") or None
    hits = semantic_search(state["plan"]["standalone_query"], top_k=TOP_K, candidates=CANDIDATES, doc_ids=doc_ids)
    # 并行节点只返回自己写的键：整状态返回会与同一超步的其他节点冲突（InvalidUpdateError）
    return {"embedding_chunks": hits, "top_dense": get_top_dense(hits)}


def get_top_dense(hits: list) -> float:
    """命中中稠密检索的最高分；没有稠密分时为 0.0"""
    dense_scores = [hit["score_dense"] for hit in hits if hit["score_dense"] is not None]
    if not dense_scores:
        return 0.0
    return max(dense_scores)


if __name__ == "__main__":
    # 运行：uv run python -m processor.query_processor.nodes.node_search_embedding
    # 依赖：Milvus、Mongo、BGE-M3（CPU 首次加载约 20 秒）
    demo_plan = {"standalone_query": "贵州茅台一季度营业收入是多少？", "intent": "announcement",
                 "entity_mentions": [], "metrics": [], "wants_summary": False}
    result = node_search_embedding(create_query_default_state(session_id="demo-search", plan=demo_plan))
    print(f"top_dense={result['top_dense']}")
    for hit in result["embedding_chunks"][:5]:
        print(hit["score_rrf"], hit["file_name"], hit["text"][:80])
