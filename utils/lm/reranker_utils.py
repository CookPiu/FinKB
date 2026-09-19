"""云端精排（百炼 qwen3-rerank）。relevance_score 只在单次请求内可比，只用于排序，不用作拒答阈值。"""

from __future__ import annotations

import requests

from common.config.settings import get_settings


def rerank(query: str, documents: list[str], top_n: int) -> list[int]:
    """返回按相关度降序的文档下标（最多 top_n 个）。"""
    if not documents:
        return []
    s = get_settings()
    resp = requests.post(
        s.rerank_endpoint,
        headers={"Authorization": f"Bearer {s.openai_api_key}", "Content-Type": "application/json"},
        json={"model": s.rerank_model, "query": query, "documents": documents, "top_n": min(top_n, len(documents))},
        timeout=10,  # 正常 0.3s 左右返回；超时即退回 RRF 排序，避免用户长时间等待
    )
    resp.raise_for_status()
    body = resp.json()
    results = body.get("results") or body.get("output", {}).get("results") or []
    return [r["index"] for r in sorted(results, key=lambda r: r["relevance_score"], reverse=True)][:top_n]
