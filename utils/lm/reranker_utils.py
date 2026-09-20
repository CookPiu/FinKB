"""
云端精排（百炼 qwen3-rerank）。
relevance_score 只在单次请求内可比，只用于排序，不用作拒答阈值。
"""
from urllib.parse import urlparse

import requests

from common.config.lm_config import lm_config
from common.config.reranker_config import reranker_config


def get_rerank_endpoint() -> str:
    """精排地址；未显式配置 RERANK_URL 时与生成模型共用工作空间主机"""
    if reranker_config.rerank_url:
        return reranker_config.rerank_url
    host = urlparse(lm_config.base_url).netloc
    if not host:
        return ""
    return f"https://{host}/compatible-api/v1/reranks"


def rerank(query: str, documents: list, top_n: int) -> list:
    """
    对候选文本精排
    :param query: 问题
    :param documents: 候选文本列表
    :param top_n: 最多返回几个
    :return: 按相关度降序的候选下标（最多 top_n 个）
    """
    if not documents:
        return []
    response = requests.post(
        get_rerank_endpoint(),
        headers={"Authorization": f"Bearer {lm_config.api_key}", "Content-Type": "application/json"},
        json={
            "model": reranker_config.rerank_model,
            "query": query,
            "documents": documents,
            "top_n": min(top_n, len(documents)),
        },
        timeout=10,  # 正常 0.3s 左右返回；超时即由调用方退回 RRF 排序，避免用户长时间等待
    )
    response.raise_for_status()
    body = response.json()
    # 兼容两种返回结构：{"results": [...]} 与 {"output": {"results": [...]}}
    results = body.get("results")
    if not results:
        results = body.get("output", {}).get("results") or []
    results = sorted(results, key=lambda r: r["relevance_score"], reverse=True)
    indexes = []
    for item in results[:top_n]:
        indexes.append(item["index"])
    return indexes
