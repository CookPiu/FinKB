"""检索评测指标：命中判定按“文件名一致且切片正文包含金标原句”。"""

from __future__ import annotations

from collections.abc import Sequence
from statistics import median
from typing import Protocol

from evaluation.dataset import Gold, norm


class HitLike(Protocol):
    file_name: str
    text: str


def is_hit(hit: HitLike, golds: Sequence[Gold]) -> bool:
    text = norm(hit.text)
    return any(hit.file_name == g.file and norm(g.quote) in text for g in golds)


def first_hit_rank(ranked: Sequence[HitLike], golds: Sequence[Gold]) -> int | None:
    """首个命中的名次（从 1 开始），未命中返回 None。"""
    for i, h in enumerate(ranked, 1):
        if is_hit(h, golds):
            return i
    return None


def file_hit_rank(ranked: Sequence[HitLike], golds: Sequence[Gold]) -> int | None:
    """只看文件是否命中（不要求原句），用于区分“找错文档”与“找对文档但切片不对”。"""
    files = {g.file for g in golds}
    for i, h in enumerate(ranked, 1):
        if h.file_name in files:
            return i
    return None


def summarize(ranks: list[int | None], ks: Sequence[int] = (1, 3, 5, 10)) -> dict[str, float]:
    n = len(ranks)
    if n == 0:
        return {"n": 0}
    out: dict[str, float] = {"n": n}
    for k in ks:
        out[f"hit@{k}"] = round(sum(1 for r in ranks if r is not None and r <= k) / n, 4)
    out["mrr@10"] = round(sum(1.0 / r for r in ranks if r is not None and r <= 10) / n, 4)
    return out


def describe(values: list[float]) -> dict[str, float]:
    if not values:
        return {"n": 0}
    v = sorted(values)
    return {
        "n": len(v),
        "min": round(v[0], 4),
        "p10": round(v[int(0.1 * (len(v) - 1))], 4),
        "median": round(median(v), 4),
        "p90": round(v[int(0.9 * (len(v) - 1))], 4),
        "max": round(v[-1], 4),
    }
