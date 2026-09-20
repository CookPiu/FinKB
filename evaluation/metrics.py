"""
检索评测指标：命中判定按“文件名一致且切片正文包含金标原句”。
命中结果 hit 是 dict，至少含 file_name、text；金标 gold 是 {"file", "quote"}。
"""
from statistics import median

from evaluation.dataset import norm


def is_hit(hit: dict, golds: list) -> bool:
    """任一金标满足：文件名一致，且规范化后的正文包含规范化后的原句"""
    text = norm(hit["text"])
    for gold in golds:
        if hit["file_name"] == gold["file"] and norm(gold["quote"]) in text:
            return True
    return False


def first_hit_rank(ranked: list, golds: list):
    """首个命中的名次（从 1 开始），未命中返回 None。"""
    for i, hit in enumerate(ranked, 1):
        if is_hit(hit, golds):
            return i
    return None


def file_hit_rank(ranked: list, golds: list):
    """只看文件是否命中（不要求原句），用于区分“找错文档”与“找对文档但切片不对”。"""
    files = {gold["file"] for gold in golds}
    for i, hit in enumerate(ranked, 1):
        if hit["file_name"] in files:
            return i
    return None


def summarize(ranks: list, ks=(1, 3, 5, 10)) -> dict:
    """
    汇总命中名次
    :param ranks: 每题的首个命中名次，未命中为 None
    :return: {"n", "hit@1", "hit@3", "hit@5", "hit@10", "mrr@10"}
    """
    n = len(ranks)
    if n == 0:
        return {"n": 0}
    out = {"n": n}
    for k in ks:
        hit_count = 0
        for rank in ranks:
            if rank is not None and rank <= k:
                hit_count += 1
        out[f"hit@{k}"] = round(hit_count / n, 4)
    # 先收集再 sum：与原实现的求和方式一致，保证浮点结果逐位相同
    reciprocals = []
    for rank in ranks:
        if rank is not None and rank <= 10:
            reciprocals.append(1.0 / rank)
    out["mrr@10"] = round(sum(reciprocals) / n, 4)
    return out


def describe(values: list) -> dict:
    """数值分布：最小、P10、中位数、P90、最大"""
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
