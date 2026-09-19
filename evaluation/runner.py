"""评测运行器。

用法：
  uv run python -m evaluation.runner --set data/eval/fin_eval_set.jsonl --check-only
  uv run python -m evaluation.runner --set data/eval/fin_eval_set.jsonl --tag M1-baseline

mode=retrieval：只评检索。每个有金标的轮次取 top-10，按“文件名 + 原句”判命中；多轮题没有会话状态，
  第 2 轮起用“历史问题 + 当前问题”直接拼接作为查询（M1 朴素基线）。
mode=answer：走完整问答链路，按规则打分（行为是否符合 expect、must_include / must_not_include、
  合规守卫复查、证据是否含金标、引用是否含金标文件），不用 LLM 裁判。
结果存档到 data/eval/results/<日期>_<tag>.json。
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from common.config.settings import PROJECT_ROOT, get_settings
from evaluation.dataset import load, load_corpus, self_check
from evaluation.metrics import describe, file_hit_rank, first_hit_rank, is_hit, summarize

TOP_K = 10
CANDIDATES = 30


def _git_rev() -> str:
    try:
        rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=PROJECT_ROOT)
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, cwd=PROJECT_ROOT)
        return rev.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")
    except OSError:
        return "unknown"


def run_retrieval(items) -> list[dict]:
    from utils.lm import embedding_utils as embedding
    from utils.search_utils import semantic_search

    embedding.warmup()
    records: list[dict] = []
    for it in items:
        history: list[str] = []
        for ti, turn in enumerate(it.resolved_turns()):
            query = " ".join([*history, turn.q])
            t0 = time.perf_counter()
            hits = semantic_search(query, top_k=TOP_K, candidates=CANDIDATES)
            latency_ms = (time.perf_counter() - t0) * 1000
            dense_scores = [h.score_dense for h in hits if h.score_dense is not None]
            records.append(
                {
                    "id": it.id,
                    "turn": ti + 1,
                    "category": it.category,
                    "expect": turn.expect,
                    "query": query,
                    "has_gold": bool(turn.gold),
                    "rank": first_hit_rank(hits, turn.gold) if turn.gold else None,
                    "file_rank": file_hit_rank(hits, turn.gold) if turn.gold else None,
                    "top_dense": max(dense_scores) if dense_scores else None,
                    "latency_ms": round(latency_ms, 1),
                    "top": [
                        {
                            "file": h.file_name,
                            "chunk_id": h.chunk_id,
                            "kind": h.kind,
                            "pages": f"{h.page_start}-{h.page_end}",
                            "dense": h.score_dense,
                            "sparse": h.score_sparse,
                            "rrf": round(h.score_rrf, 5),
                            "hit": is_hit(h, turn.gold) if turn.gold else None,
                        }
                        for h in hits[:5]
                    ],
                }
            )
            history.append(turn.q)
        print(f"  {it.id}: " + ", ".join(str(r["rank"]) for r in records if r["id"] == it.id), flush=True)
    return records


REFUSAL_KINDS = {"refuse", "realtime_notice"}


def run_answers(items) -> list[dict]:
    """逐题走完整问答链路（每题一个新会话，多轮题在同一会话内依次提问），按规则打分。"""
    from utils.guard_utils import is_negated, violations
    from processor.query_processor.main_graph import ask

    records: list[dict] = []
    for it in items:
        sid = None
        for ti, turn in enumerate(it.resolved_turns()):
            t0 = time.perf_counter()
            try:
                res = ask(turn.q, session_id=sid)
                error = None
            except Exception as e:  # noqa: BLE001 - 单题失败记为错误，不中断评测
                res, error = {"kind": "error", "answer": ""}, f"{type(e).__name__}: {e}"
            sid = res.get("session_id", sid)
            answer = res.get("answer", "")
            evidence = res.get("evidence") or []
            sources = res.get("sources") or []
            ev_hit = None
            if turn.gold:
                ev_hit = any(is_hit(_Hit(e["file_name"], e["text"]), turn.gold) for e in evidence)
            records.append(
                {
                    "id": it.id,
                    "turn": ti + 1,
                    "category": it.category,
                    "query": turn.q,
                    "expect": turn.expect,
                    "kind": res.get("kind"),
                    "behavior_ok": res.get("kind") == turn.expect,
                    "include_ok": all(s in answer for s in turn.must_include) if turn.must_include else None,
                    # 与合规守卫一致：带否定前缀的出现（如“不能保证收益”）不算
                    "exclude_hits": [
                        s for s in turn.must_not_include
                        if any(not is_negated(answer, m.start()) for m in re.finditer(re.escape(s), answer))
                    ],
                    "guard_violations": violations(answer, context=turn.q),
                    "evidence_hit": ev_hit,
                    "cited_gold_file": (any(s["file_name"] in {g.file for g in turn.gold} for s in sources)
                                        if turn.gold and res.get("kind") == "answer" else None),
                    "latency_ms": round((time.perf_counter() - t0) * 1000),
                    "error": error,
                    "answer": answer,
                    "sources": [f"{s['file_name']} p{s.get('page_start')}" for s in sources],
                }
            )
        r = [x for x in records if x["id"] == it.id]
        print(f"  {it.id}: " + ", ".join(f"{x['kind']}{'' if x['behavior_ok'] else '✗'}" for x in r), flush=True)
    return records


class _Hit:
    def __init__(self, file_name: str, text: str):
        self.file_name, self.text = file_name, text


def _rate(values: list) -> float | None:
    vals = [v for v in values if v is not None]
    return round(sum(1 for v in vals if v) / len(vals), 4) if vals else None


def summarize_answers(records: list[dict]) -> dict:
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_cat[r["category"]].append(r)
    negatives = [r for r in records if r["expect"] in REFUSAL_KINDS]
    positives = [r for r in records if r["expect"] == "answer"]
    chains = defaultdict(list)
    for r in records:
        if r["category"] == "multiturn":
            chains[r["id"]].append(r)
    return {
        "behavior_accuracy": _rate([r["behavior_ok"] for r in records]),
        "behavior_by_category": {k: _rate([r["behavior_ok"] for r in v]) for k, v in sorted(by_cat.items())},
        "negative_refusal_rate": _rate([r["kind"] in REFUSAL_KINDS for r in negatives]),
        "false_refusal_rate": _rate([r["kind"] in REFUSAL_KINDS for r in positives]),
        "must_include_pass_rate": _rate([r["include_ok"] for r in records]),
        "must_not_include_violations": sum(len(r["exclude_hits"]) for r in records),
        "guard_violations": sum(len(r["guard_violations"]) for r in records),
        "evidence_hit_rate": _rate([r["evidence_hit"] for r in records]),
        "cited_gold_file_rate": _rate([r["cited_gold_file"] for r in records]),
        "multiturn_chains_passed": f"{sum(all(t['behavior_ok'] and t['include_ok'] is not False for t in c) for c in chains.values())}/{len(chains)}",
        "errors": sum(1 for r in records if r["error"]),
        "latency_ms": describe([r["latency_ms"] for r in records]),
    }


def summarize_records(records: list[dict]) -> dict:
    by_cat: dict[str, list] = defaultdict(list)
    file_by_cat: dict[str, list] = defaultdict(list)
    for r in records:
        if not r["has_gold"]:
            continue
        key = r["category"] if r["category"] != "multiturn" else ("multiturn_t1" if r["turn"] == 1 else "multiturn_followup")
        by_cat[key].append(r["rank"])
        file_by_cat[key].append(r["file_rank"])
    all_ranks = [r["rank"] for r in records if r["has_gold"]]
    answerable = [r["top_dense"] for r in records if r["has_gold"] and r["top_dense"] is not None]
    unanswerable = [
        r["top_dense"] for r in records if r["expect"] in ("refuse", "realtime_notice") and r["top_dense"] is not None
    ]
    return {
        "by_category": {k: summarize(v) for k, v in sorted(by_cat.items())},
        "file_level_by_category": {k: summarize(v) for k, v in sorted(file_by_cat.items())},
        "overall": summarize(all_ranks),
        "top_dense_answerable": describe(answerable),
        "top_dense_unanswerable": describe(unanswerable),
        "latency_ms": describe([r["latency_ms"] for r in records]),
    }


def _print_summary(summary: dict) -> None:
    print("\n类别                     n   hit@1  hit@3  hit@5  hit@10  mrr@10   (文件级 hit@5)")
    rows = {**summary["by_category"], "overall": summary["overall"]}
    for k, s in rows.items():
        f5 = summary["file_level_by_category"].get(k, {}).get("hit@5", "")
        print(
            f"{k:<22} {s['n']:>3}  {s['hit@1']:.3f}  {s['hit@3']:.3f}  {s['hit@5']:.3f}  {s['hit@10']:.3f}   "
            f"{s['mrr@10']:.3f}   {f5}"
        )
    print(f"\n稠密最高分  可回答：{summary['top_dense_answerable']}")
    print(f"稠密最高分  不可回答：{summary['top_dense_unanswerable']}")
    print(f"检索延迟 ms：{summary['latency_ms']}")


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(description="FinKB 评测")
    p.add_argument("--set", default="data/eval/fin_eval_set.jsonl")
    p.add_argument("--tag", default="run")
    p.add_argument("--mode", choices=["retrieval", "answer"], default="retrieval")
    p.add_argument("--check-only", action="store_true", help="只做评测集自检")
    args = p.parse_args(argv)

    path = Path(args.set)
    path = path if path.is_absolute() else PROJECT_ROOT / path
    items = load(path)
    errors = self_check(items, load_corpus())
    if errors:
        print(f"评测集自检失败（{len(errors)} 处）：")
        for e in errors:
            print("  ✗", e)
        return 1
    counts: dict[str, int] = defaultdict(int)
    for it in items:
        counts[it.category] += 1
    print(f"评测集自检通过：{len(items)} 题 {dict(counts)}")
    if args.check_only:
        return 0

    s = get_settings()
    if args.mode == "answer":
        records = run_answers(items)
        summary = summarize_answers(records)
        print("\n" + json.dumps(summary, ensure_ascii=False, indent=1))
    else:
        records = run_retrieval(items)
        summary = summarize_records(records)
        _print_summary(summary)

    out_dir = PROJECT_ROOT / "data" / "eval" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{datetime.now():%Y-%m-%d}_{args.tag}.json"
    result = {
        "meta": {
            "tag": args.tag,
            "mode": args.mode,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "git": _git_rev(),
            "eval_set": path.relative_to(PROJECT_ROOT).as_posix() if path.is_relative_to(PROJECT_ROOT) else str(path),
            "n_items": len(items),
            "config": {
                "embedding": Path(s.bge_m3_path).name,
                "bge_max_length": s.bge_max_length,
                "retrieval": f"dense(COSINE,HNSW)+sparse(IP) 各取 {CANDIDATES}，RRF k=60，top {TOP_K}",
                "embed_text": "document_title · content_type · section_path + body（表格为线性化文本）",
                "chunk": {
                    "target": s.chunk_target_chars,
                    "max": s.chunk_max_chars,
                    "min": s.chunk_min_chars,
                    "table_max": s.table_max_chars,
                },
                "multiturn_query": "历史问题与当前问题直接拼接（M1 无会话状态）",
            },
        },
        "summary": summary,
        "records": records,
    }
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n结果已存档：{out.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
