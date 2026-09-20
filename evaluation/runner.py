"""
评测运行器。
用法：
  uv run python -m evaluation.runner --set data/eval/fin_eval_set.jsonl --check-only
  uv run python -m evaluation.runner --set data/eval/fin_eval_set.jsonl --tag M1-baseline

mode=retrieval：只评检索。每个有金标的轮次取 top-10，按“文件名 + 原句”判命中；多轮题没有会话状态，
  第 2 轮起用“历史问题 + 当前问题”直接拼接作为查询（M1 朴素基线）。
mode=answer：走完整问答链路，按规则打分（行为是否符合 expect、must_include / must_not_include、
  合规守卫复查、证据是否含金标、引用是否含金标文件），不用 LLM 裁判。
结果存档到 data/eval/results/<日期>_<tag>.json。
"""
import argparse
import json
import re
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

from common.config.embedding_config import embedding_config
from evaluation.dataset import load_corpus, load_eval_items, resolve_turns, self_check
from evaluation.metrics import describe, file_hit_rank, first_hit_rank, is_hit, summarize
from processor.import_processor.nodes.node_chunk import (
    CHUNK_MAX_CHARS,
    CHUNK_MIN_CHARS,
    CHUNK_TARGET_CHARS,
    TABLE_MAX_CHARS,
)
from utils.path_util import PROJECT_ROOT

TOP_K = 10
CANDIDATES = 30
REFUSAL_KINDS = {"refuse", "realtime_notice"}


def get_git_rev() -> str:
    """当前提交的短哈希，工作区有改动时加 -dirty"""
    try:
        rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=PROJECT_ROOT)
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, cwd=PROJECT_ROOT)
    except OSError:
        return "unknown"
    if dirty.stdout.strip():
        return rev.stdout.strip() + "-dirty"
    return rev.stdout.strip()


def build_top_record(hit: dict, golds: list) -> dict:
    """检索结果前几名的存档格式"""
    hit_flag = None
    if golds:
        hit_flag = is_hit(hit, golds)
    return {
        "file": hit["file_name"],
        "chunk_id": hit["chunk_id"],
        "kind": hit["kind"],
        "pages": f"{hit['page_start']}-{hit['page_end']}",
        "dense": hit["score_dense"],
        "sparse": hit["score_sparse"],
        "rrf": round(hit["score_rrf"], 5),
        "hit": hit_flag,
    }


def join_item_results(records: list, item_id: str, key: str) -> str:
    """把某题各轮记录的一个字段拼成一行，用于进度输出"""
    parts = []
    for record in records:
        if record["id"] == item_id:
            parts.append(str(record[key]))
    return ", ".join(parts)


def run_retrieval(items: list) -> list:
    """逐题逐轮检索，记录首个命中名次与前 5 名"""
    from utils.lm.embedding_utils import warmup
    from utils.search_utils import semantic_search

    warmup()
    records = []
    for item in items:
        history = []
        for ti, turn in enumerate(resolve_turns(item)):
            query = " ".join(history + [turn["q"]])
            start_ts = time.perf_counter()
            hits = semantic_search(query, top_k=TOP_K, candidates=CANDIDATES)
            latency_ms = (time.perf_counter() - start_ts) * 1000
            dense_scores = [h["score_dense"] for h in hits if h["score_dense"] is not None]
            golds = turn["gold"]
            records.append(
                {
                    "id": item["id"],
                    "turn": ti + 1,
                    "category": item["category"],
                    "expect": turn["expect"],
                    "query": query,
                    "has_gold": bool(golds),
                    "rank": first_hit_rank(hits, golds) if golds else None,
                    "file_rank": file_hit_rank(hits, golds) if golds else None,
                    "top_dense": max(dense_scores) if dense_scores else None,
                    "latency_ms": round(latency_ms, 1),
                    "top": [build_top_record(h, golds) for h in hits[:5]],
                }
            )
            history.append(turn["q"])
        print(f"  {item['id']}: " + join_item_results(records, item["id"], "rank"), flush=True)
    return records


def find_exclude_hits(answer: str, must_not_include: list) -> list:
    """
    回答里出现的禁止内容
    与合规守卫一致：带否定前缀的出现（如“不能保证收益”）不算
    """
    from utils.guard_utils import is_negated

    hits = []
    for text in must_not_include:
        for match in re.finditer(re.escape(text), answer):
            if not is_negated(answer, match.start()):
                hits.append(text)
                break
    return hits


def check_cited_gold_file(turn: dict, kind, sources: list):
    """回答题的引用来源是否含金标文件；没有金标或不是回答时为 None"""
    if not turn["gold"] or kind != "answer":
        return None
    gold_files = {gold["file"] for gold in turn["gold"]}
    return any(source["file_name"] in gold_files for source in sources)


def format_answer_progress(records: list, item_id: str) -> str:
    """某题各轮的回答类型，行为不符合预期的加 ✗"""
    parts = []
    for record in records:
        if record["id"] != item_id:
            continue
        if record["behavior_ok"]:
            parts.append(f"{record['kind']}")
        else:
            parts.append(f"{record['kind']}✗")
    return ", ".join(parts)


def run_answers(items: list) -> list:
    """逐题走完整问答链路（每题一个新会话，多轮题在同一会话内依次提问），按规则打分。"""
    from processor.query_processor.main_graph import ask
    from utils.clients.mongo_history_utils import delete_sessions
    from utils.guard_utils import violations

    records = []
    created = []
    for item in items:
        session_id = f"eval-{uuid.uuid4().hex[:8]}"
        created.append(session_id)
        for ti, turn in enumerate(resolve_turns(item)):
            start_ts = time.perf_counter()
            try:
                result = ask(turn["q"], session_id=session_id)
                error = None
            except Exception as e:  # 单题失败记为错误，不中断评测
                result = {"kind": "error", "answer": ""}
                error = f"{type(e).__name__}: {e}"
            session_id = result.get("session_id", session_id)
            answer = result.get("answer", "")
            evidence = result.get("evidence") or []
            sources = result.get("sources") or []
            evidence_hit = None
            if turn["gold"]:
                evidence_hit = any(is_hit(e, turn["gold"]) for e in evidence)
            include_ok = None
            if turn["must_include"]:
                include_ok = all(s in answer for s in turn["must_include"])
            records.append(
                {
                    "id": item["id"],
                    "turn": ti + 1,
                    "category": item["category"],
                    "query": turn["q"],
                    "expect": turn["expect"],
                    "kind": result.get("kind"),
                    "behavior_ok": result.get("kind") == turn["expect"],
                    "include_ok": include_ok,
                    "exclude_hits": find_exclude_hits(answer, turn["must_not_include"]),
                    "guard_violations": violations(answer, context=turn["q"]),
                    "evidence_hit": evidence_hit,
                    "cited_gold_file": check_cited_gold_file(turn, result.get("kind"), sources),
                    "latency_ms": round((time.perf_counter() - start_ts) * 1000),
                    "error": error,
                    "answer": answer,
                    "sources": [f"{s['file_name']} p{s.get('page_start')}" for s in sources],
                }
            )
        print(f"  {item['id']}: " + format_answer_progress(records, item["id"]), flush=True)
    delete_sessions(created)  # 评测会话不留在历史记录里
    return records


def get_rate(values: list):
    """真值比例（忽略 None）；全为 None 时返回 None"""
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return round(sum(1 for v in vals if v) / len(vals), 4)


def group_records(records: list, key: str) -> dict:
    """按记录的某个字段分组：{取值: [记录]}"""
    groups = {}
    for record in records:
        value = record[key]
        if value not in groups:
            groups[value] = []
        groups[value].append(record)
    return groups


def count_passed_chains(records: list) -> str:
    """多轮题整条对话通过数：每一轮行为都正确且 must_include 没有失败"""
    chains = group_records([r for r in records if r["category"] == "multiturn"], "id")
    passed = 0
    for chain in chains.values():
        if all(t["behavior_ok"] and t["include_ok"] is not False for t in chain):
            passed += 1
    return f"{passed}/{len(chains)}"


def summarize_answers(records: list) -> dict:
    by_category = group_records(records, "category")
    behavior_by_category = {}
    for category in sorted(by_category):
        behavior_by_category[category] = get_rate([r["behavior_ok"] for r in by_category[category]])
    negatives = [r for r in records if r["expect"] in REFUSAL_KINDS]
    positives = [r for r in records if r["expect"] == "answer"]
    return {
        "behavior_accuracy": get_rate([r["behavior_ok"] for r in records]),
        "behavior_by_category": behavior_by_category,
        "negative_refusal_rate": get_rate([r["kind"] in REFUSAL_KINDS for r in negatives]),
        "false_refusal_rate": get_rate([r["kind"] in REFUSAL_KINDS for r in positives]),
        "must_include_pass_rate": get_rate([r["include_ok"] for r in records]),
        "must_not_include_violations": sum(len(r["exclude_hits"]) for r in records),
        "guard_violations": sum(len(r["guard_violations"]) for r in records),
        "evidence_hit_rate": get_rate([r["evidence_hit"] for r in records]),
        "cited_gold_file_rate": get_rate([r["cited_gold_file"] for r in records]),
        "multiturn_chains_passed": count_passed_chains(records),
        "errors": sum(1 for r in records if r["error"]),
        "latency_ms": describe([r["latency_ms"] for r in records]),
    }


def get_category_key(record: dict) -> str:
    """多轮题的首轮与追问分开统计"""
    if record["category"] != "multiturn":
        return record["category"]
    if record["turn"] == 1:
        return "multiturn_t1"
    return "multiturn_followup"


def summarize_records(records: list) -> dict:
    ranks_by_category = {}
    file_ranks_by_category = {}
    for record in records:
        if not record["has_gold"]:
            continue
        key = get_category_key(record)
        if key not in ranks_by_category:
            ranks_by_category[key] = []
            file_ranks_by_category[key] = []
        ranks_by_category[key].append(record["rank"])
        file_ranks_by_category[key].append(record["file_rank"])
    by_category = {}
    file_level_by_category = {}
    for key in sorted(ranks_by_category):
        by_category[key] = summarize(ranks_by_category[key])
        file_level_by_category[key] = summarize(file_ranks_by_category[key])
    all_ranks = [r["rank"] for r in records if r["has_gold"]]
    answerable = []
    unanswerable = []
    for record in records:
        if record["top_dense"] is None:
            continue
        if record["has_gold"]:
            answerable.append(record["top_dense"])
        if record["expect"] in ("refuse", "realtime_notice"):
            unanswerable.append(record["top_dense"])
    return {
        "by_category": by_category,
        "file_level_by_category": file_level_by_category,
        "overall": summarize(all_ranks),
        "top_dense_answerable": describe(answerable),
        "top_dense_unanswerable": describe(unanswerable),
        "latency_ms": describe([r["latency_ms"] for r in records]),
    }


def print_summary(summary: dict):
    print("\n类别                     n   hit@1  hit@3  hit@5  hit@10  mrr@10   (文件级 hit@5)")
    rows = dict(summary["by_category"])
    rows["overall"] = summary["overall"]
    for key, s in rows.items():
        f5 = summary["file_level_by_category"].get(key, {}).get("hit@5", "")
        print(
            f"{key:<22} {s['n']:>3}  {s['hit@1']:.3f}  {s['hit@3']:.3f}  {s['hit@5']:.3f}  {s['hit@10']:.3f}   "
            f"{s['mrr@10']:.3f}   {f5}"
        )
    print(f"\n稠密最高分  可回答：{summary['top_dense_answerable']}")
    print(f"稠密最高分  不可回答：{summary['top_dense_unanswerable']}")
    print(f"检索延迟 ms：{summary['latency_ms']}")


def build_meta(args: argparse.Namespace, path: Path, n_items: int) -> dict:
    """结果存档的 meta：运行参数与影响检索结果的配置"""
    if path.is_relative_to(PROJECT_ROOT):
        eval_set = path.relative_to(PROJECT_ROOT).as_posix()
    else:
        eval_set = str(path)
    return {
        "tag": args.tag,
        "mode": args.mode,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "git": get_git_rev(),
        "eval_set": eval_set,
        "n_items": n_items,
        "config": {
            "embedding": Path(embedding_config.bge_m3_path).name,
            "bge_max_length": embedding_config.bge_max_length,
            "retrieval": f"dense(COSINE,HNSW)+sparse(IP) 各取 {CANDIDATES}，RRF k=60，top {TOP_K}",
            "embed_text": "document_title · content_type · section_path + body（表格为线性化文本）",
            "chunk": {
                "target": CHUNK_TARGET_CHARS,
                "max": CHUNK_MAX_CHARS,
                "min": CHUNK_MIN_CHARS,
                "table_max": TABLE_MAX_CHARS,
            },
            "multiturn_query": "历史问题与当前问题直接拼接（M1 无会话状态）",
        },
    }


def main(argv=None) -> int:
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
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    items = load_eval_items(path)
    errors = self_check(items, load_corpus())
    if errors:
        print(f"评测集自检失败（{len(errors)} 处）：")
        for e in errors:
            print("  ✗", e)
        return 1
    counts = {}
    for item in items:
        counts[item["category"]] = counts.get(item["category"], 0) + 1
    print(f"评测集自检通过：{len(items)} 题 {counts}")
    if args.check_only:
        return 0

    if args.mode == "answer":
        records = run_answers(items)
        summary = summarize_answers(records)
        print("\n" + json.dumps(summary, ensure_ascii=False, indent=1))
    else:
        records = run_retrieval(items)
        summary = summarize_records(records)
        print_summary(summary)

    out_dir = PROJECT_ROOT / "data" / "eval" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{datetime.now():%Y-%m-%d}_{args.tag}.json"
    result = {"meta": build_meta(args, path, len(items)), "summary": summary, "records": records}
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n结果已存档：{out.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
