"""命令行入口：uv run python cli.py check | ingest | status | search | ask。"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from common.logging.logger import set_level


def _probe(name: str, fn) -> bool:
    t0 = time.perf_counter()
    try:
        detail = fn()
        print(f"  [OK]   {name:<10} {detail}  ({(time.perf_counter() - t0) * 1000:.0f} ms)")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  [FAIL] {name:<10} {type(e).__name__}: {e}")
        return False


def cmd_check(args: argparse.Namespace) -> int:
    import requests

    from common.config.settings import get_settings
    from utils.clients import milvus_utils as milvus
    from utils.clients import minio_utils as minio
    from utils.clients import mongo_utils as mongo
    from utils.lm import lm_utils as llm
    from utils.clients import mineru_utils as mineru

    s = get_settings()

    def p_milvus():
        c = milvus.get_client()
        return f"{s.milvus_uri} server={c.get_server_version()} collections={len(c.list_collections())}"

    def p_mongo():
        db = mongo.get_db()
        db.command("ping")
        return f"db={s.mongo_db} collections={db.list_collection_names()}"

    def p_minio():
        minio.get_client()
        return f"bucket={s.minio_bucket} 就绪"

    def p_llm():
        text = llm.chat([{"role": "user", "content": "只回复 OK"}], max_tokens=5)
        return f"model={s.llm_model} reply={text.strip()!r}"

    def p_rerank():
        r = requests.post(
            s.rerank_endpoint,
            headers={"Authorization": f"Bearer {s.openai_api_key}", "Content-Type": "application/json"},
            json={"model": s.rerank_model, "query": "基金托管费", "documents": ["托管费率 0.10%", "今天天气晴"]},
            timeout=30,
        )
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
        results = r.json().get("results") or r.json().get("output", {}).get("results")
        return f"model={s.rerank_model} top={results[0] if results else None}"

    def p_mineru():
        return mineru.check_token()

    probes = [
        ("Milvus", p_milvus),
        ("MongoDB", p_mongo),
        ("MinIO", p_minio),
        ("LLM", p_llm),
        ("Rerank", p_rerank),
        ("MinerU", p_mineru),
    ]
    if args.models:
        from utils.lm import embedding_utils as embedding

        def p_bge():
            enc = embedding.encode_query("连通性检查")
            return f"dense={len(enc.dense)} sparse_nnz={len(enc.sparse)}"

        probes.append(("BGE-M3", p_bge))

    print("连通性检查：")
    ok = [_probe(n, f) for n, f in probes]
    return 0 if all(ok) else 1


def cmd_ingest(args: argparse.Namespace) -> int:
    from processor.import_processor.main_graph import import_directory

    rep = import_directory(Path(args.dir), force=args.force, reparse=args.reparse, only=args.only)
    print(f"\n运行 {rep.task_id}：扫描 {rep.scanned} 个文件")
    for action, names in rep.actions.items():
        print(f"  {action:<7} {len(names)}")
    print(f"  本次就绪 {len(rep.ready)}，失败 {len(rep.failed)}")
    for name, err in rep.failed.items():
        print(f"    ✗ {name}: {err}")
    return 1 if rep.failed else 0


def cmd_status(args: argparse.Namespace) -> int:
    from utils.clients import mongo_utils as mongo

    rows = list(
        mongo.get_db().documents.find(
            {}, {"file_name": 1, "status": 1, "stage": 1, "version": 1, "page_count": 1, "chunk_count": 1, "error": 1}
        ).sort("file_name", 1)
    )
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
        print(
            f"{r['_id']}  {r['status']:<10} {str(r.get('stage')):<9} v{r.get('version')} "
            f"p={r.get('page_count')} c={r.get('chunk_count')}  {r['file_name']}"
        )
        if r.get("error"):
            print(f"    error: {r['error']}")
    print(f"共 {len(rows)} 个文档：{counts}")
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    from utils.search_utils import semantic_search

    hits = semantic_search(args.query, top_k=args.k, kinds=args.kind, content_types=args.content_type)
    for i, h in enumerate(hits, 1):
        print(
            f"\n#{i} rrf={h.score_rrf:.4f} dense={h.score_dense} sparse={h.score_sparse} "
            f"[{h.kind}] {h.file_name} p{h.page_start}-{h.page_end}"
        )
        print(f"   § {h.section_path}")
        text = h.text if args.full else h.text[:300].replace("\n", " ⏎ ")
        print(f"   {text}")
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    from processor.query_processor.main_graph import ask

    def on_event(ev: dict) -> None:
        if ev["type"] == "delta":
            print(ev["text"], end="", flush=True)
        elif ev["type"] == "sources":
            print("\n\n引用来源：\n" + ev["text"], flush=True)

    sid = args.session
    questions = [args.question] if args.question else []
    interactive = not questions
    while True:
        if interactive:
            try:
                q = input("\n问> ").strip()
            except EOFError:
                break
            if not q or q in ("exit", "quit"):
                break
        else:
            if not questions:
                break
            q = questions.pop(0)
        result = ask(q, session_id=sid, on_event=on_event)
        sid = result["session_id"]
        print(f"\n\n[kind={result.get('kind')} session={sid}]")
    return 0


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(prog="finkb", description="FinKB 金融知识库")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("check", help="中间件与外部 API 连通性检查")
    p.add_argument("--models", action="store_true", help="同时加载 BGE-M3 并编码一次")
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("ingest", help="导入目录（可重复执行：续跑未完成文档、跳过未变化文档）")
    p.add_argument("dir")
    p.add_argument("--force", action="store_true", help="已就绪文档也从头重建（解析缓存仍然有效）")
    p.add_argument("--reparse", action="store_true", help="忽略解析缓存，重新调用 MinerU")
    p.add_argument("--only", help="只处理文件名包含该子串的文件")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("status", help="查看文档导入状态")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("search", help="语义检索调试（稠密 + 稀疏，RRF 融合）")
    p.add_argument("query")
    p.add_argument("-k", type=int, default=5)
    p.add_argument("--kind", action="append", help="按 kind 过滤，可重复")
    p.add_argument("--content-type", action="append", help="按内容类型过滤，可重复")
    p.add_argument("--full", action="store_true", help="打印完整文本")
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("ask", help="问答（流式输出）；不带问题时进入多轮交互")
    p.add_argument("question", nargs="?")
    p.add_argument("--session", help="沿用已有会话 ID（多轮追问）")
    p.set_defaults(func=cmd_ask)

    args = parser.parse_args(argv)
    set_level(logging.DEBUG if args.verbose else logging.INFO)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
