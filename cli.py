"""命令行入口：uv run python cli.py check | ingest | status | search | ask。"""
import argparse
import logging
import sys
import time
from pathlib import Path

from common.logging.logger import set_level

# 各命令用到的模块放在命令函数里导入：torch、LangGraph 等加载较慢，不用它们的命令（如 status）不必付出这个代价


def run_probe(name: str, probe) -> bool:
    """执行一项连通性检查并打印结果与耗时"""
    start_ts = time.perf_counter()
    try:
        detail = probe()
        print(f"  [OK]   {name:<10} {detail}  ({(time.perf_counter() - start_ts) * 1000:.0f} ms)")
        return True
    except Exception as e:
        print(f"  [FAIL] {name:<10} {type(e).__name__}: {e}")
        return False


def probe_milvus() -> str:
    from common.config.milvus_config import milvus_config
    from utils.clients.milvus_utils import get_milvus_client

    client = get_milvus_client()
    return f"{milvus_config.milvus_uri} server={client.get_server_version()} collections={len(client.list_collections())}"


def probe_mongo() -> str:
    from common.config.mongo_config import mongo_config
    from utils.clients.mongo_utils import get_db

    db = get_db()
    db.command("ping")
    return f"db={mongo_config.mongo_db} collections={db.list_collection_names()}"


def probe_minio() -> str:
    from common.config.minio_config import minio_config
    from utils.clients.minio_utils import get_minio_client

    get_minio_client()
    return f"bucket={minio_config.bucket} 就绪"


def probe_llm() -> str:
    from common.config.lm_config import lm_config
    from utils.lm.lm_utils import chat

    text = chat([{"role": "user", "content": "只回复 OK"}], max_tokens=5)
    return f"model={lm_config.llm_model} reply={text.strip()!r}"


def probe_rerank() -> str:
    import requests

    from common.config.lm_config import lm_config
    from common.config.reranker_config import reranker_config
    from utils.lm.reranker_utils import get_rerank_endpoint

    resp = requests.post(
        get_rerank_endpoint(),
        headers={"Authorization": f"Bearer {lm_config.api_key}", "Content-Type": "application/json"},
        json={"model": reranker_config.rerank_model, "query": "基金托管费", "documents": ["托管费率 0.10%", "今天天气晴"]},
        timeout=30,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
    results = resp.json().get("results") or resp.json().get("output", {}).get("results")
    top = None
    if results:
        top = results[0]
    return f"model={reranker_config.rerank_model} top={top}"


def probe_mineru() -> str:
    from utils.clients.mineru_utils import check_token

    return check_token()


def probe_bge() -> str:
    from utils.lm.embedding_utils import generate_query_embedding

    embedding = generate_query_embedding("连通性检查")
    return f"dense={len(embedding['dense'])} sparse_nnz={len(embedding['sparse'])}"


def cmd_check(args: argparse.Namespace) -> int:
    probes = [
        ("Milvus", probe_milvus),
        ("MongoDB", probe_mongo),
        ("MinIO", probe_minio),
        ("LLM", probe_llm),
        ("Rerank", probe_rerank),
        ("MinerU", probe_mineru),
    ]
    if args.models:
        probes.append(("BGE-M3", probe_bge))

    print("连通性检查：")
    # 每项都要执行，不能在第一项失败时短路
    results = [run_probe(name, probe) for name, probe in probes]
    if all(results):
        return 0
    return 1


def cmd_ingest(args: argparse.Namespace) -> int:
    from processor.import_processor.main_graph import import_directory

    report = import_directory(Path(args.dir), force=args.force, reparse=args.reparse, only=args.only)
    print(f"\n运行 {report['task_id']}：扫描 {report['scanned']} 个文件")
    for action, names in report["actions"].items():
        print(f"  {action:<7} {len(names)}")
    print(f"  本次就绪 {len(report['ready'])}，失败 {len(report['failed'])}")
    for name, error in report["failed"].items():
        print(f"    ✗ {name}: {error}")
    if report["failed"]:
        return 1
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    from utils.clients.mongo_utils import get_db

    fields = {"file_name": 1, "status": 1, "stage": 1, "version": 1, "page_count": 1, "chunk_count": 1, "error": 1}
    rows = list(get_db().documents.find({}, fields).sort("file_name", 1))
    counts = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
        print(
            f"{row['_id']}  {row['status']:<10} {str(row.get('stage')):<9} v{row.get('version')} "
            f"p={row.get('page_count')} c={row.get('chunk_count')}  {row['file_name']}"
        )
        if row.get("error"):
            print(f"    error: {row['error']}")
    print(f"共 {len(rows)} 个文档：{counts}")
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    from utils.search_utils import semantic_search

    hits = semantic_search(args.query, top_k=args.k, kinds=args.kind, content_types=args.content_type)
    for i, hit in enumerate(hits, 1):
        print(
            f"\n#{i} rrf={hit['score_rrf']:.4f} dense={hit['score_dense']} sparse={hit['score_sparse']} "
            f"[{hit['kind']}] {hit['file_name']} p{hit['page_start']}-{hit['page_end']}"
        )
        print(f"   § {hit['section_path']}")
        if args.full:
            text = hit["text"]
        else:
            text = hit["text"][:300].replace("\n", " ⏎ ")
        print(f"   {text}")
    return 0


def print_ask_event(event: dict):
    """问答流式事件：回答增量直接接着打印，引用来源另起一段"""
    if event["type"] == "delta":
        print(event["text"], end="", flush=True)
    elif event["type"] == "sources":
        print("\n\n引用来源：\n" + event["text"], flush=True)


def read_question():
    """交互模式读一个问题；输入结束、空行或 exit/quit 时返回 None"""
    try:
        question = input("\n问> ").strip()
    except EOFError:
        return None
    if not question or question in ("exit", "quit"):
        return None
    return question


def cmd_ask(args: argparse.Namespace) -> int:
    from processor.query_processor.main_graph import ask

    session_id = args.session
    questions = []
    if args.question:
        questions.append(args.question)
    interactive = not questions
    while True:
        if interactive:
            question = read_question()
            if question is None:
                break
        else:
            if not questions:
                break
            question = questions.pop(0)
        result = ask(question, session_id=session_id, on_event=print_ask_event)
        session_id = result["session_id"]
        print(f"\n\n[kind={result.get('kind')} session={session_id}]")
    return 0


def build_parser() -> argparse.ArgumentParser:
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
    return parser


def main(argv=None) -> int:
    # Windows 控制台默认本地代码页，中文与特殊符号会乱码或报错
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    args = build_parser().parse_args(argv)
    if args.verbose:
        set_level(logging.DEBUG)
    else:
        set_level(logging.INFO)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
