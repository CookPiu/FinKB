"""导入图（LangGraph）：一次执行处理一个文件。

node_entry → node_parse → node_normalize → node_document_split → node_bge_embedding → node_import_milvus → node_enrich

每个节点通过 utils/task_utils.run_stage 读写 documents.stage：已完成的阶段直接跳过，因此中断后重跑即从断点续跑；
同哈希且已就绪的文件在 node_entry 判定为 skip，直接结束。
某个节点失败时标记文档 failed 并抛出，图终止；import_directory 捕获后继续处理下一个文件。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from langgraph.graph import END, StateGraph

from common.logging.logger import logger
from processor.import_processor.nodes.node_bge_embedding import node_bge_embedding
from processor.import_processor.nodes.node_document_split import node_document_split
from processor.import_processor.nodes.node_enrich import node_enrich
from processor.import_processor.nodes.node_entry import node_entry
from processor.import_processor.nodes.node_import_milvus import node_import_milvus
from processor.import_processor.nodes.node_normalize import node_normalize
from processor.import_processor.nodes.node_parse import node_parse
from processor.import_processor.state import ImportGraphState, create_default_state
from utils.clients import mongo_utils as mongo
from utils.task_utils import scan

workflow = StateGraph(ImportGraphState)
workflow.add_node("node_entry", node_entry)
workflow.add_node("node_parse", node_parse)
workflow.add_node("node_normalize", node_normalize)
workflow.add_node("node_document_split", node_document_split)
workflow.add_node("node_bge_embedding", node_bge_embedding)
workflow.add_node("node_import_milvus", node_import_milvus)
workflow.add_node("node_enrich", node_enrich)
workflow.set_entry_point("node_entry")


def route_after_entry(state: ImportGraphState) -> str:
    """同哈希且已就绪的文件直接结束。"""
    return END if state.get("action") == "skip" else "node_parse"


workflow.add_conditional_edges("node_entry", route_after_entry, {"node_parse": "node_parse", END: END})
workflow.add_edge("node_parse", "node_normalize")
workflow.add_edge("node_normalize", "node_document_split")
workflow.add_edge("node_document_split", "node_bge_embedding")
workflow.add_edge("node_bge_embedding", "node_import_milvus")
workflow.add_edge("node_import_milvus", "node_enrich")
workflow.add_edge("node_enrich", END)

kb_import_app = workflow.compile()


@dataclass
class ImportReport:
    task_id: str
    scanned: int = 0
    actions: dict[str, list[str]] = field(default_factory=dict)
    ready: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)


def import_file(path: Path, *, root: Path | None = None, task_id: str = "", force: bool = False, reparse: bool = False) -> dict:
    """导入单个文件，返回图的最终状态。失败时抛出异常（文档状态已标记为 failed）。"""
    state = create_default_state(
        task_id=task_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%S"),
        local_file_path=str(path),
        root_dir=str(root or path.parent),
        force=force,
        reparse=reparse,
    )
    return kb_import_app.invoke(state)


def import_directory(root: Path, *, force: bool = False, reparse: bool = False, only: str | None = None) -> ImportReport:
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"目录不存在：{root}")
    mongo.ensure_indexes()
    report = ImportReport(task_id=datetime.now(UTC).strftime("%Y%m%dT%H%M%S"))
    files = [p for p in scan(root) if not only or only in p.name]
    report.scanned = len(files)
    for path in files:
        try:
            final = import_file(path, root=root, task_id=report.task_id, force=force, reparse=reparse)
        except Exception as e:  # noqa: BLE001 - 单个文件失败不影响其他文件
            report.failed[path.name] = str(e)
            continue
        report.actions.setdefault(final.get("action", ""), []).append(path.name)
        doc = final.get("doc")
        if doc is not None and doc.status == "ready" and final.get("action") != "skip":
            report.ready.append(path.name)
    logger.info("扫描 %d 个文件：%s，失败 %d", len(files), {k: len(v) for k, v in report.actions.items()}, len(report.failed))
    return report
