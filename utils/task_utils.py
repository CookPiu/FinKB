"""导入进度：每个文档的进度记在 documents.stage（最后完成的阶段）与 documents.status。

节点通过 run_stage() 执行自己的阶段：已完成则跳过（断点续跑），失败则标记后抛出（图终止，只影响这一个文件）。
"""

from __future__ import annotations

import traceback
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from common.logging.logger import logger
from common.models.document import STAGE_ORDER, DocStatus, DocumentRecord, Stage
from utils.clients import mongo_utils as mongo

INGEST_EXTS = {".pdf", ".doc", ".docx", ".ppt", ".pptx", ".md"}


def scan(root: Path) -> list[Path]:
    files = [
        p
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in INGEST_EXTS and not p.name.startswith(("~$", "."))
    ]
    return sorted(files, key=lambda p: p.as_posix())


def save_doc(doc: DocumentRecord, extra: dict | None = None) -> None:
    doc.updated_at = datetime.now(UTC)
    fields = {
        "stage": doc.stage.value if doc.stage else None,
        "status": doc.status.value,
        "error": doc.error,
        "updated_at": doc.updated_at,
        **(extra or {}),
    }
    mongo.get_db().documents.update_one({"_id": doc.doc_id}, {"$set": fields})


def stage_done(doc: DocumentRecord, stage: Stage) -> bool:
    return doc.stage is not None and STAGE_ORDER.index(Stage(doc.stage)) >= STAGE_ORDER.index(stage)


def mark_running(doc: DocumentRecord) -> None:
    doc.status = DocStatus.RUNNING
    save_doc(doc)


def mark_done(doc: DocumentRecord, stage: Stage, **fields) -> None:
    doc.stage = stage
    doc.status = DocStatus.READY if stage == STAGE_ORDER[-1] else DocStatus.PENDING
    doc.error = None
    for k, v in fields.items():
        setattr(doc, k, v)
    save_doc(doc, fields)


def mark_failed(doc: DocumentRecord, stage: Stage, err: BaseException | str) -> None:
    if isinstance(err, BaseException):
        logger.debug("%s", "".join(traceback.format_exception(err)))
    doc.status = DocStatus.FAILED
    doc.error = f"[{stage.value}] {err}"
    save_doc(doc)
    logger.error("%s 在 %s 阶段失败：%s", doc.file_name, stage.value, err)


def run_stage(doc: DocumentRecord, stage: Stage, fn: Callable[[DocumentRecord], dict | None]) -> bool:
    """执行一个阶段：已完成返回 False（跳过）；否则执行 fn 并记录进度，返回 True。fn 返回要写回文档的字段。"""
    if stage_done(doc, stage):
        logger.info("%-9s 跳过（已完成） %s", stage.value, doc.file_name)
        return False
    mark_running(doc)
    try:
        fields = fn(doc) or {}
    except Exception as e:
        mark_failed(doc, stage, e)
        raise
    mark_done(doc, stage, **fields)
    return True
