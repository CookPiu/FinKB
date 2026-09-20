"""
导入进度：每个文档的进度记在 Mongo documents 集合的 stage（最后完成的阶段）与 status 字段。
节点开始前先用 is_stage_done 判断：已完成就跳过（断点续跑）；失败时标记 failed 后抛出，图终止，只影响这一个文件。
"""
from datetime import datetime, timezone

from common.logging.logger import logger
from utils.clients.mongo_utils import get_db

# 支持导入的文件类型
INGEST_EXTS = {".pdf", ".doc", ".docx", ".ppt", ".pptx", ".md"}

# 流水线阶段，按执行顺序排列。enrich 放在 index 之后：它只写 Mongo（财务事实、摘要），调整它不需要重新向量化
STAGE_REGISTER = "register"
STAGE_PARSE = "parse"
STAGE_NORMALIZE = "normalize"
STAGE_CHUNK = "chunk"
STAGE_INDEX = "index"
STAGE_ENRICH = "enrich"
STAGE_ORDER = [STAGE_REGISTER, STAGE_PARSE, STAGE_NORMALIZE, STAGE_CHUNK, STAGE_INDEX, STAGE_ENRICH]

# 文档状态
STATUS_PENDING = "pending"  # 等待下一阶段
STATUS_RUNNING = "running"
STATUS_FAILED = "failed"
STATUS_READY = "ready"  # 全部阶段完成
STATUS_SUPERSEDED = "superseded"  # 同名文件内容变化后被新文档替换


def scan_files(root):
    """
    递归列出目录下所有可导入的文件（跳过 Office 临时文件 ~$xxx 与隐藏文件），按路径排序
    :param root: 目录（Path）
    :return: [Path, ...]
    """
    files = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in INGEST_EXTS:
            continue
        if path.name.startswith("~$") or path.name.startswith("."):
            continue
        files.append(path)
    files.sort(key=lambda p: p.as_posix())
    return files


def is_stage_done(doc: dict, stage: str) -> bool:
    """文档是否已完成 stage 阶段（最后完成的阶段不早于 stage）"""
    done_stage = doc.get("stage")
    if not done_stage:
        return False
    return STAGE_ORDER.index(done_stage) >= STAGE_ORDER.index(stage)


def save_doc_progress(doc: dict, fields=None):
    """把进度字段（及阶段产出的字段）写回 Mongo"""
    doc["updated_at"] = datetime.now(timezone.utc)
    update = {
        "stage": doc.get("stage"),
        "status": doc.get("status"),
        "error": doc.get("error"),
        "updated_at": doc["updated_at"],
    }
    if fields:
        update.update(fields)
    get_db().documents.update_one({"_id": doc["doc_id"]}, {"$set": update})


def mark_stage_running(doc: dict):
    doc["status"] = STATUS_RUNNING
    save_doc_progress(doc)


def mark_stage_done(doc: dict, stage: str, fields=None):
    """
    标记阶段完成：最后一个阶段完成后文档就绪，否则等待下一阶段
    :param fields: 本阶段产出、要一并写回文档的字段，如 {"page_count": 12}
    """
    doc["stage"] = stage
    if stage == STAGE_ORDER[-1]:
        doc["status"] = STATUS_READY
    else:
        doc["status"] = STATUS_PENDING
    doc["error"] = None
    if fields:
        doc.update(fields)
    save_doc_progress(doc, fields)


def mark_stage_failed(doc: dict, stage: str, error):
    """标记失败并记录原因；异常堆栈只在 DEBUG 级别输出"""
    if isinstance(error, BaseException):
        logger.debug("失败详情", exc_info=error)
    doc["status"] = STATUS_FAILED
    doc["error"] = f"[{stage}] {error}"
    save_doc_progress(doc)
    logger.error(f"{doc['file_name']} 在 {stage} 阶段失败：{error}")
