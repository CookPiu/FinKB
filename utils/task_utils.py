"""
导入进度：documents.status 记录文档当前的导入结果，失败原因写进 documents.error。
一次导入从头跑到尾，中途失败的文档下次重跑整条流水线（解析结果按文件哈希缓存在 data/artifacts，不会重复调用 MinerU）。
"""
from datetime import datetime, timezone

from common.logging.logger import logger
from utils.clients.mongo_utils import get_db

# 支持导入的文件类型
INGEST_EXTS = {".pdf", ".doc", ".docx", ".ppt", ".pptx", ".md"}

# 文档状态
STATUS_RUNNING = "running"  # 正在导入
STATUS_FAILED = "failed"  # 导入失败，原因见 error
STATUS_READY = "ready"  # 导入完成，可被检索


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


def save_doc_fields(doc: dict, fields: dict):
    """把节点产出的字段写回内存中的文档与 Mongo"""
    doc.update(fields)
    doc["updated_at"] = datetime.now(timezone.utc)
    update = dict(fields)
    update["updated_at"] = doc["updated_at"]
    get_db().documents.update_one({"_id": doc["doc_id"]}, {"$set": update})


def mark_running(doc: dict):
    save_doc_fields(doc, {"status": STATUS_RUNNING, "error": None})


def mark_ready(doc: dict, fields=None):
    """整条流水线跑完：文档就绪；fields 为最后一个节点产出的字段"""
    update = dict(fields or {})
    update["status"] = STATUS_READY
    update["error"] = None
    save_doc_fields(doc, update)


def mark_failed(doc_id: str, file_name: str, error):
    """某个节点抛异常时由 import_file 统一标记，只影响这一个文档"""
    if isinstance(error, BaseException):
        logger.debug("失败详情", exc_info=error)
    get_db().documents.update_one(
        {"_id": doc_id},
        {"$set": {"status": STATUS_FAILED, "error": str(error), "updated_at": datetime.now(timezone.utc)}},
    )
    logger.error(f"{file_name} 导入失败：{error}")
