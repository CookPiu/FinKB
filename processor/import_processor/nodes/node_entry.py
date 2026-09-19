"""节点：入口（register）。计算哈希、判定新增 / 续跑 / 跳过、原件上传 MinIO、写 documents 记录。"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from common.logging.logger import logger, node_log
from common.models.document import STAGE_ORDER, DocStatus, DocumentRecord, Stage
from processor.import_processor.state import ImportGraphState
from utils import artifact_utils as artifacts
from utils.classify_utils import guess_content_type, title_from_filename
from utils.clients import minio_utils as minio
from utils.clients import mongo_utils as mongo


class RegisterAction(StrEnum):
    NEW = "new"  # 首次导入
    RESUME = "resume"  # 上次未完成，从断点继续
    FORCE = "force"  # 已就绪但要求从头重建
    SKIP = "skip"  # 同哈希已就绪，跳过


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def doc_id_from_hash(file_hash: str) -> str:
    return file_hash[:16]


def to_mongo(rec: DocumentRecord) -> dict:
    d = rec.model_dump(mode="json")
    d["created_at"], d["updated_at"] = rec.created_at, rec.updated_at
    d["_id"] = rec.doc_id
    return d


def load(doc_id: str) -> DocumentRecord | None:
    raw = mongo.get_db().documents.find_one({"_id": doc_id})
    return DocumentRecord.model_validate(raw) if raw else None


def register(path: Path, root: Path, *, force: bool = False) -> tuple[DocumentRecord, RegisterAction]:
    db = mongo.get_db()
    file_hash = file_sha256(path)
    doc_id = doc_id_from_hash(file_hash)
    now = datetime.now(UTC)

    existing = load(doc_id)
    if existing is not None:
        # 流水线新增了阶段时，旧的“就绪”文档只补跑新阶段
        complete = existing.status == DocStatus.READY and existing.stage == STAGE_ORDER[-1]
        if complete and not force:
            return existing, RegisterAction.SKIP
        if force and existing.status == DocStatus.READY:
            # 版本号由切分节点递增（每产生一套新切片即一个新版本）
            existing.stage = Stage.REGISTER
            existing.status = DocStatus.PENDING
            existing.error = None
            existing.updated_at = now
            db.documents.replace_one({"_id": doc_id}, to_mongo(existing))
            return existing, RegisterAction.FORCE
        # 未完成：文件路径可能变化，刷新本地路径后从断点继续
        existing.local_path = str(path)
        db.documents.update_one({"_id": doc_id}, {"$set": {"local_path": str(path), "updated_at": now}})
        return existing, RegisterAction.RESUME

    rel_dir = path.parent.relative_to(root).as_posix() if path.parent != root else ""
    source_path = minio.upload_file(f"originals/{doc_id}/{path.name}", path)
    # 同名文件内容变化：新文档就绪后替换旧文档（入库节点删除旧切片）
    supersedes = [
        d["_id"]
        for d in db.documents.find(
            {"file_name": path.name, "_id": {"$ne": doc_id}, "status": {"$ne": DocStatus.SUPERSEDED.value}},
            {"_id": 1},
        )
    ]
    rec = DocumentRecord(
        doc_id=doc_id,
        file_name=path.name,
        file_ext=path.suffix.lower(),
        file_hash=file_hash,
        file_size=path.stat().st_size,
        rel_dir=rel_dir,
        local_path=str(path),
        source_path=source_path,
        content_type=guess_content_type(rel_dir, path.name),
        content_type_source="dir_rule",
        document_title=title_from_filename(path.stem),
        version=0,
        stage=Stage.REGISTER,
        status=DocStatus.PENDING,
        artifacts_dir=str(artifacts.doc_dir(doc_id)),
        supersedes=supersedes,
        created_at=now,
        updated_at=now,
    )
    db.documents.insert_one(to_mongo(rec))
    if supersedes:
        logger.info("%s 内容已变化，就绪后将替换旧文档 %s", path.name, supersedes)
    return rec, RegisterAction.NEW


@node_log("node_entry")
def node_entry(state: ImportGraphState) -> dict:
    path = Path(state["local_file_path"])
    root = Path(state.get("root_dir") or path.parent)
    doc, action = register(path, root, force=state.get("force", False))
    if action != RegisterAction.SKIP:
        logger.info("%-9s %s（%s）", "register", doc.file_name, action.value)
    return {"doc": doc, "action": action.value, "file_name": doc.file_name}
