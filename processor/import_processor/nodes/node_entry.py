"""节点：入口（register）。计算哈希、判定新增 / 重做 / 跳过、原件上传 MinIO、写 documents 记录。"""
import hashlib
from datetime import datetime, timezone
from pathlib import Path

from common.logging.logger import logger, node_log, step_log
from processor.import_processor.state import ImportGraphState, create_default_state
from utils.classify_utils import guess_content_type, title_from_filename
from utils.clients.minio_utils import upload_file
from utils.clients.mongo_utils import get_db
from utils.task_utils import STATUS_READY, STATUS_RUNNING

# 登记结果
ACTION_NEW = "new"  # 首次导入
ACTION_REDO = "redo"  # 库里已有记录但没就绪，或指定了 --force：整条流水线重做
ACTION_SKIP = "skip"  # 同哈希已就绪，跳过


def get_file_sha256(path: Path) -> str:
    """分块读取文件计算 SHA-256，避免大文件一次读入内存"""
    sha256 = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(1 << 20)
            if not block:
                break
            sha256.update(block)
    return sha256.hexdigest()


def build_doc_id(file_hash: str) -> str:
    """doc_id 取文件哈希前 16 位：内容不变则 doc_id 不变，解析缓存与切片 ID 都能复用"""
    return file_hash[:16]


def load_document(doc_id: str):
    """
    从 Mongo 读取文档记录
    :return: 文档记录 dict；不存在时返回 None
    """
    return get_db().documents.find_one({"_id": doc_id})


@step_log("build_new_document")
def build_new_document(path: Path, root: Path, file_hash: str, now) -> dict:
    """为首次导入的文件建立文档记录：原件上传 MinIO，查出同名的旧文档（入库成功后删除）"""
    doc_id = build_doc_id(file_hash)
    if path.parent != root:
        rel_dir = path.parent.relative_to(root).as_posix()
    else:
        rel_dir = ""
    source_path = upload_file(f"originals/{doc_id}/{path.name}", path)
    # 同名文件内容变化：新文档入库成功后，入库节点删掉旧文档的切片与记录
    supersedes = []
    for old in get_db().documents.find({"file_name": path.name, "_id": {"$ne": doc_id}}, {"_id": 1}):
        supersedes.append(old["_id"])
    return {
        "_id": doc_id,
        "doc_id": doc_id,
        "file_name": path.name,
        "file_ext": path.suffix.lower(),
        "file_hash": file_hash,
        "local_path": str(path),
        "source_path": source_path,
        "content_type": guess_content_type(rel_dir, path.name),
        "document_title": title_from_filename(path.stem),
        "version": 0,
        "status": STATUS_RUNNING,
        "error": None,
        "page_count": None,
        "chunk_count": None,
        "supersedes": supersedes,
        "summary": None,
        "created_at": now,
        "updated_at": now,
    }


@step_log("register_document")
def register_document(path: Path, root: Path, force: bool = False):
    """
    登记文件，判定本次要做的动作
    :return: (文档记录, 动作 new / redo / skip)
    """
    db = get_db()
    file_hash = get_file_sha256(path)
    doc_id = build_doc_id(file_hash)
    now = datetime.now(timezone.utc)

    existing = load_document(doc_id)
    if existing is not None:
        # 内容没变又已经就绪：无需再做（--force 时仍重做一遍）
        if existing["status"] == STATUS_READY and not force:
            return existing, ACTION_SKIP
        # 上次失败或要求重建：整条流水线从头再跑一次；版本号由切分节点递增
        existing["local_path"] = str(path)
        existing["status"] = STATUS_RUNNING
        existing["error"] = None
        existing["updated_at"] = now
        db.documents.update_one(
            {"_id": doc_id},
            {"$set": {"local_path": str(path), "status": STATUS_RUNNING, "error": None, "updated_at": now},
             "$unset": {"stage": ""}},  # 旧版本记过阶段，这里顺手清掉
        )
        return existing, ACTION_REDO

    doc = build_new_document(path, root, file_hash, now)
    db.documents.insert_one(doc)
    if doc["supersedes"]:
        logger.info(f"{path.name} 内容已变化，就绪后将替换旧文档 {doc['supersedes']}")
    return doc, ACTION_NEW


@step_log("validate_and_get_data")
def validate_and_get_data(state: ImportGraphState):
    """
    取出并校验登记所需的入参
    :return: 元组 (文件路径, 导入根目录, 是否强制重建)；root_dir 为空时取文件所在目录
    :raise ValueError: 没有传入文件路径
    """
    local_file_path = state.get("local_file_path")
    if not local_file_path:
        logger.error("no local_file_path found in state")
        raise ValueError("no local_file_path found in state")
    path = Path(local_file_path)
    root = Path(state.get("root_dir") or path.parent)
    return path, root, state.get("force", False)


@node_log("node_entry")
def node_entry(state: ImportGraphState):
    """
    节点功能：登记待导入文件，决定后续流程。
    同哈希且已就绪 → skip（路由直接结束）；上次失败或指定 --force → redo（整条流水线重做）；
    新文件 → new（上传原件、写入 documents 记录）。
    """
    path, root, force = validate_and_get_data(state)
    doc, action = register_document(path, root, force)
    if action != ACTION_SKIP:
        logger.info(f"register  {doc['file_name']}（{action}）")
    state["doc"] = doc
    state["action"] = action
    state["file_name"] = doc["file_name"]
    return state


if __name__ == "__main__":
    # 运行：uv run python -m processor.import_processor.nodes.node_entry <文件路径>
    # 依赖 Mongo、MinIO；新文件会写入 documents 记录并上传原件，已就绪的文件只返回 skip
    import sys

    result = node_entry(create_default_state(local_file_path=sys.argv[1]))
    logger.info(f"action={result['action']} doc_id={result['doc']['doc_id']} status={result['doc']['status']}")
