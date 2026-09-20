"""
导入服务：上传文件后在后台执行导入图，查询文档导入状态。
启动：uv run python -m api.file_import_service（或 uv run uvicorn api.file_import_service:app --port 8000）
导入任务串行执行（同一时刻只跑一个文件，避免 CPU 上的向量化互相争抢）；单个文件失败只标记该文档，不影响服务。
"""
import threading
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from common.logging.logger import logger
from processor.import_processor.main_graph import import_file
from utils.clients.mongo_utils import get_db
from utils.path_util import PROJECT_ROOT
from utils.task_utils import INGEST_EXTS

UPLOAD_DIR = PROJECT_ROOT / "data" / "uploads"
# 后台任务在线程池里执行，用锁保证同一时刻只导入一个文件
_import_lock = threading.Lock()

app = FastAPI(title="FinKB 导入服务", description="文档上传与导入", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def run_import(path: Path):
    """后台任务：导入一个已保存的上传文件"""
    with _import_lock:
        try:
            import_file(path, root=UPLOAD_DIR)
        except Exception as e:  # 文档状态已由节点标记为 failed
            logger.error(f"导入失败 {path.name}：{e}")


@app.post("/documents", summary="上传文件并触发导入", description="支持 pdf/doc/docx/ppt/pptx/md，导入在后台执行")
async def upload_documents(background_tasks: BackgroundTasks, files: list[UploadFile] = File(...)):
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for f in files:
        name = Path(f.filename or "").name
        if not name or Path(name).suffix.lower() not in INGEST_EXTS:
            results.append({"file_name": name, "accepted": False, "reason": f"不支持的格式，仅支持 {sorted(INGEST_EXTS)}"})
            continue
        dest = UPLOAD_DIR / name
        dest.write_bytes(await f.read())
        background_tasks.add_task(run_import, dest)
        results.append({"file_name": name, "accepted": True})
    return {"files": results}


@app.get("/documents", summary="文档与导入状态")
def list_documents():
    fields = {"_id": 1, "file_name": 1, "content_type": 1, "status": 1, "stage": 1, "chunk_count": 1,
              "page_count": 1, "error": 1, "source_path": 1, "updated_at": 1}
    docs = get_db().documents.find({"status": {"$ne": "superseded"}}, fields).sort("file_name", 1)
    documents = []
    for doc in docs:
        # 与原实现一致：保留 _id，另加 doc_id
        item = dict(doc)
        item["doc_id"] = doc["_id"]
        documents.append(item)
    return {"documents": documents}


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
