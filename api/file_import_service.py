"""
导入服务：上传文件后在后台执行导入图，查询导入任务进度与文档导入状态。
启动：uv run python -m api.file_import_service（或 uv run uvicorn api.file_import_service:app --port 8000）
查询服务把本服务挂在 /import 下，聊天页面的“资料导入”走的就是那一组接口，平时只启动查询服务即可。
导入任务串行执行（同一时刻只跑一个文件，避免 CPU 上的向量化互相争抢）；单个文件失败只标记该文档，不影响服务。
任务进度只保存在内存里，服务重启即清空；文档是否导入成功以 documents 集合为准。
"""
import copy
import threading
import time
import uuid
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from common.logging.logger import logger
from processor.import_processor.main_graph import import_file
from utils.clients.mongo_utils import get_db
from utils.path_util import PROJECT_ROOT
from utils.task_utils import INGEST_EXTS

UPLOAD_DIR = PROJECT_ROOT / "data" / "uploads"
MAX_TASKS = 50  # 内存里最多保留的任务数，超出时丢掉最早的已结束任务

# 任务状态
TASK_QUEUED = "queued"  # 排队等待（前一个文件还在导入）
TASK_RUNNING = "running"  # 正在导入
TASK_DONE = "done"  # 导入图跑完（含内容未变化直接跳过）
TASK_FAILED = "failed"  # 某个节点抛异常，原因见 error

# 后台任务在线程池里执行，用锁保证同一时刻只导入一个文件
_import_lock = threading.Lock()
# 导入任务进度：task_id → 任务 dict，按提交顺序保存；读写都在 _tasks_lock 内进行
_tasks = {}
_tasks_lock = threading.Lock()

app = FastAPI(title="FinKB 导入服务", description="文档上传与导入", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def new_task(file_name: str) -> dict:
    """登记一个导入任务，同时清掉超出上限的已结束任务"""
    task = {
        "task_id": uuid.uuid4().hex[:12],
        "file_name": file_name,
        "status": TASK_QUEUED,
        "action": "",  # node_entry 的登记结果 new / redo / skip
        "done_nodes": [],  # 已跑完的节点，按完成顺序
        "node_seconds": {},  # 节点名 → 耗时（秒）
        "doc_id": "",
        "chunk_count": None,
        "error": "",
        "submitted_at": time.time(),
        "node_started_at": None,  # 当前节点的开始时间
    }
    with _tasks_lock:
        _tasks[task["task_id"]] = task
        for task_id in list(_tasks):
            if len(_tasks) <= MAX_TASKS:
                break
            if _tasks[task_id]["status"] in (TASK_DONE, TASK_FAILED):
                del _tasks[task_id]
    return task


def update_task(task_id: str, fields: dict):
    with _tasks_lock:
        _tasks[task_id].update(fields)


def list_tasks() -> list:
    """任务快照（新的在前），正在跑的任务附带当前节点已用时间 running_seconds"""
    now = time.time()
    with _tasks_lock:
        tasks = copy.deepcopy(list(_tasks.values()))
    for task in tasks:
        task["running_seconds"] = None
        if task["status"] == TASK_RUNNING and task["node_started_at"]:
            task["running_seconds"] = round(now - task["node_started_at"], 1)
        del task["node_started_at"]
    tasks.reverse()
    return tasks


def run_import(task_id: str, path: Path):
    """后台任务：导入一个已保存的上传文件，每个节点跑完更新一次任务进度"""
    with _import_lock:
        update_task(task_id, {"status": TASK_RUNNING, "node_started_at": time.time()})

        def on_node(node_name: str, node_state: dict):
            now = time.time()
            with _tasks_lock:
                task = _tasks[task_id]
                task["done_nodes"].append(node_name)
                task["node_seconds"][node_name] = round(now - task["node_started_at"], 1)
                task["node_started_at"] = now
                if node_name == "node_entry":
                    task["action"] = node_state["action"]
                    task["doc_id"] = node_state["doc"]["doc_id"]

        try:
            final = import_file(path, root=UPLOAD_DIR, on_node=on_node)
            update_task(task_id, {"status": TASK_DONE, "chunk_count": final["doc"].get("chunk_count")})
        except Exception as e:  # 文档状态已由 import_file 标记为 failed
            update_task(task_id, {"status": TASK_FAILED, "error": str(e)})
            logger.error(f"导入失败 {path.name}：{e}")


@app.post("/documents", summary="上传文件并触发导入", description="支持 pdf/doc/docx/ppt/pptx/md，导入在后台执行")
async def upload_documents(background_tasks: BackgroundTasks, files: list[UploadFile] = File(...)):
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for f in files:
        name = Path(f.filename or "").name
        if not name or Path(name).suffix.lower() not in INGEST_EXTS:
            results.append({"file_name": name, "accepted": False, "reason": f"不支持的格式，仅支持 {' '.join(sorted(INGEST_EXTS))}"})
            continue
        dest = UPLOAD_DIR / name
        dest.write_bytes(await f.read())
        task = new_task(name)
        background_tasks.add_task(run_import, task["task_id"], dest)
        results.append({"file_name": name, "accepted": True, "task_id": task["task_id"]})
    return {"files": results}


@app.get("/tasks", summary="导入任务进度", description="本次服务启动以来提交的导入任务，新的在前")
def tasks():
    return {"tasks": list_tasks()}


@app.get("/documents", summary="文档与导入状态")
def list_documents():
    fields = {"_id": 1, "file_name": 1, "document_title": 1, "content_type": 1, "status": 1, "chunk_count": 1,
              "page_count": 1, "error": 1, "source_path": 1, "updated_at": 1}
    docs = get_db().documents.find({}, fields).sort("file_name", 1)
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
