"""
查询服务：聊天页面、流式问答（SSE）、历史会话。
启动：uv run python -m api.query_service（或 uv run uvicorn api.query_service:app --port 8001）
"""
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from common.logging.logger import logger
from processor.query_processor.main_graph import query_app
from processor.query_processor.state import create_query_default_state
from utils.clients.mongo_history_utils import get_messages, list_sessions
from utils.clients.mongo_utils import ensure_indexes
from utils.lm.embedding_utils import warmup
from utils.path_util import PROJECT_ROOT
from utils.sse_utils import SSEEvent, sse_pack

CHAT_PAGE = PROJECT_ROOT / "page" / "chat.html"
MAX_QUESTION_CHARS = 500


@asynccontextmanager
async def lifespan(app: FastAPI):
    """服务启动时预热：BGE-M3 在 CPU 上首次编码要十几秒；失败不阻止服务启动，首个请求时再加载"""
    try:
        ensure_indexes()
        warmup()
        logger.info("查询服务就绪（BGE-M3 已预热）")
    except Exception as e:
        logger.warning(f"启动预热失败：{e}")
    yield


app = FastAPI(title="FinKB 查询服务", description="金融知识库问答", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class QueryRequest(BaseModel):
    question: str = Field(..., description="用户问题")
    # 聊天页新会话会传 null，类型必须允许 None
    session_id: str | None = Field(None, description="沿用已有会话（多轮追问）；为空时新建")


def friendly_error(e: Exception) -> str:
    """把内部异常转成面向用户的提示，细节只写日志。"""
    name = type(e).__name__
    if name in ("PermissionDeniedError", "AuthenticationError", "RateLimitError"):
        return "大模型服务暂时不可用（账户或额度问题），请稍后再试或联系管理员。"
    if name in ("APIConnectionError", "APITimeoutError", "Timeout", "ReadTimeout", "ConnectionError"):
        return "大模型服务连接超时，请稍后重试。"
    if name in ("ServerSelectionTimeoutError", "AutoReconnect"):
        return "会话数据库连接失败，请稍后重试。"
    if name == "MilvusException":
        return "知识库检索服务连接失败，请稍后重试。"
    return "服务处理出错，请稍后重试。"


def stream_answer_events(state: dict, session_id: str):
    """
    运行查询图并把事件转成 SSE 消息：节点推送的 delta / sources 原样转发，
    图结束后用最终状态发一条 final；出错时发 error（只含面向用户的提示）
    """
    final = {}
    try:
        for mode, chunk in query_app.stream(state, stream_mode=["custom", "values"]):
            if mode == "values":
                final = chunk
            elif chunk.get("type") in (SSEEvent.DELTA, SSEEvent.SOURCES):
                data = dict(chunk)
                del data["type"]
                yield sse_pack(chunk["type"], data)
        yield sse_pack(
            SSEEvent.FINAL,
            {
                "session_id": session_id,
                "kind": final.get("kind"),
                "answer": final.get("answer", ""),
                "sources": final.get("sources", []),
            },
        )
    except Exception as e:  # 异常以 error 事件告知前端
        logger.exception(f"问答失败 session={session_id}")
        yield sse_pack(SSEEvent.ERROR, {"session_id": session_id, "message": friendly_error(e)})


@app.api_route("/", methods=["GET", "HEAD"], include_in_schema=False)
@app.api_route("/chat.html", methods=["GET", "HEAD"], include_in_schema=False)
def chat_page():
    return FileResponse(CHAT_PAGE, media_type="text/html")


@app.post("/query", summary="流式问答", description="返回 text/event-stream：delta / sources / final / error 事件")
def query(req: QueryRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="问题不能为空")
    if len(question) > MAX_QUESTION_CHARS:
        raise HTTPException(status_code=400, detail=f"问题过长（不超过 {MAX_QUESTION_CHARS} 字）")
    session_id = req.session_id or uuid.uuid4().hex[:12]
    state = create_query_default_state(session_id=session_id, original_query=question)
    return StreamingResponse(
        stream_answer_events(state, session_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/sessions", summary="最近的会话")
def sessions(limit: int = 30):
    return {"sessions": list_sessions(limit)}


@app.get("/sessions/{session_id}/messages", summary="会话的问答记录")
def session_messages(session_id: str):
    return {"session_id": session_id, "messages": get_messages(session_id)}


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8001)
