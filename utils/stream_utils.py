"""流式事件：节点通过 LangGraph custom stream 发出 {"type": "delta"|"sources"|"final", ...}。"""

from __future__ import annotations

from langgraph.config import get_stream_writer


def emit(event: dict) -> None:
    try:
        get_stream_writer()(event)
    except RuntimeError:  # 不在 stream 模式下运行时没有 writer
        pass
