from langgraph.config import get_stream_writer


def emit(event: dict):
    """
    节点向外推送流式事件 {"type": "delta" | "sources" | "final", ...}
    图以 stream_mode="custom" 运行时，事件由 query_app.stream 逐条产出；invoke 运行时没有 writer，直接忽略
    """
    try:
        writer = get_stream_writer()
        writer(event)
    except RuntimeError:
        pass
