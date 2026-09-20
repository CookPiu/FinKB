import json


class SSEEvent:
    DELTA = "delta"  # 回答增量文本
    SOURCES = "sources"  # 引用来源
    FINAL = "final"  # 结束：session_id、kind、完整回答
    ERROR = "error"  # 异常：面向用户的提示


def sse_pack(event: str, data: dict) -> str:
    """
    打包一条 SSE 消息：event: <类型>\\ndata: <JSON>\\n\\n
    default=str 让 datetime 等对象也能序列化
    """
    payload = json.dumps(data, ensure_ascii=False, default=str)
    return f"event: {event}\ndata: {payload}\n\n"
