"""SSE 事件打包：event: <类型>\\ndata: <JSON>\\n\\n。"""

from __future__ import annotations

import json
from typing import Any


class SSEEvent:
    DELTA = "delta"  # 回答增量文本
    SOURCES = "sources"  # 引用来源
    FINAL = "final"  # 结束：session_id、kind、完整回答
    ERROR = "error"  # 异常：面向用户的提示


def sse_pack(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"
