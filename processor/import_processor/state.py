"""导入图的状态：一次图执行处理一个文件。大块中间产物落盘在 data/artifacts/<doc_id>/，状态里只放引用与少量数据。"""

from __future__ import annotations

import copy
from typing import Any, TypedDict


class ImportGraphState(TypedDict, total=False):
    task_id: str  # 本次导入运行 ID
    local_file_path: str  # 待导入文件
    root_dir: str  # 导入目录（用于计算相对目录、判定内容类型）
    file_name: str
    force: bool  # 已就绪也从头重建
    reparse: bool  # 忽略解析缓存，重新调用 MinerU

    # --- node_entry 写入 ---
    doc: Any  # DocumentRecord
    action: str  # new / resume / force / skip

    # --- node_bge_embedding → node_import_milvus ---
    chunks: list  # list[Chunk]
    embeddings_content: list  # list[Encoded]，与 chunks 一一对应


graph_default_state: ImportGraphState = {
    "task_id": "",
    "local_file_path": "",
    "root_dir": "",
    "file_name": "",
    "force": False,
    "reparse": False,
    "doc": None,
    "action": "",
    "chunks": [],
    "embeddings_content": [],
}


def create_default_state(**overrides) -> ImportGraphState:
    state = copy.deepcopy(graph_default_state)
    state.update(overrides)
    return state
