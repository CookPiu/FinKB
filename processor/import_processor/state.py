"""
导入图的状态：一次图执行处理一个文件。
大块中间产物落盘在 data/artifacts/<doc_id>/，切片与向量只在节点内部传递，状态里只放文档记录与本次导入的参数。
"""
import copy
from typing import TypedDict


class ImportGraphState(TypedDict):
    task_id: str  # 本次导入运行 ID
    local_file_path: str  # 待导入文件
    root_dir: str  # 导入目录（用于计算相对目录、判定内容类型）
    file_name: str
    force: bool  # 已就绪也从头重建
    reparse: bool  # 忽略解析缓存，重新调用 MinerU

    # --- node_entry 写入 ---
    doc: dict  # 文档记录（Mongo documents 中的一条）
    action: str  # new / redo / skip


graph_default_state: ImportGraphState = {
    "task_id": "",
    "local_file_path": "",
    "root_dir": "",
    "file_name": "",
    "force": False,
    "reparse": False,
    "doc": None,
    "action": "",
}


def create_default_state(**overrides) -> ImportGraphState:
    """创建默认状态并覆盖指定字段（深拷贝，避免多次导入共享同一个列表）"""
    state = copy.deepcopy(graph_default_state)
    state.update(overrides)
    return state
