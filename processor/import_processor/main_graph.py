"""
导入图（LangGraph）：一次执行处理一个文件。
node_entry → node_parse → node_chunk → node_index

一次导入从头跑到尾：同哈希且已就绪的文件在 node_entry 判定为 skip 直接结束，其余整条流水线重做
（解析结果按文件哈希缓存在 data/artifacts，重做不会重复调用 MinerU）。
某个节点抛异常时，import_file 把该文档标记为 failed 并继续抛出；import_directory 捕获后处理下一个文件。
"""
from datetime import datetime, timezone
from pathlib import Path

from langgraph.graph import END, StateGraph

from common.logging.logger import logger
from processor.import_processor.nodes.node_chunk import node_chunk
from processor.import_processor.nodes.node_entry import ACTION_SKIP, build_doc_id, get_file_sha256, node_entry
from processor.import_processor.nodes.node_index import node_index
from processor.import_processor.nodes.node_parse import node_parse
from processor.import_processor.state import ImportGraphState, create_default_state
from utils.clients.mongo_utils import ensure_indexes
from utils.task_utils import STATUS_READY, mark_failed, scan_files

# 1. 注册节点
workflow = StateGraph(ImportGraphState)
workflow.add_node("node_entry", node_entry)
workflow.add_node("node_parse", node_parse)
workflow.add_node("node_chunk", node_chunk)
workflow.add_node("node_index", node_index)
workflow.set_entry_point("node_entry")


def route_after_entry(state: ImportGraphState) -> str:
    """同哈希且已就绪的文件直接结束，其余进入解析"""
    if state.get("action") == ACTION_SKIP:
        return END
    return "node_parse"


# 2. 入口之后按登记结果分支，其余节点顺序执行
workflow.add_conditional_edges("node_entry", route_after_entry, {"node_parse": "node_parse", END: END})
workflow.add_edge("node_parse", "node_chunk")
workflow.add_edge("node_chunk", "node_index")
workflow.add_edge("node_index", END)

# 3. 编译成可执行的图
kb_import_app = workflow.compile()


def new_task_id() -> str:
    """导入运行 ID：UTC 时间，精确到秒"""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")


def import_file(path: Path, root=None, task_id: str = "", force: bool = False, reparse: bool = False,
                on_node=None) -> dict:
    """
    导入单个文件
    :param path: 文件路径
    :param root: 导入目录（计算相对目录、判定内容类型），默认为文件所在目录
    :param on_node: 可选回调 on_node(节点名, 节点返回的状态)，每个节点跑完调用一次（导入页面据此显示进度）
    :return: 图的最终状态；失败时把文档标记为 failed 后抛出异常
    """
    if not task_id:
        task_id = new_task_id()
    if not root:
        root = path.parent
    state = create_default_state(
        task_id=task_id,
        local_file_path=str(path),
        root_dir=str(root),
        force=force,
        reparse=reparse,
    )
    try:
        final = state
        for mode, chunk in kb_import_app.stream(state, stream_mode=["updates", "values"]):
            if mode == "values":
                final = chunk
            elif on_node is not None:
                for node_name, node_state in chunk.items():
                    on_node(node_name, node_state)
        return final
    except Exception as e:
        # 节点内不做失败处理，统一在这里记进 documents.error；doc_id 由文件内容决定，重算一次即可
        mark_failed(build_doc_id(get_file_sha256(path)), path.name, e)
        raise


def list_import_files(root: Path, only=None) -> list:
    """目录下可导入的文件；only 非空时只保留文件名包含该子串的"""
    files = []
    for path in scan_files(root):
        if not only or only in path.name:
            files.append(path)
    return files


def import_directory(root: Path, force: bool = False, reparse: bool = False, only=None) -> dict:
    """
    逐个导入目录下的文件，单个文件失败不影响其他文件
    :return: 报告 {"task_id", "scanned", "actions": {动作: [文件名]}, "ready": [文件名], "failed": {文件名: 错误}}
    """
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"目录不存在：{root}")
    ensure_indexes()
    report = {"task_id": new_task_id(), "scanned": 0, "actions": {}, "ready": [], "failed": {}}
    files = list_import_files(root, only)
    report["scanned"] = len(files)
    for path in files:
        try:
            final = import_file(path, root=root, task_id=report["task_id"], force=force, reparse=reparse)
        except Exception as e:  # 单个文件失败不影响其他文件
            report["failed"][path.name] = str(e)
            continue
        action = final.get("action", "")
        if action not in report["actions"]:
            report["actions"][action] = []
        report["actions"][action].append(path.name)
        doc = final.get("doc")
        if doc is not None and doc["status"] == STATUS_READY and action != ACTION_SKIP:
            report["ready"].append(path.name)
    action_counts = {}
    for action, names in report["actions"].items():
        action_counts[action] = len(names)
    logger.info(f"扫描 {len(files)} 个文件：{action_counts}，失败 {len(report['failed'])}")
    return report


if __name__ == "__main__":
    # 运行：uv run python -m processor.import_processor.main_graph <文件路径>
    # 依赖 Mongo、MinIO、Milvus、MinerU（pdf/doc）、BGE-M3、百炼；已就绪且未变化的文件直接跳过
    import sys

    final_state = import_file(Path(sys.argv[1]))
    logger.info(f"action={final_state['action']} status={final_state['doc']['status']}")
