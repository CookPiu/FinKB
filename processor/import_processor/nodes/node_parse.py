"""
节点：解析。原件 → data/artifacts/<doc_id>/content_list.json。
pdf/doc/docx 提交 MinerU；md 走本地解析。content_list.json 已存在即视为缓存命中
（doc_id 由文件哈希生成，内容不变则产物不变），除非指定 reparse。
MinerU 偶发 "parsing failed, please try again later"，失败后重提交。
"""
import time
from pathlib import Path

from common.logging.logger import logger, node_log, step_log
from processor.import_processor.state import ImportGraphState, create_default_state
from utils.artifact_utils import CONTENT_LIST, get_doc_dir, read_json, write_json
from utils.clients.mineru_utils import SUPPORTED_EXTS, MinerUError, download_and_extract, poll_batch, submit_batch
from utils.markdown_utils import parse_markdown
from utils.task_utils import save_doc_fields

RETRIES = 2
RETRY_WAIT_S = 20


def get_page_count(content_list: list) -> int:
    """页数 = 最大页码下标 + 1（MinerU 的 page_idx 从 0 开始）"""
    if not content_list:
        return 1
    return max([block.get("page_idx", 0) for block in content_list]) + 1


@step_log("parse_with_mineru")
def parse_with_mineru(doc: dict) -> list:
    """提交 MinerU 并等待结果（含失败重提交），返回 content_list。"""
    doc_id = doc["doc_id"]
    file = {"data_id": doc_id, "path": Path(doc["local_path"])}
    error = ""
    for attempt in range(1 + RETRIES):
        if attempt:
            logger.warning(f"{doc['file_name']} 解析失败（{error}），{RETRY_WAIT_S}s 后重提交（第 {attempt} 次重试）")
            time.sleep(RETRY_WAIT_S)
        try:
            batch_id = submit_batch([file])
        except Exception as e:  # 提交失败同样重试
            error = f"提交失败：{e}"
            continue
        result = poll_batch(batch_id, [doc_id]).get(doc_id)
        if result is not None and result["state"] == "done" and result["zip_url"]:
            return read_json(download_and_extract(result["zip_url"], get_doc_dir(doc_id)))
        if result is None:
            error = "MinerU 状态 missing："
        else:
            error = f"MinerU 状态 {result['state']}：{result['error']}"
    raise MinerUError(error)


@step_log("parse_document")
def parse_document(doc: dict, reparse: bool) -> int:
    """
    解析原件并落盘 content_list.json（已有缓存且不要求重解析时直接复用）
    :return: 页数
    """
    out_path = get_doc_dir(doc["doc_id"]) / CONTENT_LIST
    if out_path.exists() and not reparse:
        logger.info(f"parse     缓存命中 {doc['file_name']}")
        return get_page_count(read_json(out_path))
    src = Path(doc["local_path"])
    if not src.is_file():
        raise FileNotFoundError(f"原件不存在：{src}")
    if doc["file_ext"] == ".md":
        content = parse_markdown(src.read_text(encoding="utf-8"))
    elif doc["file_ext"] in SUPPORTED_EXTS:
        content = parse_with_mineru(doc)
    else:
        raise ValueError(f"不支持的格式 {doc['file_ext']}")
    write_json(out_path, content)
    logger.info(f"parse     {doc['file_name']}（{len(content)} 块）")
    return get_page_count(content)


@step_log("validate_and_get_data")
def validate_and_get_data(state: ImportGraphState):
    """
    取出并校验解析所需的入参
    :return: 元组 (文档记录, 是否忽略解析缓存)
    :raise ValueError: 状态里没有文档记录（node_entry 未执行）
    """
    doc = state.get("doc")
    if not doc:
        logger.error("no doc found in state")
        raise ValueError("no doc found in state")
    return doc, state.get("reparse", False)


@node_log("node_parse")
def node_parse(state: ImportGraphState):
    """
    节点功能：把原件解析成 content_list.json，记录页数。
    上游 node_entry，下游 node_normalize；解析结果按文件哈希缓存，内容没变时不会重复调用 MinerU。
    """
    doc, reparse = validate_and_get_data(state)
    page_count = parse_document(doc, reparse)
    save_doc_fields(doc, {"page_count": page_count})
    return state


if __name__ == "__main__":
    # 运行：uv run python -m processor.import_processor.nodes.node_parse <doc_id>
    # 依赖 Mongo（读取文档记录）；没有解析缓存时会调用 MinerU
    import sys

    from processor.import_processor.nodes.node_entry import load_document

    result = node_parse(create_default_state(doc=load_document(sys.argv[1])))
    logger.info(f"page_count={result['doc']['page_count']}")
