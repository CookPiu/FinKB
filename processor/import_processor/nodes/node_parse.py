"""节点：解析。原件 → data/artifacts/<doc_id>/content_list.json。

pdf/doc/docx 提交 MinerU；md 走本地解析。content_list.json 已存在即视为缓存命中
（doc_id 由文件哈希生成，内容不变则产物不变），除非指定 reparse。
MinerU 偶发 "parsing failed, please try again later"，失败后重提交。
"""

from __future__ import annotations

import time
from pathlib import Path

from common.logging.logger import logger, node_log
from common.models.document import DocumentRecord, Stage
from processor.import_processor.state import ImportGraphState
from utils import artifact_utils as artifacts
from utils.clients import mineru_utils as mineru
from utils.markdown_utils import parse_markdown
from utils.task_utils import run_stage

RETRIES = 2
RETRY_WAIT_S = 20


def page_count(content_list: list[dict]) -> int:
    return max((b.get("page_idx", 0) for b in content_list), default=0) + 1


def parse_with_mineru(doc: DocumentRecord) -> list[dict]:
    """提交 MinerU 并等待结果（含失败重提交），返回 content_list。"""
    f = mineru.MinerUFile(data_id=doc.doc_id, path=Path(doc.local_path))
    error = ""
    for attempt in range(1 + RETRIES):
        if attempt:
            logger.warning("%s 解析失败（%s），%ds 后重提交（第 %d 次重试）", doc.file_name, error, RETRY_WAIT_S, attempt)
            time.sleep(RETRY_WAIT_S)
        try:
            batch_id = mineru.submit_batch([f])
        except Exception as e:  # noqa: BLE001 - 提交失败同样重试
            error = f"提交失败：{e}"
            continue
        r = mineru.poll_batch(batch_id, [doc.doc_id]).get(doc.doc_id)
        if r is not None and r.state == "done" and r.zip_url:
            return artifacts.read_json(mineru.download_and_extract(r.zip_url, artifacts.doc_dir(doc.doc_id)))
        error = f"MinerU 状态 {r.state if r else 'missing'}：{r.error if r else ''}"
    raise mineru.MinerUError(error)


@node_log("node_parse")
def node_parse(state: ImportGraphState) -> dict:
    doc: DocumentRecord = state["doc"]

    def parse(d: DocumentRecord) -> dict:
        out = artifacts.doc_dir(d.doc_id) / artifacts.CONTENT_LIST
        if out.exists() and not state.get("reparse"):
            logger.info("parse     缓存命中 %s", d.file_name)
            return {"page_count": page_count(artifacts.read_json(out))}
        src = Path(d.local_path)
        if not src.is_file():
            raise FileNotFoundError(f"原件不存在：{src}")
        if d.file_ext == ".md":
            content = parse_markdown(src.read_text(encoding="utf-8"))
        elif d.file_ext in mineru.SUPPORTED_EXTS:
            content = parse_with_mineru(d)
        else:
            raise ValueError(f"不支持的格式 {d.file_ext}")
        artifacts.write_json(out, content)
        logger.info("parse     %s（%d 块）", d.file_name, len(content))
        return {"page_count": page_count(content)}

    run_stage(doc, Stage.PARSE, parse)
    return {"doc": doc}
