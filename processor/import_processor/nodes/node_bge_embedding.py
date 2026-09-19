"""节点：BGE-M3 向量化。chunks.json → 稠密 + 稀疏向量（放在状态里交给入库节点）。

嵌入文本 = 资料名称 · 内容类型 · 章节路径 + 正文（表格为线性化文本）。
本节点没有独立的阶段记录：入库完成前中断，续跑时重新编码。
"""

from __future__ import annotations

import time

from common.logging.logger import logger, node_log
from common.models.document import Chunk, DocumentRecord, Stage
from processor.import_processor.state import ImportGraphState
from utils import artifact_utils as artifacts
from utils.lm import embedding_utils as embedding
from utils.task_utils import mark_failed, stage_done


def embed_text(doc: DocumentRecord, c: Chunk) -> str:
    head = " · ".join(x for x in (doc.document_title, doc.content_type, c.section_path) if x)
    return f"{head}\n{c.embed_body}"


@node_log("node_bge_embedding")
def node_bge_embedding(state: ImportGraphState) -> dict:
    doc: DocumentRecord = state["doc"]
    if stage_done(doc, Stage.INDEX):
        return {}
    try:
        chunks = [Chunk.model_validate(x) for x in artifacts.read_json(artifacts.doc_dir(doc.doc_id) / artifacts.CHUNKS)]
        if not chunks:
            raise ValueError("没有可索引的切片")
        t0 = time.perf_counter()
        vectors = embedding.encode([embed_text(doc, c) for c in chunks])
    except Exception as e:
        mark_failed(doc, Stage.INDEX, e)
        raise
    logger.info("embedding %s：%d 个切片，编码 %.1fs", doc.file_name, len(chunks), time.perf_counter() - t0)
    return {"chunks": chunks, "embeddings_content": vectors}
