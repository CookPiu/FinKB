"""节点：写入 Milvus fin_chunks。

写入顺序：先 upsert 当前版本（chunk_id 确定性生成，重跑幂等），再删除该文档其他版本和被替换文档的切片，
避免“先删后插”中途失败导致文档短暂或永久缺失。
"""

from __future__ import annotations

from common.logging.logger import logger, node_log
from common.models.document import Chunk, DocStatus, DocumentRecord, Stage
from processor.import_processor.state import ImportGraphState
from utils.clients import milvus_utils as milvus
from utils.clients import mongo_utils as mongo
from utils.lm.embedding_utils import Encoded
from utils.task_utils import run_stage


def chunk_id(doc: DocumentRecord, seq: int) -> str:
    return f"{doc.doc_id}-{doc.version}-{seq:04d}"


def _fit_bytes(s: str, limit: int) -> str:
    b = s.encode("utf-8")
    return s if len(b) <= limit else b[:limit].decode("utf-8", errors="ignore")


def build_rows(doc: DocumentRecord, chunks: list[Chunk], vectors: list[Encoded]) -> list[dict]:
    rows = []
    for c, v in zip(chunks, vectors, strict=True):
        rows.append(
            {
                "chunk_id": chunk_id(doc, c.seq),
                "doc_id": doc.doc_id,
                "version": doc.version,
                "kind": c.kind.value,
                "content_type": doc.content_type,
                "entity_ids": doc.entity_ids[:64],
                "section_path": _fit_bytes(c.section_path, milvus.SECTION_MAX_LEN),
                "page_start": c.page_start,
                "page_end": c.page_end,
                "publish_date": doc.publish_date or 0,
                "derived": c.derived,
                "text": _fit_bytes(c.text, milvus.TEXT_MAX_LEN),
                "dense": v.dense,
                "sparse": v.sparse or {0: 1e-6},
            }
        )
    return rows


def _retire_superseded(doc: DocumentRecord) -> None:
    for old in doc.supersedes:
        milvus.delete(f"doc_id == {milvus.quote_str(old)}")
        mongo.get_db().documents.update_one(
            {"_id": old}, {"$set": {"status": DocStatus.SUPERSEDED.value, "superseded_by": doc.doc_id}}
        )
        logger.info("旧文档 %s 已被 %s 替换", old, doc.file_name)


@node_log("node_import_milvus")
def node_import_milvus(state: ImportGraphState) -> dict:
    doc: DocumentRecord = state["doc"]
    chunks, vectors = state.get("chunks") or [], state.get("embeddings_content") or []

    def import_milvus(d: DocumentRecord) -> dict:
        milvus.ensure_collection()
        milvus.upsert(build_rows(d, chunks, vectors))
        q = milvus.quote_str(d.doc_id)
        milvus.delete(f"doc_id == {q} and version != {d.version}")
        n = milvus.count(f"doc_id == {q}")
        if n != len(chunks):
            raise RuntimeError(f"写入后切片数不一致：Milvus {n}，chunks.json {len(chunks)}")
        _retire_superseded(d)
        logger.info("index     %s：写入 %d 个切片（版本 %d）", d.file_name, n, d.version)
        return {"chunk_count": n}

    run_stage(doc, Stage.INDEX, import_milvus)
    return {"doc": doc, "chunks": [], "embeddings_content": []}
