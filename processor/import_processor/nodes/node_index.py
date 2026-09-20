"""
节点：建立索引。chunks.json → BGE-M3 稠密 + 稀疏向量 → 写入 Milvus fin_chunks。
嵌入文本 = 资料名称 · 内容类型 · 章节路径 + 正文（表格为线性化文本）。
写入顺序：先 upsert 当前版本（chunk_id 确定性生成，重跑幂等），再删除该文档其他版本和被替换文档的切片，
避免“先删后插”中途失败导致文档短暂或永久缺失。
"""
import time

from common.logging.logger import logger, node_log, step_log
from processor.import_processor.state import ImportGraphState, create_default_state
from utils.artifact_utils import CHUNKS, get_doc_dir, read_json
from utils.clients.milvus_utils import (
    SECTION_MAX_LEN,
    TEXT_MAX_LEN,
    count_rows,
    delete_rows,
    ensure_collection,
    list_chunk_ids,
    quote_str,
    upsert_rows,
)
from utils.lm.embedding_utils import generate_embeddings
from utils.task_utils import mark_ready


def build_embed_text(doc: dict, chunk: dict) -> str:
    """资料名称、内容类型、章节路径作为抬头（空的跳过），换行后接正文"""
    head_parts = []
    for part in (doc["document_title"], doc["content_type"], chunk["section_path"]):
        if part:
            head_parts.append(part)
    head = " · ".join(head_parts)
    return f"{head}\n{chunk['embed_body']}"


def build_chunk_id(doc: dict, seq: int) -> str:
    """chunk_id = {doc_id}-{seq:04d}：由内容与序号决定，重跑时 upsert 覆盖同一条"""
    return f"{doc['doc_id']}-{seq:04d}"


def fit_bytes(text: str, limit: int) -> str:
    """按 UTF-8 字节截断到 limit 以内（Milvus VARCHAR 按字节计长度），截断处不完整的字符丢弃"""
    data = text.encode("utf-8")
    if len(data) <= limit:
        return text
    return data[:limit].decode("utf-8", errors="ignore")


def build_rows(doc: dict, chunks: list, vectors: list) -> list:
    """
    把切片与向量组装成 Milvus 行
    :param doc: 文档记录
    :param chunks: 切片列表
    :param vectors: 与 chunks 一一对应的 {"dense", "sparse"}
    """
    if len(chunks) != len(vectors):
        raise ValueError(f"切片数与向量数不一致：{len(chunks)} != {len(vectors)}")
    rows = []
    for chunk, vector in zip(chunks, vectors):
        sparse = vector["sparse"]
        if not sparse:
            # Milvus 不接受空的稀疏向量，放一个极小权重占位
            sparse = {0: 1e-6}
        rows.append(
            {
                "chunk_id": build_chunk_id(doc, chunk["seq"]),
                "doc_id": doc["doc_id"],
                "kind": chunk["kind"],
                "content_type": doc["content_type"],
                "section_path": fit_bytes(chunk["section_path"], SECTION_MAX_LEN),
                "page_start": chunk["page_start"],
                "page_end": chunk["page_end"],
                "derived": chunk["derived"],
                "text": fit_bytes(chunk["text"], TEXT_MAX_LEN),
                "dense": vector["dense"],
                "sparse": sparse,
            }
        )
    return rows


@step_log("encode_chunks")
def encode_chunks(doc: dict, chunks: list) -> list:
    """批量编码切片，返回与 chunks 一一对应的 {"dense", "sparse"}"""
    start_ts = time.perf_counter()
    vectors = generate_embeddings([build_embed_text(doc, chunk) for chunk in chunks])
    logger.info(f"embedding {doc['file_name']}：{len(chunks)} 个切片，编码 {time.perf_counter() - start_ts:.1f}s")
    return vectors


@step_log("import_to_milvus")
def import_to_milvus(doc: dict, chunks: list, vectors: list) -> int:
    """
    先写入本次的全部切片，再删掉这次没写到的旧切片（重切后切片变少时会有多余的），最后核对条数
    :return: Milvus 中该文档的切片数
    """
    ensure_collection()
    quoted_id = quote_str(doc["doc_id"])
    old_ids = list_chunk_ids(f"doc_id == {quoted_id}")
    rows = build_rows(doc, chunks, vectors)
    upsert_rows(rows)
    new_ids = set(row["chunk_id"] for row in rows)
    stale = [chunk_id for chunk_id in old_ids if chunk_id not in new_ids]
    if stale:
        delete_rows("chunk_id in [" + ", ".join(quote_str(c) for c in stale) + "]")
        logger.info(f"删除 {len(stale)} 个不再存在的旧切片")
    count = count_rows(f"doc_id == {quoted_id}")
    if count != len(chunks):
        raise RuntimeError(f"写入后切片数不一致：Milvus {count}，chunks.json {len(chunks)}")
    logger.info(f"index     {doc['file_name']}：写入 {count} 个切片")
    return count


@step_log("validate_and_get_data")
def validate_and_get_data(state: ImportGraphState):
    """
    取出并校验建索引所需的入参
    :return: 元组 (文档记录, 切片列表)
    :raise ValueError: 状态里没有文档记录，或 chunks.json 为空
    """
    doc = state.get("doc")
    if not doc:
        logger.error("no doc found in state")
        raise ValueError("no doc found in state")
    chunks = read_json(get_doc_dir(doc["doc_id"]) / CHUNKS)
    if not chunks:
        logger.error("no chunks found, cannot build index.")
        raise ValueError("no chunks found, cannot build index.")
    return doc, chunks


@node_log("node_index")
def node_index(state: ImportGraphState):
    """
    节点功能：读取 chunks.json，编码成向量写入 Milvus。
    上游 node_chunk；导入图的最后一个节点，跑完文档状态变为 ready。
    切片与向量都是大对象，只在本节点内传递，不进状态。
    """
    doc, chunks = validate_and_get_data(state)
    count = import_to_milvus(doc, chunks, encode_chunks(doc, chunks))
    mark_ready(doc, {"chunk_count": count})
    return state


if __name__ == "__main__":
    # 运行：uv run python -m processor.import_processor.nodes.node_index <doc_id>
    # 依赖 Mongo（读文档记录）、本地 BGE-M3、Milvus；会真的写入该文档当前版本的切片
    import sys

    from processor.import_processor.nodes.node_entry import load_document

    result = node_index(create_default_state(doc=load_document(sys.argv[1])))
    logger.info(f"chunk_count={result['doc']['chunk_count']}")
