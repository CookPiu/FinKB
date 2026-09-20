"""
节点：BGE-M3 向量化。chunks.json → 稠密 + 稀疏向量（放在状态里交给入库节点）。
嵌入文本 = 资料名称 · 内容类型 · 章节路径 + 正文（表格为线性化文本）。
本节点没有独立的阶段记录：入库完成前中断，续跑时重新编码。
"""
import time

from common.logging.logger import logger, node_log
from processor.import_processor.state import ImportGraphState, create_default_state
from utils.artifact_utils import CHUNKS, get_doc_dir, read_json
from utils.lm.embedding_utils import generate_embeddings
from utils.task_utils import STAGE_INDEX, is_stage_done, mark_stage_failed


def build_embed_text(doc: dict, chunk: dict) -> str:
    """资料名称、内容类型、章节路径作为抬头（空的跳过），换行后接正文"""
    head_parts = []
    for part in (doc["document_title"], doc["content_type"], chunk["section_path"]):
        if part:
            head_parts.append(part)
    head = " · ".join(head_parts)
    return f"{head}\n{chunk['embed_body']}"


@node_log("node_bge_embedding")
def node_bge_embedding(state: ImportGraphState):
    """
    节点功能：读取 chunks.json，批量编码成稠密 + 稀疏向量。
    上游 node_document_split，下游 node_import_milvus；index 阶段已完成则跳过。
    编码失败按 index 阶段标记失败（本节点属于入库的前半段）。
    """
    doc = state["doc"]
    if is_stage_done(doc, STAGE_INDEX):
        return state
    try:
        chunks = read_json(get_doc_dir(doc["doc_id"]) / CHUNKS)
        if not chunks:
            raise ValueError("没有可索引的切片")
        start_ts = time.perf_counter()
        vectors = generate_embeddings([build_embed_text(doc, chunk) for chunk in chunks])
    except Exception as e:
        mark_stage_failed(doc, STAGE_INDEX, e)
        raise
    logger.info(f"embedding {doc['file_name']}：{len(chunks)} 个切片，编码 {time.perf_counter() - start_ts:.1f}s")
    state["chunks"] = chunks
    state["embeddings_content"] = vectors
    return state


if __name__ == "__main__":
    # 运行：uv run python -m processor.import_processor.nodes.node_bge_embedding <doc_id>
    # 依赖 Mongo（读取文档记录）与本地 BGE-M3；只在内存里把阶段退回 chunk 以触发编码，编码成功不写库
    import sys

    from processor.import_processor.nodes.node_entry import load_document

    test_doc = load_document(sys.argv[1])
    test_doc["stage"] = "chunk"
    result = node_bge_embedding(create_default_state(doc=test_doc))
    logger.info(f"切片 {len(result['chunks'])} 个，稠密维度 {len(result['embeddings_content'][0]['dense'])}")
