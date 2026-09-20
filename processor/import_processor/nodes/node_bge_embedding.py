"""
节点：BGE-M3 向量化。chunks.json → 稠密 + 稀疏向量（放在状态里交给入库节点）。
嵌入文本 = 资料名称 · 内容类型 · 章节路径 + 正文（表格为线性化文本）。
"""
import time

from common.logging.logger import logger, node_log, step_log
from processor.import_processor.state import ImportGraphState, create_default_state
from utils.artifact_utils import CHUNKS, get_doc_dir, read_json
from utils.lm.embedding_utils import generate_embeddings


def build_embed_text(doc: dict, chunk: dict) -> str:
    """资料名称、内容类型、章节路径作为抬头（空的跳过），换行后接正文"""
    head_parts = []
    for part in (doc["document_title"], doc["content_type"], chunk["section_path"]):
        if part:
            head_parts.append(part)
    head = " · ".join(head_parts)
    return f"{head}\n{chunk['embed_body']}"


@step_log("validate_and_get_data")
def validate_and_get_data(state: ImportGraphState):
    """
    取出并校验向量化所需的入参
    :return: 文档记录
    :raise ValueError: 状态里没有文档记录
    """
    doc = state.get("doc")
    if not doc:
        logger.error("no doc found in state")
        raise ValueError("no doc found in state")
    return doc


@node_log("node_bge_embedding")
def node_bge_embedding(state: ImportGraphState):
    """
    节点功能：读取 chunks.json，批量编码成稠密 + 稀疏向量。
    上游 node_document_split，下游 node_import_milvus。
    """
    doc = validate_and_get_data(state)
    chunks = read_json(get_doc_dir(doc["doc_id"]) / CHUNKS)
    if not chunks:
        raise ValueError("没有可索引的切片")
    start_ts = time.perf_counter()
    vectors = generate_embeddings([build_embed_text(doc, chunk) for chunk in chunks])
    logger.info(f"embedding {doc['file_name']}：{len(chunks)} 个切片，编码 {time.perf_counter() - start_ts:.1f}s")
    state["chunks"] = chunks
    state["embeddings_content"] = vectors
    return state


if __name__ == "__main__":
    # 运行：uv run python -m processor.import_processor.nodes.node_bge_embedding <doc_id>
    # 依赖 Mongo（读取文档记录）与本地 BGE-M3；只编码不写库
    import sys

    from processor.import_processor.nodes.node_entry import load_document

    test_doc = load_document(sys.argv[1])
    result = node_bge_embedding(create_default_state(doc=test_doc))
    logger.info(f"切片 {len(result['chunks'])} 个，稠密维度 {len(result['embeddings_content'][0]['dense'])}")
