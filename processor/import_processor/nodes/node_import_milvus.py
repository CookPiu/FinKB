"""
节点：写入 Milvus fin_chunks。
写入顺序：先 upsert 当前版本（chunk_id 确定性生成，重跑幂等），再删除该文档其他版本和被替换文档的切片，
避免“先删后插”中途失败导致文档短暂或永久缺失。
"""
from common.logging.logger import logger, node_log, step_log
from processor.import_processor.state import ImportGraphState
from utils.clients.milvus_utils import (
    SECTION_MAX_LEN,
    TEXT_MAX_LEN,
    count_rows,
    delete_rows,
    ensure_collection,
    quote_str,
    upsert_rows,
)
from utils.clients.mongo_utils import get_db
from utils.task_utils import STATUS_SUPERSEDED, save_doc_fields


def build_chunk_id(doc: dict, seq: int) -> str:
    """chunk_id = {doc_id}-{version}-{seq:04d}：同一版本重跑时覆盖同一条"""
    return f"{doc['doc_id']}-{doc['version']}-{seq:04d}"


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
                "version": doc["version"],
                "kind": chunk["kind"],
                "content_type": doc["content_type"],
                "entity_ids": doc["entity_ids"][:64],
                "section_path": fit_bytes(chunk["section_path"], SECTION_MAX_LEN),
                "page_start": chunk["page_start"],
                "page_end": chunk["page_end"],
                "publish_date": doc["publish_date"] or 0,
                "derived": chunk["derived"],
                "text": fit_bytes(chunk["text"], TEXT_MAX_LEN),
                "dense": vector["dense"],
                "sparse": sparse,
            }
        )
    return rows


@step_log("retire_superseded")
def retire_superseded(doc: dict):
    """删除被本文档替换的旧文档切片，并把旧文档标记为 superseded"""
    for old_id in doc["supersedes"]:
        delete_rows(f"doc_id == {quote_str(old_id)}")
        get_db().documents.update_one(
            {"_id": old_id}, {"$set": {"status": STATUS_SUPERSEDED, "superseded_by": doc["doc_id"]}}
        )
        logger.info(f"旧文档 {old_id} 已被 {doc['file_name']} 替换")


@step_log("import_to_milvus")
def import_to_milvus(doc: dict, chunks: list, vectors: list) -> int:
    """
    写入当前版本、删除其他版本并核对条数
    :return: Milvus 中该文档的切片数
    """
    ensure_collection()
    upsert_rows(build_rows(doc, chunks, vectors))
    quoted_id = quote_str(doc["doc_id"])
    delete_rows(f"doc_id == {quoted_id} and version != {doc['version']}")
    count = count_rows(f"doc_id == {quoted_id}")
    if count != len(chunks):
        raise RuntimeError(f"写入后切片数不一致：Milvus {count}，chunks.json {len(chunks)}")
    retire_superseded(doc)
    logger.info(f"index     {doc['file_name']}：写入 {count} 个切片（版本 {doc['version']}）")
    return count


@step_log("validate_and_get_data")
def validate_and_get_data(state: ImportGraphState):
    """
    取出并校验入库所需的入参
    :return: 元组 (文档记录, 切片列表, 向量列表)
    :raise ValueError: 文档记录、切片或向量缺失（上游节点未执行）
    """
    doc = state.get("doc")
    chunks = state.get("chunks") or []
    vectors = state.get("embeddings_content") or []
    if not doc or not chunks or not vectors:
        logger.error("doc, chunks or embeddings_content is empty, cannot import to milvus.")
        raise ValueError("doc, chunks or embeddings_content is empty, cannot import to milvus.")
    return doc, chunks, vectors


@node_log("node_import_milvus")
def node_import_milvus(state: ImportGraphState):
    """
    节点功能：把切片与向量写入 Milvus，记录 chunk_count。
    上游 node_bge_embedding，下游 node_enrich。
    结束后清空状态里的切片与向量（体积大，后续节点不需要）。
    """
    doc, chunks, vectors = validate_and_get_data(state)
    count = import_to_milvus(doc, chunks, vectors)
    save_doc_fields(doc, {"chunk_count": count})
    state["chunks"] = []
    state["embeddings_content"] = []
    return state


if __name__ == "__main__":
    # 运行：uv run python -m processor.import_processor.nodes.node_import_milvus
    # 不依赖任何服务：只用假文档、假向量构造 Milvus 行并打印，不写库
    test_doc = {"doc_id": "demo", "version": 1, "content_type": "其他", "entity_ids": [], "publish_date": None}
    test_chunks = [{"seq": 0, "kind": "text", "section_path": "第一节", "page_start": 1, "page_end": 1,
                    "derived": False, "text": "示例正文"}]
    test_rows = build_rows(test_doc, test_chunks, [{"dense": [0.0] * 4, "sparse": {}}])
    logger.info(f"{test_rows[0]['chunk_id']} sparse={test_rows[0]['sparse']}")
