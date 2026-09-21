"""需要虚拟机中间件与外部 API：uv run pytest -m integration"""

import pytest

pytestmark = pytest.mark.integration


def test_mongo_auth_and_ping():
    from utils.clients.mongo_utils import get_db

    assert get_db().command("ping")["ok"] == 1


def test_milvus_collection_ready():
    from common.config.milvus_config import milvus_config
    from utils.clients.milvus_utils import ensure_collection, get_milvus_client

    ensure_collection()
    assert get_milvus_client().has_collection(milvus_config.chunks_collection)


def test_minio_bucket():
    from common.config.minio_config import minio_config
    from utils.clients.minio_utils import get_minio_client

    assert get_minio_client().bucket_exists(minio_config.bucket)


def test_index_matches_documents():
    """每个就绪文档在 Milvus 中的切片数等于 documents.chunk_count，且只有当前版本。"""
    from utils.clients.milvus_utils import count_rows, list_chunk_ids, quote_str
    from utils.clients.mongo_utils import get_db

    ready = list(get_db().documents.find({"status": "ready"}))
    assert ready
    for d in ready:
        q = quote_str(d["_id"])
        assert count_rows(f"doc_id == {q}") == d["chunk_count"], d["file_name"]
        # chunk_id 由 doc_id 与序号决定，序号必须是 0..chunk_count-1 的连续区间
        chunk_ids = sorted(list_chunk_ids(f"doc_id == {q}"))
        expected = [f"{d['_id']}-{seq:04d}" for seq in range(d["chunk_count"])]
        assert chunk_ids == expected, d["file_name"]


def test_ready_documents_have_entity():
    """每个就绪文档都记着导入时识别出的对象，汇总出的实体覆盖全部就绪文档"""
    from utils.clients.mongo_utils import get_db
    from utils.entity_utils import get_entities

    ready = list(get_db().documents.find({"status": "ready"}, {"file_name": 1, "entity": 1}))
    assert [d["file_name"] for d in ready if not d.get("entity")] == []
    covered = {doc_id for entity in get_entities() for doc_id in entity["doc_ids"]}
    assert covered == {d["_id"] for d in ready}


def test_hybrid_search_returns_maotai_revenue_table():
    from utils.search_utils import semantic_search

    hits = semantic_search("贵州茅台2026年第一季度营业收入是多少", top_k=5)
    assert any("53,909,252,220.51" in h["text"] for h in hits)
