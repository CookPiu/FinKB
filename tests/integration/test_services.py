"""需要虚拟机中间件与外部 API：uv run pytest -m integration"""

import pytest

pytestmark = pytest.mark.integration


def test_mongo_auth_and_ping():
    from utils.clients import mongo_utils as mongo

    assert mongo.get_db().command("ping")["ok"] == 1


def test_milvus_collection_ready():
    from utils.clients import milvus_utils as milvus

    milvus.ensure_collection()
    assert milvus.get_client().has_collection(milvus.collection_name())


def test_minio_bucket():
    from common.config.settings import get_settings
    from utils.clients import minio_utils as minio

    assert minio.get_client().bucket_exists(get_settings().minio_bucket)


def test_index_matches_documents():
    """每个就绪文档在 Milvus 中的切片数等于 documents.chunk_count，且只有当前版本。"""
    from utils.clients import milvus_utils as milvus
    from utils.clients import mongo_utils as mongo

    ready = list(mongo.get_db().documents.find({"status": "ready"}))
    assert ready
    for d in ready:
        q = milvus.quote_str(d["_id"])
        assert milvus.count(f"doc_id == {q}") == d["chunk_count"], d["file_name"]
        assert milvus.count(f"doc_id == {q} and version != {d['version']}") == 0, d["file_name"]


def test_hybrid_search_returns_maotai_revenue_table():
    from utils.search_utils import semantic_search

    hits = semantic_search("贵州茅台2026年第一季度营业收入是多少", top_k=5)
    assert any("53,909,252,220.51" in h.text for h in hits)
