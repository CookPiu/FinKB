"""MongoDB 客户端单例与索引初始化。库 finkb 开启了认证，MONGO_URI 必须带账号。"""

from __future__ import annotations

import threading

from pymongo import ASCENDING, MongoClient
from pymongo.database import Database

from common.config.settings import get_settings

_client: MongoClient | None = None
_lock = threading.Lock()


def get_client() -> MongoClient:
    global _client
    if _client is None:
        with _lock:
            if _client is None:
                s = get_settings()
                if not s.mongo_uri:
                    raise RuntimeError("MONGO_URI 未配置")
                _client = MongoClient(s.mongo_uri, serverSelectionTimeoutMS=5000, tz_aware=True)
    return _client


def get_db() -> Database:
    return get_client()[get_settings().mongo_db]


def documents_by_file() -> dict[str, dict]:
    """文件名 → 文档元数据（资料名称、内容类型、原件地址），供实体过滤与证据溯源使用。"""
    return {
        d["file_name"]: d
        for d in get_db().documents.find(
            {"status": {"$ne": "superseded"}}, {"file_name": 1, "document_title": 1, "content_type": 1, "source_path": 1}
        )
    }


def ensure_indexes() -> None:
    db = get_db()
    db.documents.create_index([("file_hash", ASCENDING)])
    db.documents.create_index([("file_name", ASCENDING)])
    db.documents.create_index([("status", ASCENDING)])
    db.financial_facts.create_index([("entity_id", ASCENDING), ("item", ASCENDING)])
    db.sessions.create_index([("session_id", ASCENDING)], unique=True)
    db.messages.create_index([("session_id", ASCENDING), ("created_at", ASCENDING)])
