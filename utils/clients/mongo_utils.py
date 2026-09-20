from pymongo import ASCENDING, MongoClient

from common.config.mongo_config import mongo_config

# 全局 MongoDB 客户端单例：MongoClient 自带连接池，整个进程共用一个
_mongo_client = None


def get_mongo_client():
    """
    获取 MongoDB 客户端单例
    :return: MongoClient 实例
    :raise RuntimeError: 未配置 MONGO_URI
    """
    global _mongo_client
    if _mongo_client is None:
        if not mongo_config.mongo_uri:
            raise RuntimeError("MONGO_URI 未配置")
        # tz_aware=True：读出的时间带时区（UTC），前端再转成本地时间
        _mongo_client = MongoClient(mongo_config.mongo_uri, serverSelectionTimeoutMS=5000, tz_aware=True)
    return _mongo_client


def get_db():
    """项目数据库 finkb：documents / financial_facts / sessions / messages 四个集合"""
    return get_mongo_client()[mongo_config.mongo_db]


def get_documents_by_file():
    """
    文件名 → 文档元数据（资料名称、内容类型、原件地址），供实体过滤与证据溯源使用
    :return: {file_name: {"_id", "file_name", "document_title", "content_type", "source_path"}}
    """
    fields = {"file_name": 1, "document_title": 1, "content_type": 1, "source_path": 1}
    documents = {}
    for doc in get_db().documents.find({}, fields):
        documents[doc["file_name"]] = doc
    return documents


def ensure_indexes():
    """创建常用查询的索引（已存在时 MongoDB 直接跳过）"""
    db = get_db()
    db.documents.create_index([("file_hash", ASCENDING)])
    db.documents.create_index([("file_name", ASCENDING)])
    db.documents.create_index([("status", ASCENDING)])
    db.financial_facts.create_index([("entity_id", ASCENDING), ("item", ASCENDING)])
    db.sessions.create_index([("session_id", ASCENDING)], unique=True)
    db.messages.create_index([("session_id", ASCENDING), ("created_at", ASCENDING)])
