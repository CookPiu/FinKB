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
    文件名 → 文档元数据（资料名称、内容类型、原件地址），供证据溯源使用
    :return: {file_name: {"_id", "file_name", "document_title", "content_type", "source_path"}}
    """
    fields = {"file_name": 1, "document_title": 1, "content_type": 1, "source_path": 1}
    documents = {}
    for doc in get_db().documents.find({}, fields):
        documents[doc["file_name"]] = doc
    return documents


def get_document_entities():
    """
    已就绪文档上记录的对象（导入时由 node_chunk 识别），供实体解析汇总使用
    :return: [{"_id", "entity"}]，按文件名排序；status 取值见 utils/task_utils.py 的 STATUS_READY
    """
    fields = {"entity": 1}
    return list(get_db().documents.find({"status": "ready", "entity": {"$ne": None}}, fields).sort("file_name", 1))


def ensure_indexes():
    """创建常用查询的索引（已存在时 MongoDB 直接跳过）"""
    db = get_db()
    db.documents.create_index([("file_hash", ASCENDING)])
    db.documents.create_index([("file_name", ASCENDING)])
    db.documents.create_index([("status", ASCENDING)])
    db.sessions.create_index([("session_id", ASCENDING)], unique=True)
    db.messages.create_index([("session_id", ASCENDING), ("created_at", ASCENDING)])
