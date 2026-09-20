"""
会话记录：sessions 只存焦点对象与待澄清候选（多轮追问用）；messages 存问答历史。
"""
from datetime import datetime, timedelta, timezone

from utils.clients.mongo_utils import get_db


def load_session(session_id: str) -> dict:
    """
    读取会话状态
    :return: {"focus_entity_ids": 上一轮在谈的对象, "pending": 待澄清的候选（没有则为 None）}
    """
    doc = get_db().sessions.find_one({"session_id": session_id})
    if doc is None:
        doc = {}
    return {
        "focus_entity_ids": doc.get("focus_entity_ids", []),
        "pending": doc.get("pending_clarification"),
    }


def last_turns(messages_desc: list) -> list:
    """数据库按时间倒序取出最近的 n 条（不能取成最早的 n 条），这里反转回时间正序"""
    result = list(messages_desc)
    result.reverse()
    return result


def get_history(session_id: str, n_turns: int = 3) -> list:
    """
    最近 n 轮问答，按时间正序
    :return: [{"role": "user" | "assistant", "text": ...}]
    """
    cursor = get_db().messages.find({"session_id": session_id}, {"role": 1, "text": 1})
    cursor = cursor.sort("created_at", -1).limit(n_turns * 2)
    return last_turns(list(cursor))


def save_turn(session_id, question, answer, kind, plan, sources, focus_entity_ids, pending):
    """
    保存一轮问答，并更新会话的焦点对象与待澄清候选
    :param kind: answer / refuse / clarify / decline_advice / realtime_notice / chitchat
    :param plan: 查询计划（dict），固定话术时可能为空
    :param sources: 引用来源列表
    :param focus_entity_ids: 本轮之后的焦点对象
    :param pending: 澄清时的 {"options": 候选实体 ID, "plan": 查询计划}；否则为 None
    """
    db = get_db()
    now = datetime.now(timezone.utc)
    user_message = {"session_id": session_id, "role": "user", "text": question, "created_at": now}
    assistant_message = {
        "session_id": session_id,
        "role": "assistant",
        "text": answer,
        "kind": kind,
        "plan": plan,
        "sources": sources,
        # Mongo 时间精度为毫秒，回答晚 1ms，保证同一轮内"问在前、答在后"
        "created_at": now + timedelta(milliseconds=1),
    }
    db.messages.insert_many([user_message, assistant_message])
    db.sessions.update_one(
        {"session_id": session_id},
        {
            "$set": {"focus_entity_ids": focus_entity_ids, "pending_clarification": pending, "updated_at": now},
            # 只在新建会话时写入：标题取第一个问题
            "$setOnInsert": {"created_at": now, "title": question[:40]},
        },
        upsert=True,
    )


def list_sessions(limit: int = 30) -> list:
    """最近的会话（按最后更新时间倒序）：[{"session_id", "title", "updated_at"}]"""
    db = get_db()
    fields = {"_id": 0, "session_id": 1, "title": 1, "updated_at": 1}
    sessions = list(db.sessions.find({}, fields).sort("updated_at", -1).limit(limit))
    for session in sessions:
        # 早期会话没有记标题，取第一个问题
        if not session.get("title"):
            first = db.messages.find_one({"session_id": session["session_id"], "role": "user"},
                                         sort=[("created_at", 1)])
            if first:
                session["title"] = first.get("text", "")[:40]
            else:
                session["title"] = ""
    return sessions


def get_messages(session_id: str, limit: int = 200) -> list:
    """某个会话的全部问答，按时间正序"""
    fields = {"_id": 0, "role": 1, "text": 1, "kind": 1, "sources": 1, "created_at": 1}
    cursor = get_db().messages.find({"session_id": session_id}, fields).sort("created_at", 1)
    return list(cursor.limit(limit))


def delete_sessions(session_ids: list):
    """删除会话及其问答记录（评测结束后清理评测产生的会话）"""
    db = get_db()
    db.sessions.delete_many({"session_id": {"$in": session_ids}})
    db.messages.delete_many({"session_id": {"$in": session_ids}})
