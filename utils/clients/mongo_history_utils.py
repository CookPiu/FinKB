"""会话状态：sessions 只存焦点实体与待澄清候选；messages 存问答历史。"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from utils.clients import mongo_utils as mongo


def load(session_id: str) -> dict:
    doc = mongo.get_db().sessions.find_one({"session_id": session_id}) or {}
    return {"focus_entity_ids": doc.get("focus_entity_ids", []), "pending": doc.get("pending_clarification")}


def last_turns(messages_desc: list[dict]) -> list[dict]:
    """数据库按时间倒序取出最近 n 条（旧项目 K-23：不能取成最早的 n 条），这里反转回时间正序。"""
    return list(reversed(messages_desc))


def history(session_id: str, n_turns: int = 3) -> list[dict]:
    cur = mongo.get_db().messages.find({"session_id": session_id}, {"role": 1, "text": 1}).sort("created_at", -1)
    return last_turns(list(cur.limit(n_turns * 2)))


def save_turn(
    session_id: str,
    question: str,
    answer: str,
    *,
    kind: str,
    plan: dict | None,
    sources: list[dict],
    focus_entity_ids: list[str],
    pending: dict | None,
) -> None:
    db = mongo.get_db()
    now = datetime.now(UTC)
    db.messages.insert_many(
        [
            {"session_id": session_id, "role": "user", "text": question, "created_at": now},
            {
                "session_id": session_id,
                "role": "assistant",
                "text": answer,
                "kind": kind,
                "plan": plan,
                "sources": sources,
                # Mongo 时间精度为毫秒，回答晚 1ms，保证同一轮内“问在前、答在后”
                "created_at": now + timedelta(milliseconds=1),
            },
        ]
    )
    db.sessions.update_one(
        {"session_id": session_id},
        {
            "$set": {"focus_entity_ids": focus_entity_ids, "pending_clarification": pending, "updated_at": now},
            "$setOnInsert": {"created_at": now, "title": question[:40]},
        },
        upsert=True,
    )


def list_sessions(limit: int = 30) -> list[dict]:
    """最近的会话（按最后更新时间倒序），标题为首个问题。"""
    db = mongo.get_db()
    cur = db.sessions.find({}, {"_id": 0, "session_id": 1, "title": 1, "updated_at": 1}).sort("updated_at", -1)
    sessions = list(cur.limit(limit))
    for s in sessions:
        if not s.get("title"):  # 早期会话没有记标题，取第一个问题
            first = db.messages.find_one({"session_id": s["session_id"], "role": "user"}, sort=[("created_at", 1)])
            s["title"] = (first or {}).get("text", "")[:40]
    return sessions


def get_messages(session_id: str, limit: int = 200) -> list[dict]:
    """某个会话的全部问答，按时间正序。"""
    cur = mongo.get_db().messages.find(
        {"session_id": session_id}, {"_id": 0, "role": 1, "text": 1, "kind": 1, "sources": 1, "created_at": 1}
    ).sort("created_at", 1)
    return list(cur.limit(limit))


def delete_sessions(session_ids: list[str]) -> None:
    db = mongo.get_db()
    db.sessions.delete_many({"session_id": {"$in": session_ids}})
    db.messages.delete_many({"session_id": {"$in": session_ids}})
