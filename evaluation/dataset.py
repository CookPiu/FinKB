"""
评测集读取与自检。
每行一题（JSONL）。单轮题的 expect/gold/must_include 写在顶层；多轮题（category=multiturn）写在每一轮 turns[i] 里。
金标按“文件名 + 关键原句”，不用 chunk_id（切片 ID 随切分参数变化，金标不能随之失效）。
自检：每条 quote 必须能在对应文件的解析文本中找到（两边都做 NFKC 规范化并去掉空白后比较）。

题目 item：{"id", "category", "turns", "expect", "gold", "must_include", "must_not_include", "note"}
一轮 turn：{"q", "expect", "gold", "must_include", "must_not_include"}
金标 gold：{"file", "quote"}
"""
import json
import re
import unicodedata
from pathlib import Path

from utils.artifact_utils import CHUNKS, get_doc_dir
from utils.clients.mongo_utils import get_db

CATEGORIES = ["product", "announcement", "risk", "knowledge", "process", "negative", "compliance", "multiturn"]
EXPECTS = ["answer", "refuse", "clarify", "decline_advice", "realtime_notice"]

WHITESPACE = re.compile(r"\s+")


def norm(text: str) -> str:
    """NFKC 规范化（全角转半角等）并去掉所有空白"""
    return WHITESPACE.sub("", unicodedata.normalize("NFKC", text or ""))


def get_str(data: dict, key: str, default=None) -> str:
    """取字符串字段；缺失且没有默认值、或类型不对时报错"""
    if key not in data:
        if default is None:
            raise ValueError(f"缺少字段 {key}")
        return default
    value = data[key]
    if not isinstance(value, str):
        raise ValueError(f"字段 {key} 应为字符串：{value!r}")
    return value


def get_str_list(data: dict, key: str) -> list:
    """取字符串列表字段，缺失时为空列表"""
    value = data.get(key, [])
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise ValueError(f"字段 {key} 应为字符串列表：{value!r}")
    return list(value)


def get_expect(data: dict):
    """取 expect 字段：可以缺失或为 null，否则必须是 EXPECTS 之一"""
    value = data.get("expect")
    if value is not None and value not in EXPECTS:
        raise ValueError(f"expect 取值无效：{value!r}")
    return value


def get_object_list(data: dict, key: str, required: bool) -> list:
    """取对象列表字段；非必填的缺失时为空列表"""
    if key not in data and not required:
        return []
    value = data.get(key)
    if not isinstance(value, list) or not all(isinstance(v, dict) for v in value):
        raise ValueError(f"字段 {key} 应为对象列表：{value!r}")
    return value


def build_gold(data: dict) -> dict:
    return {"file": get_str(data, "file"), "quote": get_str(data, "quote")}


def build_turn(data: dict) -> dict:
    return {
        "q": get_str(data, "q"),
        "expect": get_expect(data),
        "gold": [build_gold(g) for g in get_object_list(data, "gold", False)],
        "must_include": get_str_list(data, "must_include"),
        "must_not_include": get_str_list(data, "must_not_include"),
    }


def build_eval_item(data: dict) -> dict:
    """
    校验一道题并补齐默认值
    :param data: 评测集一行解析出的 dict
    :return: 题目 item
    :raise ValueError: 缺少必填字段、类型不对或取值不在枚举内
    """
    if not isinstance(data, dict):
        raise ValueError(f"每行应为 JSON 对象：{data!r}")
    category = get_str(data, "category")
    if category not in CATEGORIES:
        raise ValueError(f"category 取值无效：{category!r}")
    return {
        "id": get_str(data, "id"),
        "category": category,
        "turns": [build_turn(t) for t in get_object_list(data, "turns", True)],
        "expect": get_expect(data),
        "gold": [build_gold(g) for g in get_object_list(data, "gold", False)],
        "must_include": get_str_list(data, "must_include"),
        "must_not_include": get_str_list(data, "must_not_include"),
        "note": get_str(data, "note", ""),
    }


def resolve_turns(item: dict) -> list:
    """把顶层字段并入单轮题的第一轮，调用方只需处理 turns。"""
    if item["category"] == "multiturn":
        return item["turns"]
    turn = item["turns"][0]
    return [
        {
            "q": turn["q"],
            "expect": turn["expect"] or item["expect"],
            "gold": turn["gold"] or item["gold"],
            "must_include": turn["must_include"] or item["must_include"],
            "must_not_include": turn["must_not_include"] or item["must_not_include"],
        }
    ]


def load_eval_items(path: Path) -> list:
    """读取 JSONL 评测集，某一行格式错误时报出行号"""
    items = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            items.append(build_eval_item(json.loads(line)))
        except ValueError as e:
            raise ValueError(f"{path.name} 第 {n} 行格式错误：{e}") from e
    return items


def load_corpus() -> dict:
    """文件名 → 规范化后的全文（该文件全部切片正文拼接），与检索时的切片文本同源。"""
    corpus = {}
    for doc in get_db().documents.find({"status": {"$ne": "superseded"}}, {"file_name": 1}):
        path = get_doc_dir(doc["_id"]) / CHUNKS
        if path.exists():
            chunks = json.loads(path.read_text(encoding="utf-8"))
            corpus[doc["file_name"]] = "\n".join([norm(c["text"]) for c in chunks])
    return corpus


def check_turn(tag: str, category: str, turn: dict, corpus: dict) -> list:
    """检查一轮：expect 必填，回答题要有金标，金标原句长度合适且能在对应文件中找到"""
    errors = []
    if turn["expect"] is None:
        errors.append(f"{tag}: 缺少 expect")
    if turn["expect"] == "answer" and not turn["gold"] and category not in ("compliance",):
        errors.append(f"{tag}: expect=answer 但没有 gold")
    for gold in turn["gold"]:
        quote = norm(gold["quote"])
        if not 4 <= len(quote) <= 60:
            errors.append(f"{tag}: quote 长度 {len(quote)} 超出 4~60：{gold['quote']}")
        if gold["file"] not in corpus:
            errors.append(f"{tag}: 文件不在知识库中：{gold['file']}")
        elif quote not in corpus[gold["file"]]:
            errors.append(f"{tag}: quote 在 {gold['file']} 中找不到：{gold['quote']}")
    return errors


def self_check(items: list, corpus: dict) -> list:
    """
    评测集自检
    :param items: 题目列表
    :param corpus: load_corpus() 的结果
    :return: 错误描述列表，为空表示通过
    """
    errors = []
    seen = set()
    for item in items:
        if item["id"] in seen:
            errors.append(f"{item['id']}: id 重复")
        seen.add(item["id"])
        if item["category"] != "multiturn" and len(item["turns"]) != 1:
            errors.append(f"{item['id']}: 单轮题只能有一轮")
        if item["category"] == "multiturn" and len(item["turns"]) < 2:
            errors.append(f"{item['id']}: 多轮题至少两轮")
        for ti, turn in enumerate(resolve_turns(item)):
            errors.extend(check_turn(f"{item['id']}#{ti + 1}", item["category"], turn, corpus))
    return errors
