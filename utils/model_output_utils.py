"""
回答模型的输出解析：正文 + 分隔符 + 一行 JSON 元信息。
模型先写正文，再另起一行写 <<<META>>>，最后写 {"kind": ..., "cited": [...], "notices": [...]}。
解析失败时不报错，按“普通回答、无标记”兜底，由调用方记日志——宁可少一条提示语，也不要整轮问答失败。
"""
import json
import re

from common.logging.logger import logger

META_MARK = "<<<META>>>"
KINDS = ["answer", "refuse", "clarify", "decline_advice", "realtime_notice", "chitchat"]
NOTICE_NAMES = ["risk", "realtime"]
# 兜底用：正文里出现过的引用编号
CITE_MARK = re.compile(r"[\[【]E(\d+)[\]】]")


def split_body_and_meta(raw: str):
    """
    按分隔符切开正文与元信息
    :return: 元组 (正文, 元信息文本)；没有分隔符时元信息为空串
    """
    if META_MARK not in raw:
        return raw.strip(), ""
    body, meta = raw.split(META_MARK, 1)
    return body.strip(), meta.strip()


def get_int_list(value) -> list:
    """把 cited 字段整理成整数列表，忽略非数字元素"""
    result = []
    if not isinstance(value, list):
        return result
    for item in value:
        if isinstance(item, bool):
            continue
        if isinstance(item, int):
            result.append(item)
        elif isinstance(item, str) and item.strip().isdigit():
            result.append(int(item.strip()))
    return result


def get_notices(value) -> list:
    """把 notices 字段整理成已知标记的列表"""
    result = []
    if not isinstance(value, list):
        return result
    for item in value:
        if isinstance(item, str) and item.strip() in NOTICE_NAMES and item.strip() not in result:
            result.append(item.strip())
    return result


def parse_meta(meta_text: str, body: str) -> dict:
    """
    解析元信息
    :param meta_text: <<<META>>> 之后的内容
    :param body: 正文，用于兜底推断引用编号
    :return: {"kind", "cited", "notices", "meta_ok"}；meta_ok 为 False 表示走了兜底
    """
    fallback = {
        "kind": "answer",
        "cited": [int(n) for n in CITE_MARK.findall(body)],
        "notices": [],
        "meta_ok": False,
    }
    if not meta_text:
        logger.warning("模型输出里没有元信息分隔符，按普通回答处理")
        return fallback
    match = re.search(r"\{.*\}", meta_text, re.S)
    if not match:
        logger.warning(f"元信息里没有 JSON：{meta_text[:80]}")
        return fallback
    try:
        data = json.loads(match.group(0))
    except ValueError as e:
        logger.warning(f"元信息 JSON 解析失败：{e}")
        return fallback
    if not isinstance(data, dict):
        logger.warning("元信息不是 JSON 对象")
        return fallback
    kind = data.get("kind")
    if kind not in KINDS:
        logger.warning(f"元信息里的 kind 取值不合法：{kind}")
        return fallback
    return {
        "kind": kind,
        "cited": get_int_list(data.get("cited")),
        "notices": get_notices(data.get("notices")),
        "meta_ok": True,
    }
