"""内容类型判定。M1 只用目录与文件名规则；M2 由 LLM 在枚举内确认（网页上传无目录时兜底）。"""

from __future__ import annotations

import re

from common.models.document import ContentType

# 文件名关键词优先于目录规则（同一目录下可能混放不同类型）
_FILE_RULES: list[tuple[str, ContentType]] = [
    ("消费者权益保护实施办法", ContentType.REGULATION),
    ("调查报告", ContentType.SURVEY),
    ("风险揭示书", ContentType.WEALTH_RISK),
    ("基金产品资料概要", ContentType.FUND_KFS),
    ("季度报告", ContentType.COMPANY_REPORT),
    ("年度报告", ContentType.COMPANY_REPORT),
]

_DIR_RULES: list[tuple[str, ContentType]] = [
    ("上市公司年报", ContentType.COMPANY_REPORT),
    ("基金产品", ContentType.FUND_KFS),
    ("银行理财", ContentType.WEALTH_RISK),
    ("宏观经济", ContentType.MACRO_POLICY),
    ("用户FAQ", ContentType.INVESTOR_FAQ),
]


def guess_content_type(rel_dir: str, file_name: str) -> ContentType:
    for kw, ct in _FILE_RULES:
        if kw in file_name:
            return ct
    for kw, ct in _DIR_RULES:
        if kw in rel_dir:
            return ct
    return ContentType.OTHER


_DATE_PREFIX = re.compile(r"^\d{4}-\d{2}-\d{2}\s*")


def title_from_filename(stem: str) -> str:
    """文件名去掉日期前缀作为初始资料名称；M2 由抽取结果覆盖。"""
    return _DATE_PREFIX.sub("", stem).strip()
