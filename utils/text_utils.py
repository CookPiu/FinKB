"""文本清洗：去掉 HTML 标签、Markdown 粗体与链接、逐字定位 PDF 产生的字间空格。"""

from __future__ import annotations

import re

_HTML_TAG = re.compile(r"</?(sup|sub|span|b|i|u|em|strong|br)\b[^>]*>", re.I)
_MD_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_MD_BOLD = re.compile(r"\*\*(.+?)\*\*")
_SPACES = re.compile(r"[ \t　]+")

_CJK = "一-鿿"
_SPACED_PAIR = re.compile(rf"[{_CJK}][ \t　]+(?=[{_CJK}])")
_JOINABLE = rf"{_CJK}\d.%，。、；：！？（）《》“”‘’,"
_JOIN_SPACES = re.compile(rf"(?<=[{_JOINABLE}])[ \t　]+(?=[{_JOINABLE}])")


def despace(text: str) -> str:
    """去掉逐字定位 PDF 被解析出的字间空格（“开 放 式 基 金 上 涨 1 2 1 . 0 2 %”）。

    只在“汉字-空格-汉字”占汉字数 20% 以上的文本上生效，正常文本（如“2026 年 3 月”）不受影响；
    只合并同一行内的空格，不跨换行。
    """
    cjk = len(re.findall(rf"[{_CJK}]", text))
    if cjk < 8 or len(_SPACED_PAIR.findall(text)) / cjk < 0.2:
        return text
    return _JOIN_SPACES.sub("", text)


def clean_text(text: str) -> str:
    text = _HTML_TAG.sub("", text or "")
    text = _MD_LINK.sub(r"\1", text)
    text = _MD_BOLD.sub(r"\1", text).replace("**", "")
    text = despace(text)
    text = _SPACES.sub(" ", text)
    return text.strip()
