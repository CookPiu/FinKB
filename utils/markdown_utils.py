"""Markdown 本地解析：转成与 MinerU content_list 同构的块列表，后续阶段无需区分来源。"""
import html
import re

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_TABLE_SEP = re.compile(r"^\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$")


def _split_row(line: str) -> list:
    """把一行 Markdown 表格拆成单元格文本（去掉首尾竖线）"""
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    return [c.strip() for c in line.split("|")]


def _build_table_html(rows: list) -> str:
    """单元格二维列表 → 不带表头标记的 HTML 表格（由表格解析器自行识别表头）"""
    body = ""
    for row in rows:
        cells = "".join([f"<td>{html.escape(c)}</td>" for c in row])
        body += f"<tr>{cells}</tr>"
    return f"<table>{body}</table>"


def _flush_paragraph(blocks: list, para: list):
    """把累积的段落行合成一个文本块，并清空累积"""
    if para:
        blocks.append({"type": "text", "text": "\n".join(para).strip(), "page_idx": 0})
        para.clear()


def parse_markdown(text: str) -> list:
    """
    解析 Markdown 正文
    :param text: Markdown 全文
    :return: content_list 格式的块列表（标题带 text_level，表格转成 table_body HTML，页码统一为 0）
    """
    blocks = []
    lines = text.splitlines()
    para = []
    i = 0
    while i < len(lines):
        line = lines[i]
        m = _HEADING.match(line)
        if m:
            _flush_paragraph(blocks, para)
            blocks.append({"type": "text", "text": m.group(2), "text_level": len(m.group(1)), "page_idx": 0})
            i += 1
            continue
        # 表格：本行含竖线且下一行是分隔行
        if "|" in line and i + 1 < len(lines) and _TABLE_SEP.match(lines[i + 1]):
            _flush_paragraph(blocks, para)
            rows = [_split_row(line)]
            i += 2
            while i < len(lines) and "|" in lines[i] and lines[i].strip():
                rows.append(_split_row(lines[i]))
                i += 1
            blocks.append({"type": "table", "table_body": _build_table_html(rows), "page_idx": 0})
            continue
        if not line.strip():
            _flush_paragraph(blocks, para)
        else:
            para.append(line)
        i += 1
    _flush_paragraph(blocks, para)
    return blocks
