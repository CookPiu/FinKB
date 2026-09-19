"""Markdown 本地解析：转成与 MinerU content_list 同构的块列表，后续阶段无需区分来源。"""

from __future__ import annotations

import html
import re

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_TABLE_SEP = re.compile(r"^\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$")


def _split_row(line: str) -> list[str]:
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    return [c.strip() for c in line.split("|")]


def _table_html(rows: list[list[str]]) -> str:
    body = "".join("<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table>{body}</table>"


def parse_markdown(text: str) -> list[dict]:
    blocks: list[dict] = []
    lines = text.splitlines()
    para: list[str] = []

    def flush_para() -> None:
        if para:
            blocks.append({"type": "text", "text": "\n".join(para).strip(), "page_idx": 0})
            para.clear()

    i = 0
    while i < len(lines):
        line = lines[i]
        m = _HEADING.match(line)
        if m:
            flush_para()
            blocks.append({"type": "text", "text": m.group(2), "text_level": len(m.group(1)), "page_idx": 0})
            i += 1
            continue
        if "|" in line and i + 1 < len(lines) and _TABLE_SEP.match(lines[i + 1]):
            flush_para()
            rows = [_split_row(line)]
            i += 2
            while i < len(lines) and "|" in lines[i] and lines[i].strip():
                rows.append(_split_row(lines[i]))
                i += 1
            blocks.append({"type": "table", "table_body": _table_html(rows), "page_idx": 0})
            continue
        if not line.strip():
            flush_para()
        else:
            para.append(line)
        i += 1
    flush_para()
    return blocks
