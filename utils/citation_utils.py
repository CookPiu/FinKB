"""引用渲染：保留正文中真实存在的证据编号，删除模型编造的编号，按首次出现顺序生成来源列表。"""

from __future__ import annotations

import re

from common.models.evidence import Evidence

MARK = re.compile(r"[\[【]E(\d+)[\]】]")


def finalize_citations(text: str, evidence: list[Evidence]) -> tuple[str, list[Evidence]]:
    valid = {e.eid: e for e in evidence}
    used: list[int] = []

    def repl(m: re.Match) -> str:
        n = int(m.group(1))
        if n not in valid:
            return ""
        if n not in used:
            used.append(n)
        return f"[E{n}]"

    return MARK.sub(repl, text), [valid[n] for n in used]


def render_sources(sources: list[Evidence]) -> str:
    """需求 §6.2：资料名称、内容类型、产品名称/代码、发布时间（M2 未抽取，暂缺）、来源文件名，另加页码。"""
    lines = []
    for e in sources:
        parts = [e.document_title, e.content_type]
        if e.entity_name:
            codes = f"（{'、'.join(e.entity_codes)}）" if e.entity_codes else ""
            parts.append(f"{e.entity_name}{codes}")
        parts.append(e.file_name)
        if e.page_start:
            parts.append(f"第{e.page_start}页" if e.page_start == e.page_end else f"第{e.page_start}-{e.page_end}页")
        lines.append(f"[E{e.eid}] " + "｜".join(parts))
    return "\n".join(lines)
