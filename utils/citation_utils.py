"""
证据与引用：证据在提示词里的写法、引用编号校正、来源列表渲染
证据 evidence 为 dict：eid, kind(text / table / fact / summary / image_desc), text, doc_id, file_name, document_title, content_type,
page_start, page_end, source_path, entity_name, entity_codes, derived。
模型只写编号 [E1]，引用由代码根据证据携带的来源元数据渲染：保留正文中真实存在的编号，删除模型编造的编号，
按首次出现顺序生成来源列表。
"""
import copy
import re

MARK = re.compile(r"[\[【]E(\d+)[\]】]")


def build_prompt_block(evidence: dict, max_chars: int = 1500) -> str:
    """
    一条证据在提示词里的写法：编号、资料名称、内容类型、页码，正文截断到 max_chars 字
    """
    pages = ""
    if evidence["page_start"]:
        pages = f"｜第{evidence['page_start']}-{evidence['page_end']}页"
    derived = ""
    if evidence["derived"]:
        derived = "｜据图表示意"
    head = f"[E{evidence['eid']}] 〔{evidence['document_title']}｜{evidence['content_type']}{pages}{derived}〕"
    return f"{head}\n{evidence['text'][:max_chars]}"


def build_source(evidence: dict) -> dict:
    """证据 → 引用来源：复制一份并去掉正文（推送给前端、写入会话记录）"""
    source = copy.deepcopy(evidence)
    source.pop("text")
    return source


def finalize_citations(text: str, evidence: list):
    """
    校正回答中的引用编号：证据里有的统一写成 [En]，没有的（模型编造）删除
    :param text: 模型生成的回答
    :param evidence: 本轮证据列表
    :return: 元组 (校正后的回答, 被引用的证据列表)，证据按在正文中首次出现的顺序
    """
    valid = {}
    for item in evidence:
        valid[item["eid"]] = item
    used = []
    parts = []
    last_end = 0
    for m in MARK.finditer(text):
        parts.append(text[last_end:m.start()])
        n = int(m.group(1))
        if n in valid:
            if n not in used:
                used.append(n)
            parts.append(f"[E{n}]")
        last_end = m.end()
    parts.append(text[last_end:])
    return "".join(parts), [valid[n] for n in used]


def render_sources(sources: list) -> str:
    """
    来源列表文本（需求 §6.2）：资料名称、内容类型、产品名称/代码、来源文件名，另加页码
    发布时间 M2 未抽取，暂缺
    """
    lines = []
    for source in sources:
        parts = [source["document_title"], source["content_type"]]
        if source["entity_name"]:
            codes = ""
            if source["entity_codes"]:
                codes = f"（{'、'.join(source['entity_codes'])}）"
            parts.append(f"{source['entity_name']}{codes}")
        parts.append(source["file_name"])
        if source["page_start"]:
            if source["page_start"] == source["page_end"]:
                parts.append(f"第{source['page_start']}页")
            else:
                parts.append(f"第{source['page_start']}-{source['page_end']}页")
        lines.append(f"[E{source['eid']}] " + "｜".join(parts))
    return "\n".join(lines)
