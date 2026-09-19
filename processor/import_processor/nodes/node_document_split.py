"""节点：文档切分。blocks.json → chunks.json。

- 正文按章节聚合：目标 CHUNK_TARGET 字、上限 CHUNK_MAX 字；缓冲不足 CHUNK_MIN 字时跨小节继续累积，
  标题行保留在切片正文中；超长段落按句切分；
- 表格独立成块：展示文本 = 标题/说明行 + Markdown 表格 + 表注；向量文本 = 说明 + 逐行线性化；
  线性化超过 TABLE_MAX 字时按行拆分，每段重复表头；
- 图片描述独立成块（kind=image_desc，derived=true）。
每次切分产出一套新切片，文档版本号 +1（入库节点先写新版本、再删旧版本）。
"""

from __future__ import annotations

import re

from common.config.settings import get_settings
from common.logging.logger import node_log
from common.models.document import Block, Chunk, ChunkKind, DocumentRecord, Stage
from processor.import_processor.state import ImportGraphState
from utils import artifact_utils as artifacts
from utils.table_html_utils import linearize_rows, parse_table, split_rows, to_markdown
from utils.task_utils import run_stage

_SENT_SPLIT = re.compile(r"(?<=[。！？!?；;\n])")


def split_long_text(text: str, target: int, max_chars: int) -> list[str]:
    """按句切分超长文本：累积到 target 即断开，任何片段不超过 max_chars。"""
    pieces: list[str] = []
    cur = ""
    for sent in (s for s in _SENT_SPLIT.split(text) if s):
        while len(sent) > max_chars:  # 单句超长时硬切
            if cur:
                pieces.append(cur)
                cur = ""
            pieces.append(sent[:max_chars])
            sent = sent[max_chars:]
        if cur and len(cur) + len(sent) > max_chars:
            pieces.append(cur)
            cur = ""
        cur += sent
        if len(cur) >= target:
            pieces.append(cur)
            cur = ""
    if cur:
        pieces.append(cur)
    return [p.strip() for p in pieces if p.strip()]


def _path_str(path: list[str]) -> str:
    return " > ".join(path)


def table_chunks(b: Block, max_chars: int) -> list[Chunk]:
    table = parse_table(b.table_html or "")
    if not table.grid:
        return []
    label = " · ".join(x for x in (b.caption, b.context or table.unit_hint) if x)
    groups = split_rows(table, max_chars) if table.data_rows or not table.header_rows else [[]]
    if not groups:
        groups = [[]]
    out: list[Chunk] = []
    for gi, rows in enumerate(groups):
        part = f"（续表 {gi + 1}/{len(groups)}）" if len(groups) > 1 else ""
        head = " ".join(x for x in (b.caption, part) if x)
        display = [head, b.context, to_markdown(table, rows)]
        if gi == len(groups) - 1 and b.footnote:
            display.append(b.footnote)
        lines = linearize_rows(table, rows)
        embed = ([f"[{label}]"] if label else []) + lines
        out.append(
            Chunk(
                seq=0,
                kind=ChunkKind.TABLE,
                text="\n".join(x for x in display if x),
                embed_body="\n".join(embed),
                section_path=_path_str(b.section_path),
                page_start=b.page,
                page_end=b.last_page,
                table_html=b.table_html if len(groups) == 1 else None,
            )
        )
    return out


def image_chunk(b: Block) -> Chunk:
    head = f"[图表] {b.caption}" if b.caption else "[图表]"
    parts = [head, b.text, b.footnote]
    text = "\n".join(x for x in parts if x)
    return Chunk(
        seq=0,
        kind=ChunkKind.IMAGE_DESC,
        text=text,
        embed_body=text,
        section_path=_path_str(b.section_path),
        page_start=b.page,
        page_end=b.last_page,
        derived=True,
    )


def build_chunks(blocks: list[Block], *, target: int, max_chars: int, min_chars: int, table_max: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    buf: list[Block] = []

    def buf_len() -> int:
        return sum(len(x.text) + 1 for x in buf)

    def emit_text(text: str, path: list[str], p0: int, p1: int) -> None:
        chunks.append(
            Chunk(
                seq=0,
                kind=ChunkKind.TEXT,
                text=text,
                embed_body=text,
                section_path=_path_str(path),
                page_start=p0,
                page_end=p1,
            )
        )

    def flush() -> None:
        if not buf:
            return
        # 只有标题没有正文的缓冲不单独成块，留给后续内容
        if all(x.type == "heading" for x in buf):
            return
        emit_text(
            "\n".join(x.text for x in buf),
            buf[0].section_path,
            min(x.page for x in buf),
            max(x.last_page for x in buf),
        )
        buf.clear()

    for b in blocks:
        if b.absorbed:
            continue
        if b.type in ("table", "image"):
            if all(x.type == "heading" for x in buf):
                buf.clear()  # 标题下的内容就是这张表/图，标题已记录在其 section_path 中
            elif b.type == "table" and buf_len() >= min_chars:
                flush()
            # 不足 min_chars 的零碎正文（如“□是 √否”）跨过表格继续累积，避免产生碎片切片
            chunks.extend(table_chunks(b, table_max) if b.type == "table" else [image_chunk(b)])
            continue
        if b.type == "heading":
            if buf_len() >= min_chars:
                flush()
            elif buf and all(x.type == "heading" for x in buf) and b.level is not None:
                # 连续标题：丢掉已被新标题取代的同级或更深标题
                buf[:] = [x for x in buf if (x.level or 0) < b.level]
            buf.append(b)
            continue
        if len(b.text) > max_chars:
            flush()
            prefix = "\n".join(x.text for x in buf)  # flush 后缓冲里只可能剩标题
            pieces = split_long_text(b.text, target, max_chars)
            for i, piece in enumerate(pieces):
                text = f"{prefix}\n{piece}" if i == 0 and prefix else piece
                emit_text(text, b.section_path, b.page, b.last_page)
            buf.clear()
            continue
        if buf and buf_len() + len(b.text) > max_chars:
            flush()
        buf.append(b)
        if buf_len() >= target:
            flush()
    flush()  # 文末若只剩标题则丢弃：没有可检索的内容

    for i, c in enumerate(chunks):
        c.seq = i
    return chunks


@node_log("node_document_split")
def node_document_split(state: ImportGraphState) -> dict:
    doc: DocumentRecord = state["doc"]

    def split(d: DocumentRecord) -> dict:
        s = get_settings()
        ddir = artifacts.doc_dir(d.doc_id)
        blocks = [Block.model_validate(x) for x in artifacts.read_json(ddir / artifacts.BLOCKS)]
        chunks = build_chunks(
            blocks,
            target=s.chunk_target_chars,
            max_chars=s.chunk_max_chars,
            min_chars=s.chunk_min_chars,
            table_max=s.table_max_chars,
        )
        artifacts.write_json(ddir / artifacts.CHUNKS, [c.model_dump(mode="json") for c in chunks])
        return {"chunk_count": len(chunks), "version": d.version + 1}

    run_stage(doc, Stage.CHUNK, split)
    return {"doc": doc}
