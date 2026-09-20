"""
节点：文档切分。blocks.json → chunks.json。
- 正文按章节聚合：目标 CHUNK_TARGET_CHARS 字、上限 CHUNK_MAX_CHARS 字；缓冲不足 CHUNK_MIN_CHARS 字时跨小节继续累积，
  标题行保留在切片正文中；超长段落按句切分；
- 表格独立成块：展示文本 = 标题/说明行 + Markdown 表格 + 表注；向量文本 = 说明 + 逐行线性化；
  线性化超过 TABLE_MAX_CHARS 字时按行拆分，每段重复表头；
- 图片描述独立成块（kind=image_desc，derived=true）。
每次切分产出一套新切片，文档版本号 +1（入库节点先写新版本、再删旧版本）。

切片 chunk 是 dict，字段见 new_chunk；chunks.json 写出全部字段。
"""
import re

from common.logging.logger import logger, node_log
from processor.import_processor.nodes.node_normalize import get_last_page, read_blocks
from processor.import_processor.state import ImportGraphState
from utils.artifact_utils import BLOCKS, CHUNKS, get_doc_dir, write_json
from utils.table_html_utils import get_unit_hint, linearize_rows, parse_table, split_rows, to_markdown
from utils.task_utils import STAGE_CHUNK, is_stage_done, mark_stage_done, mark_stage_failed, mark_stage_running

# 切分参数（评测结果里会记录）
CHUNK_TARGET_CHARS = 600  # 正文切片目标长度，缓冲累积到该长度即断开
CHUNK_MAX_CHARS = 1000  # 正文切片上限
CHUNK_MIN_CHARS = 200  # 缓冲不足该长度时跨小节、跨表格继续累积，避免碎片切片
TABLE_MAX_CHARS = 3000  # 表格线性化文本上限，超过时按行拆成多段

_SENT_SPLIT = re.compile(r"(?<=[。！？!?；;\n])")


def new_chunk(kind: str, text: str, embed_body: str, section_path: str, page_start: int, page_end: int) -> dict:
    """
    新建切片；字段顺序即 chunks.json 中的字段顺序
    :param kind: text / table / summary / term / image_desc
    :param text: 展示文本
    :param embed_body: 计算向量用的文本
    :param section_path: 章节路径（“ > ”连接的字符串）
    """
    return {
        "seq": 0,
        "kind": kind,
        "text": text,
        "embed_body": embed_body,
        "section_path": section_path,
        "page_start": page_start,
        "page_end": page_end,
        "derived": False,
        "table_html": None,
    }


def split_long_text(text: str, target: int, max_chars: int) -> list:
    """按句切分超长文本：累积到 target 即断开，任何片段不超过 max_chars"""
    pieces = []
    cur = ""
    for sent in _SENT_SPLIT.split(text):
        if not sent:
            continue
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


def _path_str(path: list) -> str:
    return " > ".join(path)


def build_table_chunks(block: dict, max_chars: int) -> list:
    """
    表格切片：展示文本 = 标题 + 单位说明 + Markdown 表格 + 表注；向量文本 = [标题 · 单位] + 逐行线性化。
    线性化超过 max_chars 时按行拆成多段，每段重复表头，表注只放在最后一段，原表 HTML 只在不拆分时保留。
    """
    table = parse_table(block["table_html"] or "")
    if not table["grid"]:
        return []
    unit = block["context"] or get_unit_hint(table)
    label = " · ".join([x for x in [block["caption"], unit] if x])
    data_rows = table["grid"][table["header_rows"]:]
    if data_rows or not table["header_rows"]:
        groups = split_rows(table, max_chars)
    else:
        groups = [[]]  # 只有表头：输出一个只含表头的切片
    if not groups:
        groups = [[]]
    chunks = []
    for gi, rows in enumerate(groups):
        part = ""
        if len(groups) > 1:
            part = f"（续表 {gi + 1}/{len(groups)}）"
        head = " ".join([x for x in [block["caption"], part] if x])
        display = [head, block["context"], to_markdown(table, rows)]
        if gi == len(groups) - 1 and block["footnote"]:
            display.append(block["footnote"])
        embed = []
        if label:
            embed.append(f"[{label}]")
        embed.extend(linearize_rows(table, rows))
        chunk = new_chunk(
            "table",
            "\n".join([x for x in display if x]),
            "\n".join(embed),
            _path_str(block["section_path"]),
            block["page"],
            get_last_page(block),
        )
        if len(groups) == 1:
            chunk["table_html"] = block["table_html"]
        chunks.append(chunk)
    return chunks


def build_image_chunk(block: dict) -> dict:
    """图片描述切片：“[图表] 标题” + 描述 + 脚注，标记 derived"""
    if block["caption"]:
        head = f"[图表] {block['caption']}"
    else:
        head = "[图表]"
    text = "\n".join([x for x in [head, block["text"], block["footnote"]] if x])
    chunk = new_chunk("image_desc", text, text, _path_str(block["section_path"]), block["page"], get_last_page(block))
    chunk["derived"] = True
    return chunk


def _new_text_chunk(text: str, path: list, page_start: int, page_end: int) -> dict:
    return new_chunk("text", text, text, _path_str(path), page_start, page_end)


def _buffer_len(buf: list) -> int:
    return sum(len(x["text"]) + 1 for x in buf)


def _is_all_headings(buf: list) -> bool:
    """缓冲里是否只有标题（空缓冲也算）"""
    return all(x["type"] == "heading" for x in buf)


def _flush_buffer(buf: list, chunks: list):
    """把缓冲中的块合成一个正文切片并清空缓冲；只有标题没有正文的缓冲不单独成块，留给后续内容"""
    if not buf:
        return
    if _is_all_headings(buf):
        return
    text = "\n".join([x["text"] for x in buf])
    page_start = min([x["page"] for x in buf])
    page_end = max([get_last_page(x) for x in buf])
    chunks.append(_new_text_chunk(text, buf[0]["section_path"], page_start, page_end))
    buf.clear()


def _add_long_text_chunks(block: dict, buf: list, chunks: list, target: int, max_chars: int):
    """超长段落：先输出缓冲，再按句切分；缓冲里剩下的标题行拼在第一段前面"""
    _flush_buffer(buf, chunks)
    prefix = "\n".join([x["text"] for x in buf])  # flush 后缓冲里只可能剩标题
    pieces = split_long_text(block["text"], target, max_chars)
    for i, piece in enumerate(pieces):
        if i == 0 and prefix:
            text = f"{prefix}\n{piece}"
        else:
            text = piece
        chunks.append(_new_text_chunk(text, block["section_path"], block["page"], get_last_page(block)))
    buf.clear()


def build_chunks(blocks: list, target: int, max_chars: int, min_chars: int, table_max: int) -> list:
    """
    版面块 → 切片
    :param target: 正文切片目标长度
    :param max_chars: 正文切片上限
    :param min_chars: 缓冲不足该长度时跨小节、跨表格继续累积
    :param table_max: 表格线性化文本上限
    :return: 切片 dict 列表，seq 为下标
    """
    chunks = []
    buf = []  # 待合并成一个正文切片的标题与正文块
    for b in blocks:
        if b["absorbed"]:
            continue
        if b["type"] in ("table", "image"):
            if _is_all_headings(buf):
                buf.clear()  # 标题下的内容就是这张表/图，标题已记录在其 section_path 中
            elif b["type"] == "table" and _buffer_len(buf) >= min_chars:
                _flush_buffer(buf, chunks)
            # 不足 min_chars 的零碎正文（如“□是 √否”）跨过表格继续累积，避免产生碎片切片
            if b["type"] == "table":
                chunks.extend(build_table_chunks(b, table_max))
            else:
                chunks.append(build_image_chunk(b))
            continue
        if b["type"] == "heading":
            if _buffer_len(buf) >= min_chars:
                _flush_buffer(buf, chunks)
            elif buf and _is_all_headings(buf) and b["level"] is not None:
                # 连续标题：丢掉已被新标题取代的同级或更深标题
                buf = [x for x in buf if (x["level"] or 0) < b["level"]]
            buf.append(b)
            continue
        if len(b["text"]) > max_chars:
            _add_long_text_chunks(b, buf, chunks, target, max_chars)
            continue
        if buf and _buffer_len(buf) + len(b["text"]) > max_chars:
            _flush_buffer(buf, chunks)
        buf.append(b)
        if _buffer_len(buf) >= target:
            _flush_buffer(buf, chunks)
    _flush_buffer(buf, chunks)  # 文末若只剩标题则丢弃：没有可检索的内容

    for i, c in enumerate(chunks):
        c["seq"] = i
    return chunks


@node_log("node_document_split")
def node_document_split(state: ImportGraphState):
    """
    节点功能：把版面块 blocks.json 切成检索切片 chunks.json，并把文档版本号 +1。
    上游 node_normalize；下游 node_bge_embedding 读取 chunks.json 计算向量。
    已完成本阶段的文档直接跳过（断点续跑）；失败时标记 failed 后抛出。
    """
    doc = state["doc"]
    if is_stage_done(doc, STAGE_CHUNK):
        logger.info(f"chunk     跳过（已完成） {doc['file_name']}")
        return state
    mark_stage_running(doc)
    try:
        chunk_count = split_document(doc["doc_id"])
    except Exception as e:
        mark_stage_failed(doc, STAGE_CHUNK, e)
        raise
    # 每次切分产出一套新切片，版本号 +1；入库节点按新版本写入后删除旧版本
    mark_stage_done(doc, STAGE_CHUNK, {"chunk_count": chunk_count, "version": doc["version"] + 1})
    return state


def split_document(doc_id: str) -> int:
    """切分一个文档：读 blocks.json，写 chunks.json，返回切片数"""
    doc_dir = get_doc_dir(doc_id)
    blocks = read_blocks(doc_dir / BLOCKS)
    chunks = build_chunks(blocks, CHUNK_TARGET_CHARS, CHUNK_MAX_CHARS, CHUNK_MIN_CHARS, TABLE_MAX_CHARS)
    write_json(doc_dir / CHUNKS, chunks)
    return len(chunks)


if __name__ == "__main__":
    # 运行：uv run python -m processor.import_processor.nodes.node_document_split <doc_id>
    # 只演示纯函数部分：读 data/artifacts/<doc_id>/blocks.json 切分后打印前 5 个切片。
    # 不写文件、不连 Mongo（节点本身会写 Mongo 进度与版本号）
    import sys

    test_blocks = read_blocks(get_doc_dir(sys.argv[1]) / BLOCKS)
    test_chunks = build_chunks(test_blocks, CHUNK_TARGET_CHARS, CHUNK_MAX_CHARS, CHUNK_MIN_CHARS, TABLE_MAX_CHARS)
    logger.info(f"切片 {len(test_chunks)} 个")
    for test_chunk in test_chunks[:5]:
        print(test_chunk["seq"], test_chunk["kind"], test_chunk["section_path"], test_chunk["text"][:60])
