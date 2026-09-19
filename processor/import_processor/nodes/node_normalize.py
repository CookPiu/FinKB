"""节点：规范化。content_list.json → blocks.json。

- 丢弃页眉、页脚、页码；清洗文本中的 HTML 标签、Markdown 粗体与链接（utils/text_utils.py）；
- MinerU 的标题层级基本是扁平的（除文档标题外都是 2 级），按编号样式推断层级并重建章节路径；
- 跨页断开的段落拼回一段；
- 表格：MinerU vlm 已合并大部分跨页表格，只留下空表格块，据此延长前表页码；未合并的相邻同列数续表在此合并；
  表格上方的“单位：元 币种：人民币”一类说明行并入表格；
- 图片与图表交给视觉模型描述（结果缓存在 image_desc.json），标记 derived=true。
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from common.logging.logger import logger, node_log
from common.models.document import Block, DocumentRecord, Stage
from processor.import_processor.state import ImportGraphState
from utils import artifact_utils as artifacts
from utils.image_utils import image_size
from utils.lm import lm_utils as llm
from utils.table_html_utils import merge_html, parse_table
from utils.task_utils import run_stage
from utils.text_utils import clean_text

DROP_TYPES = {"header", "footer", "page_number"}
IMAGE_TYPES = {"image", "chart"}
MIN_IMAGE_SIDE = 200  # 长边小于该像素的图片视为图标，不描述
NO_INFO = "无信息图片"
VL_WORKERS = 4

# ---------- 标题层级 ----------

_CN_NUM = "一二三四五六七八九十百零〇两"
_LEVEL_RULES: list[tuple[re.Pattern[str], int]] = [
    (re.compile(rf"^第[{_CN_NUM}\d]+\s*(部分|篇|章)"), 2),
    (re.compile(rf"^第[{_CN_NUM}\d]+\s*节"), 3),
    (re.compile(rf"^[{_CN_NUM}]+\s*[、.．]"), 3),
    (re.compile(rf"^[{_CN_NUM}]+\s+\S"), 3),
    (re.compile(rf"^[（(]\s*[{_CN_NUM}]+\s*[)）]"), 4),
    (re.compile(r"^专栏\s*\d+"), 4),
    (re.compile(r"^\d+\.\d+\.\d+"), 5),
    (re.compile(r"^\d+\.\d+(?![.\d])"), 4),
    (re.compile(r"^\d+\s*[、.．](?!\d)"), 5),
    (re.compile(r"^\d{1,2}\s+[^\d\s年月日]"), 3),  # 招商银行季报的“2 主要财务数据”
    (re.compile(r"^[（(]\s*\d+\s*[)）]"), 6),
    (re.compile(r"^[①-⑳]"), 7),
]
_NOT_HEADING_END = ("，", ",", "；", ";", "、")
ORPHAN_LEVEL = 9


def numbered_level(title: str) -> int | None:
    for pattern, level in _LEVEL_RULES:
        if pattern.match(title):
            return level
    return None


def is_heading_text(title: str) -> bool:
    return bool(title) and len(title) <= 60 and not title.endswith(_NOT_HEADING_END) and "□" not in title


class HeadingStack:
    """维护章节栈。文档首个 1 级标题是文档标题，不进入章节路径；其余 1 级标题视为子文档（如附带的产品说明书）。"""

    def __init__(self) -> None:
        self.stack: list[tuple[int, str, bool]] = []  # (level, title, numbered)
        self.seen_title = False

    def push(self, title: str, mineru_level: int) -> int:
        if mineru_level == 1:
            level, numbered = 1, True
            if not self.seen_title:
                self.seen_title = True
                self.stack = []
                return 1
        else:
            nl = numbered_level(title)
            if nl is not None:
                level, numbered = nl, True
            else:
                # 无编号标题挂在最近的有编号标题下；没有编号父级时放到最深层，
                # 让后续任何有编号标题都能把它弹出（如“重要内容提示”不应成为“第一节”的父级）
                parent = next((lv for lv, _, num in reversed(self.stack) if num and lv > 1), None)
                level, numbered = (min(parent + 1, 8) if parent else ORPHAN_LEVEL), False
        while self.stack and self.stack[-1][0] >= level:
            self.stack.pop()
        self.stack.append((level, title, numbered))
        return level

    @property
    def path(self) -> list[str]:
        return [t for _, t, _ in self.stack]


def _describe_images(blocks: list[Block], base: Path, cache_path: Path) -> None:
    cache: dict[str, str] = artifacts.read_json(cache_path) if cache_path.exists() else {}
    todo: list[Block] = []
    for b in blocks:
        if b.type != "image" or not b.img_path:
            continue
        if b.img_path in cache:
            continue
        p = base / b.img_path
        size = image_size(p) if p.exists() else None
        if size is None or max(size) < MIN_IMAGE_SIDE:
            cache[b.img_path] = NO_INFO
            continue
        todo.append(b)

    def work(b: Block) -> tuple[str, str | None]:
        ctx = b.caption or " > ".join(b.section_path[-2:])
        try:
            return b.img_path, llm.describe_image(base / b.img_path, ctx)
        except Exception as e:  # noqa: BLE001 - 单张图片失败不影响文档
            logger.warning("图片描述失败 %s：%s", b.img_path, e)
            return b.img_path, None

    if todo:
        logger.info("视觉模型描述 %d 张图片", len(todo))
        with ThreadPoolExecutor(VL_WORKERS) as pool:
            for path, desc in pool.map(work, todo):
                if desc is not None:
                    cache[path] = desc
        artifacts.write_json(cache_path, cache)
    elif not cache_path.exists():
        artifacts.write_json(cache_path, cache)

    for b in blocks:
        if b.type == "image" and b.img_path:
            desc = cache.get(b.img_path)
            b.text = "" if desc is None or NO_INFO in desc else desc


# ---------- 主流程 ----------

_UNIT_LINE = re.compile(r"^[（(]?\s*(单位|金额单位|货币单位)\s*[:：]")
_SENTENCE_END = tuple("。！？!?；;：:”」』）)]")


def _mineru_base(doc_dir: Path) -> Path:
    """图片路径相对于 MinerU 解压目录中 content_list.json 所在目录。"""
    found = sorted((doc_dir / "mineru").rglob("*content_list.json")) if (doc_dir / "mineru").exists() else []
    return found[0].parent if found else doc_dir


def normalize(raw_blocks: list[dict]) -> list[Block]:
    """纯函数部分（不含图片描述），便于单元测试。"""
    out: list[Block] = []
    heads = HeadingStack()
    last_para: Block | None = None  # 可接续的上一段正文（中间只隔页眉页脚/脚注）
    last_unit: tuple[tuple[str, ...], str] | None = None  # (所在章节, 单位说明)

    def add(b: Block) -> Block:
        b.seq = len(out)
        out.append(b)
        return b

    for raw in raw_blocks:
        t = raw.get("type", "text")
        page = int(raw.get("page_idx", 0)) + 1
        if t in DROP_TYPES:
            continue

        if t == "table":
            body = raw.get("table_body") or ""
            prev = out[-1] if out else None
            if not body.strip():
                # MinerU 已把续表并入前表，只剩空壳：前表页码延到本页
                if prev is not None and prev.type == "table":
                    prev.page_end = max(prev.last_page, page)
                continue
            caption = clean_text(" ".join(raw.get("table_caption") or []))
            footnote = clean_text(" ".join(x for x in raw.get("table_footnote") or [] if x))
            if (
                prev is not None
                and prev.type == "table"
                and page == prev.last_page + 1
                and not caption
                and parse_table(body).n_cols == parse_table(prev.table_html or "").n_cols
            ):
                prev.table_html = merge_html(prev.table_html or "", body)
                prev.page_end = page
                prev.footnote = " ".join(x for x in (prev.footnote, footnote) if x)
                continue
            context = ""
            if prev is not None and prev.type == "text" and _UNIT_LINE.match(prev.text) and len(prev.text) <= 40:
                prev.absorbed = True
                context = prev.text
                last_unit = (tuple(heads.path), context)
            elif last_unit is not None and last_unit[0] == tuple(heads.path):
                context = last_unit[1]  # 同一小节内连续的表格沿用上一条单位说明
            add(
                Block(
                    seq=0,
                    type="table",
                    page=page,
                    section_path=heads.path,
                    table_html=body,
                    caption=caption,
                    footnote=footnote,
                    context=context,
                )
            )
            last_para = None
            continue

        if t in IMAGE_TYPES:
            caption = clean_text(" ".join(raw.get("image_caption") or raw.get("chart_caption") or []))
            footnote = clean_text(" ".join(raw.get("image_footnote") or raw.get("chart_footnote") or []))
            add(
                Block(
                    seq=0,
                    type="image",
                    page=page,
                    section_path=heads.path,
                    img_path=raw.get("img_path") or None,
                    caption=caption,
                    footnote=footnote,
                    text=clean_text(raw.get("content") or ""),
                    derived=True,
                )
            )
            last_para = None
            continue

        # 文本类：text / page_footnote / list / equation / code 等
        if t == "list":
            text = "\n".join(clean_text(x) for x in raw.get("list_items") or [] if x)
        elif t == "code":
            text = (raw.get("code_body") or "").strip()
        else:
            text = clean_text(raw.get("text") or "")
        if not text:
            continue

        level = raw.get("text_level")
        if t == "text" and level and is_heading_text(text):
            lv = heads.push(text, int(level))
            if lv == 1 and not heads.stack:
                continue  # 文档标题：由 document_title 表达，不单独成块
            add(Block(seq=0, type="heading", text=text, level=lv, page=page, section_path=heads.path))
            last_para = None
            continue

        if t == "page_footnote":
            # 页脚注不打断跨页段落的接续
            add(Block(seq=0, type="text", text=f"注：{text}", page=page, section_path=heads.path))
            continue

        # 跨页断开的段落：上一段不以句末标点结尾，且本段在后续页
        if (
            last_para is not None
            and page > last_para.last_page
            and not last_para.text.endswith(_SENTENCE_END)
        ):
            last_para.text += text
            last_para.page_end = page
            continue

        last_para = add(Block(seq=0, type="text", text=text, page=page, section_path=heads.path))

    return out


@node_log("node_normalize")
def node_normalize(state: ImportGraphState) -> dict:
    doc: DocumentRecord = state["doc"]

    def process(d: DocumentRecord) -> dict:
        ddir = artifacts.doc_dir(d.doc_id)
        blocks = normalize(artifacts.read_json(ddir / artifacts.CONTENT_LIST))
        _describe_images(blocks, _mineru_base(ddir), ddir / artifacts.IMAGE_DESC)
        blocks = [b for b in blocks if b.type != "image" or b.text]
        for i, b in enumerate(blocks):
            b.seq = i
        artifacts.write_json(ddir / artifacts.BLOCKS, [b.model_dump(exclude_defaults=True) for b in blocks])
        return {}

    run_stage(doc, Stage.NORMALIZE, process)
    return {"doc": doc}
