"""
版面块：把 MinerU 的解析结果 content_list.json 整理成结构清楚的块，写出 blocks.json。
- 丢弃页眉、页脚、页码；清洗文本中的 HTML 标签、Markdown 粗体与链接（utils/text_utils.py）；
- MinerU 的标题层级基本是扁平的（除文档标题外都是 2 级），按编号样式推断层级并重建章节路径；
- 跨页断开的段落拼回一段；
- 表格：MinerU vlm 已合并大部分跨页表格，只留下空表格块，据此延长前表页码；未合并的相邻同列数续表在此合并；
  表格上方的“单位：元 币种：人民币”一类说明行并入表格；
- 图片与图表交给视觉模型描述（结果缓存在 image_desc.json），标记 derived=true。

版面块 block 是 dict，字段见 new_block。blocks.json 写出时省略取默认值的字段，读取时用 read_blocks 补齐。
调用方：node_chunk（切片）与 node_enrich（抽财务事实）。
"""
import re
from concurrent.futures import ThreadPoolExecutor

from common.logging.logger import logger, step_log
from utils.artifact_utils import BLOCKS, CONTENT_LIST, IMAGE_DESC, get_doc_dir, read_json, write_json
from utils.image_utils import image_size
from utils.lm.lm_utils import describe_image
from utils.table_html_utils import merge_html, parse_table
from utils.text_utils import clean_text

DROP_TYPES = {"header", "footer", "page_number"}
IMAGE_TYPES = {"image", "chart"}
MIN_IMAGE_SIDE = 200  # 长边小于该像素的图片视为图标，不描述
NO_INFO = "无信息图片"
VL_WORKERS = 4

# ---------- 版面块 ----------

# 可省略字段的默认值：blocks.json 写出时省略取默认值的字段（与重构前 model_dump(exclude_defaults=True) 的格式一致）
BLOCK_DEFAULTS = {
    "text": "",
    "level": None,
    "page_end": None,
    "section_path": [],
    "table_html": None,
    "caption": "",
    "footnote": "",
    "context": "",
    "img_path": None,
    "derived": False,
    "absorbed": False,
}


def new_block(block_type: str, page: int, section_path: list) -> dict:
    """
    新建版面块，其余字段取默认值；字段顺序即 blocks.json 中的字段顺序
    :param block_type: heading / text / table / image
    :param page: 起始页码（从 1 开始）
    :param section_path: 章节路径（标题列表）
    """
    return {
        "seq": 0,
        "type": block_type,
        "text": "",
        "level": None,  # 标题层级，只有 heading 有
        "page": page,
        "page_end": None,  # 跨页时的结束页码
        "section_path": section_path,
        "table_html": None,
        "caption": "",
        "footnote": "",
        "context": "",  # 表格上方的单位说明行
        "img_path": None,
        "derived": False,  # 内容由模型生成（图片描述）
        "absorbed": False,  # 已并入表格的说明行，切片时跳过
    }


def get_last_page(block: dict) -> int:
    """块的结束页码"""
    return block["page_end"] or block["page"]


def dump_blocks(blocks: list) -> list:
    """转成写入 blocks.json 的格式：省略取默认值的字段"""
    items = []
    for block in blocks:
        item = {}
        for key, value in block.items():
            if key in BLOCK_DEFAULTS and value == BLOCK_DEFAULTS[key]:
                continue
            item[key] = value
        items.append(item)
    return items


def read_blocks(path) -> list:
    """读取 blocks.json，补齐被省略的默认值字段"""
    blocks = []
    for item in read_json(path):
        block = new_block(item["type"], item["page"], [])
        block.update(item)
        blocks.append(block)
    return blocks


# ---------- 标题层级 ----------

_CN_NUM = "一二三四五六七八九十百零〇两"
# (编号样式, 层级)，按顺序匹配，先匹配到的生效
_LEVEL_RULES = [
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


def numbered_level(title: str):
    """按编号样式推断标题层级；没有编号返回 None"""
    for pattern, level in _LEVEL_RULES:
        if pattern.match(title):
            return level
    return None


def is_heading_text(title: str) -> bool:
    """MinerU 标成标题的文本是否真是标题：过长、以逗号分号结尾、含勾选框“□”的都不是"""
    return bool(title) and len(title) <= 60 and not title.endswith(_NOT_HEADING_END) and "□" not in title


class HeadingStack:
    """
    维护章节栈。文档首个 1 级标题是文档标题，不进入章节路径；其余 1 级标题视为子文档（如附带的产品说明书）。
    栈元素：{"level": 层级, "title": 标题, "numbered": 是否有编号}
    """

    def __init__(self):
        self.stack = []
        self.seen_title = False

    def push(self, title: str, mineru_level: int) -> int:
        """压入一个标题，弹出同级及更深的标题；返回推断出的层级"""
        if mineru_level == 1:
            level = 1
            numbered = True
            if not self.seen_title:
                self.seen_title = True
                self.stack = []
                return 1
        else:
            level = numbered_level(title)
            numbered = level is not None
            if level is None:
                level = self._get_unnumbered_level()
        while self.stack and self.stack[-1]["level"] >= level:
            self.stack.pop()
        self.stack.append({"level": level, "title": title, "numbered": numbered})
        return level

    def _get_unnumbered_level(self) -> int:
        """
        无编号标题挂在最近的有编号标题下；没有编号父级时放到最深层，
        让后续任何有编号标题都能把它弹出（如“重要内容提示”不应成为“第一节”的父级）
        """
        parent = None
        for item in reversed(self.stack):
            if item["numbered"] and item["level"] > 1:
                parent = item["level"]
                break
        if parent:
            return min(parent + 1, 8)
        return ORPHAN_LEVEL

    def get_path(self) -> list:
        """当前章节路径（标题列表），每次返回新列表"""
        return [item["title"] for item in self.stack]


# ---------- 规范化（纯函数部分） ----------

_UNIT_LINE = re.compile(r"^[（(]?\s*(单位|金额单位|货币单位)\s*[:：]")
_SENTENCE_END = tuple("。！？!?；;：:”」』）)]")


def normalize(raw_blocks: list) -> list:
    """
    content_list → 版面块列表（不含图片描述，便于单元测试）
    :param raw_blocks: MinerU content_list（或 Markdown 本地解析结果）
    :return: 版面块 dict 列表，seq 为下标
    """
    ctx = {
        "blocks": [],
        "heads": HeadingStack(),
        "last_para": None,  # 可接续的上一段正文（中间只隔页眉页脚/脚注）
        "last_unit": None,  # 最近一条单位说明 {"path": 所在章节, "text": 说明}
    }
    for raw in raw_blocks:
        block_type = raw.get("type", "text")
        page = int(raw.get("page_idx", 0)) + 1
        if block_type in DROP_TYPES:
            continue
        if block_type == "table":
            _add_table(ctx, raw, page)
        elif block_type in IMAGE_TYPES:
            _add_image(ctx, raw, page)
        else:
            # 文本类：text / page_footnote / list / equation / code 等
            _add_text(ctx, raw, block_type, page)
    return ctx["blocks"]


def _append_block(blocks: list, block: dict) -> dict:
    block["seq"] = len(blocks)
    blocks.append(block)
    return block


def _add_table(ctx: dict, raw: dict, page: int):
    """表格：空壳续表只延长前表页码；未合并的续表并入前表；否则新增表格块并带上单位说明"""
    blocks = ctx["blocks"]
    body = raw.get("table_body") or ""
    if blocks:
        prev = blocks[-1]
    else:
        prev = None
    if not body.strip():
        # MinerU 已把续表并入前表，只剩空壳：前表页码延到本页
        if prev is not None and prev["type"] == "table":
            prev["page_end"] = max(get_last_page(prev), page)
        return
    caption = clean_text(" ".join(raw.get("table_caption") or []))
    footnotes = raw.get("table_footnote") or []
    footnote = clean_text(" ".join([x for x in footnotes if x]))
    if _is_table_continuation(prev, page, caption, body):
        prev["table_html"] = merge_html(prev["table_html"] or "", body)
        prev["page_end"] = page
        prev["footnote"] = " ".join([x for x in [prev["footnote"], footnote] if x])
        return
    context = _get_table_context(ctx, prev)
    block = new_block("table", page, ctx["heads"].get_path())
    block["table_html"] = body
    block["caption"] = caption
    block["footnote"] = footnote
    block["context"] = context
    _append_block(blocks, block)
    ctx["last_para"] = None


def _is_table_continuation(prev, page: int, caption: str, body: str) -> bool:
    """未被 MinerU 合并的续表：紧跟在上一页的表格之后、没有标题、列数相同"""
    if prev is None or prev["type"] != "table":
        return False
    if page != get_last_page(prev) + 1 or caption:
        return False
    return parse_table(body)["n_cols"] == parse_table(prev["table_html"] or "")["n_cols"]


def _is_unit_line(block: dict) -> bool:
    """“单位：元 币种：人民币”一类的短说明行"""
    return block["type"] == "text" and bool(_UNIT_LINE.match(block["text"])) and len(block["text"]) <= 40


def _get_table_context(ctx: dict, prev) -> str:
    """表格的单位说明：紧邻上方的单位行（标记为已吸收）；否则沿用同一小节内上一条单位说明"""
    path = ctx["heads"].get_path()
    if prev is not None and _is_unit_line(prev):
        prev["absorbed"] = True
        ctx["last_unit"] = {"path": path, "text": prev["text"]}
        return prev["text"]
    last_unit = ctx["last_unit"]
    if last_unit is not None and last_unit["path"] == path:
        return last_unit["text"]  # 同一小节内连续的表格沿用上一条单位说明
    return ""


def _add_image(ctx: dict, raw: dict, page: int):
    """图片/图表：先带上 MinerU 给出的标题、脚注与文字，描述在 describe_images 中补上"""
    captions = raw.get("image_caption") or raw.get("chart_caption") or []
    footnotes = raw.get("image_footnote") or raw.get("chart_footnote") or []
    block = new_block("image", page, ctx["heads"].get_path())
    block["img_path"] = raw.get("img_path") or None
    block["caption"] = clean_text(" ".join(captions))
    block["footnote"] = clean_text(" ".join(footnotes))
    block["text"] = clean_text(raw.get("content") or "")
    block["derived"] = True
    _append_block(ctx["blocks"], block)
    ctx["last_para"] = None


def _get_raw_text(raw: dict, block_type: str) -> str:
    """文本类块的文字：列表逐项清洗后换行拼接，代码块只去首尾空白"""
    if block_type == "list":
        items = raw.get("list_items") or []
        return "\n".join([clean_text(x) for x in items if x])
    if block_type == "code":
        return (raw.get("code_body") or "").strip()
    return clean_text(raw.get("text") or "")


def _add_text(ctx: dict, raw: dict, block_type: str, page: int):
    """文本类：标题进入章节栈；页脚注加“注：”前缀；跨页断开的段落拼回上一段"""
    text = _get_raw_text(raw, block_type)
    if not text:
        return
    level = raw.get("text_level")
    if block_type == "text" and level and is_heading_text(text):
        _add_heading(ctx, text, int(level), page)
        return
    if block_type == "page_footnote":
        # 页脚注不打断跨页段落的接续（不重置 last_para）
        block = new_block("text", page, ctx["heads"].get_path())
        block["text"] = f"注：{text}"
        _append_block(ctx["blocks"], block)
        return
    # 跨页断开的段落：上一段不以句末标点结尾，且本段在后续页
    last_para = ctx["last_para"]
    if last_para is not None and page > get_last_page(last_para) and not last_para["text"].endswith(_SENTENCE_END):
        last_para["text"] += text
        last_para["page_end"] = page
        return
    block = new_block("text", page, ctx["heads"].get_path())
    block["text"] = text
    ctx["last_para"] = _append_block(ctx["blocks"], block)


def _add_heading(ctx: dict, text: str, mineru_level: int, page: int):
    heads = ctx["heads"]
    level = heads.push(text, mineru_level)
    if level == 1 and not heads.stack:
        return  # 文档标题：由 document_title 表达，不单独成块
    block = new_block("heading", page, heads.get_path())
    block["text"] = text
    block["level"] = level
    _append_block(ctx["blocks"], block)
    ctx["last_para"] = None


# ---------- 图片描述 ----------

def get_mineru_base(doc_dir):
    """图片路径相对于 MinerU 解压目录中 content_list.json 所在的目录；找不到时用文档产物目录"""
    mineru_dir = doc_dir / "mineru"
    if not mineru_dir.exists():
        return doc_dir
    found = sorted(mineru_dir.rglob("*content_list.json"))
    if not found:
        return doc_dir
    return found[0].parent


def _find_images_to_describe(blocks: list, base, cache: dict) -> list:
    """找出需要调用视觉模型的图片块；过小（图标）或读不出尺寸的图片直接在缓存中记为无信息"""
    todo = []
    for b in blocks:
        if b["type"] != "image" or not b["img_path"]:
            continue
        if b["img_path"] in cache:
            continue
        path = base / b["img_path"]
        if path.exists():
            size = image_size(path)
        else:
            size = None
        if size is None or max(size) < MIN_IMAGE_SIDE:
            cache[b["img_path"]] = NO_INFO
            continue
        todo.append(b)
    return todo


def describe_one_image(block: dict, base):
    """调用视觉模型描述一张图片；失败时返回 None（单张图片失败不影响整个文档）"""
    context = block["caption"] or " > ".join(block["section_path"][-2:])
    try:
        return describe_image(base / block["img_path"], context)
    except Exception as e:
        logger.warning(f"图片描述失败 {block['img_path']}：{e}")
        return None


@step_log("describe_images")
def describe_images(blocks: list, base, cache_path):
    """
    把图片描述写入图片块的 text。描述缓存在 image_desc.json，已缓存的图片不再调用模型；
    无信息的图片 text 置空，随后被丢弃。
    :param base: 图片相对路径的基准目录
    :param cache_path: image_desc.json 路径
    """
    if cache_path.exists():
        cache = read_json(cache_path)
    else:
        cache = {}
    todo = _find_images_to_describe(blocks, base, cache)
    if todo:
        logger.info(f"视觉模型描述 {len(todo)} 张图片")
        with ThreadPoolExecutor(VL_WORKERS) as pool:
            futures = [pool.submit(describe_one_image, b, base) for b in todo]
            for block, future in zip(todo, futures):
                desc = future.result()
                if desc is not None:
                    cache[block["img_path"]] = desc
        write_json(cache_path, cache)
    elif not cache_path.exists():
        write_json(cache_path, cache)

    for b in blocks:
        if b["type"] == "image" and b["img_path"]:
            desc = cache.get(b["img_path"])
            if desc is None or NO_INFO in desc:
                b["text"] = ""
            else:
                b["text"] = desc


@step_log("normalize_document")
def normalize_document(doc_id: str) -> list:
    """规范化一个文档：content_list.json → 版面块 → 图片描述 → 丢弃没有描述的图片 → blocks.json"""
    doc_dir = get_doc_dir(doc_id)
    blocks = normalize(read_json(doc_dir / CONTENT_LIST))
    describe_images(blocks, get_mineru_base(doc_dir), doc_dir / IMAGE_DESC)
    blocks = [b for b in blocks if b["type"] != "image" or b["text"]]
    for i, b in enumerate(blocks):
        b["seq"] = i
    write_json(doc_dir / BLOCKS, dump_blocks(blocks))
    return blocks


if __name__ == "__main__":
    # 运行：uv run python -m utils.block_utils <doc_id>
    # 只跑纯函数部分：读 data/artifacts/<doc_id>/content_list.json 规范化后打印前 10 块，
    # 不调用视觉模型、不写文件
    import sys

    test_blocks = normalize(read_json(get_doc_dir(sys.argv[1]) / CONTENT_LIST))
    logger.info(f"版面块 {len(test_blocks)} 个")
    for test_block in test_blocks[:10]:
        print(test_block["seq"], test_block["type"], test_block["section_path"], test_block["text"][:40])
