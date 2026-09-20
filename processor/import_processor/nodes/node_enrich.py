"""
节点：知识抽取。把两类"不是正文、但值得被检索到"的内容也做成切片，追加进 chunks.json。
- 财务指标事实：季报"主要会计数据和财务指标"一节的表格，用 table_html_utils 确定性解析（不经过 LLM），
  同一行的各列合并成一条 "营业收入：本报告期=…；上年同期=…（单位）〔章节〕"，一条一个切片（kind=fact）；
- 文档摘要：每份文档一次 LLM 调用（正文截断到 SUMMARY_INPUT_CHARS），做成一个切片（kind=summary）。
上游 node_chunk 产出 chunks.json；下游 node_index 统一编码入库，所以这两类内容和正文走同一条检索路径。
"""
from common.logging.logger import logger, node_log, step_log
from processor.import_processor.nodes.node_chunk import new_chunk
from processor.import_processor.state import ImportGraphState
from utils.artifact_utils import BLOCKS, CHUNKS, get_doc_dir, read_json, write_json
from utils.block_utils import get_last_page, read_blocks
from utils.lm.lm_utils import chat
from utils.load_prompt import load_prompt
from utils.table_html_utils import get_unit_hint, is_spanning_row, parse_table, row_columns, row_values

COMPANY_REPORT = "公司定期报告"  # 只有这类文档抽取财务指标事实
FACT_SECTION_KEYWORDS = ("主要会计数据",)
SUMMARY_INPUT_CHARS = 6000
SUMMARY_PREFIX = "（文档摘要）"


# ---------- 财务指标事实 ----------

def extract_facts(blocks: list) -> list:
    """
    从"主要会计数据"一节的表格抽取财务指标事实
    :param blocks: 版面块列表
    :return: 事实 dict 列表，键：unit, section, page_start, page_end, item, column, value
    """
    facts = []
    for b in blocks:
        if not _is_fact_table(b):
            continue
        table = parse_table(b["table_html"] or "")
        base = {
            "unit": b["context"] or get_unit_hint(table),
            "section": " > ".join(b["section_path"]),
            "page_start": b["page"],
            "page_end": get_last_page(b),
        }
        if table["kv"]:
            facts.extend(_extract_kv_facts(table, base))
        else:
            facts.extend(_extract_row_facts(table, base))
    return facts


def _is_fact_table(block: dict) -> bool:
    """章节路径含"主要会计数据"的表格"""
    if block["type"] != "table":
        return False
    path = " > ".join(block["section_path"])
    for keyword in FACT_SECTION_KEYWORDS:
        if keyword in path:
            return True
    return False


def _new_fact(base: dict, item: str, column: str, value: str) -> dict:
    """一条事实：行名 + 列名 + 值，带上单位、章节与页码"""
    fact = dict(base)
    fact["item"] = item
    fact["column"] = column
    fact["value"] = value
    return fact


def _extract_kv_facts(table: dict, base: dict) -> list:
    """键值表：每对非空的 (键, 值) 一条事实，列名为空"""
    facts = []
    for i in range(len(table["grid"])):
        vals = row_values(table, i)
        for j in range(0, table["n_cols"], 2):
            if vals[j] and vals[j + 1]:
                facts.append(_new_fact(base, vals[j], "", vals[j + 1]))
    return facts


def _extract_row_facts(table: dict, base: dict) -> list:
    """普通表：首列是指标名，其余每个非空单元格一条事实；首列为空的行与分组标题行跳过"""
    facts = []
    for i, cols in row_columns(table).items():
        vals = row_values(table, i)
        if not vals[0] or is_spanning_row(table, i):
            continue
        for j in range(1, table["n_cols"]):
            if vals[j]:
                facts.append(_new_fact(base, vals[0], cols[j], vals[j]))
    return facts


def group_facts(facts: list) -> list:
    """
    把一行一列的事实按 (章节, 起始页, 行名) 归拢：同一表格同一行的各列算一组
    :return: [(base, rows)]，按首次出现顺序
    """
    groups = {}
    order = []
    for fact in facts:
        key = (fact["section"], fact["page_start"], fact["item"])
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(fact)
    return [(groups[key][0], groups[key]) for key in order]


def build_fact_text(base: dict, rows: list) -> str:
    """一组事实合并成一句：行名：列=值；…（单位）〔章节〕"""
    cells = []
    for row in rows:
        if row["column"]:
            cells.append(f"{row['column']}={row['value']}")
        else:
            cells.append(row["value"])
    text = f"{base['item']}：{'；'.join(cells)}"
    if base["unit"]:
        text += f"（{base['unit']}）"
    text += f"〔{base['section']}〕"
    return text


@step_log("build_fact_chunks")
def build_fact_chunks(blocks: list) -> list:
    """版面块 → 财务指标事实切片，一组一个切片"""
    chunks = []
    for base, rows in group_facts(extract_facts(blocks)):
        text = build_fact_text(base, rows)
        chunks.append(new_chunk("fact", text, text, base["section"], base["page_start"], base["page_end"]))
    return chunks


# ---------- 文档摘要 ----------

def build_summary_input(chunks: list) -> str:
    """按顺序取正文切片拼接，截断到 SUMMARY_INPUT_CHARS 字"""
    parts = []
    size = 0
    for c in chunks:
        if c["kind"] != "text":
            continue
        parts.append(c["text"])
        size += len(c["text"])
        if size >= SUMMARY_INPUT_CHARS:
            break
    return "\n".join(parts)[:SUMMARY_INPUT_CHARS]


@step_log("build_summary_chunk")
def build_summary_chunk(doc: dict, chunks: list) -> dict:
    """调用 LLM 生成文档摘要，做成一个切片；页码留 0，表示不指向具体某一页"""
    text = build_summary_input(chunks)
    prompt = load_prompt("doc_summary", title=doc["document_title"], content_type=doc["content_type"], text=text)
    summary = chat([{"role": "user", "content": prompt}], max_tokens=1000).strip()
    return new_chunk("summary", SUMMARY_PREFIX + summary, summary, "", 0, 0)


# ---------- 节点 ----------

@step_log("validate_and_get_data")
def validate_and_get_data(state: ImportGraphState):
    """
    取出并校验抽取所需的入参
    :return: 文档记录
    :raise ValueError: 状态里没有文档记录，或切片产物不存在
    """
    doc = state.get("doc")
    if not doc:
        logger.error("no doc found in state")
        raise ValueError("no doc found in state")
    chunks_path = get_doc_dir(doc["doc_id"]) / CHUNKS
    if not chunks_path.is_file():
        logger.error(f"chunks.json not found: {chunks_path}")
        raise ValueError(f"chunks.json not found: {chunks_path}")
    return doc


@node_log("node_enrich")
def node_enrich(state: ImportGraphState):
    """
    节点功能：把财务指标事实与文档摘要做成切片，追加进 chunks.json。
    上游 node_chunk；下游 node_index 统一编码入库。
    """
    doc = validate_and_get_data(state)
    enrich_document(doc)
    return state


@step_log("enrich_document")
def enrich_document(doc: dict) -> int:
    """抽取事实与摘要，追加进 chunks.json 并重排序号，返回追加的切片数"""
    doc_dir = get_doc_dir(doc["doc_id"])
    chunks = read_json(doc_dir / CHUNKS)
    # 重跑时先去掉上一次追加的，避免累积
    chunks = [c for c in chunks if c["kind"] not in ("fact", "summary")]
    extra = []
    if doc["content_type"] == COMPANY_REPORT:
        extra.extend(build_fact_chunks(read_blocks(doc_dir / BLOCKS)))
    extra.append(build_summary_chunk(doc, chunks))
    chunks.extend(extra)
    for i, chunk in enumerate(chunks):
        chunk["seq"] = i
    write_json(doc_dir / CHUNKS, chunks)
    logger.info(f"enrich    {doc['file_name']}：追加 {len(extra)} 个切片（事实 {len(extra) - 1}、摘要 1）")
    return len(extra)


if __name__ == "__main__":
    # 运行：uv run python -m processor.import_processor.nodes.node_enrich <doc_id>
    # 只演示财务事实：读 data/artifacts/<doc_id>/blocks.json 抽取后打印前 5 个切片。
    # 不调用 LLM、不写文件
    import sys

    test_chunks = build_fact_chunks(read_blocks(get_doc_dir(sys.argv[1]) / BLOCKS))
    logger.info(f"财务指标事实切片 {len(test_chunks)} 个")
    for test_chunk in test_chunks[:5]:
        print(test_chunk["page_start"], test_chunk["text"][:90])
