"""
节点：知识抽取。从已切分的文档中抽取结构化知识，只写 Mongo（放在入库之后，调整它不需要重新向量化）。
- 财务指标事实：季报“主要会计数据和财务指标”一节的表格，用 table_html_utils 确定性解析（不经过 LLM），
  每个“行 × 列”一条事实，写入 financial_facts；
- 文档摘要：每份文档一次 LLM 调用（正文截断到 SUMMARY_INPUT_CHARS），写入 documents.summary。
"""
from common.logging.logger import logger, node_log, step_log
from processor.import_processor.nodes.node_normalize import get_last_page, read_blocks
from processor.import_processor.state import ImportGraphState
from utils.artifact_utils import BLOCKS, CHUNKS, get_doc_dir, read_json
from utils.clients.mongo_utils import get_db
from utils.entity_utils import get_entity_by_file
from utils.lm.lm_utils import chat
from utils.load_prompt import load_prompt
from utils.table_html_utils import get_unit_hint, is_spanning_row, parse_table, row_columns, row_values
from utils.task_utils import mark_ready

COMPANY_REPORT = "公司定期报告"  # 只有这类文档抽取财务指标事实
FACT_SECTION_KEYWORDS = ("主要会计数据",)
SUMMARY_INPUT_CHARS = 6000


# ---------- 财务指标事实 ----------

def extract_facts(doc: dict, blocks: list) -> list:
    """
    从“主要会计数据”一节的表格抽取财务指标事实
    :param doc: 文档记录（用到 doc_id、file_name）
    :param blocks: 版面块列表
    :return: 事实 dict 列表，键：doc_id, entity_id, file_name, unit, section, page_start, page_end, item, column, value
    """
    entity = get_entity_by_file(doc["file_name"])
    if entity:
        entity_id = entity["id"]
    else:
        entity_id = None
    facts = []
    for b in blocks:
        if not _is_fact_table(b):
            continue
        table = parse_table(b["table_html"] or "")
        base = {
            "doc_id": doc["doc_id"],
            "entity_id": entity_id,
            "file_name": doc["file_name"],
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
    """章节路径含“主要会计数据”的表格"""
    if block["type"] != "table":
        return False
    path = " > ".join(block["section_path"])
    return any(k in path for k in FACT_SECTION_KEYWORDS)


def _new_fact(base: dict, item: str, column: str, value: str) -> dict:
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


@step_log("save_facts")
def save_facts(doc: dict, facts: list):
    """覆盖写入文档的财务指标事实（先删后插，重跑不产生重复）"""
    db = get_db()
    db.financial_facts.delete_many({"doc_id": doc["doc_id"]})
    if facts:
        db.financial_facts.insert_many(facts)
    logger.info(f"enrich    {doc['file_name']}：财务指标事实 {len(facts)} 条")


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


@step_log("summarize")
def summarize(doc: dict, chunks: list) -> str:
    """调用 LLM 生成文档摘要"""
    text = build_summary_input(chunks)
    prompt = load_prompt("doc_summary", title=doc["document_title"], content_type=doc["content_type"], text=text)
    return chat([{"role": "user", "content": prompt}], max_tokens=1000).strip()


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
    节点功能：公司定期报告抽取财务指标事实写入 financial_facts；每份文档生成摘要写入 documents.summary。
    上游 node_import_milvus；导入图的最后一个节点，跑完文档状态变为 ready。
    """
    doc = validate_and_get_data(state)
    summary = enrich_document(doc)
    mark_ready(doc, {"summary": summary})
    return state


@step_log("enrich_document")
def enrich_document(doc: dict) -> str:
    """抽取并保存财务指标事实（仅公司定期报告），返回文档摘要"""
    doc_dir = get_doc_dir(doc["doc_id"])
    if doc["content_type"] == COMPANY_REPORT:
        blocks = read_blocks(doc_dir / BLOCKS)
        save_facts(doc, extract_facts(doc, blocks))
    chunks = read_json(doc_dir / CHUNKS)
    return summarize(doc, chunks)


if __name__ == "__main__":
    # 运行：uv run python -m processor.import_processor.nodes.node_enrich <doc_id> <文件名>
    # 只演示财务事实抽取：读 data/artifacts/<doc_id>/blocks.json 抽取后打印前 10 条（依赖 data/entities.json）。
    # 不调用 LLM、不写 Mongo（节点本身会写 financial_facts 与文档摘要）
    import sys

    test_doc = {"doc_id": sys.argv[1], "file_name": sys.argv[2]}
    test_facts = extract_facts(test_doc, read_blocks(get_doc_dir(test_doc["doc_id"]) / BLOCKS))
    logger.info(f"财务指标事实 {len(test_facts)} 条")
    for test_fact in test_facts[:10]:
        print(test_fact["item"], "|", test_fact["column"], "|", test_fact["value"], "|", test_fact["unit"])
