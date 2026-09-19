"""节点：知识抽取。从已切分的文档中抽取结构化知识，只写 Mongo（放在入库之后，调整它不需要重新向量化）。

- 财务指标事实：季报“主要会计数据和财务指标”一节的表格，用 table_html_utils 确定性解析（不经过 LLM），
  每个“行 × 列”一条事实，写入 financial_facts；
- 文档摘要：每份文档一次 LLM 调用（正文截断到 SUMMARY_INPUT_CHARS），写入 documents.summary。
"""

from __future__ import annotations

from common.logging.logger import logger, node_log
from common.models.document import Block, Chunk, ChunkKind, ContentType, DocumentRecord, Stage
from processor.import_processor.state import ImportGraphState
from utils import artifact_utils as artifacts
from utils.clients import mongo_utils as mongo
from utils.entity_utils import entity_by_file
from utils.lm import lm_utils as llm
from utils.load_prompt import load_prompt
from utils.table_html_utils import is_spanning_row, parse_table, row_columns, row_values
from utils.task_utils import run_stage

FACT_SECTION_KEYWORDS = ("主要会计数据",)
SUMMARY_INPUT_CHARS = 6000


def extract_facts(doc: DocumentRecord, blocks: list[Block]) -> list[dict]:
    entity = entity_by_file(doc.file_name)
    facts: list[dict] = []
    for b in blocks:
        if b.type != "table" or not any(k in " > ".join(b.section_path) for k in FACT_SECTION_KEYWORDS):
            continue
        table = parse_table(b.table_html or "")
        base = {
            "doc_id": doc.doc_id,
            "entity_id": entity.id if entity else None,
            "file_name": doc.file_name,
            "unit": b.context or table.unit_hint,
            "section": " > ".join(b.section_path),
            "page_start": b.page,
            "page_end": b.last_page,
        }
        if table.kv:
            for i in range(len(table.grid)):
                vals = row_values(table, i)
                for j in range(0, table.n_cols, 2):
                    if vals[j] and vals[j + 1]:
                        facts.append({**base, "item": vals[j], "column": "", "value": vals[j + 1]})
            continue
        for i, cols in row_columns(table).items():
            vals = row_values(table, i)
            if not vals[0] or is_spanning_row(table, i):
                continue
            for j in range(1, table.n_cols):
                if vals[j]:
                    facts.append({**base, "item": vals[0], "column": cols[j], "value": vals[j]})
    return facts


def summarize(doc: DocumentRecord, chunks: list[Chunk]) -> str:
    parts, size = [], 0
    for c in chunks:
        if c.kind != ChunkKind.TEXT:
            continue
        parts.append(c.text)
        size += len(c.text)
        if size >= SUMMARY_INPUT_CHARS:
            break
    text = "\n".join(parts)[:SUMMARY_INPUT_CHARS]
    prompt = load_prompt("doc_summary", title=doc.document_title, content_type=doc.content_type, text=text)
    return llm.chat([{"role": "user", "content": prompt}], max_tokens=1000).strip()


@node_log("node_enrich")
def node_enrich(state: ImportGraphState) -> dict:
    doc: DocumentRecord = state["doc"]

    def enrich(d: DocumentRecord) -> dict:
        ddir = artifacts.doc_dir(d.doc_id)
        if d.content_type == ContentType.COMPANY_REPORT:
            blocks = [Block.model_validate(x) for x in artifacts.read_json(ddir / artifacts.BLOCKS)]
            facts = extract_facts(d, blocks)
            db = mongo.get_db()
            db.financial_facts.delete_many({"doc_id": d.doc_id})
            if facts:
                db.financial_facts.insert_many(facts)
            logger.info("enrich    %s：财务指标事实 %d 条", d.file_name, len(facts))
        chunks = [Chunk.model_validate(x) for x in artifacts.read_json(ddir / artifacts.CHUNKS)]
        return {"summary": summarize(d, chunks)}

    run_stage(doc, Stage.ENRICH, enrich)
    return {"doc": doc}
