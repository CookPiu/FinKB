"""导入链路的数据模型：文档记录、规范化块、切片。"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field


class Stage(StrEnum):
    REGISTER = "register"
    PARSE = "parse"
    NORMALIZE = "normalize"
    CHUNK = "chunk"
    INDEX = "index"
    ENRICH = "enrich"


# 流水线阶段顺序；documents.stage 记录最后完成的阶段。
# enrich 放在 index 之后：它只写 Mongo（财务事实、摘要），新增或调整它不需要重新向量化。
STAGE_ORDER: list[Stage] = [Stage.REGISTER, Stage.PARSE, Stage.NORMALIZE, Stage.CHUNK, Stage.INDEX, Stage.ENRICH]


def next_stage(done: Stage | str | None) -> Stage | None:
    """已完成 done 阶段后应执行的下一阶段；全部完成返回 None。"""
    if done is None:
        return STAGE_ORDER[0]
    idx = STAGE_ORDER.index(Stage(done))
    return STAGE_ORDER[idx + 1] if idx + 1 < len(STAGE_ORDER) else None


class DocStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    FAILED = "failed"
    READY = "ready"
    SUPERSEDED = "superseded"


class ContentType(StrEnum):
    COMPANY_REPORT = "公司定期报告"
    FUND_KFS = "基金产品资料概要"
    WEALTH_RISK = "理财风险揭示书"
    MACRO_POLICY = "宏观经济与政策"
    REGULATION = "法规制度"
    INVESTOR_FAQ = "投资者教育FAQ"
    SURVEY = "调查报告"
    OTHER = "其他"


class ChunkKind(StrEnum):
    TEXT = "text"
    TABLE = "table"
    SUMMARY = "summary"
    TERM = "term"
    IMAGE_DESC = "image_desc"


BlockType = Literal["heading", "text", "table", "image", "list", "equation", "code"]


class Block(BaseModel):
    """normalize 阶段产物中的一个版面块。页码从 1 开始。"""

    seq: int
    type: BlockType
    text: str = ""
    level: int | None = None
    page: int
    page_end: int | None = None
    section_path: list[str] = Field(default_factory=list)
    table_html: str | None = None
    caption: str = ""
    footnote: str = ""
    context: str = ""
    img_path: str | None = None
    derived: bool = False
    absorbed: bool = False

    @property
    def last_page(self) -> int:
        return self.page_end or self.page


class Chunk(BaseModel):
    """chunk 阶段产物。chunk_id 在 index 阶段按 {doc_id}-{version}-{seq} 生成。"""

    seq: int
    kind: ChunkKind
    text: str
    embed_body: str
    section_path: str = ""
    page_start: int
    page_end: int
    derived: bool = False
    table_html: str | None = None


class DocumentRecord(BaseModel):
    """Mongo documents 集合中的一条记录（_id 即 doc_id）。"""

    doc_id: str
    file_name: str
    file_ext: str
    file_hash: str
    file_size: int
    rel_dir: str = ""
    local_path: str
    source_path: str = ""
    content_type: str = ContentType.OTHER
    content_type_source: str = "dir_rule"
    document_title: str
    # 切片版本：chunk 阶段每次产出新切片集时 +1；index 先写新版本再删旧版本
    version: int = 0
    stage: Stage | None = None
    status: DocStatus = DocStatus.PENDING
    error: str | None = None
    artifacts_dir: str
    page_count: int | None = None
    chunk_count: int | None = None
    supersedes: list[str] = Field(default_factory=list)
    # 以下字段由 M2 的 enrich 阶段填充
    institution_name: str | None = None
    publish_date: int | None = None
    report_period: str | None = None
    entity_ids: list[str] = Field(default_factory=list)
    summary: str | None = None
    created_at: datetime
    updated_at: datetime
