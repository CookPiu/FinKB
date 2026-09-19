"""证据对象：全程携带来源元数据，引用由代码据此渲染（模型只写编号 [E1]）。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class Evidence:
    eid: int
    kind: str  # fact / summary / chunk
    text: str
    doc_id: str
    file_name: str
    document_title: str
    content_type: str
    page_start: int | None = None
    page_end: int | None = None
    source_path: str = ""
    entity_name: str = ""
    entity_codes: list[str] = field(default_factory=list)
    score_dense: float | None = None
    derived: bool = False

    def prompt_block(self, max_chars: int = 1500) -> str:
        pages = f"｜第{self.page_start}-{self.page_end}页" if self.page_start else ""
        derived = "｜据图表示意" if self.derived else ""
        head = f"[E{self.eid}] 〔{self.document_title}｜{self.content_type}{pages}{derived}〕"
        return f"{head}\n{self.text[:max_chars]}"

    def source(self) -> dict:
        d = asdict(self)
        d.pop("text")
        return d

    @classmethod
    def from_dict(cls, d: dict) -> Evidence:
        return cls(**d)
