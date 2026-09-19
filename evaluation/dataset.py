"""评测集读取与自检。

每行一题（JSONL）。单轮题的 expect/gold/must_include 写在顶层；多轮题（category=multiturn）写在每一轮 turns[i] 里。
金标按“文件名 + 关键原句”，不用 chunk_id（切片 ID 随切分参数变化，金标不能随之失效）。
自检：每条 quote 必须能在对应文件的解析文本中找到（两边都做 NFKC 规范化并去掉空白后比较）。
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from utils.clients import mongo_utils as mongo
from utils import artifact_utils as artifacts

Category = Literal["product", "announcement", "risk", "knowledge", "process", "negative", "compliance", "multiturn"]
Expect = Literal["answer", "refuse", "clarify", "decline_advice", "realtime_notice"]

_WS = re.compile(r"\s+")


def norm(text: str) -> str:
    return _WS.sub("", unicodedata.normalize("NFKC", text or ""))


class Gold(BaseModel):
    file: str
    quote: str


class Turn(BaseModel):
    q: str
    expect: Expect | None = None
    gold: list[Gold] = Field(default_factory=list)
    must_include: list[str] = Field(default_factory=list)
    must_not_include: list[str] = Field(default_factory=list)


class EvalItem(BaseModel):
    id: str
    category: Category
    turns: list[Turn]
    expect: Expect | None = None
    gold: list[Gold] = Field(default_factory=list)
    must_include: list[str] = Field(default_factory=list)
    must_not_include: list[str] = Field(default_factory=list)
    note: str = ""

    def resolved_turns(self) -> list[Turn]:
        """把顶层字段并入单轮题的第一轮，调用方只需处理 turns。"""
        if self.category == "multiturn":
            return self.turns
        t = self.turns[0]
        return [
            Turn(
                q=t.q,
                expect=t.expect or self.expect,
                gold=t.gold or self.gold,
                must_include=t.must_include or self.must_include,
                must_not_include=t.must_not_include or self.must_not_include,
            )
        ]


def load(path: Path) -> list[EvalItem]:
    items = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            items.append(EvalItem.model_validate_json(line))
        except ValueError as e:
            raise ValueError(f"{path.name} 第 {n} 行格式错误：{e}") from e
    return items


def load_corpus() -> dict[str, str]:
    """文件名 → 规范化后的全文（该文件全部切片正文拼接），与检索时的切片文本同源。"""
    corpus: dict[str, str] = {}
    for d in mongo.get_db().documents.find({"status": {"$ne": "superseded"}}, {"file_name": 1}):
        p = artifacts.doc_dir(d["_id"]) / artifacts.CHUNKS
        if p.exists():
            corpus[d["file_name"]] = "\n".join(norm(c["text"]) for c in json.loads(p.read_text(encoding="utf-8")))
    return corpus


def self_check(items: list[EvalItem], corpus: dict[str, str]) -> list[str]:
    errors: list[str] = []
    seen: set[str] = set()
    for it in items:
        if it.id in seen:
            errors.append(f"{it.id}: id 重复")
        seen.add(it.id)
        if it.category != "multiturn" and len(it.turns) != 1:
            errors.append(f"{it.id}: 单轮题只能有一轮")
        if it.category == "multiturn" and len(it.turns) < 2:
            errors.append(f"{it.id}: 多轮题至少两轮")
        for ti, turn in enumerate(it.resolved_turns()):
            tag = f"{it.id}#{ti + 1}"
            if turn.expect is None:
                errors.append(f"{tag}: 缺少 expect")
            if turn.expect == "answer" and not turn.gold and it.category not in ("compliance",):
                errors.append(f"{tag}: expect=answer 但没有 gold")
            for g in turn.gold:
                q = norm(g.quote)
                if not 4 <= len(q) <= 60:
                    errors.append(f"{tag}: quote 长度 {len(q)} 超出 4~60：{g.quote}")
                if g.file not in corpus:
                    errors.append(f"{tag}: 文件不在知识库中：{g.file}")
                elif q not in corpus[g.file]:
                    errors.append(f"{tag}: quote 在 {g.file} 中找不到：{g.quote}")
    return errors
