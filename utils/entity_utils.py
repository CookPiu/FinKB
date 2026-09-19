"""实体表与实体解析。

实体只有十几个，手写在 data/entities.json（全称、别名、代码、类型、对应文件名），解析在内存中完成：
代码精确匹配 → 别名包含匹配（取最长别名；最长别名并列于多个实体时判为有歧义）。
文件名 → doc_id 的映射在运行时从 Mongo documents 读取，因此重新导入后配置不用改。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from common.config.settings import PROJECT_ROOT

ENTITIES_FILE = PROJECT_ROOT / "data" / "entities.json"

EntityType = Literal["company", "fund", "wealth_product", "document"]


class Entity(BaseModel):
    id: str
    name: str
    type: EntityType
    codes: list[str] = Field(default_factory=list)
    aliases: list[str] = Field(default_factory=list)
    files: list[str] = Field(default_factory=list)

    @property
    def labels(self) -> list[str]:
        return [self.name, *self.aliases]


@dataclass
class Resolution:
    """confirmed：已确认；ambiguous：有歧义需澄清；unknown：提到了库外实体；none：没有提到实体。"""

    status: Literal["confirmed", "ambiguous", "unknown", "none"]
    entities: list[Entity] = field(default_factory=list)
    candidates: list[Entity] = field(default_factory=list)
    unknown_mentions: list[str] = field(default_factory=list)


# 泛指词不是实体（规划器偶尔会把它们当作提及）
GENERIC_MENTIONS = {"基金", "理财", "理财产品", "产品", "银行", "公司", "报告", "公告", "股票", "ETF", "私募基金", "债券基金"}


class EntityRegistry:
    def __init__(self, entities: list[Entity]):
        self.entities = entities
        self.by_id = {e.id: e for e in entities}

    @classmethod
    def load(cls, path: Path = ENTITIES_FILE) -> EntityRegistry:
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls([Entity.model_validate(x) for x in data])

    def match_mention(self, mention: str) -> list[Entity]:
        """返回与一个提及最匹配的实体（可能多个，表示有歧义）；无匹配返回空列表。"""
        text = mention.strip().upper()
        if not text or text in GENERIC_MENTIONS:
            return []
        by_code = [e for e in self.entities if any(c.upper() in text for c in e.codes)]
        if by_code:
            return by_code
        best_len, best = 0, []
        for e in self.entities:
            for label in e.labels:
                lab = label.upper()
                # 别名出现在提及里，或提及是别名/全称的一部分（至少 2 个字）
                if lab in text or (len(text) >= 2 and text in lab):
                    n = len(lab) if lab in text else len(text)
                    if n > best_len:
                        best_len, best = n, [e]
                    elif n == best_len and e not in best:
                        best.append(e)
        return best

    def resolve(self, mentions: list[str]) -> Resolution:
        confirmed: list[Entity] = []
        ambiguous: list[Entity] = []
        unknown: list[str] = []
        for m in mentions:
            if m.strip().upper() in GENERIC_MENTIONS:
                continue
            found = self.match_mention(m)
            if len(found) == 1:
                if found[0] not in confirmed:
                    confirmed.append(found[0])
            elif len(found) > 1:
                ambiguous.extend(e for e in found if e not in ambiguous)
            else:
                unknown.append(m)
        if confirmed:
            return Resolution("confirmed", entities=confirmed, unknown_mentions=unknown)
        if ambiguous:
            return Resolution("ambiguous", candidates=ambiguous, unknown_mentions=unknown)
        if unknown:
            return Resolution("unknown", unknown_mentions=unknown)
        return Resolution("none")

    def pick_option(self, reply: str, options: list[Entity]) -> Entity | None:
        """解析用户对澄清问题的回复：序号（1/第一个）或选项名称中的关键词（如“债券那只”）。"""
        text = reply.strip()
        ordinals = {"1": 0, "一": 0, "2": 1, "二": 1, "3": 2, "三": 2, "4": 3, "四": 3}
        m = re.fullmatch(r"第?\s*([1-4一二三四])\s*(个|只|款|项)?[。.！!]?", text)
        if m and ordinals[m.group(1)] < len(options):
            return options[ordinals[m.group(1)]]
        core = re.sub(r"[，。,.！!？?\s]|那只|那个|那款|这只|这个|的|吧|呢|是|就", "", text)
        if len(core) < 2:
            return None
        hits = [e for e in options if any(core in label or label in core for label in e.labels)]
        if len(hits) == 1:
            return hits[0]
        # 取回复中至少 2 个连续字出现在选项名称里的唯一选项
        hits = [e for e in options if any(core[i : i + 2] in "".join(e.labels) for i in range(len(core) - 1))]
        return hits[0] if len(hits) == 1 else None


@lru_cache(maxsize=1)
def registry() -> EntityRegistry:
    return EntityRegistry.load()


def entity_by_file(file_name: str) -> Entity | None:
    return next((e for e in registry().entities if file_name in e.files), None)
