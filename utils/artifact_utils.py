"""导入产物的落盘约定：data/artifacts/<doc_id>/ 下每个阶段一个文件，写入先写临时文件再原子替换。"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from common.config.settings import get_settings

CONTENT_LIST = "content_list.json"
BLOCKS = "blocks.json"
CHUNKS = "chunks.json"
IMAGE_DESC = "image_desc.json"


def doc_dir(doc_id: str) -> Path:
    return get_settings().artifacts_path / doc_id


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))
