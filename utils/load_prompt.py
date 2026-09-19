"""加载 common/prompt/<name>.prompt 并渲染占位符。"""

from __future__ import annotations

from functools import lru_cache

from common.config.settings import PROJECT_ROOT

PROMPT_DIR = PROJECT_ROOT / "common" / "prompt"


@lru_cache(maxsize=None)
def _read(name: str) -> str:
    path = PROMPT_DIR / f"{name}.prompt"
    if not path.exists():
        raise FileNotFoundError(f"提示词文件不存在：{path}")
    return path.read_text(encoding="utf-8").strip()


def load_prompt(name: str, **kwargs) -> str:
    raw = _read(name)
    return raw.format(**kwargs) if kwargs else raw
