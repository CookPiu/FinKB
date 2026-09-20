# 精排配置：百炼 qwen3-rerank（云端）
import os
from dataclasses import dataclass

from dotenv import load_dotenv

from utils.path_util import PROJECT_ROOT

load_dotenv(PROJECT_ROOT / ".env")


@dataclass
class RerankerConfig:
    rerank_model: str
    rerank_url: str  # 留空时由 OPENAI_BASE_URL 的主机推导，见 utils/lm/reranker_utils.get_rerank_endpoint


reranker_config = RerankerConfig(
    rerank_model=os.getenv("RERANK_MODEL", "qwen3-rerank"),
    rerank_url=os.getenv("RERANK_URL", ""),
)
