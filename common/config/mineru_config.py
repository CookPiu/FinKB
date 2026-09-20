# MinerU 精准解析 API 配置
import os
from dataclasses import dataclass

from dotenv import load_dotenv

from utils.path_util import PROJECT_ROOT

load_dotenv(PROJECT_ROOT / ".env")


@dataclass
class MinerUConfig:
    base_url: str
    api_token: str
    model_version: str  # vlm / pipeline
    poll_timeout_s: int  # 轮询解析结果的最长等待秒数


mineru_config = MinerUConfig(
    base_url=os.getenv("MINERU_BASE_URL", "https://mineru.net/api/v4"),
    api_token=os.getenv("MINERU_API_TOKEN", ""),
    model_version=os.getenv("MINERU_MODEL_VERSION", "vlm"),
    poll_timeout_s=int(os.getenv("MINERU_POLL_TIMEOUT_S", "1800")),
)
