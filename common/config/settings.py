"""集中配置：所有可调项从项目根目录的 .env 读取，字段说明见 .env.example。"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # 百炼（OpenAI 兼容模式）
    openai_api_key: str = ""
    openai_base_url: str = ""
    llm_model: str = "qwen3.8-flash"
    vl_model: str = "qwen3-vl-flash"
    llm_temperature: float = 0.0

    # 精排
    rerank_provider: Literal["cloud", "local"] = "cloud"
    rerank_model: str = "qwen3-rerank"
    rerank_url: str = ""

    # MinerU
    mineru_base_url: str = "https://mineru.net/api/v4"
    mineru_api_token: str = ""
    mineru_model_version: str = "vlm"
    mineru_poll_timeout_s: int = 1800

    # 本地模型
    bge_m3_path: str = "BAAI/bge-m3"
    bge_device: str = "cpu"
    bge_fp16: bool = False
    bge_max_length: int = 2048
    bge_batch_size: int = 8
    bge_reranker_path: str = ""

    # Milvus
    milvus_uri: str = "http://192.168.10.150:19530"
    milvus_collection: str = "fin_chunks"

    # MongoDB
    mongo_uri: str = ""
    mongo_db: str = "finkb"

    # MinIO
    minio_endpoint: str = "192.168.10.150:9000"
    minio_access_key: str = ""
    minio_secret_key: str = ""
    minio_bucket: str = "finkb-files"
    minio_secure: bool = False

    # 本地路径
    artifacts_dir: Path = Field(default=Path("data/artifacts"))

    # 切分参数
    chunk_target_chars: int = 600
    chunk_max_chars: int = 1000
    chunk_min_chars: int = 200
    table_max_chars: int = 3000

    @property
    def artifacts_path(self) -> Path:
        p = self.artifacts_dir
        return p if p.is_absolute() else PROJECT_ROOT / p

    @property
    def rerank_endpoint(self) -> str:
        """精排端点；未显式配置时与生成模型共用工作空间主机。"""
        if self.rerank_url:
            return self.rerank_url
        host = urlparse(self.openai_base_url).netloc
        return f"https://{host}/compatible-api/v1/reranks" if host else ""


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
