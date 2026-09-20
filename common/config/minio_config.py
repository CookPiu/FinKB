# MinIO 配置：存放原始文件，前端按链接打开原件
import os
from dataclasses import dataclass

from dotenv import load_dotenv

from utils.path_util import PROJECT_ROOT

load_dotenv(PROJECT_ROOT / ".env")


@dataclass
class MinioConfig:
    endpoint: str  # 主机:端口，不带协议
    access_key: str
    secret_key: str
    bucket: str
    secure: bool  # 是否 https


minio_config = MinioConfig(
    endpoint=os.getenv("MINIO_ENDPOINT", "192.168.10.150:9000"),
    access_key=os.getenv("MINIO_ACCESS_KEY", ""),
    secret_key=os.getenv("MINIO_SECRET_KEY", ""),
    bucket=os.getenv("MINIO_BUCKET", "finkb-files"),
    secure=os.getenv("MINIO_SECURE", "0").lower() in ("1", "true"),
)
