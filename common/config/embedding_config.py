# 向量模型配置：本地 BGE-M3（本机无 CUDA，只跑 CPU）
import os
from dataclasses import dataclass

from dotenv import load_dotenv

from utils.path_util import PROJECT_ROOT

load_dotenv(PROJECT_ROOT / ".env")


@dataclass
class EmbeddingConfig:
    bge_m3_path: str  # 本地模型目录；填 BAAI/bge-m3 时自动下载
    bge_device: str  # cpu / cuda:0
    bge_fp16: bool  # 半精度只在 GPU 上有意义
    bge_max_length: int  # 切片编码的最大 token 数
    bge_batch_size: int  # 每批编码的切片数


embedding_config = EmbeddingConfig(
    bge_m3_path=os.getenv("BGE_M3_PATH", "BAAI/bge-m3"),
    bge_device=os.getenv("BGE_DEVICE", "cpu"),
    # .env 里写 1/0 或 true/false
    bge_fp16=os.getenv("BGE_FP16", "0").lower() in ("1", "true"),
    bge_max_length=int(os.getenv("BGE_MAX_LENGTH", "2048")),
    bge_batch_size=int(os.getenv("BGE_BATCH_SIZE", "8")),
)
