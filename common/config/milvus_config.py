# Milvus 向量库配置
import os
from dataclasses import dataclass

from dotenv import load_dotenv

from utils.path_util import PROJECT_ROOT

load_dotenv(PROJECT_ROOT / ".env")


@dataclass
class MilvusConfig:
    milvus_uri: str  # 服务地址
    chunks_collection: str  # 切片集合（文本、表格、摘要、图片描述都在这一个集合）


milvus_config = MilvusConfig(
    milvus_uri=os.getenv("MILVUS_URI", "http://192.168.10.150:19530"),
    chunks_collection=os.getenv("MILVUS_COLLECTION", "fin_chunks"),
)
