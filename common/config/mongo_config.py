# MongoDB 配置：库 finkb 开启了认证，连接串必须带账号
import os
from dataclasses import dataclass

from dotenv import load_dotenv

from utils.path_util import PROJECT_ROOT

load_dotenv(PROJECT_ROOT / ".env")


@dataclass
class MongoConfig:
    mongo_uri: str
    mongo_db: str


mongo_config = MongoConfig(
    mongo_uri=os.getenv("MONGO_URI", ""),
    mongo_db=os.getenv("MONGO_DB", "finkb"),
)
