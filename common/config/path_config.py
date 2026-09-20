# 本地路径配置
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from utils.path_util import PROJECT_ROOT

load_dotenv(PROJECT_ROOT / ".env")


@dataclass
class PathConfig:
    artifacts_dir: Path  # 导入中间产物目录，每个文档一个子目录


# 相对路径按项目根目录解析；写绝对路径时 PROJECT_ROOT / 绝对路径 仍得到该绝对路径
path_config = PathConfig(
    artifacts_dir=PROJECT_ROOT / os.getenv("ARTIFACTS_DIR", "data/artifacts"),
)
