"""项目路径：以本文件位置推导项目根目录，不受启动时工作目录影响。"""
from pathlib import Path

# utils/path_util.py 的上两级就是项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
