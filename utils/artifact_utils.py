"""导入中间产物的落盘约定：data/artifacts/<doc_id>/ 下每个阶段一个文件。"""
import json
import os

from common.config.path_config import path_config

CONTENT_LIST = "content_list.json"  # 解析结果（MinerU 或本地 Markdown 解析）
BLOCKS = "blocks.json"  # 规范化后的版面块
CHUNKS = "chunks.json"  # 切片
IMAGE_DESC = "image_desc.json"  # 图片描述缓存 {图片相对路径: 描述}


def get_doc_dir(doc_id: str):
    """文档的产物目录（Path）"""
    return path_config.artifacts_dir / doc_id


def write_json(path, data):
    """先写临时文件再替换，避免写到一半中断留下损坏的文件"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp_path, path)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))
