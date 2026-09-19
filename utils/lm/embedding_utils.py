"""BGE-M3 稠密 + 稀疏编码。模型加载需数十秒（CPU），模块级单例，服务启动时调用 warmup() 预热。"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from common.logging.logger import logger
from common.config.settings import get_settings

_model = None
_lock = threading.Lock()


@dataclass
class Encoded:
    dense: list[float]
    sparse: dict[int, float]


def get_model():
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                from FlagEmbedding import BGEM3FlagModel  # 延迟导入：torch 加载较慢

                s = get_settings()
                t0 = time.perf_counter()
                _model = BGEM3FlagModel(
                    s.bge_m3_path,
                    normalize_embeddings=True,
                    use_fp16=s.bge_fp16,
                    devices=s.bge_device,
                )
                logger.info("BGE-M3 加载完成，用时 %.1fs（device=%s）", time.perf_counter() - t0, s.bge_device)
    return _model


def encode(texts: list[str], *, max_length: int | None = None, batch_size: int | None = None) -> list[Encoded]:
    """编码一批文本；稀疏向量的键转为 int 以便写入 Milvus。"""
    if not texts:
        return []
    s = get_settings()
    out = get_model().encode(
        texts,
        batch_size=batch_size or s.bge_batch_size,
        max_length=max_length or s.bge_max_length,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
    )
    result: list[Encoded] = []
    for dense, lex in zip(out["dense_vecs"], out["lexical_weights"]):
        sparse = {int(k): float(v) for k, v in lex.items() if float(v) > 0}
        result.append(Encoded(dense=[float(x) for x in dense], sparse=sparse))
    return result


def encode_query(text: str) -> Encoded:
    return encode([text], max_length=512, batch_size=1)[0]


def warmup() -> None:
    encode_query("预热")
