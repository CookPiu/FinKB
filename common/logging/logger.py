"""项目日志：所有模块直接导入 logger 使用；节点函数用 node_log、步骤函数用 step_log 装饰。

用标准库 logging（不引入 loguru）；日志行带调用方的模块名与函数名。
"""

from __future__ import annotations

import logging
import sys
import time
from collections.abc import Mapping
from functools import wraps

LOG_FORMAT = "%(asctime)s %(levelname)-5s %(module)s:%(funcName)s: %(message)s"
NOISY_LOGGERS = ("httpx", "httpx2", "httpcore", "urllib3", "pymongo", "openai")


class _NodeFilter(logging.Filter):
    """node_log / step_log 发出的日志，位置显示为节点（步骤）名，而不是装饰器内部的 wrapper。"""

    def filter(self, record: logging.LogRecord) -> bool:
        node = getattr(record, "node", None)
        if node:
            record.module = record.funcName = node
        return True


def init_logger(level: int = logging.INFO) -> logging.Logger:
    root = logging.getLogger()
    if not root.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt="%H:%M:%S"))
        handler.addFilter(_NodeFilter())
        root.addHandler(handler)
    root.setLevel(level)
    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
    return logging.getLogger("finkb")


logger = init_logger()


def set_level(level: int) -> None:
    logging.getLogger().setLevel(level)


def _trace_id(state) -> str:
    if isinstance(state, Mapping):
        return str(state.get("session_id") or state.get("file_name") or state.get("task_id") or "-")
    return "-"


def node_log(node_name: str):
    """节点日志：记录开始、完成（耗时）与异常，不吞异常。"""

    def deco(func):
        @wraps(func)
        def wrapper(state, *args, **kwargs):
            trace_id = _trace_id(state)
            extra = {"node": node_name}
            start = time.perf_counter()
            logger.debug("开始 %s", trace_id, extra=extra)
            try:
                result = func(state, *args, **kwargs)
            except Exception:
                logger.error("异常 %s", trace_id, extra=extra)
                raise
            logger.info("完成 %s，耗时 %dms", trace_id, (time.perf_counter() - start) * 1000, extra=extra)
            return result

        return wrapper

    return deco


def step_log(step_name: str):
    """步骤日志：记录开始、完成（耗时）与异常，不吞异常。"""

    def deco(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            extra = {"node": step_name}
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
            except Exception:
                logger.error("异常", extra=extra)
                raise
            logger.debug("完成，耗时 %dms", (time.perf_counter() - start) * 1000, extra=extra)
            return result

        return wrapper

    return deco
