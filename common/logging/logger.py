"""
项目日志工具
基于标准库 logging，输出到控制台（stderr），格式：时间 | 级别 | 文件:函数:行号 - 消息。
所有模块直接 from common.logging.logger import logger 使用；节点函数加 @node_log，耗时步骤加 @step_log。
"""
import logging
import sys
import time
from functools import wraps

LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(filename)s:%(funcName)s:%(lineno)d - %(message)s"
# 这些第三方库的 INFO 日志很多（每个 HTTP 请求一行），只保留警告以上
# httpx2 / httpcore2 是 pymilvus 自带的 httpx 副本，日志器名字不同，要单独压一次
NOISY_LOGGERS = ["httpx", "httpx2", "httpcore", "httpcore2", "urllib3", "pymongo", "openai"]


def init_logger():
    """
    初始化全局日志配置（只在首次导入时添加 handler，避免重复打印）
    :return: 项目统一使用的 logger
    """
    root = logging.getLogger()
    if not root.handlers:
        # Windows 下输出被重定向（服务日志、管道）时默认用本地代码页，中文会乱码，统一改为 UTF-8
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt="%H:%M:%S"))
        root.addHandler(handler)
    root.setLevel(logging.INFO)
    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
    return logging.getLogger("finkb")


logger = init_logger()


def set_level(level):
    """调整日志级别（命令行 -v 时切到 DEBUG）"""
    logging.getLogger().setLevel(level)


def _trace_id(state):
    """从图状态里取追踪 ID：查询图用 session_id，导入图用文件名"""
    if isinstance(state, dict):
        return str(state.get("session_id") or state.get("file_name") or state.get("task_id") or "-")
    return "-"


def node_log(node_name: str):
    """
    节点日志装饰器：记录节点开始、完成（耗时）与异常，不吞异常
    :param node_name: 节点名，与 add_node 注册的名字一致
    """
    def deco(func):
        @wraps(func)
        def wrapper(state, *args, **kwargs):
            trace_id = _trace_id(state)
            start_ts = time.time()
            logger.debug(f"[{node_name}] 节点开始，追踪ID={trace_id}")
            try:
                result = func(state, *args, **kwargs)
            except Exception:
                logger.error(f"[{node_name}] 节点异常，追踪ID={trace_id}")
                raise
            cost_ms = int((time.time() - start_ts) * 1000)
            logger.info(f"[{node_name}] 节点完成，追踪ID={trace_id}，耗时={cost_ms}ms")
            return result
        return wrapper
    return deco


def step_log(step_name: str):
    """
    步骤日志装饰器：记录步骤完成耗时与异常，不吞异常
    :param step_name: 步骤名
    """
    def deco(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_ts = time.time()
            try:
                result = func(*args, **kwargs)
            except Exception:
                logger.error(f"[{step_name}] 步骤异常")
                raise
            cost_ms = int((time.time() - start_ts) * 1000)
            logger.debug(f"[{step_name}] 步骤完成，耗时={cost_ms}ms")
            return result
        return wrapper
    return deco
