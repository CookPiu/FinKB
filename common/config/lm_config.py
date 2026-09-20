# 大模型配置：百炼 OpenAI 兼容模式（文本对话 + 图片描述）
import os
from dataclasses import dataclass

from dotenv import load_dotenv

from utils.path_util import PROJECT_ROOT

# 读取项目根目录下的 .env；系统环境变量里已有的同名配置优先
load_dotenv(PROJECT_ROOT / ".env")


@dataclass
class LLMConfig:
    api_key: str  # 百炼 API Key
    base_url: str  # 兼容模式地址，形如 https://<WorkspaceId>.cn-beijing.maas.aliyuncs.com/compatible-mode/v1
    llm_model: str  # 文本模型（查询规划、回答生成、摘要）
    vl_model: str  # 视觉模型（图片描述）
    llm_temperature: float


lm_config = LLMConfig(
    api_key=os.getenv("OPENAI_API_KEY", ""),
    base_url=os.getenv("OPENAI_BASE_URL", ""),
    llm_model=os.getenv("LLM_MODEL", "qwen3.8-flash"),
    vl_model=os.getenv("VL_MODEL", "qwen3-vl-flash"),
    llm_temperature=float(os.getenv("LLM_TEMPERATURE", "0")),
)
