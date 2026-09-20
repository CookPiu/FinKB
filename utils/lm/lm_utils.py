"""百炼 OpenAI 兼容模式客户端：文本对话（含流式）与图片描述。"""
import base64
import mimetypes

from openai import OpenAI

from common.config.lm_config import lm_config
from utils.load_prompt import load_prompt

# 千问系列默认可能开启思考链；本项目的调用一律关闭，减少耗时与无关输出
NO_THINKING = {"enable_thinking": False}

# 全局客户端单例
_llm_client = None


def get_llm_client():
    """
    获取 OpenAI 兼容客户端单例
    :raise RuntimeError: 未配置 OPENAI_API_KEY / OPENAI_BASE_URL
    """
    global _llm_client
    if _llm_client is None:
        if not lm_config.api_key or not lm_config.base_url:
            raise RuntimeError("OPENAI_API_KEY / OPENAI_BASE_URL 未配置")
        _llm_client = OpenAI(api_key=lm_config.api_key, base_url=lm_config.base_url, timeout=120, max_retries=2)
    return _llm_client


def chat(messages: list, max_tokens=None, json_mode: bool = False) -> str:
    """
    一次性对话
    :param messages: [{"role": "system" | "user" | "assistant", "content": ...}]
    :param max_tokens: 最多生成的 token 数，None 表示不限制
    :param json_mode: True 时要求模型输出 JSON 对象
    :return: 模型回复文本
    """
    params = {
        "model": lm_config.llm_model,
        "messages": messages,
        "temperature": lm_config.llm_temperature,
        "extra_body": NO_THINKING,
    }
    if max_tokens:
        params["max_tokens"] = max_tokens
    if json_mode:
        params["response_format"] = {"type": "json_object"}
    response = get_llm_client().chat.completions.create(**params)
    return response.choices[0].message.content or ""


def chat_stream(messages: list):
    """
    流式对话：逐段产出模型生成的文本（空片段已跳过）
    :param messages: 同 chat
    """
    stream = get_llm_client().chat.completions.create(
        model=lm_config.llm_model,
        messages=messages,
        temperature=lm_config.llm_temperature,
        stream=True,
        extra_body=NO_THINKING,
    )
    for chunk in stream:
        if not chunk.choices:
            continue
        text = chunk.choices[0].delta.content
        if text:
            yield text


def describe_image(image_path, context: str = "") -> str:
    """
    调用视觉模型描述一张图片
    :param image_path: 本地图片路径（Path）
    :param context: 图片标题或所在章节，帮助模型理解
    :return: 描述文本
    """
    mime = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
    image_b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
    prompt = load_prompt("image_desc")
    if context:
        prompt = prompt + f"\n图片所在位置或标题：{context}"
    content = [
        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image_b64}"}},
        {"type": "text", "text": prompt},
    ]
    response = get_llm_client().chat.completions.create(
        model=lm_config.vl_model,
        messages=[{"role": "user", "content": content}],
        temperature=0,
        extra_body=NO_THINKING,
    )
    return (response.choices[0].message.content or "").strip()
