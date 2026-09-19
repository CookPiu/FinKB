"""百炼 OpenAI 兼容模式客户端：文本对话与图片描述。"""

from __future__ import annotations

import base64
import mimetypes
import threading
from pathlib import Path

from openai import OpenAI

from common.config.settings import get_settings
from utils.load_prompt import load_prompt

_client: OpenAI | None = None
_lock = threading.Lock()

# 千问系列默认可能开启思考链；抽取、描述类调用一律关闭
NO_THINKING = {"enable_thinking": False}


def get_client() -> OpenAI:
    global _client
    if _client is None:
        with _lock:
            if _client is None:
                s = get_settings()
                if not (s.openai_api_key and s.openai_base_url):
                    raise RuntimeError("OPENAI_API_KEY / OPENAI_BASE_URL 未配置")
                _client = OpenAI(api_key=s.openai_api_key, base_url=s.openai_base_url, timeout=120, max_retries=2)
    return _client


def chat(messages: list[dict], *, model: str | None = None, temperature: float | None = None, **kwargs) -> str:
    s = get_settings()
    resp = get_client().chat.completions.create(
        model=model or s.llm_model,
        messages=messages,
        temperature=s.llm_temperature if temperature is None else temperature,
        extra_body=NO_THINKING,
        **kwargs,
    )
    return resp.choices[0].message.content or ""


def describe_image(image_path: Path, context: str = "") -> str:
    """调用视觉模型描述图片；context 为图片标题或所在章节，帮助模型理解。"""
    s = get_settings()
    mime = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
    b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
    prompt = load_prompt("image_desc") + (f"\n图片所在位置或标题：{context}" if context else "")
    resp = get_client().chat.completions.create(
        model=s.vl_model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        temperature=0,
        extra_body=NO_THINKING,
    )
    return (resp.choices[0].message.content or "").strip()
