from utils.path_util import PROJECT_ROOT

PROMPT_DIR = PROJECT_ROOT / "common" / "prompt"


def load_prompt(name: str, **kwargs) -> str:
    """
    加载 common/prompt/<name>.prompt 并渲染占位符
    :param name: 提示词文件名（不带 .prompt 后缀）
    :param kwargs: 占位符的值，键名与文件里的 {占位符} 一致；不传则原样返回
    :return: 渲染后的提示词
    """
    prompt_path = PROMPT_DIR / f"{name}.prompt"
    if not prompt_path.exists():
        raise FileNotFoundError(f"提示词文件不存在：{prompt_path}")
    raw_prompt = prompt_path.read_text(encoding="utf-8").strip()
    if kwargs:
        return raw_prompt.format(**kwargs)
    return raw_prompt
