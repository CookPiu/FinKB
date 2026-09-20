"""内容类型判定。M1 只用目录与文件名规则；M2 由 LLM 在枚举内确认（网页上传无目录时兜底）。"""
import re

# 内容类型（documents.content_type 与 Milvus content_type 字段的取值）
CONTENT_TYPE_COMPANY_REPORT = "公司定期报告"
CONTENT_TYPE_FUND_KFS = "基金产品资料概要"
CONTENT_TYPE_WEALTH_RISK = "理财风险揭示书"
CONTENT_TYPE_MACRO_POLICY = "宏观经济与政策"
CONTENT_TYPE_REGULATION = "法规制度"
CONTENT_TYPE_INVESTOR_FAQ = "投资者教育FAQ"
CONTENT_TYPE_SURVEY = "调查报告"
CONTENT_TYPE_OTHER = "其他"

# 文件名关键词优先于目录规则（同一目录下可能混放不同类型）
FILE_RULES = [
    ("消费者权益保护实施办法", CONTENT_TYPE_REGULATION),
    ("调查报告", CONTENT_TYPE_SURVEY),
    ("风险揭示书", CONTENT_TYPE_WEALTH_RISK),
    ("基金产品资料概要", CONTENT_TYPE_FUND_KFS),
    ("季度报告", CONTENT_TYPE_COMPANY_REPORT),
    ("年度报告", CONTENT_TYPE_COMPANY_REPORT),
]

DIR_RULES = [
    ("上市公司年报", CONTENT_TYPE_COMPANY_REPORT),
    ("基金产品", CONTENT_TYPE_FUND_KFS),
    ("银行理财", CONTENT_TYPE_WEALTH_RISK),
    ("宏观经济", CONTENT_TYPE_MACRO_POLICY),
    ("用户FAQ", CONTENT_TYPE_INVESTOR_FAQ),
]

# 文件名开头的日期，如 "2026-04-25 "
DATE_PREFIX = re.compile(r"^\d{4}-\d{2}-\d{2}\s*")


def guess_content_type(rel_dir: str, file_name: str) -> str:
    """
    按文件名关键词、再按所在目录判定内容类型
    :param rel_dir: 文件相对导入目录的目录（posix 形式，可为空）
    :param file_name: 文件名
    :return: 内容类型字符串，都不匹配时为 "其他"
    """
    for keyword, content_type in FILE_RULES:
        if keyword in file_name:
            return content_type
    for keyword, content_type in DIR_RULES:
        if keyword in rel_dir:
            return content_type
    return CONTENT_TYPE_OTHER


def title_from_filename(stem: str) -> str:
    """文件名去掉日期前缀作为初始资料名称；M2 由抽取结果覆盖。"""
    return DATE_PREFIX.sub("", stem).strip()
