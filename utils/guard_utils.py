"""
合规守卫：禁用表达与投资建议措辞检查（需求说明 §7.1、§7.2）
资料原文和合规回答里常出现"不保证本金和收益""不存在稳赚不赔"，简单子串匹配会大量误报，
因此命中位置前 NEGATION_WINDOW 个字符内有否定词的不算违规。
流式输出按句缓冲（遇到句末标点才检查并放行），违规句整句替换为合规表述。
"""
import re

BANNED = [
    "一定赚钱",
    "保证收益",
    "保本无风险",
    "稳赚不赔",
    "一定会上涨",
    "现在必须买入",
    "保本保息",
    "零风险",
    "稳赚",
    "包赚",
    "肯定涨",
]
NEGATIONS = ["不承诺", "不存在", "并不", "并非", "绝非", "没有", "不能", "不会", "不得", "不是", "不", "非", "无"]
NEGATION_WINDOW = 4
ADVICE = re.compile(r"(建议|推荐)(您|你)?(现在|立即|马上)?(买入|卖出|持有|赎回|加仓|减仓|抄底|清仓|购买|入手)")
REPLACEMENT = "（此处原有不符合合规要求的表述，已删除。金融产品存在风险，不存在保证收益的承诺。）"
SENTENCE_END = "。！？!?\n"


# 句中出现这些词时，禁用表达是在描述骗局或违规行为（如"非法集资通常保本保息"），不算系统在承诺
WARNING_CONTEXT = ("非法集资", "集资诈骗", "诈骗", "骗局", "警惕", "谨防", "不法分子", "违规承诺")


def is_negated(text: str, start: int) -> bool:
    """text[start] 处的命中前 NEGATION_WINDOW 个字符内是否有否定词"""
    prefix = text[max(0, start - NEGATION_WINDOW):start]
    return any(n in prefix for n in NEGATIONS)


def violations(sentence: str, context: str = "") -> list:
    """
    检查一句话中的违规表达
    :param sentence: 待检查的句子
    :param context: 用于判断"描述骗局"语境的额外文本（用户问题、已输出的内容）
    :return: 命中的禁用表达与投资建议措辞（可重复）；无违规为空列表
    """
    hits = []
    warning_context = any(w in sentence or w in context for w in WARNING_CONTEXT)
    for word in BANNED:
        for m in re.finditer(re.escape(word), sentence):
            if not is_negated(sentence, m.start()) and not warning_context:
                hits.append(word)
    for m in ADVICE.finditer(sentence):
        if not is_negated(sentence, m.start()):
            hits.append(m.group(0))
    return hits


def find_sentence_end(text: str) -> int:
    """最早出现的句末标点位置；没有则返回 -1"""
    end = -1
    for char in SENTENCE_END:
        idx = text.find(char)
        if idx >= 0 and (end < 0 or idx < end):
            end = idx
    return end


class StreamGuard:
    """按句缓冲的流式守卫：feed() 返回可以放行的文本，flush() 放行剩余部分"""

    def __init__(self, context: str = ""):
        self.buf = ""
        # 用户问题；已放行的句子会陆续追加进来，用于判断"描述骗局"的语境
        self.context = context
        self.hits = []

    def check_sentence(self, sentence: str) -> str:
        """检查一句：合规则原样返回，违规则整句替换（保留句末换行）"""
        found = violations(sentence, self.context)
        self.context += sentence
        if not found:
            return sentence
        self.hits.extend(found)
        if sentence.endswith("\n"):
            return REPLACEMENT + "\n"
        return REPLACEMENT

    def feed(self, delta: str) -> str:
        """追加一段流式文本，返回已凑成完整句子、检查过的部分"""
        self.buf += delta
        out = []
        idx = find_sentence_end(self.buf)
        while idx >= 0:
            out.append(self.check_sentence(self.buf[:idx + 1]))
            self.buf = self.buf[idx + 1:]
            idx = find_sentence_end(self.buf)
        return "".join(out)

    def flush(self) -> str:
        """流结束时检查并放行缓冲区里剩下的不完整句子"""
        rest = self.buf
        self.buf = ""
        if not rest:
            return ""
        return self.check_sentence(rest)
