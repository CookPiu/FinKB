"""
实体表与实体解析
实体只有十几个，手写在 data/entities.json（全称、别名、代码、类型、对应文件名），解析在内存中完成：
代码精确匹配 → 别名包含匹配（取最长别名；最长别名并列于多个实体时判为有歧义）。
文件名 → doc_id 的映射在运行时从 Mongo documents 读取，因此重新导入后配置不用改。
实体为 dict：{"id", "name", "type"(company / fund / wealth_product / document), "codes", "aliases", "files"}。
"""
import json
import re

from utils.path_util import PROJECT_ROOT

ENTITIES_FILE = PROJECT_ROOT / "data" / "entities.json"

# 泛指词不是实体（规划器偶尔会把它们当作提及）
GENERIC_MENTIONS = {"基金", "理财", "理财产品", "产品", "银行", "公司", "报告", "公告", "股票", "ETF", "私募基金", "债券基金"}

# 澄清回复里的序号 → 选项下标
ORDINALS = {"1": 0, "一": 0, "2": 1, "二": 1, "3": 2, "三": 2, "4": 3, "四": 3}
ORDINAL_REPLY = re.compile(r"第?\s*([1-4一二三四])\s*(个|只|款|项)?[。.！!]?")
# 澄清回复里与选项名称无关的标点和口语词，去掉后剩下的才是关键词
REPLY_NOISE = re.compile(r"[，。,.！!？?\s]|那只|那个|那款|这只|这个|的|吧|呢|是|就")

# 实体表缓存：首次使用时读取文件
_entities = None


def load_entities(path=ENTITIES_FILE) -> list:
    """
    读取实体表，缺省的代码、别名、文件补为空列表
    :param path: 实体表文件
    :return: 实体 dict 列表
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    entities = []
    for item in data:
        entities.append({
            "id": item["id"],
            "name": item["name"],
            "type": item["type"],
            "codes": item.get("codes", []),
            "aliases": item.get("aliases", []),
            "files": item.get("files", []),
        })
    return entities


def get_entities() -> list:
    """实体表（进程内只读一次文件）"""
    global _entities
    if _entities is None:
        _entities = load_entities()
    return _entities


def get_entity_map() -> dict:
    """实体 ID → 实体"""
    entity_map = {}
    for entity in get_entities():
        entity_map[entity["id"]] = entity
    return entity_map


def get_entity_by_file(file_name: str):
    """
    按文件名找实体（导入时给文档打实体标签）
    :param file_name: 资料文件名
    :return: 实体 dict；文件不属于任何实体时返回 None
    """
    for entity in get_entities():
        if file_name in entity["files"]:
            return entity
    return None


def get_labels(entity: dict) -> list:
    """实体的全部名称：全称在前，别名在后"""
    return [entity["name"]] + entity["aliases"]


def match_by_code(text: str, entities: list) -> list:
    """提及里包含实体代码（如 600519）即匹配；text 已转大写"""
    found = []
    for entity in entities:
        if any(code.upper() in text for code in entity["codes"]):
            found.append(entity)
    return found


def match_by_label(text: str, entities: list) -> list:
    """
    按名称匹配：别名出现在提及里，或提及是别名/全称的一部分（至少 2 个字）
    取匹配长度最长的实体，长度并列时全部返回（表示有歧义）
    :param text: 已转大写的提及
    """
    best_len = 0
    best = []
    for entity in entities:
        for label in get_labels(entity):
            label = label.upper()
            if label in text:
                length = len(label)
            elif len(text) >= 2 and text in label:
                length = len(text)
            else:
                continue
            if length > best_len:
                best_len = length
                best = [entity]
            elif length == best_len and entity not in best:
                best.append(entity)
    return best


def match_mention(mention: str, entities: list) -> list:
    """
    找出与一个提及最匹配的实体：代码优先，其次名称
    :return: 实体列表，多个表示有歧义；无匹配或泛指词返回空列表
    """
    text = mention.strip().upper()
    if not text or text in GENERIC_MENTIONS:
        return []
    by_code = match_by_code(text, entities)
    if by_code:
        return by_code
    return match_by_label(text, entities)


def build_resolution(status: str, entities: list, candidates: list, unknown_mentions: list) -> dict:
    """实体解析结果"""
    return {
        "status": status,
        "entities": entities,
        "candidates": candidates,
        "unknown_mentions": unknown_mentions,
    }


def resolve_mentions(mentions: list, entities: list) -> dict:
    """
    解析规划器给出的实体提及
    :param mentions: 提及列表（问题原文片段）
    :param entities: 实体表
    :return: {"status", "entities", "candidates", "unknown_mentions"}，status 取值：
             confirmed 已确认 / ambiguous 有歧义需澄清 / unknown 提到了库外实体 / none 没有提到实体；
             只要有一个提及能确认就算 confirmed
    """
    confirmed = []
    ambiguous = []
    unknown = []
    for mention in mentions:
        if mention.strip().upper() in GENERIC_MENTIONS:
            continue
        found = match_mention(mention, entities)
        if len(found) == 1:
            if found[0] not in confirmed:
                confirmed.append(found[0])
        elif len(found) > 1:
            for entity in found:
                if entity not in ambiguous:
                    ambiguous.append(entity)
        else:
            unknown.append(mention)
    if confirmed:
        return build_resolution("confirmed", confirmed, [], unknown)
    if ambiguous:
        return build_resolution("ambiguous", [], ambiguous, unknown)
    if unknown:
        return build_resolution("unknown", [], [], unknown)
    return build_resolution("none", [], [], [])


def is_label_match(core: str, entity: dict) -> bool:
    """回复关键词与实体的某个名称互相包含"""
    for label in get_labels(entity):
        if core in label or label in core:
            return True
    return False


def has_common_bigram(core: str, entity: dict) -> bool:
    """回复关键词中有任意 2 个连续字出现在实体的名称里"""
    all_labels = "".join(get_labels(entity))
    for i in range(len(core) - 1):
        if core[i:i + 2] in all_labels:
            return True
    return False


def pick_option(reply: str, options: list):
    """
    解析用户对澄清问题的回复：序号（1 / 第一个）或选项名称中的关键词（如"债券那只"）
    :param reply: 用户回复
    :param options: 上一轮给出的候选实体
    :return: 选中的实体；无法唯一确定时返回 None
    """
    text = reply.strip()
    match = ORDINAL_REPLY.fullmatch(text)
    if match and ORDINALS[match.group(1)] < len(options):
        return options[ORDINALS[match.group(1)]]
    core = REPLY_NOISE.sub("", text)
    if len(core) < 2:
        return None
    hits = [entity for entity in options if is_label_match(core, entity)]
    if len(hits) == 1:
        return hits[0]
    # 名称互相包含对不上时，取回复中至少 2 个连续字出现在选项名称里的唯一选项
    hits = [entity for entity in options if has_common_bigram(core, entity)]
    if len(hits) == 1:
        return hits[0]
    return None
