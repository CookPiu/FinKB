"""
实体与实体解析
实体不单独维护：导入时 node_chunk 让模型识别文档讲的对象，ground_entity 只保留原文里找得到依据的字段，
记在 documents.entity；查询时 get_entities 从已就绪文档现算实体列表，新导入的资料立即生效，不用重启服务。
解析在内存中完成：代码精确匹配 → 别名包含匹配（取最长别名；最长别名并列于多个实体时判为有歧义）。
实体为 dict：{"id", "name", "type"(company / fund / wealth_product / document), "codes", "aliases", "doc_ids"}。
"""
import re

from utils.clients.mongo_utils import get_document_entities

ENTITY_TYPES = ["company", "fund", "wealth_product", "document"]
MAX_ALIASES = 8  # 每份文档最多保留的别名数

# 泛指词不是实体（规划器偶尔会把它们当作提及）
GENERIC_MENTIONS = {"基金", "理财", "理财产品", "产品", "银行", "公司", "报告", "公告", "股票", "ETF", "私募基金", "债券基金"}

# 模糊匹配前去掉的套话：公司后缀、基金与理财的通用字样、口语指代
BOILERPLATE = re.compile(
    r"股份有限公司|有限责任公司|有限公司|交易型开放式指数证券投资基金|证券投资基金|投资基金|联接基金|基金"
    r"|理财产品|理财计划|理财|产品|那只|那款|这只|这款|的"
)
FUZZY_MIN_CHARS = 4  # 模糊匹配要求的最短公共片段

# 澄清回复里的序号 → 选项下标
ORDINALS = {"1": 0, "一": 0, "2": 1, "二": 1, "3": 2, "三": 2, "4": 3, "四": 3}
ORDINAL_REPLY = re.compile(r"第?\s*([1-4一二三四])\s*(个|只|款|项)?[。.！!]?")
# 澄清回复里与选项名称无关的标点和口语词，去掉后剩下的才是关键词
REPLY_NOISE = re.compile(r"[，。,.！!？?\s]|那只|那个|那款|这只|这个|的|吧|呢|是|就")


# ---------- 导入时：校验模型识别的文档对象 ----------

def squash(text: str) -> str:
    """去掉空白、全角括号改半角，用于比对名称是否出现在原文里"""
    return re.sub(r"\s+", "", text).replace("（", "(").replace("）", ")")


def is_abbreviation(alias: str, name: str) -> bool:
    """alias 的每个字按顺序都出现在 name 里，如“招行”之于“招商银行股份有限公司”"""
    pos = 0
    for ch in alias:
        pos = name.find(ch, pos)
        if pos < 0:
            return False
        pos += 1
    return True


def clean_str_list(value) -> list:
    """模型输出的字符串列表：不是列表时为空，去掉非字符串、空白项与重复项"""
    if not isinstance(value, list):
        return []
    items = []
    for item in value:
        if isinstance(item, str) and item.strip() and item.strip() not in items:
            items.append(item.strip())
    return items


def ground_entity(raw: dict, text: str, title: str) -> dict:
    """
    校验模型识别的文档对象，只保留原文里找得到依据的字段：
    - 全称必须出现在原文或资料名称中，否则按不针对具体对象的资料处理（document 类，全称取资料名称）；
    - 代码必须在原文中逐字出现，document 类不记代码；
    - 别名必须出现在原文中，或是全称、资料名称的缩写；泛指词和“本行”“本公司”这类自称不要。
    :param raw: 模型输出的 JSON 对象（无法解析时传 {}）
    :param text: 喂给模型的原文
    :param title: 资料名称
    :return: {"type", "name", "codes", "aliases"}
    """
    if not isinstance(raw, dict):
        raw = {}
    source = squash(text + title)
    entity_type = raw.get("type")
    name = raw.get("name")
    if not isinstance(name, str) or not name.strip() or squash(name) not in source or entity_type not in ENTITY_TYPES:
        entity_type = "document"
        name = title
    name = name.strip()
    codes = []
    if entity_type != "document":
        codes = [code for code in clean_str_list(raw.get("codes")) if squash(code) in source]
    aliases = []
    for alias in clean_str_list(raw.get("aliases")):
        if alias == name or len(alias) < 2 or alias.upper() in GENERIC_MENTIONS or alias.startswith("本"):
            continue
        if squash(alias) in source or is_abbreviation(alias, name) or is_abbreviation(alias, title):
            aliases.append(alias)
    return {"type": entity_type, "name": name, "codes": codes, "aliases": aliases[:MAX_ALIASES]}


# ---------- 查询时：从文档汇总实体 ----------

def add_unique(target: list, items: list):
    for item in items:
        if item not in target:
            target.append(item)


def build_entities(docs: list) -> list:
    """
    文档上记录的对象 → 实体列表：代码有交集或全称相同的文档归为同一个实体（如同一家公司的季报和年报）
    :param docs: [{"_id", "entity"}]，按文件名排序；没有 entity 的文档跳过
    :return: 实体列表；全称、类型取最早一份文档的，id 取最小的代码，没有代码时取全称，同一对象每次算出的 id 相同
    """
    groups = []
    for doc in docs:
        entity = doc.get("entity")
        if not entity:
            continue
        matched = []
        for group in groups:
            if group["name"] == entity["name"] or set(group["codes"]) & set(entity["codes"]):
                matched.append(group)
        if matched:
            # 这份文档可能把原先分开的几组连到一起，全部并进第一组
            group = matched[0]
            for other in matched[1:]:
                add_unique(group["codes"], other["codes"])
                add_unique(group["aliases"], other["aliases"])
                add_unique(group["doc_ids"], other["doc_ids"])
                groups.remove(other)
        else:
            group = {"name": entity["name"], "type": entity["type"], "codes": [], "aliases": [], "doc_ids": []}
            groups.append(group)
        add_unique(group["codes"], entity["codes"])
        add_unique(group["aliases"], entity["aliases"])
        add_unique(group["doc_ids"], [doc["_id"]])
    entities = []
    for group in groups:
        if group["codes"]:
            entity_id = min(group["codes"])
        else:
            entity_id = group["name"]
        entities.append({
            "id": entity_id,
            "name": group["name"],
            "type": group["type"],
            "codes": group["codes"],
            "aliases": group["aliases"],
            "doc_ids": group["doc_ids"],
        })
    return entities


def get_entities() -> list:
    """实体列表：每次从已就绪文档现算"""
    return build_entities(get_document_entities())


def get_entity_map() -> dict:
    """实体 ID → 实体"""
    entity_map = {}
    for entity in get_entities():
        entity_map[entity["id"]] = entity
    return entity_map


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
    by_label = match_by_label(text, entities)
    if by_label:
        return by_label
    # 去掉“基金”“理财”“那只”等字样再比一次，如“华夏基金”“华夏的那只基金”→“华夏”；
    # 这些字样说明问的是产品或公司，所以只比公司、基金、理财产品（避免“货币基金”→“货币”对上货币政策报告）
    core = BOILERPLATE.sub("", text)
    if core != text and len(core) >= 2:
        products = [entity for entity in entities if entity["type"] != "document"]
        by_core = match_by_label(core, products)
        if by_core:
            return by_core
    return match_by_fuzzy(text, entities)


def common_substring_len(a: str, b: str) -> int:
    """两个字符串最长公共片段的长度"""
    best = 0
    prev = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        cur = [0] * (len(b) + 1)
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                cur[j] = prev[j - 1] + 1
                best = max(best, cur[j])
        prev = cur
    return best


def match_by_fuzzy(text: str, entities: list) -> list:
    """
    名称互相包含对不上时的兜底（别名由模型生成，覆盖不全）：两边去掉套话后，
    最长公共片段至少 FUZZY_MIN_CHARS 个字且占提及一半以上才算匹配，如“华夏自由现金流ETF联接”对“华夏自由现金流联接”；
    取公共片段最长的实体，长度并列时全部返回（表示有歧义）
    :param text: 已转大写的提及
    """
    core = BOILERPLATE.sub("", text)
    best_len = 0
    best = []
    for entity in entities:
        for label in get_labels(entity):
            length = common_substring_len(core, BOILERPLATE.sub("", label.upper()))
            if length < FUZZY_MIN_CHARS or length * 2 < len(core):
                continue
            if length > best_len:
                best_len = length
                best = [entity]
            elif length == best_len and entity not in best:
                best.append(entity)
    return best


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
    :param entities: 实体列表
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
