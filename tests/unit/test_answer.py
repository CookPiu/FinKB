from utils.citation_utils import finalize_citations, render_sources
from utils.guard_utils import REPLACEMENT, StreamGuard, violations
from utils.entity_utils import build_entities, ground_entity, pick_option, resolve_mentions
from utils.clients.mongo_history_utils import last_turns

# ---------- 合规守卫：正反两类用例 ----------


def test_banned_phrases_are_blocked():
    assert violations("这只基金保证收益，放心买。") == ["保证收益"]
    assert "稳赚不赔" in violations("这款理财稳赚不赔。")
    assert violations("茅台股价一定会上涨。") == ["一定会上涨"]
    assert violations("建议您现在买入华夏债券C。") == ["建议您现在买入"]


def test_negated_phrases_are_allowed():
    assert violations("本产品不保证本金和收益，也不保证收益。") == []
    assert violations("金融产品不存在稳赚不赔的情况。") == []
    assert violations("理财产品并非零风险。") == []
    assert violations("我们不建议您买入任何具体产品。") == []
    assert violations("私募基金不得承诺最低收益，更不能保证收益。") == []


def test_describing_scams_is_allowed_but_promising_is_not():
    assert violations("非法集资通常保本保息，以高息、返利为诱饵。") == []
    assert violations("要警惕声称稳赚不赔的推销。") == []
    assert violations("这款产品保本保息，年化8%。") == ["保本保息"]
    # 分点回答中，语境词出现在问题或前文里
    g = StreamGuard(context="怎么分辨非法集资？")
    assert g.feed("* 收益承诺：通常保本保息。") == "* 收益承诺：通常保本保息。" and g.hits == []


def test_stream_guard_replaces_whole_sentence_and_keeps_others():
    g = StreamGuard()
    out = g.feed("茅台一季度营收增长6.54%。这只股票稳赚") + g.feed("不赔！历史业绩不代表未来") + g.flush()
    assert out == "茅台一季度营收增长6.54%。" + REPLACEMENT + "历史业绩不代表未来"
    assert g.hits == ["稳赚不赔", "稳赚"]


# ---------- 引用 ----------


def _ev(eid, title="华夏债券基金产品资料概要", page=3):
    return {"eid": eid, "kind": "chunk", "text": "t", "doc_id": "d", "file_name": f"{title}.pdf",
            "document_title": title, "content_type": "基金产品资料概要", "page_start": page, "page_end": page,
            "source_path": "", "entity_name": "华夏债券C", "entity_codes": ["001003"],
            "derived": False}


def test_citations_keep_valid_drop_invented_and_order_by_first_use():
    text, used = finalize_citations("托管费0.20%[E2]，管理费0.60%【E1】，另见[E9]。[E2]", [_ev(1), _ev(2, page=4)])
    assert text == "托管费0.20%[E2]，管理费0.60%[E1]，另见。[E2]"
    assert [e["eid"] for e in used] == [2, 1]
    assert render_sources(used).splitlines()[0] == (
        "[E2] 华夏债券基金产品资料概要｜基金产品资料概要｜华夏债券C（001003）｜华夏债券基金产品资料概要.pdf｜第4页"
    )


# ---------- 实体解析 ----------

ENTITIES = [
    {"id": "bond", "name": "华夏债券投资基金（华夏债券C）", "type": "fund", "codes": ["001001", "001003"],
     "aliases": ["华夏债券C", "华夏债券", "华夏"], "doc_ids": []},
    {"id": "fcf", "name": "华夏国证自由现金流ETF发起式联接基金", "type": "fund", "codes": ["023917"],
     "aliases": ["华夏自由现金流", "华夏"], "doc_ids": []},
    {"id": "maotai", "name": "贵州茅台酒股份有限公司", "type": "company", "codes": ["600519"],
     "aliases": ["贵州茅台", "茅台"], "doc_ids": []},
]


def test_resolve_by_code_alias_and_ambiguity():
    assert resolve_mentions(["001003"], ENTITIES)["entities"][0]["id"] == "bond"
    assert resolve_mentions(["华夏债券C"], ENTITIES)["entities"][0]["id"] == "bond"
    assert resolve_mentions(["茅台"], ENTITIES)["entities"][0]["id"] == "maotai"
    amb = resolve_mentions(["华夏的那只基金"], ENTITIES)
    assert amb["status"] == "ambiguous" and {e["id"] for e in amb["candidates"]} == {"bond", "fcf"}
    assert resolve_mentions(["宁德时代"], ENTITIES)["status"] == "unknown"
    assert resolve_mentions(["基金"], ENTITIES)["status"] == "none"  # 泛指词不算实体
    assert resolve_mentions([], ENTITIES)["status"] == "none"


def test_pick_option_for_clarification():
    opts = [ENTITIES[0], ENTITIES[1]]
    assert pick_option("第一个", opts)["id"] == "bond"
    assert pick_option("2", opts)["id"] == "fcf"
    assert pick_option("债券那只", opts)["id"] == "bond"
    assert pick_option("自由现金流的", opts)["id"] == "fcf"
    assert pick_option("茅台营收多少", opts) is None


# ---------- 文档对象：导入时校验，查询时汇总 ----------

KFS_TEXT = "基金名称 华夏债券投资基金 基金简称 华夏债券 基金代码 001001 下属基金简称 华夏债券A 华夏债券C 下属基金代码 001001 001003 基金管理人 华夏基金管理有限公司"


def test_ground_entity_keeps_fields_found_in_text():
    raw = {"type": "fund", "name": "华夏债券投资基金", "codes": ["001001", "001003"],
           "aliases": ["华夏债券C", "华夏债", "华夏债券"]}
    entity = ground_entity(raw, KFS_TEXT, "华夏债券投资基金（华夏债券C）基金产品资料概要更新")
    assert entity == {"type": "fund", "name": "华夏债券投资基金", "codes": ["001001", "001003"],
                      "aliases": ["华夏债券C", "华夏债", "华夏债券"]}  # “华夏债”是全称的缩写，保留
    company = ground_entity({"type": "company", "name": "招商银行股份有限公司", "codes": ["600036"],
                             "aliases": ["招行", "招商银行"]}, "A 股代码：600036 招商银行股份有限公司", "招商银行季报")
    assert company["codes"] == ["600036"] and company["aliases"] == ["招行", "招商银行"]


def test_ground_entity_drops_what_text_does_not_support():
    raw = {"type": "fund", "name": "华夏债券投资基金", "codes": ["001001", "999999"],
           "aliases": ["基金", "本基金", "易方达", "华夏债券C"]}
    entity = ground_entity(raw, KFS_TEXT, "华夏债券资料概要")
    assert entity["codes"] == ["001001"]  # 原文里没有的代码去掉
    assert entity["aliases"] == ["华夏债券C"]  # 泛指、自称、原文与全称里都没有的别名去掉
    # 全称在原文里找不到、类型不合法或输出无法解析：按不针对具体对象的资料处理，全称取资料名称，不记代码
    fallback = ground_entity({"type": "fund", "name": "华夏成长混合", "codes": ["001001"]}, KFS_TEXT, "资料甲")
    assert fallback == {"type": "document", "name": "资料甲", "codes": [], "aliases": []}
    assert ground_entity({"type": "stock", "name": "华夏债券投资基金"}, KFS_TEXT, "资料甲")["type"] == "document"
    assert ground_entity({}, KFS_TEXT, "资料甲")["name"] == "资料甲"


def test_build_entities_merges_documents_of_the_same_object():
    docs = [
        {"_id": "q1", "entity": {"type": "company", "name": "贵州茅台酒股份有限公司", "codes": ["600519"], "aliases": ["茅台"]}},
        {"_id": "kfs", "entity": {"type": "fund", "name": "华夏债券投资基金", "codes": ["001001", "001003"], "aliases": []}},
        {"_id": "h1", "entity": {"type": "company", "name": "贵州茅台酒股份有限公司", "codes": [], "aliases": ["贵州茅台"]}},
        {"_id": "kfs2", "entity": {"type": "fund", "name": "华夏债券基金", "codes": ["001003"], "aliases": ["华夏债券C"]}},
        {"_id": "faq", "entity": {"type": "document", "name": "投资者问答", "codes": [], "aliases": []}},
        {"_id": "old"},  # 还没识别过对象的文档跳过
    ]
    entities = {e["id"]: e for e in build_entities(docs)}
    assert set(entities) == {"600519", "001001", "投资者问答"}  # id 取最小代码，没有代码取全称
    assert entities["600519"]["doc_ids"] == ["q1", "h1"] and entities["600519"]["aliases"] == ["茅台", "贵州茅台"]
    assert entities["001001"]["doc_ids"] == ["kfs", "kfs2"] and entities["001001"]["name"] == "华夏债券投资基金"
    assert entities["投资者问答"]["type"] == "document"


def test_fuzzy_match_rescues_variants_but_not_other_objects():
    auto = [
        {"id": "fcf", "name": "华夏国证自由现金流交易型开放式指数证券投资基金发起式联接基金", "type": "fund",
         "codes": ["023917"], "aliases": ["华夏国证自由现金流ETF发起式联接", "华夏自由现金流联接"], "doc_ids": []},
        {"id": "zzys", "name": "易方达智造优势混合型证券投资基金", "type": "fund", "codes": ["011300"],
         "aliases": ["易方达智造优势"], "doc_ids": []},
        {"id": "cmb", "name": "招商银行股份有限公司", "type": "company", "codes": ["600036"], "aliases": ["招行"], "doc_ids": []},
        {"id": "ldy", "name": "中国建设银行广东省分行利得盈2013年第17期人民币非保本理财", "type": "wealth_product",
         "codes": [], "aliases": ["建行利得盈2013年第17期"], "doc_ids": []},
        {"id": "bond", "name": "华夏债券投资基金", "type": "fund", "codes": ["001001"], "aliases": ["华夏债券C"], "doc_ids": []},
        {"id": "mp", "name": "中国货币政策执行报告", "type": "document", "codes": [], "aliases": [], "doc_ids": []},
    ]
    for mention, entity_id in [("华夏自由现金流ETF联接", "fcf"), ("华夏国证自由现金流联接基金", "fcf"), ("建行利得盈理财", "ldy")]:
        assert resolve_mentions([mention], auto)["entities"][0]["id"] == entity_id, mention
    # 去掉“基金”“那只”后的品牌名对上多只产品：请用户确认
    for mention in ["华夏基金", "华夏的那只基金"]:
        amb = resolve_mentions([mention], auto)
        assert amb["status"] == "ambiguous" and {e["id"] for e in amb["candidates"]} == {"fcf", "bond"}, mention
    # 只有品牌或行业字样相同的库外对象不能被模糊匹配拉进来
    for mention in ["易方达蓝筹精选混合", "工商银行", "工商银行股份有限公司", "华夏成长混合"]:
        assert resolve_mentions([mention], auto)["status"] == "unknown", mention
    # 泛指的产品类别与资料类别不是对象：既不能对上货币政策报告或投教资料，也不算库外对象
    for mention in ["货币基金", "投资者教育资料", "季报"]:
        assert resolve_mentions([mention], auto)["status"] == "none", mention


# ---------- 会话历史：取最近 n 条（旧项目 K-23） ----------


def test_last_turns_returns_recent_in_chronological_order():
    desc = [{"text": "第3轮"}, {"text": "第2轮"}, {"text": "第1轮"}]  # 数据库按时间倒序取出的最近记录
    assert [m["text"] for m in last_turns(desc)] == ["第1轮", "第2轮", "第3轮"]
