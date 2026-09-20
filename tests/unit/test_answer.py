from utils.citation_utils import finalize_citations, render_sources
from utils.guard_utils import REPLACEMENT, StreamGuard, violations
from utils.entity_utils import pick_option, resolve_mentions
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
            "source_path": "", "entity_name": "华夏债券C", "entity_codes": ["001003"], "score_dense": None,
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
     "aliases": ["华夏债券C", "华夏债券", "华夏"], "files": []},
    {"id": "fcf", "name": "华夏国证自由现金流ETF发起式联接基金", "type": "fund", "codes": ["023917"],
     "aliases": ["华夏自由现金流", "华夏"], "files": []},
    {"id": "maotai", "name": "贵州茅台酒股份有限公司", "type": "company", "codes": ["600519"],
     "aliases": ["贵州茅台", "茅台"], "files": []},
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


# ---------- 会话历史：取最近 n 条（旧项目 K-23） ----------


def test_last_turns_returns_recent_in_chronological_order():
    desc = [{"text": "第3轮"}, {"text": "第2轮"}, {"text": "第1轮"}]  # 数据库按时间倒序取出的最近记录
    assert [m["text"] for m in last_turns(desc)] == ["第1轮", "第2轮", "第3轮"]
