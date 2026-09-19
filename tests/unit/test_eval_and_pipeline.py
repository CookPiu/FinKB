from dataclasses import dataclass

from common.models.document import Stage, next_stage
from evaluation.dataset import EvalItem, Gold, norm, self_check
from evaluation.metrics import file_hit_rank, first_hit_rank, summarize
from utils.classify_utils import guess_content_type, title_from_filename
from processor.import_processor.nodes.node_entry import doc_id_from_hash
from utils.search_utils import build_filter, rrf_fuse


@dataclass
class H:
    file_name: str
    text: str


def test_next_stage_order():
    assert next_stage(None) == Stage.REGISTER
    assert next_stage(Stage.REGISTER) == Stage.PARSE
    assert next_stage("chunk") == Stage.INDEX
    assert next_stage(Stage.INDEX) == Stage.ENRICH
    assert next_stage(Stage.ENRICH) is None


def test_doc_id_is_deterministic_prefix():
    h = "ab" * 32
    assert doc_id_from_hash(h) == h[:16]


def test_content_type_rules():
    assert guess_content_type("用户FAQ", "中国人民银行金融消费者权益保护实施办法.pdf") == "法规制度"
    assert guess_content_type("用户FAQ", "消费者金融素养问卷调查报告.pdf") == "调查报告"
    assert guess_content_type("用户FAQ", "基金基础知识.pdf") == "投资者教育FAQ"
    assert guess_content_type("上市公司年报", "招商银行二○二六年第一季度报告.pdf") == "公司定期报告"
    assert guess_content_type("", "未知.pdf") == "其他"
    assert title_from_filename("2026-04-25 平安银行股份有限公司2026年第一季度报告") == "平安银行股份有限公司2026年第一季度报告"


def test_norm_handles_fullwidth_and_whitespace():
    assert norm("营业收入 （元）\n53,909") == norm("营业收入(元)53,909")


def test_hit_requires_file_and_quote():
    golds = [Gold(file="a.pdf", quote="托管费 0.20%")]
    ranked = [H("b.pdf", "托管费0.20%"), H("a.pdf", "管理费0.60%"), H("a.pdf", "固定费率 托管费0.20% 基金托管人")]
    assert first_hit_rank(ranked, golds) == 3
    assert file_hit_rank(ranked, golds) == 2


def test_summarize():
    s = summarize([1, 3, None, 6])
    assert s["hit@1"] == 0.25 and s["hit@5"] == 0.5 and s["hit@10"] == 0.75
    assert s["mrr@10"] == round((1 + 1 / 3 + 1 / 6) / 4, 4)


def test_self_check_flags_missing_quote_and_file():
    corpus = {"a.pdf": norm("本基金不提供任何保证。投资者可能损失投资本金。")}
    items = [
        EvalItem.model_validate(
            {"id": "x1", "category": "product", "turns": [{"q": "q"}], "expect": "answer",
             "gold": [{"file": "a.pdf", "quote": "本基金不提供 任何保证"}]}
        ),
        EvalItem.model_validate(
            {"id": "x2", "category": "product", "turns": [{"q": "q"}], "expect": "answer",
             "gold": [{"file": "a.pdf", "quote": "保证最低收益率"}]}
        ),
        EvalItem.model_validate(
            {"id": "x3", "category": "product", "turns": [{"q": "q"}], "expect": "answer",
             "gold": [{"file": "b.pdf", "quote": "本基金不提供任何保证"}]}
        ),
    ]
    errors = self_check(items, corpus)
    assert len(errors) == 2
    assert errors[0].startswith("x2") and errors[1].startswith("x3")


def test_multiturn_turn_fields_resolved():
    it = EvalItem.model_validate(
        {"id": "mt", "category": "multiturn", "turns": [
            {"q": "华夏的基金管理费多少？", "expect": "clarify"},
            {"q": "债券那只", "expect": "answer", "gold": [{"file": "a.pdf", "quote": "固定费率0.60%"}]},
        ]}
    )
    turns = it.resolved_turns()
    assert [t.expect for t in turns] == ["clarify", "answer"]


def test_build_filter_and_rrf():
    assert build_filter(kinds=["table"], content_types=["公司定期报告"]) == (
        'kind in ["table"] and content_type in ["公司定期报告"]'
    )
    assert build_filter(entity_ids=["e1"]) == 'ARRAY_CONTAINS_ANY(entity_ids, ["e1"])'
    row = dict(doc_id="d", version=1, kind="text", content_type="c", section_path="", page_start=1, page_end=1,
               derived=False, text="t", entity_ids=[])
    dense = [{"chunk_id": "a", "score": 0.9, **row}, {"chunk_id": "b", "score": 0.8, **row}]
    sparse = [{"chunk_id": "b", "score": 0.3, **row}]
    fused = rrf_fuse(dense, sparse)
    assert [h.chunk_id for h in fused] == ["b", "a"]  # 两路都召回的排前面
    assert fused[0].score_dense == 0.8 and fused[0].score_sparse == 0.3 and fused[1].score_sparse is None
