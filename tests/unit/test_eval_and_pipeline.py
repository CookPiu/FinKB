from evaluation.dataset import build_eval_item, norm, resolve_turns, self_check
from evaluation.metrics import file_hit_rank, first_hit_rank, summarize
from processor.import_processor.nodes.node_entry import build_doc_id
from utils.classify_utils import guess_content_type, title_from_filename
from utils.task_utils import INGEST_EXTS, scan_files


def make_hit(file_name, text):
    return {"file_name": file_name, "text": text}


def test_scan_files_picks_supported_files(tmp_path):
    (tmp_path / "b.pdf").write_text("x", encoding="utf-8")
    (tmp_path / "a.md").write_text("x", encoding="utf-8")
    (tmp_path / "note.txt").write_text("x", encoding="utf-8")  # 不支持的格式
    (tmp_path / "~$tmp.docx").write_text("x", encoding="utf-8")  # Office 临时文件
    (tmp_path / ".hidden.pdf").write_text("x", encoding="utf-8")  # 隐藏文件
    sub = tmp_path / "子目录"
    sub.mkdir()
    (sub / "c.docx").write_text("x", encoding="utf-8")

    names = [p.name for p in scan_files(tmp_path)]
    assert names == ["a.md", "b.pdf", "c.docx"]  # 递归、按路径排序
    assert ".txt" not in INGEST_EXTS


def test_doc_id_is_deterministic_prefix():
    h = "ab" * 32
    assert build_doc_id(h) == h[:16]


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
    golds = [{"file": "a.pdf", "quote": "托管费 0.20%"}]
    ranked = [
        make_hit("b.pdf", "托管费0.20%"),
        make_hit("a.pdf", "管理费0.60%"),
        make_hit("a.pdf", "固定费率 托管费0.20% 基金托管人"),
    ]
    assert first_hit_rank(ranked, golds) == 3
    assert file_hit_rank(ranked, golds) == 2


def test_summarize():
    s = summarize([1, 3, None, 6])
    assert s["hit@1"] == 0.25 and s["hit@5"] == 0.5 and s["hit@10"] == 0.75
    assert s["mrr@10"] == round((1 + 1 / 3 + 1 / 6) / 4, 4)


def test_self_check_flags_missing_quote_and_file():
    corpus = {"a.pdf": norm("本基金不提供任何保证。投资者可能损失投资本金。")}
    items = [
        build_eval_item(
            {"id": "x1", "category": "product", "turns": [{"q": "q"}], "expect": "answer",
             "gold": [{"file": "a.pdf", "quote": "本基金不提供 任何保证"}]}
        ),
        build_eval_item(
            {"id": "x2", "category": "product", "turns": [{"q": "q"}], "expect": "answer",
             "gold": [{"file": "a.pdf", "quote": "保证最低收益率"}]}
        ),
        build_eval_item(
            {"id": "x3", "category": "product", "turns": [{"q": "q"}], "expect": "answer",
             "gold": [{"file": "b.pdf", "quote": "本基金不提供任何保证"}]}
        ),
    ]
    errors = self_check(items, corpus)
    assert len(errors) == 2
    assert errors[0].startswith("x2") and errors[1].startswith("x3")


def test_multiturn_turn_fields_resolved():
    it = build_eval_item(
        {"id": "mt", "category": "multiturn", "turns": [
            {"q": "华夏的基金管理费多少？", "expect": "clarify"},
            {"q": "债券那只", "expect": "answer", "gold": [{"file": "a.pdf", "quote": "固定费率0.60%"}]},
        ]}
    )
    turns = resolve_turns(it)
    assert [t["expect"] for t in turns] == ["clarify", "answer"]
