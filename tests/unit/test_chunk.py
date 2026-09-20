from processor.import_processor.nodes.node_chunk import build_chunks, split_long_text
from utils.block_utils import new_block

MAOTAI = (
    "<table><tr><td>项目</td><td>本报告期</td><td>上年同期</td><td>本报告期比上年同期增减变动幅度(%)</td></tr>"
    "<tr><td>营业收入</td><td>53,909,252,220.51</td><td>50,600,957,885.78</td><td>6.54</td></tr></table>"
)


def block(seq, typ, page, path=(), fields=None):
    """构造版面块：其余字段取默认值，fields 覆盖指定字段"""
    b = new_block(typ, page, list(path))
    b["seq"] = seq
    if fields:
        b.update(fields)
    return b


def text(seq, s, page=1, path=("一、概况",)):
    return block(seq, "text", page, path, {"text": s})


def head(seq, s, level, path, page=1):
    return block(seq, "heading", page, path, {"text": s, "level": level})


def build(blocks, table_max=3000):
    return build_chunks(blocks, target=600, max_chars=1000, min_chars=200, table_max=table_max)


def test_split_long_text_bounds():
    s = "。".join(["这是一句测试文本" * 5] * 60) + "。"
    pieces = split_long_text(s, 600, 1000)
    assert all(len(p) <= 1000 for p in pieces)
    assert "".join(pieces) == s
    assert len(pieces) > 1


def test_split_long_text_hard_cuts_single_long_sentence():
    s = "无标点" * 800
    pieces = split_long_text(s, 600, 1000)
    assert all(len(p) <= 1000 for p in pieces)
    assert "".join(pieces) == s


def test_text_grouped_by_target_and_max():
    blocks = [text(i, "字" * 250, page=i + 1) for i in range(6)]
    chunks = build(blocks)
    assert all(c["kind"] == "text" for c in chunks)
    assert all(len(c["text"]) <= 1000 for c in chunks)
    assert chunks[0]["page_start"] == 1 and chunks[0]["page_end"] == 3  # 3 × 251 ≥ 600 时断开
    assert sum(c["text"].count("字") for c in chunks) == 1500


def test_heading_starts_new_chunk_when_buffer_large_enough():
    blocks = [
        head(0, "一、产品概况", 3, ["一、产品概况"]),
        text(1, "甲" * 300, path=["一、产品概况"]),
        head(2, "二、投资策略", 3, ["二、投资策略"]),
        text(3, "乙" * 300, path=["二、投资策略"]),
    ]
    chunks = build(blocks)
    assert [c["section_path"] for c in chunks] == ["一、产品概况", "二、投资策略"]
    assert chunks[1]["text"].startswith("二、投资策略\n")


def test_small_sections_are_merged_with_heading_lines():
    blocks = [
        head(0, "（一）风险揭示", 4, ["（一）风险揭示"]),
        text(1, "本基金不提供任何保证。", path=["（一）风险揭示"]),
        head(2, "（二）重要提示", 4, ["（二）重要提示"]),
        text(3, "基金的过往业绩不代表未来表现。", path=["（二）重要提示"]),
    ]
    chunks = build(blocks)
    assert len(chunks) == 1
    assert chunks[0]["text"] == "（一）风险揭示\n本基金不提供任何保证。\n（二）重要提示\n基金的过往业绩不代表未来表现。"


def test_table_is_its_own_chunk_with_unit_and_pages():
    blocks = [
        head(0, "(一)主要会计数据和财务指标", 4, ["一、主要财务数据", "(一)主要会计数据和财务指标"]),
        block(1, "text", 1, (), {"text": "单位：元 币种：人民币", "absorbed": True}),
        block(
            2,
            "table",
            1,
            ["一、主要财务数据", "(一)主要会计数据和财务指标"],
            {"page_end": 2, "table_html": MAOTAI, "context": "单位：元 币种：人民币"},
        ),
    ]
    chunks = build(blocks)
    assert len(chunks) == 1  # 只有标题的缓冲不单独成块，被吸收的单位行不重复出现
    c = chunks[0]
    assert c["kind"] == "table"
    assert (c["page_start"], c["page_end"]) == (1, 2)
    assert c["section_path"] == "一、主要财务数据 > (一)主要会计数据和财务指标"
    assert "单位：元 币种：人民币" in c["text"] and "53,909,252,220.51" in c["text"]
    assert c["embed_body"].splitlines()[0] == "[单位：元 币种：人民币]"
    assert "营业收入" in c["embed_body"] and "本报告期=53,909,252,220.51" in c["embed_body"]


def test_small_text_carries_across_table():
    blocks = [
        text(0, "第一季度财务报表是否经审计 □是 √否", page=1),
        block(1, "table", 1, (), {"table_html": MAOTAI}),
        text(2, "对公司将非经常性损益项目认定的说明。□适用 √不适用", page=2),
    ]
    chunks = build(blocks)
    assert [c["kind"] for c in chunks] == ["table", "text"]
    assert chunks[1]["text"].startswith("第一季度财务报表是否经审计")
    assert (chunks[1]["page_start"], chunks[1]["page_end"]) == (1, 2)


def test_large_table_split_repeats_header():
    rows = "".join(f"<tr><td>项目{i}</td><td>{i}</td></tr>" for i in range(200))
    html = f"<table><tr><td>项目</td><td>金额</td></tr>{rows}</table>"
    blocks = [block(0, "table", 3, (), {"table_html": html, "footnote": "注：示例", "caption": "表1"})]
    chunks = build(blocks, table_max=500)
    assert len(chunks) > 1
    for i, c in enumerate(chunks):
        assert c["text"].startswith(f"表1 （续表 {i + 1}/{len(chunks)}）")
        assert "| 项目 | 金额 |" in c["text"]
        assert len(c["embed_body"]) <= 500 + 10
    assert chunks[-1]["text"].endswith("注：示例") and "注：示例" not in chunks[0]["text"]
    body = "\n".join(c["embed_body"] for c in chunks)
    assert all(f"项目=项目{i}；" in body for i in range(200))


def test_image_desc_chunk_is_derived():
    blocks = [block(0, "image", 2, (), {"text": "柱状图，显示2021-2025年GDP。", "caption": "图1 国内生产总值", "derived": True})]
    c = build(blocks)[0]
    assert c["kind"] == "image_desc" and c["derived"]
    assert c["text"].startswith("[图表] 图1 国内生产总值")


def test_long_paragraph_after_heading_keeps_heading():
    blocks = [head(0, "四、风险揭示", 3, ["四、风险揭示"]), text(1, "风险。" * 600, path=["四、风险揭示"])]
    chunks = build(blocks)
    assert chunks[0]["text"].startswith("四、风险揭示\n")
    assert all(len(c["text"]) <= 1000 + len("四、风险揭示\n") for c in chunks)
