from processor.import_processor.nodes.node_normalize import HeadingStack, normalize, numbered_level
from utils.image_utils import image_size
from utils.text_utils import clean_text, despace

TABLE = "<table><tr><td>项目</td><td>本期</td></tr><tr><td>营业收入</td><td>1</td></tr></table>"


def t(text, page=0, level=None, typ="text"):
    d = {"type": typ, "text": text, "page_idx": page}
    if level:
        d["text_level"] = level
    return d


def test_clean_text_strips_markup():
    assert clean_text("一<sub>、</sub>综合") == "一、综合"
    assert clean_text("**什么是ETF？**") == "什么是ETF？"
    assert clean_text("[**如何买卖**](http://www.sse.com.cn/x)ETF") == "如何买卖ETF"
    assert clean_text("统计公报<sup>[1</sup>") == "统计公报[1"


def test_despace_spaced_characters_and_numbers():
    s = "沪深3 0 0 指 数 上 涨 1 2 1 <sub>.</sub> 0 2 % <sub>，</sub> 开 放 式 股 票 型 基 金上 涨 1 2 1 <sub>.</sub> 4 %"
    assert clean_text(s) == "沪深300指数上涨121.02%，开放式股票型基金上涨121.4%"


def test_despace_leaves_normal_text_alone():
    s = "2026 年 3 月末，本集团资产总额 60,339.62 亿元，较上年末增长 1.8%；Exchange Traded Funds 简称 ETF。"
    assert despace(s) == s


def test_despace_does_not_join_across_lines():
    s = "开 放 式 股 票 型 基 金 上 涨 1 2 1 . 4 %\n2 0 . 9 2 %"
    assert despace(s) == "开放式股票型基金上涨121.4%\n20.92%"


def test_numbered_levels():
    assert numbered_level("第一部分 货币信贷概况") == 2
    assert numbered_level("第一节 主要财务数据") == 3
    assert numbered_level("一、主要财务数据") == 3
    assert numbered_level("三 工业和建筑业") == 3
    assert numbered_level("(一)主要会计数据和财务指标") == 4
    assert numbered_level("（二） 基金运作相关费用") == 4
    assert numbered_level("1.1 主要会计数据和财务指标") == 4
    assert numbered_level("1.5.1 总体业绩") == 5
    assert numbered_level("1、认购") == 5
    assert numbered_level("1. 人工智能行业") == 5
    assert numbered_level("（1）贷款业务") == 6
    assert numbered_level("① 市场风险") == 7
    assert numbered_level("重要内容提示") is None


def test_heading_stack_builds_paths_from_flat_levels():
    hs = HeadingStack()
    hs.push("平安银行2026年第一季度报告", 1)
    hs.push("第一节 主要财务数据", 2)
    hs.push("1.5 管理层讨论与分析", 2)
    hs.push("1.5.2 零售业务", 2)
    hs.push("（2）大财富管理业务", 2)
    hs.push("存款业务", 2)  # 无编号：挂在最近的有编号标题下
    assert hs.path == ["第一节 主要财务数据", "1.5 管理层讨论与分析", "1.5.2 零售业务", "（2）大财富管理业务", "存款业务"]
    hs.push("私行财富", 2)  # 同级无编号标题互为兄弟
    assert hs.path[-2:] == ["（2）大财富管理业务", "私行财富"]
    hs.push("1.5.3 对公业务", 2)
    assert hs.path == ["第一节 主要财务数据", "1.5 管理层讨论与分析", "1.5.3 对公业务"]


def test_orphan_unnumbered_heading_is_popped_by_numbered():
    hs = HeadingStack()
    hs.push("平安银行2026年第一季度报告", 1)
    hs.push("重要内容提示", 2)
    assert hs.path == ["重要内容提示"]
    hs.push("第一节 主要财务数据", 2)
    assert hs.path == ["第一节 主要财务数据"]
    hs.push("2 主要财务数据", 2)
    assert hs.path == ["2 主要财务数据"]
    hs.push("2.1 本集团主要会计数据及财务指标", 2)
    assert hs.path == ["2 主要财务数据", "2.1 本集团主要会计数据及财务指标"]


def test_title_not_in_path_and_second_level1_starts_subdocument():
    blocks = normalize(
        [
            t("风险揭示书", level=1),
            t("一、风险说明", level=2),
            t("正文一。"),
            t("理财产品说明书", page=3, level=1),
            t("一、产品要素", page=3, level=2),
            t("正文二。", page=3),
        ]
    )
    texts = [b for b in blocks if b.type == "text"]
    assert texts[0].section_path == ["一、风险说明"]
    assert texts[1].section_path == ["理财产品说明书", "一、产品要素"]


def test_drops_page_furniture_and_merges_paragraph_across_pages():
    blocks = normalize(
        [
            t("基金管理人依照恪尽职守的原则管理基金财产，但不保证", page=3),
            {"type": "page_number", "text": "4 / 5", "page_idx": 3},
            {"type": "header", "text": "页眉", "page_idx": 4},
            t("基金一定盈利，也不保证最低收益。", page=4),
            t("基金的过往业绩不代表未来表现。", page=4),
        ]
    )
    assert [b.text for b in blocks] == [
        "基金管理人依照恪尽职守的原则管理基金财产，但不保证基金一定盈利，也不保证最低收益。",
        "基金的过往业绩不代表未来表现。",
    ]
    assert blocks[0].page == 4 and blocks[0].page_end == 5


def test_page_footnote_does_not_break_paragraph_merge():
    blocks = normalize(
        [
            t("零售客户数达到", page=6),
            {"type": "page_footnote", "text": "<sup>1</sup> 零售客户数包含借记卡。", "page_idx": 6},
            t("1.3亿户。", page=7),
        ]
    )
    assert blocks[0].text == "零售客户数达到1.3亿户。"
    assert blocks[1].text == "注：1 零售客户数包含借记卡。"


def test_unit_line_absorbed_and_empty_continuation_extends_pages():
    blocks = normalize(
        [
            t("(一)主要会计数据和财务指标", level=2),
            t("单位：元 币种：人民币"),
            {"type": "table", "table_body": TABLE, "page_idx": 0, "table_caption": [], "table_footnote": []},
            {"type": "page_number", "text": "1", "page_idx": 0},
            {"type": "table", "page_idx": 1, "table_caption": [], "table_footnote": []},
        ]
    )
    unit, table = blocks[1], blocks[2]
    assert unit.absorbed
    assert table.context == "单位：元 币种：人民币"
    assert table.page == 1 and table.page_end == 2
    assert table.section_path == ["(一)主要会计数据和财务指标"]


def test_bracketed_unit_line_and_inheritance_within_section():
    blocks = normalize(
        [
            t("1.1 主要会计数据和财务指标", level=2),
            t("（货币单位：人民币百万元）"),
            {"type": "table", "table_body": TABLE, "page_idx": 0},
            {"type": "table", "table_body": TABLE.replace("营业收入", "净利润"), "page_idx": 0},
            t("1.2 非经常性损益项目和金额", level=2),
            {"type": "table", "table_body": TABLE, "page_idx": 0},
        ]
    )
    tables = [b for b in blocks if b.type == "table"]
    assert tables[0].context == "（货币单位：人民币百万元）"
    assert tables[1].context == "（货币单位：人民币百万元）"  # 同一小节沿用
    assert tables[2].context == ""  # 换了小节不再沿用


def test_unmerged_cross_page_table_is_merged():
    cont = "<table><tr><td>净利润</td><td>2</td></tr></table>"
    blocks = normalize(
        [
            {"type": "table", "table_body": TABLE, "page_idx": 0},
            {"type": "footer", "text": "x", "page_idx": 0},
            {"type": "table", "table_body": cont, "page_idx": 1},
        ]
    )
    assert len(blocks) == 1
    assert "净利润" in blocks[0].table_html and blocks[0].page_end == 2


def test_heading_like_fragments_become_text():
    blocks = normalize([t("理财产品过往业绩不代表其未来表现，", level=2), t("√适用 □不适用", level=2)])
    assert [b.type for b in blocks] == ["text", "text"]


def test_image_size_png(tmp_path):
    import struct
    import zlib

    raw = b"\x00" + b"\x00" * 3 * 300
    png = (
        b"\x89PNG\r\n\x1a\n"
        + struct.pack(">I", 13)
        + b"IHDR"
        + struct.pack(">IIBBBBB", 300, 1, 8, 2, 0, 0, 0)
        + struct.pack(">I", 0)
        + struct.pack(">I", 0)
        + b"IDAT"
        + zlib.compress(raw)
    )
    p = tmp_path / "a.png"
    p.write_bytes(png)
    assert image_size(p) == (300, 1)
