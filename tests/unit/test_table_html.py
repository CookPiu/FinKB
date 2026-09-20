from utils.table_html_utils import (
    get_column_names,
    get_unit_hint,
    linearize_rows,
    merge_html,
    parse_table,
    split_rows,
    to_markdown,
)

MAOTAI = (
    "<table>"
    "<tr><td>项目</td><td>本报告期</td><td>上年同期</td><td>本报告期比上年同期增减变动幅度(%)</td></tr>"
    "<tr><td>营业收入</td><td>53,909,252,220.51</td><td>50,600,957,885.78</td><td>6.54</td></tr>"
    "<tr><td>基本每股收益(元/股)</td><td>0.00</td><td>0.00</td><td>不适用</td></tr>"
    "</table>"
)


def test_simple_header_and_linearize():
    t = parse_table(MAOTAI)
    assert t["header_rows"] == 1
    assert t["n_cols"] == 4
    lines = linearize_rows(t)
    assert lines[0] == (
        "项目=营业收入；本报告期=53,909,252,220.51；上年同期=50,600,957,885.78；"
        "本报告期比上年同期增减变动幅度(%)=6.54"
    )


def test_equal_adjacent_values_are_kept():
    # 相邻两列都是 0.00 不是合并单元格，不能被当作副本去掉
    t = parse_table(MAOTAI)
    assert linearize_rows(t)[1] == "项目=基本每股收益(元/股)；本报告期=0.00；上年同期=0.00；本报告期比上年同期增减变动幅度(%)=不适用"


def test_rowspan_colspan_expansion_and_multilevel_header():
    html = (
        "<table>"
        '<tr><td rowspan="2">项目</td><td colspan="2">2026年3月31日</td></tr>'
        "<tr><td>金额</td><td>占比</td></tr>"
        '<tr><td rowspan="2">贷款</td><td>100</td><td>50%</td></tr>'
        "<tr><td>200</td><td>30%</td></tr>"
        "</table>"
    )
    t = parse_table(html)
    assert t["header_rows"] == 2
    assert t["grid"] == [
        ["项目", "2026年3月31日", "2026年3月31日"],
        ["项目", "金额", "占比"],
        ["贷款", "100", "50%"],
        ["贷款", "200", "30%"],
    ]
    assert get_column_names(t) == ["项目", "2026年3月31日·金额", "2026年3月31日·占比"]
    # 纵向合并的分组名在每行重复，便于单行检索
    assert linearize_rows(t)[1] == "项目=贷款；2026年3月31日·金额=200；2026年3月31日·占比=30%"


def test_colspan_in_data_row_not_duplicated():
    html = (
        "<table><tr><td>项目</td><td>本期</td><td>上期</td></tr>"
        '<tr><td>合计</td><td colspan="2">不适用</td></tr>'
        '<tr><td colspan="3">非经常性损益项目</td></tr></table>'
    )
    t = parse_table(html)
    lines = linearize_rows(t)
    assert lines[0] == "项目=合计；本期=不适用"
    assert lines[1] == "非经常性损益项目"


def test_rowspan_at_row_end():
    html = (
        '<table><tr><td>a</td><td>b</td><td rowspan="2">c</td></tr>'
        "<tr><td>d</td><td>e</td></tr></table>"
    )
    t = parse_table(html)
    assert t["grid"][1] == ["d", "e", "c"]


def test_br_and_entities_in_cells():
    html = "<table><tr><td>名称</td><td>值</td></tr><tr><td>托管费<br>（年费率）</td><td>0.10&#37;</td></tr></table>"
    t = parse_table(html)
    assert t["grid"][1] == ["托管费 （年费率）", "0.10%"]


def test_th_header_detection():
    html = "<table><tr><th>A</th><th>B</th></tr><tr><th>A1</th><th>B1</th></tr><tr><td>1</td><td>2</td></tr></table>"
    assert parse_table(html)["header_rows"] == 2


def test_single_row_table_has_no_header():
    t = parse_table("<table><tr><td>基金代码</td><td>001001</td></tr></table>")
    assert t["header_rows"] == 0
    assert linearize_rows(t) == ["基金代码=001001"]


def test_four_column_kv_table():
    # 基金产品资料概要“产品概况”：首行含数值，不能当表头
    html = (
        "<table><tr><td>基金简称</td><td>华夏债券</td><td>基金代码</td><td>001001</td></tr>"
        "<tr><td>下属基金简称</td><td>华夏债券C</td><td>下属基金代码</td><td>001003</td></tr>"
        '<tr><td>其他</td><td colspan="3">存续期间内基金份额持有人数量连续60个工作日达不到100人</td></tr></table>'
    )
    t = parse_table(html)
    assert t["header_rows"] == 0 and t["kv"]
    assert linearize_rows(t) == [
        "基金简称=华夏债券；基金代码=001001",
        "下属基金简称=华夏债券C；下属基金代码=001003",
        "其他=存续期间内基金份额持有人数量连续60个工作日达不到100人",
    ]


def test_two_column_kv_with_long_value():
    html = (
        "<table><tr><td>投资目标</td><td>在强调本金安全的前提下,追求较高的当期收入和总回报,实现基金资产的长期增值。</td></tr>"
        "<tr><td>业绩比较基准</td><td>中证综合债券指数</td></tr></table>"
    )
    t = parse_table(html)
    assert t["header_rows"] == 0 and t["kv"]
    assert linearize_rows(t)[1] == "业绩比较基准=中证综合债券指数"


def test_numeric_first_row_is_not_header():
    html = (
        "<table><tr><td>报告期末普通股股东总数</td><td>243,159</td><td>表决权恢复的优先股股东总数</td><td>0</td></tr>"
        "<tr><td>股东名称</td><td>持股数量</td><td>持股比例(%)</td><td>股份性质</td></tr>"
        "<tr><td>中国贵州茅台酒厂(集团)有限责任公司</td><td>678,291,955</td><td>54.07</td><td>无限售</td></tr></table>"
    )
    t = parse_table(html)
    assert t["header_rows"] == 0
    assert not t["kv"]  # 第三行偶数位置是数值，不是键值表


def test_subheader_row_switches_column_names():
    # 茅台季报主表：表体中间出现第二组表头，其后的“总资产”不能标成“本报告期/上年同期”
    html = (
        "<table><tr><td>项目</td><td>本报告期</td><td>上年同期</td><td>增减变动幅度(%)</td></tr>"
        "<tr><td>营业收入</td><td>53,909,252,220.51</td><td>50,600,957,885.78</td><td>6.54</td></tr>"
        "<tr><td></td><td>本报告期末</td><td>上年度末</td><td>本报告期末比上年度末增减变动幅度(%)</td></tr>"
        "<tr><td>总资产</td><td>319,918,844,905.58</td><td>303,834,844,021.44</td><td>5.29</td></tr></table>"
    )
    lines = linearize_rows(parse_table(html))
    assert len(lines) == 2
    assert lines[1] == (
        "项目=总资产；本报告期末=319,918,844,905.58；上年度末=303,834,844,021.44；"
        "本报告期末比上年度末增减变动幅度(%)=5.29"
    )


def test_unit_in_first_header_cell():
    html = (
        "<table><tr><td>(人民币百万元,特别注明除外)</td><td>2026年1-3月</td><td>2025年1-3月</td></tr>"
        "<tr><td>营业收入</td><td>86,940</td><td>83,751</td></tr></table>"
    )
    t = parse_table(html)
    assert get_unit_hint(t) == "(人民币百万元,特别注明除外)"
    assert linearize_rows(t) == ["项目=营业收入；2026年1-3月=86,940；2025年1-3月=83,751"]


def test_parenthesized_negative_is_numeric_for_header_check():
    html = "<table><tr><td>股东权益</td><td>(1.3%)</td></tr><tr><td>股本</td><td>19,406</td></tr></table>"
    t = parse_table(html)
    assert t["header_rows"] == 0 and t["kv"]


def test_markdown_blanks_horizontal_span_duplicates():
    html = "<table><tr><td>项目</td><td>本期</td><td>上期</td></tr><tr><td>说明</td><td colspan=\"2\">不适用</td></tr></table>"
    md = to_markdown(parse_table(html))
    assert md.splitlines()[-1] == "| 说明 | 不适用 |  |"


def test_markdown_escapes_pipe():
    t = parse_table("<table><tr><td>A</td><td>B</td></tr><tr><td>x|y</td><td>1</td></tr></table>")
    md = to_markdown(t)
    assert md.splitlines()[0] == "| A | B |"
    assert "x\\|y" in md


def test_split_rows_respects_budget_and_keeps_all_rows():
    rows = "".join(f"<tr><td>项目{i}</td><td>{i * 1000}</td></tr>" for i in range(40))
    t = parse_table(f"<table><tr><td>项目</td><td>金额</td></tr>{rows}</table>")
    groups = split_rows(t, max_chars=120)
    assert len(groups) > 1
    all_rows = []
    for g in groups:
        all_rows.extend(g)
    assert all_rows == list(range(1, 41))
    for g in groups:
        assert sum(len(x) + 1 for x in linearize_rows(t, g)) <= 120


def test_merge_html_drops_repeated_header():
    a = "<table><tr><td>项目</td><td>本期</td></tr><tr><td>营业收入</td><td>1</td></tr></table>"
    b = "<table><tr><td>项目</td><td>本期</td></tr><tr><td>净利润</td><td>2</td></tr></table>"
    t = parse_table(merge_html(a, b))
    assert t["grid"] == [["项目", "本期"], ["营业收入", "1"], ["净利润", "2"]]


def test_merge_html_keeps_rows_without_header():
    a = "<table><tr><td>项目</td><td>本期</td></tr><tr><td>营业收入</td><td>1</td></tr></table>"
    b = "<table><tr><td>净利润</td><td>2</td></tr></table>"
    t = parse_table(merge_html(a, b))
    assert t["grid"][-1] == ["净利润", "2"]
    assert len(t["grid"]) == 3
