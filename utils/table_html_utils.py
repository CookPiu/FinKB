"""
HTML 表格解析：展开 rowspan/colspan 成规则网格，识别表头，输出 Markdown（展示用）与线性化文本（算向量用）。
只用标准库 html.parser，行为可控；不引入 pandas。

parse_table 返回的表格是一个 dict：
- grid：二维列表，合并单元格已展开，每行等宽；
- origin：与 grid 同形，每个位置来自第几个源单元格（0 表示空位）；同一行相邻位置 origin 相同说明是 colspan 展开的副本；
- header_rows：表头行数；n_cols：列数；
- kv：是否键值表（如基金“产品概况”：基金简称 | 华夏债券 | 基金代码 | 001001），按 (键, 值) 成对线性化。
"""
import re
from html.parser import HTMLParser

_WS = re.compile(r"\s+")
# 表头首格的单位说明，如“(人民币百万元,特别注明除外)”
_UNIT_HEADER = re.compile(r"^[（(][^（()）]*(元|币|股|%|单位)[^（()）]*[)）]$")
# 数值单元格：千分位、小数、百分号、括号负数
_NUMERIC = re.compile(r"^[-+−(（]?[\d,，]+(\.\d+)?%?[)）]?$")
_TR = re.compile(r"<tr\b.*?</tr>", re.S | re.I)
_TABLE_INNER = re.compile(r"^\s*<table\b[^>]*>(.*)</table>\s*$", re.S | re.I)


def _parse_span(value) -> int:
    """解析 rowspan / colspan 属性，缺失或非法时按 1 处理"""
    try:
        return max(1, int(value or 1))
    except ValueError:
        return 1


class _TableParser(HTMLParser):
    """
    收集第一层表格的行与单元格；嵌套表格的文字并入外层单元格。
    行：{"cells": [单元格], "in_thead": bool}；单元格：{"text", "rowspan", "colspan", "is_th"}
    """

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.rows = []
        self._depth = 0  # 表格嵌套深度
        self._in_thead = False
        self._cell = None  # 正在收集文字的单元格
        self._buf = []

    def handle_starttag(self, tag, attrs):
        attr_map = dict(attrs)
        if tag == "table":
            self._depth += 1
            return
        if self._depth > 1:
            # 嵌套表格：结构标签只当作分隔空格
            if tag in ("br", "p", "div", "td", "th", "tr"):
                self._buf.append(" ")
            return
        if tag == "thead":
            self._in_thead = True
        elif tag == "tr":
            self.rows.append({"cells": [], "in_thead": self._in_thead})
        elif tag in ("td", "th"):
            if not self.rows:
                self.rows.append({"cells": [], "in_thead": self._in_thead})
            self._finish_cell()
            self._cell = {
                "text": "",
                "rowspan": _parse_span(attr_map.get("rowspan")),
                "colspan": _parse_span(attr_map.get("colspan")),
                "is_th": tag == "th",
            }
            self._buf = []
        elif tag in ("br", "p", "div"):
            self._buf.append(" ")

    def handle_endtag(self, tag):
        if tag == "table":
            self._depth -= 1
            if self._depth == 0:
                self._finish_cell()
            return
        if self._depth > 1:
            return
        if tag in ("td", "th"):
            self._finish_cell()
        elif tag == "thead":
            self._in_thead = False

    def handle_data(self, data):
        if self._cell is not None:
            self._buf.append(data)

    def _finish_cell(self):
        """结束当前单元格：合并空白后追加到最后一行"""
        if self._cell is not None:
            self._cell["text"] = _WS.sub(" ", "".join(self._buf)).strip()
            self.rows[-1]["cells"].append(self._cell)
            self._cell = None
            self._buf = []


def parse_table(html: str) -> dict:
    """
    解析 HTML 表格
    :param html: <table>...</table> 文本
    :return: 表格 dict（键见模块说明）
    """
    parser = _TableParser()
    parser.feed(html)
    parser.close()
    rows = [r for r in parser.rows if r["cells"]]
    grid, origin = _expand_spans(rows)
    header_rows = _detect_header_rows(rows, grid)
    if header_rows and not _looks_like_header(grid[:header_rows]):
        header_rows = 0
    if grid:
        n_cols = len(grid[0])
    else:
        n_cols = 0
    table = {"grid": grid, "origin": origin, "header_rows": header_rows, "n_cols": n_cols, "kv": False}
    table["kv"] = header_rows == 0 and _looks_like_kv(table)
    return table


def _expand_spans(rows: list):
    """
    按占位表展开合并单元格：被 rowspan / colspan 占住的位置由源单元格文本填充
    :return: (grid, origin)，两者同形且每行等宽
    """
    occupied = {}  # (行, 列) -> (文本, 源单元格编号)，记录被上方 rowspan 占住的位置
    grid = []
    origin = []
    cell_id = 0
    for i, row in enumerate(rows):
        out = {}  # 列号 -> (文本, 源单元格编号)
        j = 0
        for cell in row["cells"]:
            while (i, j) in occupied:
                out[j] = occupied.pop((i, j))
                j += 1
            cell_id += 1
            for dc in range(cell["colspan"]):
                out[j + dc] = (cell["text"], cell_id)
                for dr in range(1, cell["rowspan"]):
                    occupied[(i + dr, j + dc)] = (cell["text"], cell_id)
            j += cell["colspan"]
        # 行内单元格之后仍被上方 rowspan 占住的列
        for key in sorted([k for k in occupied if k[0] == i]):
            out[key[1]] = occupied.pop(key)
        if out:
            width = max(out) + 1
        else:
            width = 0
        texts = []
        ids = []
        for c in range(width):
            if c in out:
                texts.append(out[c][0])
                ids.append(out[c][1])
            else:
                texts.append("")
                ids.append(0)
        grid.append(texts)
        origin.append(ids)
    # rowspan 超出表格末尾的部分丢弃；补齐列宽
    width = 0
    for r in grid:
        if len(r) > width:
            width = len(r)
    grid = [r + [""] * (width - len(r)) for r in grid]
    origin = [r + [0] * (width - len(r)) for r in origin]
    return grid, origin


def _detect_header_rows(rows: list, grid: list) -> int:
    """表头行数：优先 thead，其次开头连续的全 th 行，最后取首行最大 rowspan；至少留一行数据"""
    if len(grid) <= 1:
        return 0
    thead = sum(1 for r in rows if r["in_thead"])
    if thead:
        return min(thead, len(grid) - 1)
    th_rows = 0
    for r in rows:
        if r["cells"] and all(c["is_th"] for c in r["cells"]):
            th_rows += 1
        else:
            break
    if th_rows:
        return min(th_rows, len(grid) - 1)
    # 首行有纵向合并时，表头行数取首行最大 rowspan
    first_span = max(c["rowspan"] for c in rows[0]["cells"])
    return min(first_span, len(grid) - 1)


def _looks_like_header(rows: list) -> bool:
    """候选表头中出现数值或长文本时，它更可能是数据行（如键值表的首行）"""
    for row in rows:
        for v in row:
            if _NUMERIC.match(v.replace(" ", "")) or len(v) > 30:
                return False
    return True


def _looks_like_kv(table: dict) -> bool:
    """偶数列、每行偶数位置都是短标签时视为键值表"""
    n_cols = table["n_cols"]
    if n_cols % 2 or len(table["grid"]) < 1:
        return False
    for i in range(len(table["grid"])):
        vals = row_values(table, i)
        for j in range(0, n_cols, 2):
            key = vals[j]
            if j > 0 and not key and not any(vals[j:]):
                break  # 行尾被合并单元格占满
            if not key or len(key) > 40 or _NUMERIC.match(key.replace(" ", "")):
                return False
    return True


def get_column_names(table: dict) -> list:
    """多层表头逐列自上而下拼接，去掉 colspan 展开产生的重复。首列若是单位说明，列名改为“项目”"""
    names = []
    for j in range(table["n_cols"]):
        parts = []
        for r in range(table["header_rows"]):
            v = table["grid"][r][j]
            if v and v not in parts:
                parts.append(v)
        names.append("·".join(parts))
    if names and _UNIT_HEADER.match(names[0]):
        names[0] = "项目"
    return names


def get_unit_hint(table: dict) -> str:
    """表头首格里的单位说明，如“(人民币百万元,特别注明除外)”；没有返回空串"""
    grid = table["grid"]
    if table["header_rows"] and grid and _UNIT_HEADER.match(grid[0][0]):
        return grid[0][0]
    return ""


def _data_row_indices(table: dict) -> list:
    return list(range(table["header_rows"], len(table["grid"])))


def row_values(table: dict, i: int) -> list:
    """第 i 行各列的值，横向合并（colspan）产生的副本位置置空"""
    row = table["grid"][i]
    if table["origin"]:
        org = table["origin"][i]
    else:
        org = None
    out = []
    for j, v in enumerate(row):
        dup = org is not None and j > 0 and org[j] == org[j - 1] and org[j] != 0
        if dup:
            out.append("")
        else:
            out.append(v)
    return out


def is_spanning_row(table: dict, i: int) -> bool:
    """整行只有一个源单元格（跨全部列），通常是分组标题行"""
    vals = [v for v in row_values(table, i) if v]
    filled = [v for v in table["grid"][i] if v]
    return len(vals) == 1 and table["n_cols"] > 1 and len(filled) > 1


def _is_subheader_row(table: dict, i: int, first_col: str) -> bool:
    """表体中间的第二组表头，如季报主表中的“| | 本报告期末 | 上年度末 | 增减 |”"""
    vals = row_values(table, i)
    filled = [v for v in vals if v]
    if len(filled) < 2 or (vals[0] and vals[0] != first_col):
        return False
    for v in filled:
        if _NUMERIC.match(v.replace(" ", "")) or v in ("-", "—") or len(v) > 30:
            return False
    return True


def row_columns(table: dict) -> dict:
    """
    每个数据行适用的列名：遇到表体中的子表头行后，其后各行改用子表头
    :return: {行号: 列名列表}，子表头行本身不在结果中
    """
    cols = get_column_names(table)
    out = {}
    for i in _data_row_indices(table):
        if table["header_rows"] and not table["kv"] and _is_subheader_row(table, i, cols[0]):
            vals = row_values(table, i)
            cols = [v or cols[j] for j, v in enumerate(vals)]
            continue
        out[i] = cols
    return out


def linearize_rows(table: dict, rows=None) -> list:
    """
    每个数据行输出一行“列名=值”，供计算向量
    :param rows: 行号列表，None 表示全部数据行
    :return: 线性化文本列表（空行、子表头行不输出）
    """
    columns = row_columns(table)
    if rows is None:
        rows = _data_row_indices(table)
    out = []
    for i in rows:
        vals = row_values(table, i)
        if not any(vals) or i not in columns:
            continue
        if is_spanning_row(table, i):
            out.append([v for v in vals if v][0])
            continue
        if table["kv"]:
            parts = _linearize_kv_row(table, vals)
        else:
            parts = _linearize_normal_row(columns[i], vals)
        out.append("；".join(parts))
    return out


def _linearize_kv_row(table: dict, vals: list) -> list:
    """键值表的一行：按 (键, 值) 成对输出“键=值”，只有一边有值时原样输出"""
    parts = []
    for j in range(0, table["n_cols"], 2):
        key = vals[j]
        value = vals[j + 1]
        if key and value:
            parts.append(f"{key}={value}")
        elif key:
            parts.append(key)
        elif value:
            parts.append(value)
    return parts


def _linearize_normal_row(cols: list, vals: list) -> list:
    """普通表的一行：非空值输出“列名=值”，列名为空或与值相同时只输出值"""
    parts = []
    for name, val in zip(cols, vals):
        if not val:
            continue
        if name and name != val:
            parts.append(f"{name}={val}")
        else:
            parts.append(val)
    return parts


def _md_cell(v: str) -> str:
    return v.replace("|", "\\|").replace("\n", " ")


def _md_line(values: list) -> str:
    return "| " + " | ".join([_md_cell(v) for v in values]) + " |"


def to_markdown(table: dict, rows=None) -> str:
    """
    展示用 Markdown：纵向合并的值逐行重复，横向合并只在首列显示
    :param rows: 数据行号列表，None 表示全部数据行
    """
    if not table["grid"]:
        return ""
    if table["header_rows"]:
        head = get_column_names(table)
    else:
        head = [""] * table["n_cols"]
    if rows is None:
        rows = _data_row_indices(table)
    lines = [_md_line(head), "|" + "---|" * table["n_cols"]]
    for i in rows:
        lines.append(_md_line(row_values(table, i)))
    return "\n".join(lines)


def split_rows(table: dict, max_chars: int) -> list:
    """按线性化长度把数据行分组，每组不超过 max_chars（单行超长时独占一组）；返回行号列表的列表"""
    groups = []
    cur = []
    size = 0
    for i in _data_row_indices(table):
        n = sum(len(x) + 1 for x in linearize_rows(table, [i]))
        if cur and size + n > max_chars:
            groups.append(cur)
            cur = []
            size = 0
        cur.append(i)
        size += n
    if cur:
        groups.append(cur)
    return groups


def merge_html(first: str, second: str) -> str:
    """把跨页续表 second 的行接到 first 之后；续表重复了表头时去掉重复的表头行"""
    table_a = parse_table(first)
    table_b = parse_table(second)
    drop = 0
    header_rows = table_a["header_rows"]
    if header_rows and table_b["grid"][:header_rows] == table_a["grid"][:header_rows]:
        drop = header_rows
    match_first = _TABLE_INNER.match(first)
    match_second = _TABLE_INNER.match(second)
    if match_first:
        inner_first = match_first.group(1)
    else:
        inner_first = first
    if match_second:
        inner_second = match_second.group(1)
    else:
        inner_second = second
    rows_second = _TR.findall(inner_second)[drop:]
    return f"<table>{inner_first}{''.join(rows_second)}</table>"
