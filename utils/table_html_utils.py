"""HTML 表格解析：展开 rowspan/colspan 成规则网格，识别表头，输出 Markdown（展示）与线性化文本（向量）。

只用标准库 html.parser，行为可控；不引入 pandas。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from html.parser import HTMLParser

_WS = re.compile(r"\s+")


@dataclass
class _Cell:
    text: str
    rowspan: int = 1
    colspan: int = 1
    is_th: bool = False


@dataclass
class _Row:
    cells: list[_Cell] = field(default_factory=list)
    in_thead: bool = False


def _span(value: str | None) -> int:
    try:
        return max(1, int(value or 1))
    except ValueError:
        return 1


class _TableParser(HTMLParser):
    """收集第一层表格的行与单元格；嵌套表格的文字并入外层单元格。"""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[_Row] = []
        self._depth = 0
        self._in_thead = False
        self._cell: _Cell | None = None
        self._buf: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        a = dict(attrs)
        if tag == "table":
            self._depth += 1
            return
        if self._depth > 1:
            if tag in ("br", "p", "div", "td", "th", "tr"):
                self._buf.append(" ")
            return
        if tag == "thead":
            self._in_thead = True
        elif tag == "tr":
            self.rows.append(_Row(in_thead=self._in_thead))
        elif tag in ("td", "th"):
            if not self.rows:
                self.rows.append(_Row(in_thead=self._in_thead))
            self._finish_cell()
            self._cell = _Cell("", _span(a.get("rowspan")), _span(a.get("colspan")), tag == "th")
            self._buf = []
        elif tag in ("br", "p", "div"):
            self._buf.append(" ")

    def handle_endtag(self, tag: str) -> None:
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

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._buf.append(data)

    def _finish_cell(self) -> None:
        if self._cell is not None:
            self._cell.text = _WS.sub(" ", "".join(self._buf)).strip()
            self.rows[-1].cells.append(self._cell)
            self._cell = None
            self._buf = []


@dataclass
class Table:
    grid: list[list[str]]
    header_rows: int
    # 与 grid 同形：每个位置来自哪个源单元格；同一行相邻位置 origin 相同说明是 colspan 展开的副本
    origin: list[list[int]] = field(default_factory=list)
    # 键值表（如基金“产品概况”：基金简称 | 华夏债券 | 基金代码 | 001001），按 (键, 值) 成对线性化
    kv: bool = False

    @property
    def n_cols(self) -> int:
        return len(self.grid[0]) if self.grid else 0

    @property
    def data_rows(self) -> list[list[str]]:
        return self.grid[self.header_rows :]

    def column_names(self) -> list[str]:
        """多层表头逐列自上而下拼接，去掉 colspan 展开产生的重复。首列若是单位说明，列名改为“项目”。"""
        names = []
        for j in range(self.n_cols):
            parts: list[str] = []
            for r in range(self.header_rows):
                v = self.grid[r][j]
                if v and v not in parts:
                    parts.append(v)
            names.append("·".join(parts))
        if names and _UNIT_HEADER.match(names[0]):
            names[0] = "项目"
        return names

    @property
    def unit_hint(self) -> str:
        """表头首格里的单位说明，如“(人民币百万元,特别注明除外)”。"""
        if self.header_rows and self.grid and _UNIT_HEADER.match(self.grid[0][0]):
            return self.grid[0][0]
        return ""


_UNIT_HEADER = re.compile(r"^[（(][^（()）]*(元|币|股|%|单位)[^（()）]*[)）]$")


def parse_table(html: str) -> Table:
    p = _TableParser()
    p.feed(html)
    p.close()
    rows = [r for r in p.rows if r.cells]
    grid, origin = _expand(rows)
    header_rows = _detect_header_rows(rows, grid)
    if header_rows and not _looks_like_header(grid[:header_rows]):
        header_rows = 0
    table = Table(grid=grid, header_rows=header_rows, origin=origin)
    table.kv = header_rows == 0 and _looks_like_kv(table)
    return table


_NUMERIC = re.compile(r"^[-+−(（]?[\d,，]+(\.\d+)?%?[)）]?$")


def _looks_like_header(rows: list[list[str]]) -> bool:
    """候选表头中出现数值或长文本时，它更可能是数据行（如键值表的首行）。"""
    for row in rows:
        for v in row:
            if _NUMERIC.match(v.replace(" ", "")) or len(v) > 30:
                return False
    return True


def _looks_like_kv(table: Table) -> bool:
    """偶数列、每行偶数位置都是短标签时视为键值表。"""
    if table.n_cols % 2 or len(table.grid) < 1:
        return False
    for i, row in enumerate(table.grid):
        vals = _row_values(table, i)
        for j in range(0, table.n_cols, 2):
            key = vals[j]
            if j > 0 and not key and not any(vals[j:]):
                break  # 行尾被合并单元格占满
            if not key or len(key) > 40 or _NUMERIC.match(key.replace(" ", "")):
                return False
    return True


def _expand(rows: list[_Row]) -> tuple[list[list[str]], list[list[int]]]:
    """按占位表展开合并单元格：被 rowspan / colspan 占住的位置由源单元格文本填充。"""
    occupied: dict[tuple[int, int], tuple[str, int]] = {}
    grid: list[list[str]] = []
    origin: list[list[int]] = []
    cell_id = 0
    for i, row in enumerate(rows):
        out: dict[int, tuple[str, int]] = {}
        j = 0
        for cell in row.cells:
            while (i, j) in occupied:
                out[j] = occupied.pop((i, j))
                j += 1
            cell_id += 1
            for dc in range(cell.colspan):
                out[j + dc] = (cell.text, cell_id)
                for dr in range(1, cell.rowspan):
                    occupied[(i + dr, j + dc)] = (cell.text, cell_id)
            j += cell.colspan
        # 行内单元格之后仍被上方 rowspan 占住的列
        for key in sorted(k for k in occupied if k[0] == i):
            out[key[1]] = occupied.pop(key)
        width = max(out) + 1 if out else 0
        grid.append([out[c][0] if c in out else "" for c in range(width)])
        origin.append([out[c][1] if c in out else 0 for c in range(width)])
    # rowspan 超出表格末尾的部分丢弃；补齐列宽
    width = max((len(r) for r in grid), default=0)
    grid = [r + [""] * (width - len(r)) for r in grid]
    origin = [r + [0] * (width - len(r)) for r in origin]
    return grid, origin


def _detect_header_rows(rows: list[_Row], grid: list[list[str]]) -> int:
    if len(grid) <= 1:
        return 0
    thead = sum(1 for r in rows if r.in_thead)
    if thead:
        return min(thead, len(grid) - 1)
    th_rows = 0
    for r in rows:
        if r.cells and all(c.is_th for c in r.cells):
            th_rows += 1
        else:
            break
    if th_rows:
        return min(th_rows, len(grid) - 1)
    # 首行有纵向合并时，表头行数取首行最大 rowspan
    first_span = max(c.rowspan for c in rows[0].cells)
    return min(first_span, len(grid) - 1)



def _data_row_indices(table: Table) -> list[int]:
    return list(range(table.header_rows, len(table.grid)))


def row_values(table: Table, i: int) -> list[str]:
    """第 i 行各列的值，横向合并产生的副本位置为空。"""
    return _row_values(table, i)


def is_spanning_row(table: Table, i: int) -> bool:
    return _is_spanning_row(table, i)


def _row_values(table: Table, i: int) -> list[str]:
    """去掉 colspan 展开产生的横向副本后，按列返回 (列号, 值)；空值与副本位置置空。"""
    row, org = table.grid[i], table.origin[i] if table.origin else None
    out = []
    for j, v in enumerate(row):
        dup = org is not None and j > 0 and org[j] == org[j - 1] and org[j] != 0
        out.append("" if dup else v)
    return out


def _is_spanning_row(table: Table, i: int) -> bool:
    """整行只有一个源单元格（跨全部列），通常是分组标题行。"""
    vals = [v for v in _row_values(table, i) if v]
    return len(vals) == 1 and table.n_cols > 1 and sum(1 for v in table.grid[i] if v) > 1


def _is_subheader_row(table: Table, i: int, first_col: str) -> bool:
    """表体中间的第二组表头，如季报主表中的“| | 本报告期末 | 上年度末 | 增减 |”。"""
    vals = _row_values(table, i)
    filled = [v for v in vals if v]
    if len(filled) < 2 or (vals[0] and vals[0] != first_col):
        return False
    return all(not _NUMERIC.match(v.replace(" ", "")) and v not in ("-", "—") and len(v) <= 30 for v in filled)


def row_columns(table: Table) -> dict[int, list[str]]:
    """每个数据行适用的列名：遇到表体中的子表头行后，其后各行改用子表头。子表头行本身不在结果中。"""
    cols = table.column_names()
    out: dict[int, list[str]] = {}
    for i in _data_row_indices(table):
        if table.header_rows and not table.kv and _is_subheader_row(table, i, cols[0]):
            vals = _row_values(table, i)
            cols = [v or cols[j] for j, v in enumerate(vals)]
            continue
        out[i] = cols
    return out


def linearize_rows(table: Table, rows: list[int] | None = None) -> list[str]:
    """每个数据行输出一行“列名=值”，供计算向量。rows 为行号列表，缺省为全部数据行。"""
    columns = row_columns(table)
    out: list[str] = []
    for i in _data_row_indices(table) if rows is None else rows:
        vals = _row_values(table, i)
        if not any(vals) or i not in columns:
            continue
        cols = columns[i]
        if _is_spanning_row(table, i):
            out.append(next(v for v in vals if v))
            continue
        if table.kv:
            pairs = [(vals[j], vals[j + 1]) for j in range(0, table.n_cols, 2)]
            parts = [f"{k}={v}" if k and v else (k or v) for k, v in pairs if k or v]
        else:
            parts = [f"{name}={val}" if name and name != val else val for name, val in zip(cols, vals) if val]
        out.append("；".join(parts))
    return out


def _md_cell(v: str) -> str:
    return v.replace("|", "\\|").replace("\n", " ")


def to_markdown(table: Table, rows: list[int] | None = None) -> str:
    """展示用 Markdown。纵向合并的值逐行重复，横向合并只在首列显示。rows 为数据行号列表，缺省为全部。"""
    if not table.grid:
        return ""
    head = table.column_names() if table.header_rows else [""] * table.n_cols
    body = _data_row_indices(table) if rows is None else rows
    lines = ["| " + " | ".join(_md_cell(h) for h in head) + " |", "|" + "---|" * table.n_cols]
    lines += ["| " + " | ".join(_md_cell(v) for v in _row_values(table, i)) + " |" for i in body]
    return "\n".join(lines)


def split_rows(table: Table, max_chars: int) -> list[list[int]]:
    """按线性化长度把数据行分组，每组不超过 max_chars（单行超长时独占一组）。"""
    groups: list[list[int]] = []
    cur: list[int] = []
    size = 0
    for i in _data_row_indices(table):
        n = sum(len(x) + 1 for x in linearize_rows(table, [i]))
        if cur and size + n > max_chars:
            groups.append(cur)
            cur, size = [], 0
        cur.append(i)
        size += n
    if cur:
        groups.append(cur)
    return groups


_TR = re.compile(r"<tr\b.*?</tr>", re.S | re.I)
_TABLE_INNER = re.compile(r"^\s*<table\b[^>]*>(.*)</table>\s*$", re.S | re.I)


def merge_html(first: str, second: str) -> str:
    """把跨页续表 second 的行接到 first 之后；续表重复了表头时去掉重复的表头行。"""
    a, b = parse_table(first), parse_table(second)
    drop = 0
    if a.header_rows and b.grid[: a.header_rows] == a.grid[: a.header_rows]:
        drop = a.header_rows
    m1, m2 = _TABLE_INNER.match(first), _TABLE_INNER.match(second)
    inner1 = m1.group(1) if m1 else first
    inner2 = m2.group(1) if m2 else second
    rows2 = _TR.findall(inner2)[drop:]
    return f"<table>{inner1}{''.join(rows2)}</table>"
