from utils.model_output_utils import parse_meta, split_body_and_meta


def parse(raw: str):
    """按节点里的用法：先切分，再解析元信息"""
    body, meta_text = split_body_and_meta(raw)
    return body, parse_meta(meta_text, body)


def test_normal_output_is_split_and_parsed():
    raw = '华夏债券C的托管费固定费率为0.20% [E5]。\n<<<META>>>\n{"kind": "answer", "cited": [5], "notices": ["risk"]}'
    body, meta = parse(raw)
    assert body == "华夏债券C的托管费固定费率为0.20% [E5]。"
    assert meta == {"kind": "answer", "cited": [5], "notices": ["risk"], "meta_ok": True}


def test_meta_can_be_wrapped_in_code_fence():
    raw = '不提供投资建议。\n<<<META>>>\n```json\n{"kind": "decline_advice", "cited": [], "notices": ["risk"]}\n```'
    body, meta = parse(raw)
    assert body == "不提供投资建议。"
    assert meta["kind"] == "decline_advice" and meta["meta_ok"]


def test_missing_meta_falls_back_to_answer_and_cited_from_body():
    body, meta = parse("营业收入为 100 元[E1]，净利润为 10 元【E3】。")
    assert body.endswith("。")
    # 没有元信息时按普通回答处理，引用编号从正文里捡
    assert meta["kind"] == "answer" and meta["cited"] == [1, 3] and meta["meta_ok"] is False


def test_broken_json_falls_back():
    body, meta = parse('正文[E2]\n<<<META>>>\n{"kind": "answer", "cited": [2,,}')
    assert meta["meta_ok"] is False and meta["kind"] == "answer" and meta["cited"] == [2]


def test_unknown_kind_falls_back():
    _, meta = parse('正文\n<<<META>>>\n{"kind": "随便编的", "cited": [], "notices": []}')
    assert meta["meta_ok"] is False and meta["kind"] == "answer"


def test_dirty_fields_are_cleaned():
    _, meta = parse('正文\n<<<META>>>\n{"kind": "refuse", "cited": ["2", 3, "x", true], "notices": ["risk", "无关", "risk"]}')
    # 字符串数字保留、非数字丢弃、布尔值不算编号；提示语去重且只保留已知标记
    assert meta["cited"] == [2, 3] and meta["notices"] == ["risk"] and meta["kind"] == "refuse"


def test_notices_not_a_list():
    _, meta = parse('正文\n<<<META>>>\n{"kind": "answer", "cited": [], "notices": "risk"}')
    assert meta["notices"] == []
