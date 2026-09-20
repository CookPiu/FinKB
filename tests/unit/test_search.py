from utils.search_utils import build_filter, new_hit


def make_row(chunk_id, score):
    """构造一行 Milvus 混合检索结果"""
    return {"chunk_id": chunk_id, "score": score, "doc_id": "d", "kind": "text", "content_type": "c",
            "section_path": "", "page_start": 1, "page_end": 1, "derived": False, "text": "t"}


def test_build_filter():
    assert build_filter(kinds=["table"], content_types=["公司定期报告"]) == (
        'kind in ["table"] and content_type in ["公司定期报告"]'
    )
    assert build_filter(doc_ids=["d1", "d2"]) == 'doc_id in ["d1", "d2"]'
    assert build_filter() == ""


def test_build_filter_escapes_quotes():
    # 文档 ID 里的双引号必须转义，否则表达式会被截断
    assert build_filter(doc_ids=['a"b']) == 'doc_id in ["a\\"b"]'


def test_new_hit_keeps_fields_and_leaves_source_empty():
    hit = new_hit(make_row("c1", 0.016393))
    assert hit["chunk_id"] == "c1" and hit["kind"] == "text" and hit["text"] == "t"
    assert hit["score"] == 0.01639  # 融合分保留 5 位
    # 来源元数据由 attach_doc_meta 补，建立时为空
    assert hit["file_name"] == "" and hit["document_title"] == "" and hit["source_path"] == ""
