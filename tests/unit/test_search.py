from utils.search_utils import build_filter, rrf_fuse


def make_row(chunk_id, score):
    """构造一行 Milvus 检索结果"""
    return {"chunk_id": chunk_id, "score": score, "doc_id": "d", "version": 1, "kind": "text", "content_type": "c",
            "section_path": "", "page_start": 1, "page_end": 1, "derived": False, "text": "t", "entity_ids": []}


def test_build_filter_and_rrf():
    assert build_filter(kinds=["table"], content_types=["公司定期报告"]) == (
        'kind in ["table"] and content_type in ["公司定期报告"]'
    )
    assert build_filter(entity_ids=["e1"]) == 'ARRAY_CONTAINS_ANY(entity_ids, ["e1"])'
    dense = [make_row("a", 0.9), make_row("b", 0.8)]
    sparse = [make_row("b", 0.3)]
    fused = rrf_fuse(dense, sparse)
    assert [h["chunk_id"] for h in fused] == ["b", "a"]  # 两路都召回的排前面
    assert fused[0]["score_dense"] == 0.8 and fused[0]["score_sparse"] == 0.3 and fused[1]["score_sparse"] is None
