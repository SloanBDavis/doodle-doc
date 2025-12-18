from __future__ import annotations

from doodle_doc.search.fusion import reciprocal_rank_fusion


def test_rrf_single_list():
    results = [[("a", 1.0), ("b", 0.8), ("c", 0.5)]]
    fused = reciprocal_rank_fusion(results, k=60)

    assert fused[0][0] == "a"
    assert fused[1][0] == "b"
    assert fused[2][0] == "c"


def test_rrf_multiple_lists():
    list1 = [("a", 1.0), ("b", 0.8), ("c", 0.5)]
    list2 = [("b", 1.0), ("a", 0.8), ("d", 0.5)]

    fused = reciprocal_rank_fusion([list1, list2], k=60)

    # "a" and "b" should be top 2 (appear in both lists)
    top_keys = {fused[0][0], fused[1][0]}
    assert "a" in top_keys
    assert "b" in top_keys


def test_rrf_empty_list():
    fused = reciprocal_rank_fusion([], k=60)
    assert fused == []
