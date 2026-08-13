from app.hybrid import rrf_fuse


def test_rrf_prefers_docs_ranked_by_both():
    dense = [(10, 0.9), (20, 0.8), (30, 0.7)]                  # id 10 top
    sparse = [(20, 5.0), (10, 4.0), (40, 3.0)]                 # id 20 top
    fused = rrf_fuse([dense, sparse], k=4)
    ids = [fid for fid, _ in fused]
    # Both 10 and 20 appear in both rankings → they should top the list
    assert set(ids[:2]) == {10, 20}
    # 30 and 40 each appear only in one ranking
    assert set(ids[2:]) == {30, 40}


def test_rrf_score_uses_rank_not_raw_score():
    # Symmetric ranks (doc 1 first in a, doc 2 first in b) → identical fused scores
    # regardless of raw score magnitude.
    a = [(1, 999.0), (2, 0.0001)]
    b = [(2, 0.0001), (1, 999.0)]
    fused = dict(rrf_fuse([a, b]))
    assert abs(fused[1] - fused[2]) < 1e-9


def test_rrf_handles_empty_rankings():
    assert rrf_fuse([[], []]) == []
    assert rrf_fuse([[(1, 1.0)], []]) == [(1, 1.0 / 61)]
