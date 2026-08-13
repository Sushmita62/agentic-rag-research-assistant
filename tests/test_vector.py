from pathlib import Path

import numpy as np

from app.vector import VectorIndex


def _unit(v):
    return v / np.linalg.norm(v)


def test_add_search_persist(tmp_path: Path):
    idx = VectorIndex(dim=4, path=tmp_path / "test.index")

    # Three orthogonal-ish unit vectors
    vecs = np.stack([
        _unit(np.array([1, 0, 0, 0], dtype=np.float32)),
        _unit(np.array([0, 1, 0, 0], dtype=np.float32)),
        _unit(np.array([0.9, 0.1, 0, 0], dtype=np.float32)),   # close to vec[0]
    ])
    ids = idx.add(vecs)
    assert ids == [0, 1, 2]
    assert idx.ntotal == 3

    # Query near vec[0] → top hit is id 0, then id 2 (close), then id 1 (far)
    q = _unit(np.array([1, 0.05, 0, 0], dtype=np.float32))
    hits = idx.search(q, k=3)
    assert [h[0] for h in hits] == [0, 2, 1]
    assert hits[0][1] > hits[1][1] > hits[2][1]

    # Persist, reload, same result
    idx.save()
    idx2 = VectorIndex(dim=4, path=tmp_path / "test.index")
    assert idx2.ntotal == 3
    hits2 = idx2.search(q, k=3)
    assert [h[0] for h in hits2] == [0, 2, 1]


def test_ids_are_sequential_across_batches(tmp_path: Path):
    idx = VectorIndex(dim=4, path=tmp_path / "t.index")
    idx.add(np.random.rand(2, 4).astype(np.float32))
    ids = idx.add(np.random.rand(3, 4).astype(np.float32))
    assert ids == [2, 3, 4]
