"""First run downloads bge-small (~130MB) — takes ~30s. Cached after that."""
from pathlib import Path

import numpy as np

from app import storage
from app.storage import get_engine, reset_engine, EmbedCache, session
from app.embeddings import Embedder


def test_embed_shape_and_cache(tmp_path: Path):
    reset_engine()
    get_engine(tmp_path / "t.db")

    e = Embedder()
    v1 = e.embed(["deep learning is powerful", "cats sleep a lot"])
    assert v1.shape == (2, 384)
    assert v1.dtype == np.float32
    # bge outputs are L2-normalized when normalize_embeddings=True
    assert np.allclose(np.linalg.norm(v1, axis=1), 1.0, atol=1e-3)

    # Cache populated
    with session() as s:
        assert s.query(EmbedCache).count() == 2

    # Second call: same vectors, no re-encoding needed
    v2 = e.embed(["cats sleep a lot", "deep learning is powerful"])
    assert np.allclose(v2[0], v1[1])
    assert np.allclose(v2[1], v1[0])

    # Semantic sanity: two similar sentences closer than two unrelated ones
    sim = Embedder()
    vs = sim.embed([
        "dogs bark loudly at strangers",
        "canines vocalize when they see unfamiliar people",
        "the stock market closed higher today",
    ])
    close = float(vs[0] @ vs[1])
    far = float(vs[0] @ vs[2])
    assert close > far

    reset_engine()
