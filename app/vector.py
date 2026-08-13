"""FAISS IndexIDMap(IndexFlatIP). Exact search — right choice up to ~1M vectors."""
from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np

from app.config import settings


class VectorIndex:
    def __init__(self, dim: int = 384, path: Path | None = None):
        self.dim = dim
        self.path = path or settings.faiss_path
        self.index = self._load_or_new()

    def _load_or_new(self):
        if self.path.exists():
            idx = faiss.read_index(str(self.path))
            assert idx.d == self.dim, f"dim mismatch: index={idx.d} expected={self.dim}"
            return idx
        return faiss.IndexIDMap(faiss.IndexFlatIP(self.dim))

    @property
    def ntotal(self) -> int:
        return int(self.index.ntotal)

    def add(self, vectors: np.ndarray) -> list[int]:
        """Append vectors, return the faiss_ids assigned (sequential from current ntotal)."""
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        assert vectors.ndim == 2 and vectors.shape[1] == self.dim
        start = self.ntotal
        ids = np.arange(start, start + len(vectors), dtype=np.int64)
        self.index.add_with_ids(vectors, ids)
        return ids.tolist()

    def search(self, query: np.ndarray, k: int = 10) -> list[tuple[int, float]]:
        if query.ndim == 1:
            query = query[None, :]
        query = np.ascontiguousarray(query, dtype=np.float32)
        scores, ids = self.index.search(query, k)
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.path))
