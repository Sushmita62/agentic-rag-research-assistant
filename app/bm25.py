"""BM25 keyword retrieval. Rebuilds from full SQLite corpus on each ingest.

Sparse retrieval catches what dense misses: acronyms, model names, dataset
names, exact phrases. RRF-fused with FAISS in the hybrid step.
"""
from __future__ import annotations

import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from app.config import settings


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    def __init__(self, path: Path | None = None):
        self.path = path or settings.bm25_path
        self.bm25: BM25Okapi | None = None
        self.faiss_ids: list[int] = []
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            with open(self.path, "rb") as f:
                d = pickle.load(f)
            self.bm25 = d["bm25"]
            self.faiss_ids = d["faiss_ids"]

    def rebuild(self, corpus: list[tuple[int, str]]) -> None:
        """corpus = [(faiss_id, text), ...] — the full current corpus."""
        if not corpus:
            self.bm25 = None
            self.faiss_ids = []
            return
        self.faiss_ids = [fid for fid, _ in corpus]
        tokens = [tokenize(t) for _, t in corpus]
        self.bm25 = BM25Okapi(tokens)

    def search(self, query: str, k: int = 30) -> list[tuple[int, float]]:
        if self.bm25 is None or not self.faiss_ids:
            return []
        scores = self.bm25.get_scores(tokenize(query))
        # Return top-k by rank. No score threshold: BM25 IDF can be 0 or negative
        # for terms in most/all docs; downstream RRF uses rank, and rerank filters.
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(self.faiss_ids[i], float(scores[i])) for i in top_idx]

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "wb") as f:
            pickle.dump({"bm25": self.bm25, "faiss_ids": self.faiss_ids}, f)
