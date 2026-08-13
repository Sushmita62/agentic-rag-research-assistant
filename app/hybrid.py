"""Hybrid retrieval: dense (FAISS) + sparse (BM25) fused with Reciprocal Rank Fusion.

RRF combines rankings from heterogeneous rankers without needing to normalize scores.
Formula: score(doc) = Σ 1 / (k_const + rank_i(doc))  over all rankers.
"""
from __future__ import annotations

from sqlalchemy import select

from app.bm25 import BM25Index
from app.embeddings import Embedder
from app.storage import Chunk, session
from app.vector import VectorIndex


def rrf_fuse(
    rankings: list[list[tuple[int, float]]],
    k: int = 20,
    k_const: int = 60,
) -> list[tuple[int, float]]:
    """Each ranking is [(faiss_id, score), ...] ordered best→worst. Returns top-k fused."""
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, (fid, _) in enumerate(ranking):
            fused[fid] = fused.get(fid, 0.0) + 1.0 / (k_const + rank + 1)
    return sorted(fused.items(), key=lambda x: x[1], reverse=True)[:k]


class HybridRetriever:
    def __init__(
        self,
        vector_index: VectorIndex | None = None,
        bm25_index: BM25Index | None = None,
        embedder: Embedder | None = None,
    ):
        self.vector = vector_index or VectorIndex()
        self.bm25 = bm25_index or BM25Index()
        self.embedder = embedder or Embedder()

    def retrieve(
        self,
        query: str,
        k: int = 20,
        per_source: int = 30,
        paper_ids: list[str] | None = None,
    ) -> list[tuple[int, float]]:
        dense_vec = self.embedder.embed([query])
        dense_hits = self.vector.search(dense_vec, k=per_source)
        sparse_hits = self.bm25.search(query, k=per_source)
        # Oversample when we'll post-filter, so filtered top-k isn't starved.
        fuse_k = k * 3 if paper_ids else k
        fused = rrf_fuse([dense_hits, sparse_hits], k=fuse_k)
        if paper_ids:
            with session() as s:
                allowed = {r[0] for r in s.execute(
                    select(Chunk.faiss_id).where(
                        Chunk.paper_id.in_(paper_ids), Chunk.faiss_id.isnot(None)
                    )
                ).all()}
            fused = [(fid, sc) for fid, sc in fused if fid in allowed]
        return fused[:k]
