"""Full retrieval pipeline: hybrid (BM25 + FAISS) → RRF → cross-encoder rerank.

Usage:
    python scripts/search.py "your question"
    python scripts/search.py "your question" 10
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select

from app.hybrid import HybridRetriever
from app.reranker import Reranker
from app.storage import Chunk, Paper, get_engine, session


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        return
    query = sys.argv[1]
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    get_engine()
    retriever = HybridRetriever()

    if retriever.vector.ntotal == 0:
        print("Index is empty. Run: python scripts/ingest_folder.py data/pdfs")
        return

    # Stage 1: hybrid (BM25 + dense, fused via RRF) → top-20 candidates
    candidates = retriever.retrieve(query, k=20, per_source=30)
    if not candidates:
        print("No results.")
        return

    faiss_ids = [fid for fid, _ in candidates]
    rrf_scores = dict(candidates)

    with session() as s:
        rows = s.execute(
            select(Chunk, Paper).join(Paper).where(Chunk.faiss_id.in_(faiss_ids))
        ).all()
        by_fid = {c.faiss_id: {"text": c.text, "page": c.page,
                               "section": c.section, "title": p.title}
                  for c, p in rows}

    # Stage 2: cross-encoder rerank → top-k
    pairs = [(fid, by_fid[fid]["text"]) for fid in faiss_ids if fid in by_fid]
    reranked = Reranker().rerank(query, pairs, top_k=k)

    print(f"\nQuery: {query!r}   (index: {retriever.vector.ntotal} chunks)\n")
    for rank, (fid, ce_score) in enumerate(reranked, 1):
        r = by_fid[fid]
        snippet = " ".join(r["text"].split())[:240]
        print(f"#{rank}  ce={ce_score:+.3f}  rrf={rrf_scores[fid]:.4f}  "
              f"p.{r['page']}  [{r['section']}]  {r['title'][:60]}")
        print(f"     {snippet}...\n")


if __name__ == "__main__":
    main()
