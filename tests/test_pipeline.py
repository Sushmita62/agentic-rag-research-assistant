"""End-to-end: ingest → hybrid retrieve → rerank returns the right chunk."""
from pathlib import Path

import pymupdf
from sqlalchemy import select

from app.bm25 import BM25Index
from app.embeddings import Embedder
from app.hybrid import HybridRetriever
from app.ingest import ingest_pdf
from app.reranker import Reranker
from app.storage import Chunk, get_engine, reset_engine, session
from app.vector import VectorIndex


def _make_paper(path: Path) -> None:
    doc = pymupdf.open()
    filler = ("Neural networks are trained with gradient descent methods "
              "using backpropagation over many epochs. ") * 6

    pages = [
        ("Introduction",
         "1 Introduction\n\n" + filler + " The MERZBOW-9 dataset was released in 2024."),
        ("Methods",
         "2 Methods\n\n" + filler + " We use scaled dot-product attention throughout."),
        ("Results",
         "3 Results\n\n" + filler + " Model accuracy reached 87.4 percent on the benchmark."),
    ]
    for _, text in pages:
        page = doc.new_page()
        rect = pymupdf.Rect(72, 72, page.rect.width - 72, page.rect.height - 72)
        page.insert_textbox(rect, text, fontsize=11)
    doc.save(path)
    doc.close()


def test_hybrid_plus_rerank_end_to_end(tmp_path: Path):
    reset_engine()
    get_engine(tmp_path / "app.db")

    pdf = tmp_path / "paper.pdf"
    _make_paper(pdf)

    vi = VectorIndex(dim=384, path=tmp_path / "faiss.index")
    bm = BM25Index(path=tmp_path / "bm25.pkl")
    emb = Embedder()

    ingest_pdf(pdf, vector_index=vi, embedder=emb, bm25_index=bm)

    retriever = HybridRetriever(vector_index=vi, bm25_index=bm, embedder=emb)

    # BM25 should nail the exact-term query (dense would miss "MERZBOW-9")
    cands = retriever.retrieve("MERZBOW-9 dataset", k=20)
    assert len(cands) > 0

    with session() as s:
        rows = s.execute(
            select(Chunk).where(Chunk.faiss_id.in_([f for f, _ in cands]))
        ).scalars().all()
        by_fid = {c.faiss_id: {"text": c.text, "section": c.section} for c in rows}

    pairs = [(fid, by_fid[fid]["text"]) for fid, _ in cands if fid in by_fid]

    top = Reranker().rerank("MERZBOW-9 dataset", pairs, top_k=3)
    top_chunk = by_fid[top[0][0]]
    assert "MERZBOW-9" in top_chunk["text"]
    assert top_chunk["section"] == "introduction"

    top2 = Reranker().rerank("what accuracy did the model achieve", pairs, top_k=3)
    assert "87.4" in by_fid[top2[0][0]]["text"]

    reset_engine()
