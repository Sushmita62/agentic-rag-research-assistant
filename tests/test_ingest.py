from pathlib import Path

import pymupdf
import numpy as np

from app.ingest import ingest_pdf
from app.storage import Chunk, Paper, get_engine, reset_engine, session
from app.vector import VectorIndex
from app.embeddings import Embedder


def _make_paper(path: Path) -> None:
    doc = pymupdf.open()
    para = ("Deep learning models have transformed natural language processing "
            "over the past decade with attention-based architectures. ") * 8
    for hdr, body in [
        ("Attention Is All You Need", para),
        ("1 Introduction", para),
        ("2 Methods", para),
        ("3 Results", para + " Accuracy reached 92 percent on the benchmark."),
    ]:
        page = doc.new_page()
        rect = pymupdf.Rect(72, 72, page.rect.width - 72, page.rect.height - 72)
        page.insert_textbox(rect, hdr + "\n\n" + body, fontsize=11)
    doc.save(path)
    doc.close()


def test_ingest_end_to_end(tmp_path: Path):
    reset_engine()
    get_engine(tmp_path / "app.db")

    pdf = tmp_path / "paper.pdf"
    _make_paper(pdf)

    vi = VectorIndex(dim=384, path=tmp_path / "faiss.index")
    embedder = Embedder()

    pid = ingest_pdf(pdf, vector_index=vi, embedder=embedder)

    with session() as s:
        p = s.get(Paper, pid)
        assert p is not None
        assert p.status == "indexed"
        assert p.num_pages == 4
        assert p.title.startswith("Attention")

        chunks = s.query(Chunk).filter(Chunk.paper_id == pid).all()
        assert len(chunks) >= 1
        assert all(c.faiss_id is not None for c in chunks)
        assert vi.ntotal == len(chunks)

    # Idempotent: second call returns same id, does not add more vectors
    prior_n = vi.ntotal
    pid2 = ingest_pdf(pdf, vector_index=vi, embedder=embedder)
    assert pid2 == pid
    assert vi.ntotal == prior_n

    # Sanity: query the index with a chunk's own embedding → gets that chunk back top-1
    with session() as s:
        c = s.query(Chunk).filter(Chunk.paper_id == pid).first()
        v = embedder.embed([c.text])
        hits = vi.search(v, k=1)
        assert hits[0][0] == c.faiss_id

    reset_engine()
