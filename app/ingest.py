"""End-to-end: PDF → paper + chunks in SQLite + vectors in FAISS. Idempotent."""
from __future__ import annotations

from pathlib import Path

from app.bm25 import BM25Index
from app.chunker import chunk_document
from app.config import settings
from app.embeddings import Embedder
from app.pdf import extract_metadata, extract_pages, paper_id as compute_paper_id
from app.sections import detect_sections
from app.storage import Chunk, Paper, session
from app.vector import VectorIndex


# Common watermark / boilerplate prefixes on page 1 that are NOT the title.
_TITLE_SKIP_PREFIXES = (
    "provided proper attribution",
    "reproduce the tables",                 # 2nd line of Google's arXiv attribution
    "solely for use",
    "arxiv:",
    "preprint",
    "under review",
    "copyright",
    "abstract",
    "published as",
    "to appear",
    "https://",
    "http://",
    "www.",
)


def _looks_like_title(line: str) -> bool:
    """Heuristic: title-case-ish, not a URL/id, not too many digits, not a fragment."""
    if len(line) < 15:                              # skip very short fragments
        return False
    if sum(c.isdigit() for c in line) > len(line) * 0.3:
        return False
    words = line.split()
    if len(words) < 3:                              # real titles are ≥3 words
        return False
    if not words[0][0].isupper():                   # sentence continuation → skip
        return False
    return True


def _guess_title(pdf_path: Path, pages) -> str:
    """PDF metadata title first (best signal); fall back to text with watermark skip-list."""
    meta_title = extract_metadata(pdf_path).get("title", "")
    if meta_title and len(meta_title) >= 8:
        return meta_title[:200]

    if not pages:
        return pdf_path.stem
    for line in pages[0].text.splitlines():
        line = line.strip()
        if len(line) < 8 or line.isdigit():
            continue
        low = line.lower()
        if any(low.startswith(skip) for skip in _TITLE_SKIP_PREFIXES):
            continue
        if not _looks_like_title(line):
            continue
        return line[:200]
    return pdf_path.stem


def ingest_pdf(
    pdf_path: Path,
    vector_index: VectorIndex | None = None,
    embedder: Embedder | None = None,
    bm25_index: BM25Index | None = None,
) -> str:
    pid = compute_paper_id(pdf_path)

    with session() as s:
        existing = s.get(Paper, pid)
        if existing and existing.status == "indexed":
            return pid                                          # idempotent

    pages = extract_pages(pdf_path)
    sections = detect_sections(pages)
    chunks = chunk_document(sections, paper_id=pid)

    title = _guess_title(pdf_path, pages)

    with session() as s:
        p = s.get(Paper, pid)
        if p is None:
            p = Paper(id=pid, title=title, filename=pdf_path.name,
                      num_pages=len(pages), status="processing")
            s.add(p)
        else:
            p.status = "processing"
            # drop any prior chunks for a clean rebuild
            s.query(Chunk).filter(Chunk.paper_id == pid).delete()

    if not chunks:
        with session() as s:
            s.get(Paper, pid).status = "indexed"
        return pid

    embedder = embedder or Embedder()
    vectors = embedder.embed([c.text for c in chunks])

    vector_index = vector_index or VectorIndex()
    faiss_ids = vector_index.add(vectors)
    vector_index.save()

    with session() as s:
        for c, fid in zip(chunks, faiss_ids):
            s.add(Chunk(
                id=c.id, paper_id=c.paper_id, page=c.page, section=c.section,
                text=c.text, token_count=c.token_count,
                embedding_model=settings.embedding_model, faiss_id=fid,
            ))
        s.get(Paper, pid).status = "indexed"

    # Rebuild BM25 from full corpus. ponytail: O(N) rebuild per ingest,
    # switch to incremental if ingest of new papers gets slow at 10k+ chunks.
    bm25_index = bm25_index or BM25Index()
    with session() as s:
        corpus = [(c.faiss_id, c.text) for c in s.query(Chunk).filter(Chunk.faiss_id.isnot(None)).all()]
    bm25_index.rebuild(corpus)
    bm25_index.save()

    return pid
