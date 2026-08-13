"""FastAPI app. Wraps ingestion + graph as HTTP endpoints.

Endpoints:
    GET  /health              → basic status
    POST /papers/upload       → multipart PDF → indexed paper
    GET  /papers              → list of indexed papers
    POST /research/query      → RAG answer with citations + trace
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any


_CITE_RE = re.compile(r"\[([A-Za-z0-9]+_\d+_\d+)\]")


def _prettify_citations(markdown: str, meta_by_id: dict[str, dict]) -> str:
    """Replace [chunk_id] with [Paper title, p.N] inline. Leave unknown ids alone."""
    def repl(m: re.Match) -> str:
        cid = m.group(1)
        d = meta_by_id.get(cid)
        if not d:
            return m.group(0)
        title = (d.get("title") or "").strip()[:60]
        page = d.get("page", "?")
        return f"[{title}, p.{page}]"
    return _CITE_RE.sub(repl, markdown)

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy import select

from app.compare import DEFAULT_DIMENSIONS, compare_papers
from app.config import settings
from app.ingest import ingest_pdf
from app.litreview import literature_review
from app.storage import Chunk, Paper, get_engine, session


# ── Lazy singletons: heavy models load once on first query, not per request ──
_graph = None


def _get_graph():
    global _graph
    if _graph is None:
        from app.graph import build_graph
        _graph = build_graph()
    return _graph


# ── Response models ──────────────────────────────────────────────────────────
class PaperInfo(BaseModel):
    id: str
    title: str
    filename: str
    num_pages: int
    status: str


class UploadResponse(BaseModel):
    paper_id: str
    status: str


class Citation(BaseModel):
    chunk_id: str


class ClaimOut(BaseModel):
    text: str
    verdict: str                # SUPPORTED / PARTIAL / UNSUPPORTED
    citations: list[Citation]


class EvidenceChunk(BaseModel):
    chunk_id: str
    page: int
    section: str
    title: str
    text: str


class TraceStepOut(BaseModel):
    node: str
    detail: str = ""
    latency_ms: int = 0


class QueryResponse(BaseModel):
    answer: str
    abstained: bool
    abstain_reason: str | None = None
    claims: list[ClaimOut] = []
    evidence: list[EvidenceChunk] = []
    trace: list[TraceStepOut] = []


class QueryRequest(BaseModel):
    question: str


class CompareRequest(BaseModel):
    paper_ids: list[str]
    dimensions: list[str] | None = None


class LitReviewRequest(BaseModel):
    topic: str
    paper_ids: list[str] | None = None


app = FastAPI(title="AI Research Assistant", version="0.1.0")


@app.get("/health")
def health() -> dict[str, Any]:
    get_engine()
    with session() as s:
        n_papers = s.query(Paper).count()
    return {"ok": True, "indexed_papers": n_papers}


@app.get("/papers", response_model=list[PaperInfo])
def list_papers() -> list[PaperInfo]:
    get_engine()
    with session() as s:
        rows = s.query(Paper).order_by(Paper.uploaded_at.desc()).all()
        return [PaperInfo(id=p.id, title=p.title, filename=p.filename,
                          num_pages=p.num_pages, status=p.status) for p in rows]


@app.post("/papers/upload", response_model=UploadResponse)
async def upload_paper(file: UploadFile = File(...)) -> UploadResponse:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only .pdf files accepted")

    max_bytes = settings.max_upload_mb * 1024 * 1024
    dest_dir = settings.data_dir / "pdfs"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / file.filename

    # Stream-copy with a hard byte cap to prevent memory bombs.
    written = 0
    with dest.open("wb") as f:
        while chunk := await file.read(1024 * 1024):
            written += len(chunk)
            if written > max_bytes:
                dest.unlink(missing_ok=True)
                raise HTTPException(413, f"File exceeds {settings.max_upload_mb} MB cap")
            f.write(chunk)

    # Magic-byte check — prevents rename attacks
    with dest.open("rb") as f:
        head = f.read(5)
    if head != b"%PDF-":
        dest.unlink(missing_ok=True)
        raise HTTPException(400, "File does not look like a PDF")

    try:
        paper_id = ingest_pdf(dest)
    except Exception as e:
        raise HTTPException(500, f"Ingest failed: {e}")

    return UploadResponse(paper_id=paper_id, status="indexed")


@app.post("/research/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    get_engine()
    graph = _get_graph()
    result = graph.invoke({"query": req.question, "trace": []})

    claims: list[ClaimOut] = []
    composed = result.get("composed")
    verdicts = result.get("verdicts", [])
    if composed:
        for c, v in zip(composed.claims, verdicts):
            claims.append(ClaimOut(
                text=c.text, verdict=v,
                citations=[Citation(chunk_id=x.chunk_id) for x in c.citations],
            ))

    evidence: list[EvidenceChunk] = []
    for fid in result.get("reranked_chunk_ids", []):
        d = result.get("chunks_by_fid", {}).get(fid)
        if d:
            evidence.append(EvidenceChunk(
                chunk_id=d["chunk_id"], page=d["page"], section=d["section"],
                title=d["title"], text=d["text"],
            ))

    return QueryResponse(
        answer=result.get("final_answer", ""),
        abstained=result.get("abstained", False),
        abstain_reason=result.get("abstain_reason"),
        claims=claims,
        evidence=evidence,
        trace=[TraceStepOut(**t) for t in result.get("trace", [])],
    )


@app.get("/dimensions/default", response_model=list[str])
def default_dimensions() -> list[str]:
    return list(DEFAULT_DIMENSIONS)


@app.post("/research/compare")
def compare(req: CompareRequest) -> dict:
    if not req.paper_ids:
        raise HTTPException(400, "paper_ids required")
    get_engine()
    table = compare_papers(req.paper_ids, dimensions=req.dimensions)

    # Enrich cells with chunk metadata so the UI can render page + snippet.
    all_ids = [c.citation.chunk_id for row in table.rows for c in row.cells.values() if c.citation]
    chunk_lookup: dict[str, dict] = {}
    if all_ids:
        with session() as s:
            for c in s.query(Chunk).filter(Chunk.id.in_(all_ids)).all():
                chunk_lookup[c.id] = {"page": c.page, "section": c.section, "text": c.text}

    return {
        "dimensions": table.dimensions,
        "rows": [
            {
                "paper_id": row.paper_id, "paper_title": row.paper_title,
                "cells": {
                    dim: {
                        "value": cell.value,
                        "citation": (
                            {"chunk_id": cell.citation.chunk_id,
                             **chunk_lookup.get(cell.citation.chunk_id, {})}
                            if cell.citation else None
                        ),
                    }
                    for dim, cell in row.cells.items()
                },
            }
            for row in table.rows
        ],
    }


@app.post("/research/literature-review")
def lit_review(req: LitReviewRequest) -> dict:
    if not req.topic.strip():
        raise HTTPException(400, "topic required")
    get_engine()
    review = literature_review(req.topic, paper_ids=req.paper_ids)

    # Enrich citations with page metadata
    cit_ids = [c.chunk_id for c in review.citations]
    meta_by_id: dict[str, dict] = {}
    if cit_ids:
        with session() as s:
            rows = s.execute(
                select(Chunk, Paper).join(Paper).where(Chunk.id.in_(cit_ids))
            ).all()
            for c, p in rows:
                meta_by_id[c.id] = {"page": c.page, "section": c.section,
                                    "title": p.title, "text": c.text}

    return {
        "topic": review.topic,
        "review_markdown": _prettify_citations(review.review_markdown, meta_by_id),
        "per_paper_summaries": [s.model_dump() for s in review.per_paper_summaries],
        "citations": [
            {"chunk_id": c.chunk_id, **meta_by_id.get(c.chunk_id, {})}
            for c in review.citations
        ],
    }
