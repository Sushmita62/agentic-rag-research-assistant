"""PDF → per-page text. PyMuPDF handles reading order for most 2-column layouts."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import NamedTuple

import pymupdf  # PyMuPDF


class Page(NamedTuple):
    num: int   # 1-indexed
    text: str


def extract_pages(pdf_path: Path) -> list[Page]:
    doc = pymupdf.open(pdf_path)
    try:
        return [Page(num=i + 1, text=p.get_text("text")) for i, p in enumerate(doc)]
    finally:
        doc.close()


def paper_id(pdf_path: Path) -> str:
    """Deterministic 12-char id from file bytes. Same file → same id → dedupe."""
    h = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
    return h[:12]


def extract_metadata(pdf_path: Path) -> dict:
    """PDF-level metadata (title/author/subject). Empty strings when missing."""
    doc = pymupdf.open(pdf_path)
    try:
        m = doc.metadata or {}
        return {
            "title": (m.get("title") or "").strip(),
            "author": (m.get("author") or "").strip(),
        }
    finally:
        doc.close()
