"""Re-run title extraction for already-indexed papers. No re-embedding.

Usage:
    python scripts/refresh_titles.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.ingest import _guess_title
from app.pdf import extract_pages
from app.storage import Paper, get_engine, session


def main() -> None:
    get_engine()
    pdfs_dir = settings.data_dir / "pdfs"

    with session() as s:
        papers = s.query(Paper).all()
        for p in papers:
            pdf = pdfs_dir / p.filename
            if not pdf.exists():
                print(f"skip {p.id}: {p.filename} not on disk")
                continue
            pages = extract_pages(pdf)
            new_title = _guess_title(pdf, pages)
            if new_title != p.title:
                print(f"→ {p.id}: {p.title[:60]!r} → {new_title[:60]!r}")
                p.title = new_title
            else:
                print(f"= {p.id}: {p.title[:60]!r}")


if __name__ == "__main__":
    main()
