"""Ingest every PDF in a folder into the default index.

Usage:  python scripts/ingest_folder.py data/pdfs
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ingest import ingest_pdf


def main() -> None:
    folder = Path(sys.argv[1] if len(sys.argv) > 1 else "data/pdfs")
    pdfs = sorted(folder.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs in {folder}")
        return
    for pdf in pdfs:
        print(f"→ {pdf.name}", flush=True)
        pid = ingest_pdf(pdf)
        print(f"  indexed as {pid}", flush=True)


if __name__ == "__main__":
    main()
