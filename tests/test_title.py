"""Title extraction — PDF metadata preferred, watermarks skipped."""
from pathlib import Path

import pymupdf

from app.ingest import _guess_title
from app.pdf import extract_pages


def _pdf(path: Path, lines: list[str], meta_title: str = "") -> None:
    doc = pymupdf.open()
    page = doc.new_page()
    rect = pymupdf.Rect(72, 72, page.rect.width - 72, page.rect.height - 72)
    page.insert_textbox(rect, "\n".join(lines), fontsize=12)
    if meta_title:
        doc.set_metadata({"title": meta_title})
    doc.save(path)
    doc.close()


def test_prefers_pdf_metadata_title(tmp_path):
    p = tmp_path / "x.pdf"
    _pdf(p, ["Attention Is All You Need"], meta_title="Attention Is All You Need")
    assert _guess_title(p, extract_pages(p)) == "Attention Is All You Need"


def test_skips_arxiv_watermark(tmp_path):
    p = tmp_path / "x.pdf"
    _pdf(p, [
        "Provided proper attribution is provided, Google hereby grants...",
        "The Real Paper Title",
        "Body of the abstract goes here.",
    ])
    assert _guess_title(p, extract_pages(p)) == "The Real Paper Title"


def test_falls_back_to_filename_when_nothing_usable(tmp_path):
    p = tmp_path / "my_paper.pdf"
    _pdf(p, ["arxiv:1234.5678", "Preprint", "1"])
    assert _guess_title(p, extract_pages(p)) == "my_paper"
