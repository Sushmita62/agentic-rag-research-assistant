from pathlib import Path

import pymupdf

from app.pdf import extract_pages, paper_id


def _make_pdf(path: Path, pages: list[str]) -> None:
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        page.insert_text((72, 72), text, fontsize=12)
    doc.save(path)
    doc.close()


def test_extract_pages(tmp_path: Path):
    pdf = tmp_path / "sample.pdf"
    _make_pdf(pdf, ["Hello page one.", "Second page here.", "Third."])

    pages = extract_pages(pdf)

    assert len(pages) == 3
    assert pages[0].num == 1
    assert "Hello page one" in pages[0].text
    assert "Second page" in pages[1].text
    assert pages[2].num == 3


def test_paper_id_deterministic(tmp_path: Path):
    pdf = tmp_path / "x.pdf"
    _make_pdf(pdf, ["same content"])
    assert paper_id(pdf) == paper_id(pdf)
    assert len(paper_id(pdf)) == 12
