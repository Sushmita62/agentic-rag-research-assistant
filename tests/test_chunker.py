from app.pdf import Page
from app.sections import detect_sections
from app.chunker import chunk_document


def test_chunks_have_unique_ids_and_valid_pages():
    # 3 pages of an "Introduction" section, enough tokens to force multiple chunks
    para = ("Deep learning models have transformed the field of natural language "
            "processing over the past decade. ") * 30
    pages = [
        Page(1, "1 Introduction\n" + para),
        Page(2, para),
        Page(3, para + "\n2 Methods\nWe use gradient descent."),
    ]
    secs = detect_sections(pages)
    chunks = chunk_document(secs, paper_id="test1234abcd", target=200, overlap=30)

    assert len(chunks) > 1
    assert len({c.id for c in chunks}) == len(chunks)          # unique
    for c in chunks:
        assert c.paper_id == "test1234abcd"
        assert 1 <= c.page <= 3
        assert 40 <= c.token_count <= 200
        assert c.id.startswith("test1234abcd_")

    # chunks belonging to different pages should exist (page-tracking works)
    pages_seen = {c.page for c in chunks}
    assert len(pages_seen) >= 2


def test_chunk_id_format():
    pages = [Page(5, "Abstract\n" + "word " * 200)]
    secs = detect_sections(pages)
    chunks = chunk_document(secs, paper_id="abc123", target=100, overlap=10)
    assert chunks[0].id == "abc123_005_00"


def test_tiny_section_dropped():
    pages = [Page(1, "Abstract\ntiny.")]
    secs = detect_sections(pages)
    chunks = chunk_document(secs, paper_id="p", target=500, overlap=75, min_tokens=40)
    assert chunks == []
