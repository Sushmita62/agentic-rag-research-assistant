from app.api import _prettify_citations


def test_inline_citations_are_replaced():
    md = "The model uses attention [abc123_004_02] and dropout [abc123_005_00]."
    meta = {
        "abc123_004_02": {"title": "Attention Is All You Need", "page": 4},
        "abc123_005_00": {"title": "Attention Is All You Need", "page": 5},
    }
    out = _prettify_citations(md, meta)
    assert "[Attention Is All You Need, p.4]" in out
    assert "[Attention Is All You Need, p.5]" in out
    assert "abc123" not in out


def test_unknown_chunk_id_left_alone():
    md = "Claim [unknown_001_00]."
    assert _prettify_citations(md, {}) == "Claim [unknown_001_00]."
