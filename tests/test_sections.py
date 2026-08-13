from app.pdf import Page
from app.sections import clean_text, detect_sections


def test_clean_dehyphenates():
    assert clean_text("trans-\nformer") == "transformer"


def test_clean_collapses_whitespace():
    assert clean_text("a   b\n\n\n\nc") == "a b\n\nc"


def test_sections_split_on_headings():
    pages = [
        Page(1, "Abstract\nWe propose a novel model.\n\n1 Introduction\nDeep learning has..."),
        Page(2, "2 Methods\nWe train on ImageNet.\n\n3 Results\nAccuracy reached 92%."),
        Page(3, "References\n[1] Smith et al."),
    ]
    secs = detect_sections(pages)
    names = [s.name for s in secs]
    assert names == ["abstract", "introduction", "methods", "results", "references"]

    intro = next(s for s in secs if s.name == "introduction")
    assert "Deep learning" in intro.text
    assert intro.start_page == 1

    results = next(s for s in secs if s.name == "results")
    assert "92%" in results.text
    assert results.start_page == 2


def test_no_headings_falls_back_to_body():
    pages = [Page(1, "Just some text.\nNo headings here.")]
    secs = detect_sections(pages)
    assert len(secs) == 1
    assert secs[0].name == "body"
    assert secs[0].start_page == 1
    assert secs[0].end_page == 1
