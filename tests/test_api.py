"""FastAPI endpoint tests. Graph is monkey-patched to skip the real LLM."""
from unittest.mock import patch

import pymupdf
from fastapi.testclient import TestClient


def _make_pdf(path, body: str = "Some content about transformer models."):
    doc = pymupdf.open()
    page = doc.new_page()
    rect = pymupdf.Rect(72, 72, page.rect.width - 72, page.rect.height - 72)
    page.insert_textbox(rect, "Introduction\n\n" + body * 20, fontsize=11)
    doc.save(path)
    doc.close()


def test_health(tmp_path, monkeypatch):
    monkeypatch.setattr("app.config.settings.db_path", tmp_path / "app.db")
    from app.storage import reset_engine
    reset_engine()

    from app.api import app
    client = TestClient(app)

    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["ok"] is True


def test_rejects_non_pdf(tmp_path, monkeypatch):
    monkeypatch.setattr("app.config.settings.db_path", tmp_path / "app.db")
    from app.storage import reset_engine
    reset_engine()

    from app.api import app
    client = TestClient(app)

    r = client.post(
        "/papers/upload",
        files={"file": ("evil.exe", b"not a pdf", "application/pdf")},
    )
    assert r.status_code == 400


def test_query_returns_shape(tmp_path, monkeypatch):
    monkeypatch.setattr("app.config.settings.db_path", tmp_path / "app.db")
    from app.storage import reset_engine
    reset_engine()

    # Stub the graph so we don't hit Groq or load any models.
    fake_result = {
        "final_answer": "The transformer uses attention.",
        "abstained": False,
        "abstain_reason": None,
        "composed": type("A", (), {"claims": []})(),
        "verdicts": [],
        "reranked_chunk_ids": [],
        "chunks_by_fid": {},
        "trace": [{"node": "retrieve", "detail": "stub", "latency_ms": 1}],
    }
    with patch("app.api._get_graph") as gg:
        gg.return_value = type("G", (), {"invoke": lambda self, _: fake_result})()
        from app.api import app
        client = TestClient(app)
        r = client.post("/research/query", json={"question": "what?"})

    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "The transformer uses attention."
    assert body["abstained"] is False
    assert body["trace"][0]["node"] == "retrieve"
