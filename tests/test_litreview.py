"""Literature review workflow. Mocked LLM + services."""
import json
from unittest.mock import MagicMock

from app.litreview import _compose_review, PaperSummary, literature_review


class _FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)

    def complete(self, messages, json_mode=False, temperature=0.0):
        return self.responses.pop(0)


def test_compose_review_rejects_bogus_citations():
    summaries = [
        PaperSummary(paper_id="pA", paper_title="A",
                     problem="p", method="m", findings="f", limitations="",
                     supporting_chunk_ids=["real_1"]),
    ]
    llm = _FakeLLM([
        json.dumps({"review_markdown": "Review [FAKE]",
                    "citations": [{"chunk_id": "FAKE"}]}),
        json.dumps({"review_markdown": "Review [real_1]",
                    "citations": [{"chunk_id": "real_1"}]}),
    ])
    out = _compose_review(llm, "topic", summaries)
    assert "real_1" in out.review_markdown
    assert out.citations[0].chunk_id == "real_1"


def test_compose_review_returns_placeholder_on_repeated_failure():
    summaries = [PaperSummary(paper_id="pA", paper_title="A",
                              supporting_chunk_ids=["real_1"])]
    llm = _FakeLLM([
        json.dumps({"review_markdown": "x", "citations": [{"chunk_id": "FAKE"}]}),
        json.dumps({"review_markdown": "x", "citations": [{"chunk_id": "FAKE2"}]}),
    ])
    out = _compose_review(llm, "topic", summaries)
    assert "failed" in out.review_markdown.lower()


def test_literature_review_end_to_end(tmp_path):
    from app.storage import Chunk, Paper, get_engine, reset_engine, session
    reset_engine()
    get_engine(tmp_path / "app.db")

    with session() as s:
        s.add(Paper(id="pA", title="Paper A", filename="a.pdf", num_pages=1, status="indexed"))
        s.add(Chunk(id="pA_001_00", paper_id="pA", page=1, section="body",
                    text="We use CNN for X.", token_count=5,
                    embedding_model="test", faiss_id=100))

    retriever = MagicMock()
    retriever.retrieve.return_value = [(100, 1.0)]
    reranker = MagicMock()
    reranker.rerank.side_effect = lambda q, pairs, top_k: pairs[:top_k]

    llm = _FakeLLM([
        # per-paper summary
        json.dumps({"problem": "p", "method": "CNN", "findings": "good",
                    "limitations": ""}),
        # compose review
        json.dumps({"review_markdown": "## Review\n\nPaper A used CNN [pA_001_00].",
                    "citations": [{"chunk_id": "pA_001_00"}]}),
    ])

    review = literature_review("X", paper_ids=["pA"],
                                retriever=retriever, reranker=reranker, llm=llm)

    assert review.topic == "X"
    assert len(review.per_paper_summaries) == 1
    assert review.per_paper_summaries[0].method == "CNN"
    assert "pA_001_00" in review.review_markdown
    assert review.citations[0].chunk_id == "pA_001_00"

    reset_engine()
