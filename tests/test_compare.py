"""Compare workflow. LLM + retriever + reranker mocked. Real Cosmos-free SQLite."""
import json
from unittest.mock import MagicMock

from app.compare import _extract_cell, compare_papers


class _FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)

    def complete(self, messages, json_mode=False, temperature=0.0):
        return self.responses.pop(0)


def test_extract_cell_valid_value_with_citation():
    chunks = [{"chunk_id": "c1", "text": "We use a CNN."}]
    llm = _FakeLLM([json.dumps({"value": "CNN", "chunk_id": "c1"})])
    cell = _extract_cell(llm, "Method", chunks)
    assert cell.value == "CNN"
    assert cell.citation.chunk_id == "c1"


def test_extract_cell_not_reported():
    chunks = [{"chunk_id": "c1", "text": "The weather is nice."}]
    llm = _FakeLLM([json.dumps({"value": "NOT_REPORTED", "chunk_id": None})])
    cell = _extract_cell(llm, "Method", chunks)
    assert cell.value == "NOT_REPORTED"
    assert cell.citation is None


def test_extract_cell_rejects_bogus_chunk_id():
    """Model says 'we found the answer' but cites a chunk_id not in the allowed set →
    treat as NOT_REPORTED (no fabricated citation)."""
    chunks = [{"chunk_id": "real", "text": "content"}]
    llm = _FakeLLM([json.dumps({"value": "Transformer", "chunk_id": "MADE_UP"})])
    cell = _extract_cell(llm, "Method", chunks)
    assert cell.value == "NOT_REPORTED"
    assert cell.citation is None


def test_extract_cell_rejects_value_without_citation():
    chunks = [{"chunk_id": "c1", "text": "content"}]
    llm = _FakeLLM([json.dumps({"value": "Something", "chunk_id": None})])
    cell = _extract_cell(llm, "Method", chunks)
    assert cell.value == "NOT_REPORTED"
    assert cell.citation is None


def test_compare_papers_uses_paper_filter(tmp_path):
    from app.storage import Chunk, Paper, get_engine, reset_engine, session
    reset_engine()
    get_engine(tmp_path / "app.db")

    with session() as s:
        for pid, title, cid, txt, fid in [
            ("pA", "Paper A", "pA_001_00", "We use CNN.", 100),
            ("pB", "Paper B", "pB_001_00", "We use Transformer.", 200),
        ]:
            s.add(Paper(id=pid, title=title, filename=f"{pid}.pdf", num_pages=1, status="indexed"))
            s.add(Chunk(id=cid, paper_id=pid, page=1, section="body", text=txt,
                        token_count=5, embedding_model="test", faiss_id=fid))

    retriever = MagicMock()
    # Paper A gets its chunk; Paper B gets its chunk
    retriever.retrieve.side_effect = lambda q, k, per_source, paper_ids: (
        [(100, 1.0)] if paper_ids == ["pA"] else [(200, 1.0)]
    )
    reranker = MagicMock()
    reranker.rerank.side_effect = lambda q, pairs, top_k: pairs[:top_k]

    llm = _FakeLLM([
        json.dumps({"value": "CNN", "chunk_id": "pA_001_00"}),         # Paper A / Method
        json.dumps({"value": "Transformer", "chunk_id": "pB_001_00"}), # Paper B / Method
    ])

    table = compare_papers(["pA", "pB"], dimensions=["Method"],
                           retriever=retriever, reranker=reranker, llm=llm)

    assert len(table.rows) == 2
    assert table.rows[0].cells["Method"].value == "CNN"
    assert table.rows[1].cells["Method"].value == "Transformer"

    reset_engine()
