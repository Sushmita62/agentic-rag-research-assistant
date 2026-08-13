"""LangGraph flow test. LLM + retriever + reranker all mocked. Just verifies the
graph reaches the right terminal state and appends trace steps.
"""
import json
from unittest.mock import MagicMock

import numpy as np

from app.answer import ComposedAnswer
from app.graph import build_graph


def _fake_retriever(faiss_ids):
    r = MagicMock()
    r.retrieve.return_value = [(fid, 1.0 / (i + 1)) for i, fid in enumerate(faiss_ids)]
    return r


def _fake_reranker(top_fids):
    r = MagicMock()
    r.rerank.return_value = [(fid, 1.0) for fid in top_fids]
    return r


def _fake_llm(responses):
    calls = iter(responses)

    class L:
        def complete(self, messages, json_mode=False, temperature=0.0):
            return next(calls)
    return L()


def _seed_db(tmp_path):
    """Insert one Paper + two Chunks so retrieve_node's SQL works."""
    from app.storage import Chunk, Paper, get_engine, reset_engine, session
    reset_engine()
    get_engine(tmp_path / "app.db")
    with session() as s:
        s.add(Paper(id="pX", title="Test paper", filename="x.pdf", num_pages=1, status="indexed"))
        s.add(Chunk(id="pX_001_00", paper_id="pX", page=1, section="body",
                    text="The model reached 92 percent accuracy.",
                    token_count=10, embedding_model="test", faiss_id=100))
        s.add(Chunk(id="pX_001_01", paper_id="pX", page=1, section="body",
                    text="The model was evaluated on ImageNet.",
                    token_count=10, embedding_model="test", faiss_id=101))
    reset_engine()
    get_engine(tmp_path / "app.db")


def test_graph_finalizes_on_supported_answer(tmp_path):
    _seed_db(tmp_path)

    compose_json = json.dumps({
        "summary": "The model reached 92 percent accuracy.",
        "claims": [{"text": "The model reached 92 percent accuracy.",
                    "citations": [{"chunk_id": "pX_001_00"}]}],
    })
    verify_json = json.dumps({"verdict": "SUPPORTED", "reason": "ok"})

    llm = _fake_llm([compose_json, verify_json, verify_json, verify_json])
    graph = build_graph(
        retriever=_fake_retriever([100, 101]),
        reranker=_fake_reranker([100, 101]),
        llm=llm,
    )

    result = graph.invoke({"query": "accuracy?", "trace": []})

    assert result["abstained"] is False
    assert "92 percent" in result["final_answer"]
    trace_nodes = [t["node"] for t in result["trace"]]
    assert trace_nodes == ["retrieve", "compose", "verify", "validate", "finalize"]


def test_graph_abstains_when_unsupported(tmp_path):
    _seed_db(tmp_path)

    compose_json = json.dumps({
        "summary": "The moon is made of cheese.",
        "claims": [{"text": "The moon is made of green cheese.",
                    "citations": [{"chunk_id": "pX_001_00"}]}],
    })
    # Multi-vote: 3 UNSUPPORTED verdicts
    verify_json = json.dumps({"verdict": "UNSUPPORTED", "reason": "no"})

    llm = _fake_llm([compose_json, verify_json, verify_json, verify_json])
    graph = build_graph(
        retriever=_fake_retriever([100, 101]),
        reranker=_fake_reranker([100, 101]),
        llm=llm,
    )

    result = graph.invoke({"query": "is the moon cheese?", "trace": []})

    assert result["abstained"] is True
    assert "sufficient" in result["final_answer"].lower()
    trace_nodes = [t["node"] for t in result["trace"]]
    assert trace_nodes[-1] == "abstain"
