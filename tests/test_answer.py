"""Tests for the answer composer. LLM is mocked — no network."""
import json

import pytest

from app.answer import ComposedAnswer, compose_answer


class _FakeLLM:
    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.calls = 0

    def complete(self, messages, json_mode=False, temperature=0.0):
        self.calls += 1
        return self.responses.pop(0)


def test_valid_answer_passes():
    chunks = [{"chunk_id": "p1_001_00", "text": "The model achieved 92% accuracy."}]
    llm = _FakeLLM([json.dumps({
        "summary": "The model reached 92% accuracy.",
        "claims": [{"text": "The model achieved 92% accuracy.",
                    "citations": [{"chunk_id": "p1_001_00"}]}],
    })])
    answer = compose_answer("What accuracy?", chunks, llm=llm)
    assert isinstance(answer, ComposedAnswer)
    assert answer.claims[0].citations[0].chunk_id == "p1_001_00"
    assert llm.calls == 1


def test_bogus_chunk_id_triggers_retry_then_recovers():
    chunks = [{"chunk_id": "real_id", "text": "x"}]
    llm = _FakeLLM([
        # first: cites fake id → rejected
        json.dumps({"summary": "s", "claims": [
            {"text": "c1", "citations": [{"chunk_id": "MADE_UP"}]}]}),
        # second: cites real id → passes
        json.dumps({"summary": "s", "claims": [
            {"text": "c1", "citations": [{"chunk_id": "real_id"}]}]}),
    ])
    answer = compose_answer("q", chunks, llm=llm, max_retries=1)
    assert llm.calls == 2
    assert answer.claims[0].citations[0].chunk_id == "real_id"


def test_schema_violation_missing_citations_triggers_retry():
    chunks = [{"chunk_id": "id1", "text": "x"}]
    llm = _FakeLLM([
        json.dumps({"summary": "s", "claims": [{"text": "c", "citations": []}]}),
        json.dumps({"summary": "s", "claims": [
            {"text": "c", "citations": [{"chunk_id": "id1"}]}]}),
    ])
    answer = compose_answer("q", chunks, llm=llm, max_retries=1)
    assert llm.calls == 2
    assert answer.claims[0].citations


def test_gives_up_after_retries():
    chunks = [{"chunk_id": "id1", "text": "x"}]
    bad = json.dumps({"summary": "s", "claims": [
        {"text": "c", "citations": [{"chunk_id": "NOPE"}]}]})
    llm = _FakeLLM([bad, bad])
    with pytest.raises(RuntimeError):
        compose_answer("q", chunks, llm=llm, max_retries=1)
