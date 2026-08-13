"""Mocked LLM. Verifies majority-vote logic."""
import json

from app.answer import Citation, Claim
from app.verifier import verify_claim


class _FakeLLM:
    """Cycles through predetermined verdicts to simulate 3 different-stance calls."""
    def __init__(self, verdicts: list[str]):
        self.verdicts = list(verdicts)

    def complete(self, messages, json_mode=False, temperature=0.0):
        v = self.verdicts.pop(0)
        return json.dumps({"verdict": v, "reason": "test"})


def _claim():
    return Claim(text="X happened.", citations=[Citation(chunk_id="c1")])


def test_all_supported():
    llm = _FakeLLM(["SUPPORTED", "SUPPORTED", "SUPPORTED"])
    assert verify_claim(_claim(), {"c1": "X did happen."}, llm=llm) == "SUPPORTED"


def test_majority_wins():
    llm = _FakeLLM(["SUPPORTED", "SUPPORTED", "UNSUPPORTED"])
    assert verify_claim(_claim(), {"c1": "X"}, llm=llm) == "SUPPORTED"


def test_all_unsupported():
    llm = _FakeLLM(["UNSUPPORTED", "UNSUPPORTED", "UNSUPPORTED"])
    assert verify_claim(_claim(), {"c1": ""}, llm=llm) == "UNSUPPORTED"


def test_invalid_verdict_becomes_unsupported():
    llm = _FakeLLM(["GARBAGE", "SUPPORTED", "UNSUPPORTED"])
    # Two UNSUPPORTED (one from garbage, one real) vs one SUPPORTED
    assert verify_claim(_claim(), {"c1": ""}, llm=llm) == "UNSUPPORTED"
