"""Multi-vote adversarial claim verification.

For each claim + cited chunks, spawn 3 verifier LLM calls with DIFFERENT prompts:
  1. Supportive — assume it's supported, defend it
  2. Adversarial — assume it isn't, attack it
  3. Neutral — impartial judge

Majority vote decides. Catches "LLM confidently agrees with itself" failures that
single-verifier setups miss.
"""
from __future__ import annotations

import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

from app.answer import Claim
from app.llm import LLMClient


Verdict = Literal["SUPPORTED", "PARTIAL", "UNSUPPORTED"]
VALID_VERDICTS = {"SUPPORTED", "PARTIAL", "UNSUPPORTED"}


_PROMPT_BASE = """You are checking whether a CLAIM is supported by EVIDENCE (from a research paper).

CLAIM: {claim}

EVIDENCE:
{evidence}

{stance}

Respond with valid JSON only:
{{"verdict": "SUPPORTED" | "PARTIAL" | "UNSUPPORTED", "reason": "<1 sentence>"}}

- SUPPORTED: evidence explicitly states or clearly implies the claim.
- PARTIAL: evidence mentions the topic but does not fully back the claim.
- UNSUPPORTED: evidence does not back the claim (or contradicts it).
"""

_STANCE_SUPPORTIVE = "You are inclined to trust the claim. If the evidence plausibly backs it, say SUPPORTED."
_STANCE_ADVERSARIAL = "You are a skeptical reviewer. Default to UNSUPPORTED unless the evidence explicitly and precisely backs the claim."
_STANCE_NEUTRAL = "You are an impartial judge. Weigh the evidence strictly on what it says."


def _one_vote(llm: LLMClient, prompt: str) -> Verdict:
    try:
        raw = llm.complete([{"role": "user", "content": prompt}], json_mode=True)
        obj = json.loads(raw)
        v = obj.get("verdict", "").upper()
        return v if v in VALID_VERDICTS else "UNSUPPORTED"
    except Exception:
        return "UNSUPPORTED"


def verify_claim(
    claim: Claim,
    chunks_by_id: dict[str, str],
    llm: LLMClient | None = None,
) -> Verdict:
    llm = llm or LLMClient()
    evidence = "\n\n---\n\n".join(
        chunks_by_id.get(c.chunk_id, "") for c in claim.citations
    ) or "(no evidence)"
    prompts = [
        _PROMPT_BASE.format(claim=claim.text, evidence=evidence, stance=s)
        for s in (_STANCE_SUPPORTIVE, _STANCE_ADVERSARIAL, _STANCE_NEUTRAL)
    ]
    with ThreadPoolExecutor(max_workers=3) as ex:
        votes = list(ex.map(lambda p: _one_vote(llm, p), prompts))
    return Counter(votes).most_common(1)[0][0]


def verify_all(
    claims: list[Claim],
    chunks_by_id: dict[str, str],
    llm: LLMClient | None = None,
) -> list[Verdict]:
    llm = llm or LLMClient()
    return [verify_claim(c, chunks_by_id, llm) for c in claims]
