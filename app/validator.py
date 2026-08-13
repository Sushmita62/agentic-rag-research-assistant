"""Deterministic citation validator. No LLM. Third layer of the citation contract.

For each claim: check that at least one of its cited chunks lexically overlaps
the claim text. Cheap sanity check that catches the LLM confidently pointing at
an unrelated chunk. NLI entailment is a future upgrade (Phase 4+).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from app.answer import ComposedAnswer


_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "of", "to", "in", "on", "at", "for", "with", "by", "as", "and", "or",
    "but", "if", "not", "no", "this", "that", "these", "those", "it", "its",
    "we", "our", "their", "his", "her", "them", "which", "who", "what",
}
_WORD_RE = re.compile(r"[A-Za-z0-9]+")


def _content_tokens(text: str) -> set[str]:
    return {w for w in (t.lower() for t in _WORD_RE.findall(text)) if w not in _STOPWORDS and len(w) > 1}


def _overlap_ratio(claim: str, chunk: str) -> float:
    ct, kt = _content_tokens(claim), _content_tokens(chunk)
    if not ct:
        return 0.0
    return len(ct & kt) / len(ct)


Support = Literal["SUPPORTED", "PARTIAL", "UNSUPPORTED"]


@dataclass
class ClaimReport:
    text: str
    support: Support
    best_chunk_id: str
    best_overlap: float
    cited_ids: list[str]


@dataclass
class ValidationReport:
    per_claim: list[ClaimReport]
    supported_ratio: float                              # supported / total claims
    should_abstain: bool


def validate(
    answer: ComposedAnswer,
    chunks_by_id: dict[str, str],
    supported_threshold: float = 0.30,
    partial_threshold: float = 0.15,
    abstain_ratio: float = 0.5,
) -> ValidationReport:
    per_claim: list[ClaimReport] = []
    supported_count = 0

    for claim in answer.claims:
        best_id, best = "", 0.0
        cited_ids = [c.chunk_id for c in claim.citations]
        for cid in cited_ids:
            r = _overlap_ratio(claim.text, chunks_by_id.get(cid, ""))
            if r > best:
                best, best_id = r, cid
        support: Support = (
            "SUPPORTED" if best >= supported_threshold
            else "PARTIAL" if best >= partial_threshold
            else "UNSUPPORTED"
        )
        if support == "SUPPORTED":
            supported_count += 1
        per_claim.append(ClaimReport(claim.text, support, best_id, best, cited_ids))

    ratio = supported_count / len(per_claim) if per_claim else 0.0
    return ValidationReport(per_claim, ratio, should_abstain=ratio < abstain_ratio)
