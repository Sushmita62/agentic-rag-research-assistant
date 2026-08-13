"""Metric functions. Pure — no LLM, no I/O."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class QuestionMetrics:
    id: str
    verdict_correct: bool
    recall_at_5: float | None                # None for ABSTAIN questions
    citation_precision: float | None         # None if answer was abstained
    must_contain_hit_ratio: float | None
    latency_ms: int


@dataclass
class AggregateMetrics:
    n: int
    abstain_accuracy: float
    recall_at_5: float                       # avg over non-abstain questions
    citation_precision: float                # avg over non-abstain questions
    must_contain_hit_ratio: float
    p50_latency_ms: int
    p95_latency_ms: int


def _recall_at_k(evidence_pages: list[int], gold_pages: list[int], k: int = 5) -> float:
    if not gold_pages:
        return 1.0                           # trivially recalled
    top_k_pages = set(evidence_pages[:k])
    return 1.0 if any(g in top_k_pages for g in gold_pages) else 0.0


def _citation_precision(citation_pages: list[int], gold_pages: list[int]) -> float:
    if not citation_pages:
        return 0.0
    if not gold_pages:
        return 1.0
    hits = sum(1 for p in citation_pages if p in gold_pages)
    return hits / len(citation_pages)


def _must_contain(answer: str, keywords: list[str]) -> float:
    if not keywords:
        return 1.0
    ans = answer.lower()
    return sum(1 for k in keywords if k.lower() in ans) / len(keywords)


def score_question(bench: dict, result: dict) -> QuestionMetrics:
    """
    bench = one benchmark record. result = one graph output (dict).
    """
    expected_abstain = bench["expected_verdict"] == "ABSTAIN"
    actual_abstain = bool(result.get("abstained"))
    verdict_correct = expected_abstain == actual_abstain

    if expected_abstain:
        return QuestionMetrics(
            id=bench["id"], verdict_correct=verdict_correct,
            recall_at_5=None, citation_precision=None,
            must_contain_hit_ratio=None,
            latency_ms=result.get("total_latency_ms", 0),
        )

    evidence_pages = [e["page"] for e in result.get("evidence", [])]
    citation_pages = []
    for claim in result.get("claims", []):
        for cite in claim.get("citations", []):
            # citations reference chunk_ids; look up page from evidence set
            for e in result.get("evidence", []):
                if e["chunk_id"] == cite["chunk_id"]:
                    citation_pages.append(e["page"])
                    break

    return QuestionMetrics(
        id=bench["id"], verdict_correct=verdict_correct,
        recall_at_5=_recall_at_k(evidence_pages, bench["gold_pages"], k=5),
        citation_precision=_citation_precision(citation_pages, bench["gold_pages"]),
        must_contain_hit_ratio=_must_contain(result.get("answer", ""), bench["must_contain"]),
        latency_ms=result.get("total_latency_ms", 0),
    )


def aggregate(per_question: list[QuestionMetrics]) -> AggregateMetrics:
    n = len(per_question)
    if n == 0:
        return AggregateMetrics(0, 0, 0, 0, 0, 0, 0)

    verdict_hits = sum(1 for q in per_question if q.verdict_correct)

    non_abstain = [q for q in per_question if q.recall_at_5 is not None]
    recall = sum(q.recall_at_5 for q in non_abstain) / len(non_abstain) if non_abstain else 0.0
    cp = sum(q.citation_precision for q in non_abstain) / len(non_abstain) if non_abstain else 0.0
    mc = sum(q.must_contain_hit_ratio for q in non_abstain) / len(non_abstain) if non_abstain else 0.0

    latencies = sorted(q.latency_ms for q in per_question)
    p50 = latencies[len(latencies) // 2]
    p95 = latencies[min(int(len(latencies) * 0.95), len(latencies) - 1)]

    return AggregateMetrics(
        n=n,
        abstain_accuracy=verdict_hits / n,
        recall_at_5=recall,
        citation_precision=cp,
        must_contain_hit_ratio=mc,
        p50_latency_ms=p50,
        p95_latency_ms=p95,
    )
