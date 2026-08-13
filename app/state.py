"""LangGraph state schema. TypedDict + reducer for the trace list."""
from __future__ import annotations

from operator import add
from typing import Any, TypedDict

from typing_extensions import Annotated


class TraceStep(TypedDict, total=False):
    node: str
    detail: str
    latency_ms: int
    meta: dict


class ResearchState(TypedDict, total=False):
    # inputs
    query: str

    # retrieval
    reranked_chunk_ids: list[int]           # faiss_ids in rerank order
    chunks_by_fid: dict[int, dict]          # faiss_id → {chunk_id, text, page, section, title}

    # generation
    composed: Any                           # ComposedAnswer
    verdicts: list[str]                     # per-claim SUPPORTED/PARTIAL/UNSUPPORTED (from multi-vote)
    validation: Any                         # ValidationReport (lexical)

    # output
    final_answer: str
    abstained: bool
    abstain_reason: str

    # observability — trace list is auto-appended across nodes via `add` reducer
    trace: Annotated[list[TraceStep], add]
