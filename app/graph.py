"""LangGraph orchestration.

  retrieve → compose → verify → validate → decide → END

Trace steps auto-append via state's `add` reducer. Nodes are thin — real logic
lives in services (hybrid, reranker, answer, verifier, validator).
"""
from __future__ import annotations

import time
from typing import Callable

from langgraph.graph import END, StateGraph
from sqlalchemy import select

from app.answer import compose_answer
from app.hybrid import HybridRetriever
from app.llm import LLMClient
from app.reranker import Reranker
from app.state import ResearchState
from app.storage import Chunk, Paper, session
from app.validator import validate
from app.verifier import verify_all


def _step(node: str, detail: str, t0: float, **meta) -> dict:
    return {"node": node, "detail": detail,
            "latency_ms": int((time.perf_counter() - t0) * 1000), "meta": meta}


def make_retrieve_node(retriever: HybridRetriever, reranker: Reranker,
                       rerank_k: int = 5) -> Callable:
    def node(state: ResearchState) -> dict:
        t0 = time.perf_counter()
        cands = retriever.retrieve(state["query"], k=20, per_source=30)
        fids = [f for f, _ in cands]

        with session() as s:
            rows = s.execute(
                select(Chunk, Paper).join(Paper).where(Chunk.faiss_id.in_(fids))
            ).all()
            by_fid = {c.faiss_id: {"chunk_id": c.id, "text": c.text, "page": c.page,
                                   "section": c.section, "title": p.title}
                      for c, p in rows}

        pairs = [(f, by_fid[f]["text"]) for f in fids if f in by_fid]
        top = reranker.rerank(state["query"], pairs, top_k=rerank_k)
        top_fids = [f for f, _ in top]

        return {
            "reranked_chunk_ids": top_fids,
            "chunks_by_fid": {f: by_fid[f] for f in top_fids},
            "trace": [_step("retrieve", f"{len(cands)} candidates → {len(top)} reranked",
                            t0, top_ids=top_fids)],
        }
    return node


def make_compose_node(llm: LLMClient) -> Callable:
    def node(state: ResearchState) -> dict:
        t0 = time.perf_counter()
        top_chunks = [state["chunks_by_fid"][f] for f in state["reranked_chunk_ids"]]
        answer = compose_answer(state["query"], top_chunks, llm=llm)
        return {
            "composed": answer,
            "trace": [_step("compose", f"{len(answer.claims)} claims drafted", t0)],
        }
    return node


def make_verify_node(llm: LLMClient) -> Callable:
    def node(state: ResearchState) -> dict:
        t0 = time.perf_counter()
        answer = state["composed"]
        chunks_by_id = {c["chunk_id"]: c["text"] for c in state["chunks_by_fid"].values()}
        verdicts = verify_all(answer.claims, chunks_by_id, llm=llm)
        return {
            "verdicts": verdicts,
            "trace": [_step("verify", f"multi-vote verdicts: {verdicts}", t0)],
        }
    return node


def _validate_node(state: ResearchState) -> dict:
    t0 = time.perf_counter()
    answer = state["composed"]
    chunks_by_id = {c["chunk_id"]: c["text"] for c in state["chunks_by_fid"].values()}
    report = validate(answer, chunks_by_id)
    return {
        "validation": report,
        "trace": [_step("validate",
                        f"{report.supported_ratio:.0%} lexically supported", t0)],
    }


def _decide(state: ResearchState) -> str:
    verdicts = state.get("verdicts", [])
    n_supported_llm = sum(1 for v in verdicts if v == "SUPPORTED")
    llm_ratio = n_supported_llm / len(verdicts) if verdicts else 0.0
    lex = state.get("validation")
    lex_ratio = lex.supported_ratio if lex else 0.0
    # Abstain if EITHER signal says the answer is weakly grounded.
    if llm_ratio < 0.5 or lex_ratio < 0.5:
        return "abstain"
    return "finalize"


def _finalize_node(state: ResearchState) -> dict:
    return {"final_answer": state["composed"].summary, "abstained": False,
            "trace": [{"node": "finalize", "detail": "answer accepted", "latency_ms": 0}]}


def _abstain_node(state: ResearchState) -> dict:
    verdicts = state.get("verdicts", [])
    llm_r = sum(1 for v in verdicts if v == "SUPPORTED") / len(verdicts) if verdicts else 0.0
    lex_r = state["validation"].supported_ratio if state.get("validation") else 0.0
    reason = f"insufficient grounded evidence (LLM {llm_r:.0%}, lexical {lex_r:.0%})"
    return {
        "final_answer": "The provided documents do not contain sufficient information to answer this question with confidence.",
        "abstained": True,
        "abstain_reason": reason,
        "trace": [{"node": "abstain", "detail": reason, "latency_ms": 0}],
    }


def build_graph(
    retriever: HybridRetriever | None = None,
    reranker: Reranker | None = None,
    llm: LLMClient | None = None,
):
    retriever = retriever or HybridRetriever()
    reranker = reranker or Reranker()
    llm = llm or LLMClient()

    g = StateGraph(ResearchState)
    g.add_node("retrieve", make_retrieve_node(retriever, reranker))
    g.add_node("compose", make_compose_node(llm))
    g.add_node("verify", make_verify_node(llm))
    g.add_node("validate", _validate_node)
    g.add_node("finalize", _finalize_node)
    g.add_node("abstain", _abstain_node)

    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "compose")
    g.add_edge("compose", "verify")
    g.add_edge("verify", "validate")
    g.add_conditional_edges("validate", _decide, {"finalize": "finalize", "abstain": "abstain"})
    g.add_edge("finalize", END)
    g.add_edge("abstain", END)

    return g.compile()
