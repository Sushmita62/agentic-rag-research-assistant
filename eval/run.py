"""Run the benchmark end-to-end against the currently indexed papers.
Writes eval/results.jsonl and eval/report.md.

Prereq: attention.pdf (arxiv 1706.03762) must be ingested.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.graph import build_graph
from app.storage import get_engine
from eval.metrics import aggregate, score_question


BENCH = Path(__file__).parent / "benchmark.jsonl"
RESULTS = Path(__file__).parent / "results.jsonl"
REPORT = Path(__file__).parent / "report.md"


def _serialize_result(result: dict, latency_ms: int) -> dict:
    composed = result.get("composed")
    verdicts = result.get("verdicts", [])
    claims = []
    if composed:
        for c, v in zip(composed.claims, verdicts):
            claims.append({
                "text": c.text, "verdict": v,
                "citations": [{"chunk_id": x.chunk_id} for x in c.citations],
            })
    evidence = []
    for fid in result.get("reranked_chunk_ids", []):
        d = result.get("chunks_by_fid", {}).get(fid)
        if d:
            evidence.append({
                "chunk_id": d["chunk_id"], "page": d["page"],
                "section": d["section"], "title": d["title"],
            })
    return {
        "answer": result.get("final_answer", ""),
        "abstained": bool(result.get("abstained")),
        "abstain_reason": result.get("abstain_reason"),
        "claims": claims,
        "evidence": evidence,
        "total_latency_ms": latency_ms,
        "trace": result.get("trace", []),
    }


def _write_report(agg, per_q) -> None:
    lines: list[str] = []
    lines.append("# Evaluation Report\n")
    lines.append(f"- Questions: **{agg.n}**")
    lines.append(f"- Abstain accuracy: **{agg.abstain_accuracy:.0%}**")
    lines.append(f"- Recall@5 (SUPPORTED q's): **{agg.recall_at_5:.0%}**")
    lines.append(f"- Citation precision: **{agg.citation_precision:.0%}**")
    lines.append(f"- Must-contain hit ratio: **{agg.must_contain_hit_ratio:.0%}**")
    lines.append(f"- Latency p50 / p95: **{agg.p50_latency_ms} ms / {agg.p95_latency_ms} ms**")
    lines.append("\n## Per-question\n")
    lines.append("| ID | Verdict OK | Recall@5 | CitePrec | MustContain | Latency (ms) |")
    lines.append("|---|---|---|---|---|---|")
    for q in per_q:
        r = "-" if q.recall_at_5 is None else f"{q.recall_at_5:.0%}"
        c = "-" if q.citation_precision is None else f"{q.citation_precision:.0%}"
        m = "-" if q.must_contain_hit_ratio is None else f"{q.must_contain_hit_ratio:.0%}"
        ok = "✓" if q.verdict_correct else "✗"
        lines.append(f"| {q.id} | {ok} | {r} | {c} | {m} | {q.latency_ms} |")
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def _load_completed() -> dict[str, dict]:
    """Return {id: record} for previously successful runs. Errored rows are ignored so they retry."""
    if not RESULTS.exists():
        return {}
    done: dict[str, dict] = {}
    for line in RESULTS.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
            if not rec.get("result", {}).get("error"):
                done[rec["bench"]["id"]] = rec
        except Exception:
            continue
    return done


def main() -> None:
    if not BENCH.exists():
        print(f"missing {BENCH}")
        sys.exit(1)

    completed = _load_completed()
    if completed:
        print(f"resuming — {len(completed)} question(s) already completed")

    get_engine()
    graph = None                                                # lazy: only build if we have work
    print(f"benchmark: {BENCH}")

    per_q = []
    all_records: list[dict] = []                                # rewrite RESULTS at end (dedup)

    for line in BENCH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        bench = json.loads(line)

        if bench["id"] in completed:
            rec = completed[bench["id"]]
            print(f"✓ {bench['id']}: (cached)")
            per_q.append(score_question(bench, rec["result"]))
            all_records.append(rec)
            continue

        if graph is None:
            graph = build_graph()

        print(f"→ {bench['id']}: {bench['question'][:60]}", flush=True)
        t0 = time.perf_counter()
        try:
            result = graph.invoke({"query": bench["question"], "trace": []})
            serialized = _serialize_result(result, int((time.perf_counter() - t0) * 1000))
        except Exception as e:
            print(f"  ERROR: {e}")
            serialized = {"answer": "", "abstained": False, "claims": [],
                          "evidence": [], "total_latency_ms": 0, "error": str(e)}
        all_records.append({"bench": bench, "result": serialized})
        per_q.append(score_question(bench, serialized))
        m = per_q[-1]
        r = "-" if m.recall_at_5 is None else f"R@5={m.recall_at_5:.0%}"
        print(f"  verdict={'✓' if m.verdict_correct else '✗'}  {r}  {m.latency_ms}ms")

    with RESULTS.open("w", encoding="utf-8") as out:
        for rec in all_records:
            out.write(json.dumps(rec) + "\n")

    agg = aggregate(per_q)
    _write_report(agg, per_q)
    print()
    print("=" * 60)
    print(f"Abstain accuracy:   {agg.abstain_accuracy:.0%}")
    print(f"Recall@5:           {agg.recall_at_5:.0%}")
    print(f"Citation precision: {agg.citation_precision:.0%}")
    print(f"Must-contain:       {agg.must_contain_hit_ratio:.0%}")
    print(f"Latency p50 / p95:  {agg.p50_latency_ms}ms / {agg.p95_latency_ms}ms")
    print(f"Report:             {REPORT}")


if __name__ == "__main__":
    main()
