"""End-to-end Q&A via LangGraph. Retrieve → compose → verify → validate → decide.

Usage:
    python scripts/ask.py "your question"
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.graph import build_graph
from app.storage import get_engine


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        return
    query = sys.argv[1]

    get_engine()
    graph = build_graph()
    result = graph.invoke({"query": query, "trace": []})

    print("=" * 60)
    if result.get("abstained"):
        print(f"⚠  ABSTAIN — {result.get('abstain_reason', '')}")
        print()
    print(result["final_answer"])

    composed = result.get("composed")
    verdicts = result.get("verdicts", [])
    if composed:
        print()
        print("Claims:")
        for claim, v in zip(composed.claims, verdicts):
            mark = {"SUPPORTED": "✓", "PARTIAL": "~", "UNSUPPORTED": "✗"}.get(v, "?")
            print(f"  {mark} [{v:12s}]  {claim.text}")
            print(f"      citations: {', '.join(c.chunk_id for c in claim.citations)}")

    print()
    print("Trace:")
    for step in result.get("trace", []):
        print(f"  · {step['node']:10s} {step.get('latency_ms', 0):4d}ms  {step.get('detail', '')}")


if __name__ == "__main__":
    main()
