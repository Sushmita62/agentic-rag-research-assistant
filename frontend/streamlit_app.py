"""Streamlit UI. Three tabs: Ask, Compare, Literature Review.

Talks to the FastAPI backend only. Citations displayed as "Paper title, p. N"
by joining chunk_id → evidence lookup.
"""
from __future__ import annotations

import os

import pandas as pd
import requests
import streamlit as st


API = os.getenv("API_URL", "http://127.0.0.1:8000")


st.set_page_config(page_title="Research Assistant", page_icon="📄", layout="wide")
st.title("📄 AI Research Assistant")
st.caption("Hybrid retrieval · cross-encoder rerank · multi-vote verify · LangGraph · Groq")


# ── Sidebar: papers ──────────────────────────────────────────────────────────
def _load_papers() -> list[dict]:
    try:
        return requests.get(f"{API}/papers", timeout=10).json()
    except Exception as e:
        st.sidebar.error(f"API unreachable at {API}: {e}")
        return []


with st.sidebar:
    st.header("Papers")

    up = st.file_uploader("Upload a PDF", type=["pdf"])
    if up is not None:
        if st.button("Ingest", type="primary"):
            with st.spinner("Extracting, chunking, embedding, indexing..."):
                r = requests.post(
                    f"{API}/papers/upload",
                    files={"file": (up.name, up.getvalue(), "application/pdf")},
                    timeout=600,
                )
            if r.ok:
                st.success(f"Indexed as {r.json()['paper_id']}")
                st.rerun()
            else:
                st.error(r.text)

    papers = _load_papers()
    st.divider()
    if not papers:
        st.info("No papers yet. Upload one above.")
    else:
        st.markdown(f"**{len(papers)} indexed**")
        for p in papers:
            st.markdown(f"- **{p['title'][:60]}**  \n  <sub>`{p['id']}` · {p['num_pages']}p · {p['status']}</sub>",
                        unsafe_allow_html=True)


title_by_id = {p["id"]: p["title"] for p in papers}


# ── Helpers ──────────────────────────────────────────────────────────────────
def _cite_label(chunk_id: str, evidence_lookup: dict[str, dict]) -> str:
    """Turn a raw chunk_id into a human-readable label using paper title + page."""
    e = evidence_lookup.get(chunk_id)
    if not e:
        return f"`{chunk_id}`"
    title = e.get("title", "")[:40]
    page = e.get("page", "?")
    return f"**{title}**, p. {page}"


# ── Tabs ─────────────────────────────────────────────────────────────────────
tab_ask, tab_cmp, tab_lit = st.tabs(["🔎 Ask", "⚖️ Compare", "📚 Literature Review"])


# ── Ask ──────────────────────────────────────────────────────────────────────
with tab_ask:
    q = st.text_input("Ask a question about the indexed papers",
                      placeholder="e.g. what is scaled dot-product attention?")
    if st.button("Ask", type="primary", disabled=not q, key="ask_btn"):
        with st.spinner("retrieve → compose → verify → validate → decide"):
            r = requests.post(f"{API}/research/query", json={"question": q}, timeout=300)
        if not r.ok:
            st.error(r.text)
        else:
            resp = r.json()

            if resp.get("abstained"):
                st.warning(f"⚠ Abstained — {resp.get('abstain_reason', '')}")
            else:
                st.success("Answer grounded")

            st.markdown("### Answer")
            st.write(resp["answer"])

            # chunk_id → evidence-record lookup (title + page + text + section)
            elookup = {e["chunk_id"]: e for e in resp.get("evidence", [])}

            if resp.get("claims"):
                st.markdown("### Claims")
                for c in resp["claims"]:
                    color = {"SUPPORTED": "✅", "PARTIAL": "🟡", "UNSUPPORTED": "❌"}.get(
                        c["verdict"], "❓")
                    cites = "  ·  ".join(_cite_label(x["chunk_id"], elookup)
                                          for x in c["citations"])
                    st.markdown(f"- {color} **[{c['verdict']}]** {c['text']}  \n"
                                f"  <sub>Sources: {cites}</sub>", unsafe_allow_html=True)

            if resp.get("evidence"):
                with st.expander(f"Evidence chunks ({len(resp['evidence'])})"):
                    for e in resp["evidence"]:
                        st.markdown(f"**{e['title'][:60]}** · p. {e['page']} · "
                                    f"[{e['section']}] · `{e['chunk_id']}`")
                        st.caption(" ".join(e["text"].split())[:400] + "...")
                        st.divider()

            if resp.get("trace"):
                with st.expander(f"Decision trace ({len(resp['trace'])} nodes)"):
                    for t in resp["trace"]:
                        st.markdown(
                            f"- **{t['node']}** — {t['latency_ms']}ms — {t.get('detail', '')}")


# ── Compare ──────────────────────────────────────────────────────────────────
with tab_cmp:
    if len(papers) < 2:
        st.info("Upload at least 2 papers to compare.")
    else:
        selected = st.multiselect(
            "Papers to compare",
            options=[p["id"] for p in papers],
            format_func=lambda pid: title_by_id.get(pid, pid)[:60],
            default=[p["id"] for p in papers[:min(3, len(papers))]],
        )

        try:
            defaults = requests.get(f"{API}/dimensions/default", timeout=5).json()
        except Exception:
            defaults = ["Method", "Dataset", "Metric", "Result", "Limitations"]

        dims_text = st.text_input(
            "Dimensions (comma-separated)",
            value=", ".join(defaults),
        )
        dims = [d.strip() for d in dims_text.split(",") if d.strip()]

        if st.button("Compare", type="primary",
                     disabled=(len(selected) < 2 or not dims), key="cmp_btn"):
            with st.spinner(f"Comparing {len(selected)} papers across {len(dims)} dimensions..."):
                r = requests.post(f"{API}/research/compare",
                                  json={"paper_ids": selected, "dimensions": dims},
                                  timeout=600)
            if not r.ok:
                st.error(r.text)
            else:
                data = r.json()
                # Build a DataFrame with values + citation superscripts
                display_rows = []
                citation_details: dict[tuple[str, str], dict] = {}
                for row in data["rows"]:
                    d_row = {"Paper": row["paper_title"][:50]}
                    for dim in data["dimensions"]:
                        cell = row["cells"][dim]
                        val = cell["value"]
                        cit = cell["citation"]
                        if cit and val != "NOT_REPORTED":
                            d_row[dim] = f"{val} [p.{cit['page']}]"
                            citation_details[(row["paper_id"], dim)] = cit
                        else:
                            d_row[dim] = val
                    display_rows.append(d_row)
                st.markdown("### Comparison")
                st.dataframe(pd.DataFrame(display_rows), use_container_width=True)

                with st.expander("Citation details"):
                    if not citation_details:
                        st.caption("No citations returned.")
                    for (pid, dim), cit in citation_details.items():
                        st.markdown(f"**{title_by_id.get(pid, pid)[:40]}** · {dim} · "
                                    f"p. {cit['page']} · `{cit['chunk_id']}`")
                        st.caption(" ".join((cit.get("text") or "").split())[:300] + "...")
                        st.divider()


# ── Literature Review ────────────────────────────────────────────────────────
with tab_lit:
    if not papers:
        st.info("Upload at least one paper to generate a review.")
    else:
        topic = st.text_input(
            "Review topic",
            placeholder="e.g. attention mechanisms for sequence transduction",
        )
        scope = st.multiselect(
            "Papers in scope (empty = all)",
            options=[p["id"] for p in papers],
            format_func=lambda pid: title_by_id.get(pid, pid)[:60],
            default=[],
        )

        if st.button("Generate", type="primary", disabled=not topic, key="lit_btn"):
            with st.spinner("Per-paper summaries → thematic composition..."):
                payload = {"topic": topic}
                if scope:
                    payload["paper_ids"] = scope
                r = requests.post(f"{API}/research/literature-review",
                                  json=payload, timeout=900)
            if not r.ok:
                st.error(r.text)
            else:
                data = r.json()
                st.markdown("### Review")
                st.markdown(data["review_markdown"])

                if data.get("citations"):
                    with st.expander(f"Citations ({len(data['citations'])})"):
                        for c in data["citations"]:
                            title = c.get("title", "")[:50]
                            page = c.get("page", "?")
                            st.markdown(f"- **{title}** · p. {page} · `{c['chunk_id']}`")
                            st.caption(" ".join((c.get("text") or "").split())[:300] + "...")

                if data.get("per_paper_summaries"):
                    with st.expander("Per-paper summaries (extracted before composition)"):
                        for s in data["per_paper_summaries"]:
                            st.markdown(f"**{s['paper_title'][:60]}**")
                            if s.get("problem"): st.markdown(f"- **Problem:** {s['problem']}")
                            if s.get("method"): st.markdown(f"- **Method:** {s['method']}")
                            if s.get("findings"): st.markdown(f"- **Findings:** {s['findings']}")
                            if s.get("limitations"): st.markdown(f"- **Limitations:** {s['limitations']}")
                            st.divider()
