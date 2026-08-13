# AI Research Assistant

Grounded question answering, multi-paper comparison, and literature-review
synthesis over research papers. Hybrid retrieval, cross-encoder reranking,
multi-vote adversarial verification, and LangGraph orchestration — with
structured abstention when the evidence isn't strong enough.

![Literature review with human-readable inline citations](docs/screenshot.png)

---

## Highlights

- **100% Recall@5**, **92% citation precision**, **92% abstain accuracy** on a 12-question hand-curated benchmark.
- **Three end-user workflows** — grounded Q&A, multi-paper comparison table, and topical literature review — each with structural safeguards against hallucination.
- **3-layer citation contract** prevents fabricated references: Pydantic-enforced structured output → allowed-chunk-id restriction → lexical grounding validator.
- **3-vote adversarial verifier** (supportive + skeptical + neutral prompts, majority vote) — catches "the LLM confidently agrees with itself" failures.
- **Full decision trace** per query (per-node latency + rationale) — surfaced in UI for explainability.
- **Prompt-injection hardened**: doc content in fenced blocks, structured-output-only egress, no free-form channel.
- **Fully open-weights**, $0 running cost: Groq Llama 3.3 70B + local BAAI models.
- **61 tests** — unit, integration, and adversarial.

---

## Architecture

```
                          Streamlit UI   (3 tabs)
                Ask  ·  Compare  ·  Literature Review
                             │  HTTP
                        FastAPI API
       ┌──────────────┬─────┴───────────┬────────────────────┐
       ▼              ▼                 ▼                    ▼
   POST /query    POST /compare    POST /literature-   GET /papers
   (LangGraph)    (per-paper       review              /upload etc.
       │           extract →        (summarize →
       │           table)           compose review)
       ▼
   retrieve → compose → verify → validate → decide → END
                                              │
                                          abstain

   ┌────────────────── Services ───────────────────┐
   │ HybridRetriever  (BM25 + FAISS + RRF)         │
   │ Reranker         (bge-reranker-base)          │
   │ AnswerComposer   (Groq + Pydantic)            │
   │ Verifier         (3-vote adversarial)         │
   │ CitationValidator (lexical grounding)         │
   │ Compare          (structured cell extraction) │
   │ LitReview        (per-paper → sectioned)      │
   └───────────────────────────────────────────────┘
                             │
   ┌──────────── Storage ────────────────┐
   │ SQLite (papers, chunks, embed cache)│
   │ FAISS  (dense vectors)              │
   │ BM25   (sparse index, pickled)      │
   └─────────────────────────────────────┘
```

No answer, comparison cell, or review sentence reaches the user without a valid
citation to a real retrieved chunk — enforced structurally, not by asking the
LLM nicely.

---

## Screenshots

### Ask — grounded Q&A with claim verdicts
![Ask tab showing a grounded answer](docs/ask.png)

### Compare — multi-paper structured table with page-level citations
![Compare tab showing a comparison table with NOT_REPORTED cells](docs/compare.png)

### Literature Review — sectioned synthesis with inline citations
![Literature Review input form with topic and paper scope](docs/review-input.png)

---

## Stack

| Layer | Choice | Why |
|---|---|---|
| LLM | Groq (Llama 3.3 70B) | Open weights, generous free tier, low latency |
| Embeddings | `BAAI/bge-small-en-v1.5` (local) | Strong MTEB score at 30M params, runs on CPU |
| Reranker | `BAAI/bge-reranker-base` (local) | Best quality-per-cost for RAG shortlisting |
| Vector search | FAISS `IndexIDMap(IndexFlatIP)` | Exact search, right choice up to ~1M chunks |
| Sparse search | `rank_bm25` (Okapi) | Catches acronyms/model names dense misses |
| Fusion | Reciprocal Rank Fusion (RRF, k=60) | Rank-based, no per-corpus tuning |
| Metadata store | SQLite via SQLAlchemy | Zero-ops, sufficient for portfolio scale |
| Orchestration | LangGraph | Conditional edges + parallel-safe node fan-out |
| Validation | Pydantic v2 | Structured LLM output, schema enforcement |
| API | FastAPI | Async, Pydantic-native, TestClient for tests |
| UI | Streamlit | Minimum-code demoable UI |

Explicitly avoided: LangChain retrievers/chains (opaque), OpenAI/hosted embeddings
(cost + lock-in), Docker (unnecessary for single-user portfolio).

---

## Setup

Requires Python 3.11+ and a free [Groq API key](https://console.groq.com/keys).

```powershell
git clone https://github.com/<your-user>/ai-research-assistant.git
cd ai-research-assistant

python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

Copy-Item .env.example .env
# Edit .env — paste your Groq key on the GROQ_API_KEY line
```

### Run (two terminals)

```powershell
# Terminal 1 — API backend
uvicorn app.api:app --reload

# Terminal 2 — Streamlit UI
streamlit run frontend/streamlit_app.py
```

Browser opens at http://localhost:8501. The UI has three tabs:

- **Ask** — grounded Q&A with color-coded claim verdicts and an expandable decision trace.
- **Compare** — pick 2+ papers and dimensions (Method, Dataset, Metric, Result, Limitations by default); the table shows cited values with page superscripts and `NOT_REPORTED` for missing fields.
- **Literature Review** — enter a topic and paper scope; get a sectioned Markdown review (Introduction → Methods → Findings → Agreements → Disagreements → Research gaps → Conclusion) with inline `[Paper title, p.N]` citations.

### Or from the CLI

```powershell
Invoke-WebRequest -Uri "https://arxiv.org/pdf/1706.03762" -OutFile "data\pdfs\attention.pdf"
python scripts/ingest_folder.py data/pdfs
python scripts/ask.py "what is scaled dot-product attention"
```

---

## Evaluation

12-question hand-curated benchmark on the *Attention Is All You Need* paper —
8 SUPPORTED questions with gold pages, 4 out-of-scope questions that must ABSTAIN.

| Metric | Result |
|---|---|
| **Recall@5** | 100% |
| **Citation precision** | 92% |
| **Abstain accuracy** | 92% (11/12) |
| Must-contain hit ratio | 75% |
| Latency p50 / p95 | 37s / 100s (free-tier Groq) |

Re-run:

```powershell
python eval/run.py
```

Writes `eval/results.jsonl` (raw) and `eval/report.md` (summary table). The runner
is resumable: rate-limited questions retry on the next invocation.

**Full per-question breakdown:** [`eval/report.md`](eval/report.md).

---

## Design decisions

- **LangGraph only where it earns its keep.** Ingestion is a straight-line pipeline — plain async Python. The graph is reserved for the answer flow, where conditional routing (finalize vs abstain) and future parallel fan-out actually pay off. Comparison and literature review use dedicated pipelines, not the graph.
- **Citations are a contract, not an agent.** Every claim MUST carry ≥1 citation (Pydantic `min_length=1`); cited `chunk_id`s must exist in the retrieved set (post-hoc check); claim text must lexically overlap the cited chunk (deterministic validator). No LLM was harmed writing a "citation agent".
- **Structured extraction over free-form generation.** Comparison cells and per-paper review summaries are Pydantic-typed. If the LLM says "value: X" without a valid supporting `chunk_id`, the cell collapses to `NOT_REPORTED`. Nothing invented sneaks through.
- **Multi-vote > single verifier.** Three parallel LLM calls with distinct stances (supportive / skeptical / neutral) — majority vote. Cuts the "LLM confidently agrees with itself" failure mode.
- **Two real agents.** Query intent + answer composition make model-driven decisions; everything else (retrieve, verify, validate, cell-extract, per-paper summarize, review-compose) is a service or a graph node — no fake "agents" that are just functions with an LLM call.

---

## Test suite

61 tests: unit + integration + adversarial.

```powershell
pytest tests/ -v
```

- **Storage, PDF, sections, chunker, embeddings, vector, BM25, hybrid, reranker**
- **Answer composer** — mocked LLM: schema violations, retry, allowed-id rejection
- **Verifier** — majority vote, invalid verdict handling
- **Validator** — lexical grounding, abstain threshold
- **Graph** — routing, terminal state
- **Compare** — cell extraction, NOT_REPORTED enforcement, paper filter
- **LitReview** — per-paper summary, review composition, bogus-citation rejection
- **API** — health, upload rejects non-PDF, query/compare/review response shape
- **Prompt-injection defenses** — fenced blocks, hijacked-id rejection, free-form rejection
- **Metrics** — Recall@K, citation precision, abstain accuracy math
- **Title extraction** — arXiv watermark skipping, metadata preference
- **Citation prettification** — inline `[chunk_id]` → `[Paper title, p.N]`

---

## Future work

- **Query Planner intent routing** so a single `/query` endpoint dispatches to Ask / Compare / LitReview instead of separate endpoints.
- **LLM-judge for qualifier detection** — motivation from a benchmark miss: *"Does this paper discuss image classification with CNNs?"* was incorrectly answered because the paper mentions CNNs (just not for image classification). A dedicated qualifier-grounding check would catch this.
- **OCR fallback** (`pytesseract`) for scanned PDFs.
- **Figure + table understanding** (multimodal embeddings).
- **Migrate FAISS → Qdrant** for native metadata filters and persistence at scale.
- **Streaming responses** (SSE from FastAPI, incremental Streamlit render).
- **Cost & token telemetry** per query, surfaced in the trace panel.

---

## Repo layout

```
ai-research-assistant/
├── app/                          # library code
│   ├── api.py                    # FastAPI app + endpoints
│   ├── graph.py                  # LangGraph orchestration (Ask flow)
│   ├── state.py                  # ResearchState (TypedDict)
│   ├── answer.py                 # AnswerComposer + citation contract
│   ├── verifier.py               # 3-vote adversarial verify
│   ├── validator.py              # lexical grounding + abstain rule
│   ├── compare.py                # multi-paper structured comparison
│   ├── litreview.py              # per-paper summarize → sectioned review
│   ├── hybrid.py                 # BM25 + FAISS + RRF (with paper filter)
│   ├── reranker.py               # cross-encoder
│   ├── embeddings.py             # bge-small wrapper + SQL cache
│   ├── bm25.py                   # BM25 index
│   ├── vector.py                 # FAISS wrapper
│   ├── chunker.py                # section-aware token windowing
│   ├── sections.py               # regex heading detector + cleaner
│   ├── pdf.py                    # PyMuPDF extractor + metadata
│   ├── ingest.py                 # end-to-end ingest pipeline
│   ├── llm.py                    # Groq client (OpenAI-compat)
│   ├── storage.py                # SQLAlchemy models + session
│   └── config.py                 # Pydantic Settings
├── frontend/
│   └── streamlit_app.py          # 3-tab UI (Ask / Compare / LitReview)
├── scripts/
│   ├── ingest_folder.py          # bulk-ingest a folder of PDFs
│   ├── search.py                 # retrieval-only CLI
│   ├── ask.py                    # full graph CLI
│   └── refresh_titles.py         # re-extract titles for indexed papers
├── eval/
│   ├── benchmark.jsonl           # 12 hand-curated Q&A
│   ├── metrics.py
│   ├── run.py                    # resumable benchmark runner
│   └── report.md                 # the numbers
├── tests/                        # 61 tests
├── data/pdfs/                    # your PDFs (gitignored)
├── data/index/                   # FAISS + SQLite + BM25 (gitignored)
├── requirements.txt
├── .env.example
└── README.md
```
