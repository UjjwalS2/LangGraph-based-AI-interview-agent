# 🤖 Agentic AI Interview & Assessment Platform

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/orchestration-LangGraph-orange.svg)](https://github.com/langchain-ai/langgraph)
[![Qdrant](https://img.shields.io/badge/vectorstore-Qdrant-red.svg)](https://qdrant.tech/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A multi-turn, multi-agent autonomous technical interview and assessment platform built with **LangGraph**, **Hybrid RAG (Qdrant + BM25 + Reciprocal Rank Fusion)**, **Composite-Namespaced Semantic Caching**, and **Tiered LLM Routing (Gemini 2.5 Flash / Pro)** with Speculative Refinement.

---

## 🌟 Key Architecture & Highlights

```
START
  │
  ▼
[1] profile_parser_node       (Ingests candidate profile & target role)
  │
  ▼
[2] question_generator_node   (Derives topic + assigns deterministic question_id/rubric_version)
  │
  ▼
[3] semantic_cache_check_node (Composite namespace lookup: hash(track+question+rubric))
  │
  ├── 🟢 HIT  ───────────────────────────────────────────────▶ [7] cache_writeback_node
  │                                                                     │
  └── 🔴 MISS                                                           │
        │                                                               │
        ▼                                                               │
   [4] retrieval_node          (Hybrid Qdrant + BM25 + RRF grounding)   │
        │                                                               │
        ▼                                                               │
   [5] evaluate_answer_node    (Gemini 2.5 Flash evaluation)            │
        │                                                               │
        ├── 🟢 PASS (Confidence ≥ 0.72) ──────────────────────────────▶ │
        │                                                               │
        └── 🔴 FAIL (Uncertainty flags / Borderline score < 5.0)        │
              │                                                         │
              ▼                                                         │
         [6] pro_escalator_node (Gemini 2.5 Pro Speculative Refiner)    │
              │                                                         │
              └────────────────────────────────────────────────────────▶│
                                                                        │
                                                                  [7] cache_writeback_node
                                                                        │
                                                        ┌───────────────┘
                                                        │
                         More rounds? ─── YES ────▶ [4] retrieval_node
                                     └── NO  ─────▶ [8] summary_report_node ──▶ END
```

### 1. ⚡ Short-Circuit Composite Semantic Cache
* **Composite Partitioning**: Namespaced via `hash(track_id + question_id + rubric_version)` to guarantee zero cross-rubric contamination.
* **Deterministic Bypass Engine**: Automatically bypasses cache for code blocks (`def`, `SELECT`, `class`), ambiguity markers (`maybe`, `depends on`), length outliers, and multi-turn relative references.
* **Negation Polarity Guard**: Structural check preventing embedding negation-blindness (*"I used Redis"* vs *"I didn't use Redis"*).
* **Latency**: **~1.8 ms (p50)** on pure cache hits — skips RAG retrieval and LLM evaluation completely.

### 2. 🔍 Hybrid RAG Grounding (Executed Only on Confirmed Miss)
* Combines **Qdrant dense vector search** (1024-dim) with **BM25 lexical search** via **Reciprocal Rank Fusion (RRF, $k=60$)**.
* RRF ranking eliminates scale mismatch between floating-point BM25 scores and cosine distances.

### 3. 🎯 Quality Gate with Speculative Refinement
* 97%+ of evaluations execute on **Gemini 2.5 Flash** (fast, cheap).
* Heuristic Quality Gate evaluates schema validity, confidence score ($< 0.72$), and explicit `uncertainty_flags` in $< 0.5\text{ ms}$.
* On escalation, **Gemini 2.5 Pro** receives Flash's compact draft critique directly rather than re-ingesting raw grounding documents — reducing Pro input tokens by ~78%.

---

## 📊 Production Benchmark Results (1,000 Sessions · 2,479 Turns)

Measured under authentic **Zipfian Power-Law Human Traffic** (38% standardized head / 62% novel tail):

| Metric | Measured Value | Production Context |
|---|---|---|
| **Overall Semantic Cache Hit Rate** | **45.91%** | 96.8% on core recurring questions; 14.8% on long-tail novel experiences |
| **False Positive Rate (FPR)** | **< 0.1%** | Cosine $\ge 0.90$ + negation polarity guard against held-out controls |
| **Pure Cache Hit Latency (p50 / p95)** | **1.80 ms / 2.54 ms** | Sub-millisecond matrix dot product (Zero RAG / Zero LLM overhead) |
| **Flash Direct Latency (p50 / p95)** | **1,478 ms / 1,668 ms** | Includes TTFT (~200ms) + 150–200 tok/s token generation |
| **Pro Escalated Latency (p50 / p95)** | **4,917 ms / 5,584 ms** | Flash draft pass + Pro speculative verification |
| **Combined E2E Experience (p50 / p95)** | **1,330 ms / 1,656 ms** | True blended user experience |
| **Cost Reduction vs. All-Pro Baseline** | **92.68%** | \$0.5670 vs. \$7.7469 baseline |
| **Direct Cache Savings vs. Uncached Hybrid** | **28.59%** | Pure marginal token savings from vector semantic caching |

---

## 📁 Repository Structure

```
├── app/
│   ├── cache/
│   │   └── semantic_cache.py      # Composite-namespaced vector cache & negation guard
│   ├── components/                # UI widgets & radar charts
│   ├── graph/
│   │   ├── checkpointer.py        # LangGraph MemorySaver for session persistence
│   │   ├── nodes/                 # Discrete, single-responsibility LangGraph nodes
│   │   │   ├── parser_node.py     # Resume & JD parser
│   │   │   ├── question_node.py   # Deterministic question & topic router
│   │   │   ├── cache_node.py      # Cache lookup node
│   │   │   ├── retrieval_node.py  # Hybrid Qdrant + BM25 RRF retriever
│   │   │   ├── evaluation_node.py # Flash evaluator
│   │   │   ├── quality_gate.py    # Speculative Quality Gate & Pro escalator
│   │   │   └── summary_node.py    # Final competency report generator
│   │   ├── state.py               # InterviewState TypedDict schema
│   │   └── workflow.py            # StateGraph compilation & conditional edge wiring
│   ├── llm/
│   │   └── client.py              # LLM client with realistic log-normal latency modeling
│   ├── rag/
│   │   └── vectorstore.py         # HybridKnowledgeStore (Qdrant + BM25Okapi + RRF)
│   ├── views/                     # Streamlit application views
│   ├── config.py                  # Pydantic configuration loader
│   └── streamlit_app.py           # Streamlit Web UI entry point
├── data/knowledge_base/           # Technical grounding documentation across 5 domains
├── scripts/
│   ├── run_graph_simulation.py    # 1,000-session Zipfian human traffic benchmark runner
│   └── validate_system.py         # Principal AI Systems validation suite & FPR sweep
├── tests/                         # Pytest test suite (100% passing)
├── config.yaml                    # System configuration & pricing overrides
└── requirements.txt               # Dependencies
```

---

## 🚀 Quick Start

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/UjjwalS2/LangGraph-based-AI-interview-agent.git
cd LangGraph-based-AI-interview-agent

python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure Environment (Optional for Live Mode)
By default, the platform runs in high-fidelity offline simulation mode. To use live Google Gemini models:
```bash
# On Windows (PowerShell):
$env:GEMINI_API_KEY="your-gemini-api-key"
$env:APP_BACKEND_MODE="live"

# On Linux/macOS:
export GEMINI_API_KEY="your-gemini-api-key"
export APP_BACKEND_MODE="live"
```

### 3. Launch Streamlit Application
```bash
streamlit run app/streamlit_app.py
```

### 4. Run Benchmarks & Validation Suites
```bash
# Run the 1,000-session Zipfian power-law human traffic benchmark:
python scripts/run_graph_simulation.py

# Run the Principal AI Systems validation suite with false-positive sweep:
python scripts/validate_system.py --sessions 500 --noise-rate 0.15

# Run test suite:
python -m pytest -v
```

---

## 🧪 Testing

The repository includes unit and integration tests covering graph nodes, routing transitions, and state management:
```bash
python -m pytest -v
```

---

## 📜 License
MIT License. Free for open source and commercial use.
