# LangGraph Agentic Technical Interview & Assessment Platform

## Overview
An advanced, production-style autonomous AI technical interview platform built with **LangGraph**, **LangChain**, **Qdrant**, **Google GenAI (Gemini Flash/Pro)**, and **Streamlit**.

The platform orchestrates a stateful multi-agent DAG that adaptively evaluates candidates on technical skills, verifies domain grounding via hybrid RAG, leverages vector semantic caching, and escalates to high-reasoning models on complex edge cases.

---

## Key Architecture & Features

1. **Stateful LangGraph DAG Orchestrator:**
   - Typed graph state (`InterviewState`) tracking candidate competencies, turn history, grounding passages, and evaluations.
   - Built-in `MemorySaver` checkpointing for session persistence and time-travel replay.

2. **Adaptive RAG Knowledge Retrieval:**
   - Multi-stage hybrid search combining Qdrant dense vectors and BM25 lexical keywords with Reciprocal Rank Fusion ($k=60$).
   - Grounding 29 technical domains (ML, NLP/LLM, Deep Learning, Python Internals, SQL & Distributed System Design).

3. **Vector Semantic Cache & Quality-Gated Model Routing:**
   - Vector cache with partitioned matrix dot-products ($\text{Cosine} \ge 0.90$) for sub-millisecond evaluation reuse.
   - Quality-Gate conditional edge: verifies evaluation completeness and dynamically escalates from Gemini Flash to Gemini Pro when uncertainty or schema violations are detected.

4. **Streamlit UI & Real-Time Graph Visualizer:**
   - Live LangGraph DAG rendering with active node highlights.
   - Turn-by-turn interview cards with live scorecards and 3-column rubric breakdown (Covered, Missing, Inaccuracies).
   - Telemetry analytics tracking token costs, latency distributions, and routing savings.

---

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment (Optional for live Gemini API)
```bash
cp .env.example .env
# Add GEMINI_API_KEY="your-api-key" (Runs in deterministic offline demo mode if omitted)
```

### 3. Launch Streamlit Platform
```bash
streamlit run app/streamlit_app.py
```

### 4. Run Automated Test Suite
```bash
pytest -v tests/
```
