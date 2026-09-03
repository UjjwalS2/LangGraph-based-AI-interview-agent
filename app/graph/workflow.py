"""
LangGraph StateGraph Workflow Builder.
Compiles the multi-agent graph with conditional edges, quality gating, and state persistence.

Correct execution order (cache-first short-circuit):
  START
    → profile_parser_node       (parse candidate profile, assign focus_areas)
    → question_generator_node   (assign deterministic question_id + rubric_version; NO RAG needed here)
    → semantic_cache_check_node (composite namespace lookup: hash(track+question+rubric) × answer vector)
        ↓ HIT  → cache_writeback_node  ← short-circuits: zero RAG, zero LLM calls
        ↓ MISS → retrieval_node        (hybrid Qdrant + BM25 + RRF — only on confirmed miss)
                  → evaluate_answer_node (Flash LLM with grounding context)
                      → quality_gate_decision
                          ↓ PASS → cache_writeback_node
                          ↓ FAIL → pro_escalator_node → cache_writeback_node
    → cache_writeback_node (persist eval + record granular E2E telemetry)
        ↓ more rounds → retrieval_node
        ↓ done        → summary_report_node → END
"""

from typing import Literal
from langgraph.graph import StateGraph, START, END
from app.graph.state import InterviewState
from app.graph.checkpointer import get_memory_checkpointer
from app.graph.nodes.parser_node import profile_parser_node
from app.graph.nodes.retrieval_node import retrieval_node
from app.graph.nodes.question_node import question_generator_node
from app.graph.nodes.cache_node import semantic_cache_check_node
from app.graph.nodes.evaluation_node import evaluate_answer_node
from app.graph.nodes.quality_gate import (
    quality_gate_decision,
    pro_escalator_node,
    cache_writeback_node,
)
from app.graph.nodes.summary_node import summary_report_node


def cache_hit_router(state: InterviewState) -> Literal["cache_writeback_node", "retrieval_node"]:
    """
    Conditional Edge: Short-circuits the pipeline on a semantic cache hit.

    HIT  → cache_writeback_node:
        The cached evaluation is returned immediately. Hybrid RAG (Qdrant + BM25)
        and all LLM calls are completely skipped. E2E latency = cache_lookup_lat only.

    MISS → retrieval_node:
        Only on a confirmed miss does the pipeline proceed to fetch grounding
        documents (Qdrant + BM25 + RRF) and run LLM evaluation.
    """
    if state.get("cache_hit"):
        return "cache_writeback_node"
    return "retrieval_node"


def turn_progression_router(state: InterviewState) -> Literal["retrieval_node", "summary_report_node"]:
    """Conditional Edge: Checks if more rounds remain or transitions to final report."""
    if state.get("is_complete", False):
        return "summary_report_node"
    return "retrieval_node"


def build_interview_graph(with_checkpointer: bool = True):
    """Builds and compiles the complete LangGraph StateGraph."""
    builder = StateGraph(InterviewState)

    # 1. Register all nodes
    builder.add_node("profile_parser_node", profile_parser_node)
    builder.add_node("question_generator_node", question_generator_node)
    builder.add_node("semantic_cache_check_node", semantic_cache_check_node)
    builder.add_node("retrieval_node", retrieval_node)
    builder.add_node("evaluate_answer_node", evaluate_answer_node)
    builder.add_node("pro_escalator_node", pro_escalator_node)
    builder.add_node("cache_writeback_node", cache_writeback_node)
    builder.add_node("summary_report_node", summary_report_node)

    # 2. Pre-cache path: profile → question metadata → cache lookup
    builder.add_edge(START, "profile_parser_node")
    builder.add_edge("profile_parser_node", "question_generator_node")
    builder.add_edge("question_generator_node", "semantic_cache_check_node")

    # 3. Cache short-circuit: HIT skips retrieval+LLM, MISS proceeds to RAG
    builder.add_conditional_edges(
        "semantic_cache_check_node",
        cache_hit_router,
        {
            "cache_writeback_node": "cache_writeback_node",   # HIT path: no RAG, no LLM
            "retrieval_node": "retrieval_node",               # MISS path: full pipeline
        },
    )

    # 4. Miss path: grounding retrieval → LLM evaluation
    builder.add_edge("retrieval_node", "evaluate_answer_node")

    # 5. Quality Gate: Flash passes → writeback | Flash uncertain → Pro escalation
    builder.add_conditional_edges(
        "evaluate_answer_node",
        quality_gate_decision,
        {
            "pro_escalator_node": "pro_escalator_node",
            "cache_writeback_node": "cache_writeback_node",
        },
    )
    builder.add_edge("pro_escalator_node", "cache_writeback_node")

    # 6. Multi-turn loop: next round → retrieval_node | done → summary
    builder.add_conditional_edges(
        "cache_writeback_node",
        turn_progression_router,
        {
            "retrieval_node": "retrieval_node",
            "summary_report_node": "summary_report_node",
        },
    )

    builder.add_edge("summary_report_node", END)

    # 7. Compile with optional checkpointer
    checkpointer = get_memory_checkpointer() if with_checkpointer else None
    return builder.compile(checkpointer=checkpointer)


# Singleton compiled graph instance
_GLOBAL_GRAPH = None


def get_compiled_graph():
    global _GLOBAL_GRAPH
    if _GLOBAL_GRAPH is None:
        _GLOBAL_GRAPH = build_interview_graph(with_checkpointer=True)
    return _GLOBAL_GRAPH
