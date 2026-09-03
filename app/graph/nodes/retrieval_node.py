"""
Adaptive RAG Retrieval Node for LangGraph.

Only executes on a confirmed semantic cache MISS (per graph wiring in workflow.py).
Queries Hybrid Qdrant + BM25 store for grounding passages for the current round's topic.
Topic is read from state (already derived by question_generator_node before the cache check).
"""

from typing import Any, Dict
from app.graph.state import InterviewState
from app.rag.vectorstore import get_hybrid_store


def retrieval_node(state: InterviewState) -> Dict[str, Any]:
    """
    Node: Retrieves top technical grounding passages for the confirmed-miss turn.
    Uses current_topic already set by question_generator_node to ensure
    composite key consistency (track_id = current_topic must match across cache check and retrieval).
    """
    history = list(state.get("graph_history", []))
    history.append("retrieval_node")

    store = get_hybrid_store()

    # Use the topic already determined by question_generator_node.
    # Fallback formula is a safety net only; should not normally trigger.
    current_topic = state.get("current_topic")
    if not current_topic:
        focus_areas = state.get("focus_areas", ["machine_learning", "nlp_llm", "system_design"])
        round_num = state.get("round_number", 1)
        current_topic = focus_areas[(round_num - 1) % len(focus_areas)]

    import time as _time
    t0 = _time.perf_counter()

    # Search knowledge store for grounding passages
    passages = store.search_hybrid(
        query=f"Core architectural mechanism, tradeoffs, and production considerations in {current_topic}",
        topic_filter=current_topic,
        top_k=3,
    )
    retrieval_lat = (_time.perf_counter() - t0) * 1000.0

    doc_payloads = [
        {
            "chunk_id": p.chunk_id,
            "doc_id": p.doc_id,
            "topic": p.topic,
            "text": p.text,
            "score": p.score,
        }
        for p in passages
    ]

    return {
        "current_topic": current_topic,
        "grounding_docs": doc_payloads,
        "retrieval_latency_ms": retrieval_lat,
        "active_node": "retrieval_node",
        "graph_history": history,
    }

