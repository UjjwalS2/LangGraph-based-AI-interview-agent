"""
Semantic Cache Check Node for LangGraph.
Checks vector semantic cache for evaluation reuse to bypass LLM generation on repeated answer rubrics.
"""

from typing import Any, Dict
from app.graph.state import InterviewState
from app.cache.semantic_cache import get_semantic_cache


def semantic_cache_check_node(state: InterviewState) -> Dict[str, Any]:
    """Node: Checks semantic cache for evaluation similarity."""
    history = list(state.get("graph_history", []))
    history.append("semantic_cache_check_node")

    cache = get_semantic_cache()
    ans = state.get("candidate_answer", "")
    topic = state.get("current_topic", "general")
    q_id = state.get("question_id", "q0")
    rubric_ver = state.get("rubric_version", "v1")

    cached_result = cache.lookup(
        query=ans.strip(),
        track_id=topic,
        question_id=q_id,
        rubric_version=rubric_ver,
    )

    lookup_lat = cached_result.get("lookup_latency_ms", 1.2)
    bypassed = cached_result.get("bypassed", False)
    bypass_reason = cached_result.get("bypass_reason", "")

    if cached_result.get("hit"):
        eval_data = dict(cached_result.get("response", {}))
        eval_data["model_used"] = "cache"
        eval_data["cache_hit"] = True
        eval_data["latency_ms"] = lookup_lat
        return {
            "cache_hit": True,
            "cache_bypassed": False,
            "cache_bypass_reason": "",
            "model_used": "cache",
            "current_evaluation": eval_data,
            "cache_lookup_latency_ms": lookup_lat,
            "active_node": "semantic_cache_check_node",
            "graph_history": history,
        }

    return {
        "cache_hit": False,
        "cache_bypassed": bypassed,
        "cache_bypass_reason": bypass_reason,
        "cache_lookup_latency_ms": lookup_lat,
        "active_node": "semantic_cache_check_node",
        "graph_history": history,
    }
