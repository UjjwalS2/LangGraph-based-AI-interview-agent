"""
Adaptive Question Generation & Topic Routing Node for LangGraph.

Runs BEFORE the semantic cache check so that:
  - question_id and rubric_version are set deterministically for the composite cache key
  - current_topic is derived from focus_areas + round_number (same formula as retrieval_node)
  - Question text is generated from topic only (no grounding docs needed at this stage)

Grounding documents are fetched by retrieval_node ONLY on a confirmed cache miss.
"""

from typing import Any, Dict
from app.graph.state import InterviewState
from app.llm.client import get_llm_client


def question_generator_node(state: InterviewState) -> Dict[str, Any]:
    """
    Node: Derives topic, assigns deterministic question_id/rubric_version,
    and generates a grounding-free question stub for the cache key lookup.
    """
    history = list(state.get("graph_history", []))
    history.append("question_generator_node")

    # ── Topic Selection: deterministic from focus_areas + round_number ─────────
    # Must use the same formula as retrieval_node so the composite cache key
    # (track_id + question_id + rubric_version) is consistent across turns.
    focus_areas = state.get("focus_areas", ["machine_learning", "nlp_llm", "system_design"])
    round_num = state.get("round_number", 1)
    topic_idx = (round_num - 1) % len(focus_areas)
    current_topic = focus_areas[topic_idx]

    # ── Deterministic question_id + rubric_version (no LLM needed) ────────────
    q_id = state.get("question_id") or f"{current_topic}_r{round_num}"
    rubric_ver = state.get("rubric_version") or "v1"

    # ── Lightweight Question Generation (topic-only, no grounding docs) ────────
    # Grounding docs (Qdrant + BM25) are fetched by retrieval_node only on a
    # confirmed cache miss. This call is intentionally shallow.
    client = get_llm_client()
    exp_level = state.get("experience_level", "Senior")

    prompt = f"""
    You are an expert technical interviewer conducting a {exp_level}-level interview.
    Formulate an in-depth technical question for the topic below.
    
    TOPIC: {current_topic}
    
    Return JSON with:
    - topic: (string)
    - question: (string)
    - expected_concepts: (list of strings)
    """

    res = client.generate_json(prompt=prompt, model_tier="flash")
    data = res.get("data", {})

    q_text = data.get("question") or (
        f"In {current_topic}, explain the mathematical mechanics, "
        f"primary tradeoffs, and production scaling practices."
    )
    concepts = data.get("expected_concepts") or [
        "Algorithmic derivation", "Tradeoffs", "Production monitoring"
    ]

    return {
        "current_topic": current_topic,
        "current_question": q_text,
        "question_id": q_id,
        "rubric_version": rubric_ver,
        "expected_concepts": concepts,
        "active_node": "question_generator_node",
        "graph_history": history,
        "total_input_tokens": state.get("total_input_tokens", 0) + res.get("input_tokens", 150),
        "total_output_tokens": state.get("total_output_tokens", 0) + res.get("output_tokens", 100),
    }

