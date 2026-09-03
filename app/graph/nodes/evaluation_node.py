"""
Candidate Answer Evaluation Node for LangGraph.
Evaluates candidate answer against ground-truth references using Gemini Flash.
"""

from typing import Any, Dict
from app.graph.state import InterviewState
from app.llm.client import get_llm_client


def evaluate_answer_node(state: InterviewState) -> Dict[str, Any]:
    """Node: Evaluates answer with Gemini Flash."""
    history = list(state.get("graph_history", []))
    history.append("evaluate_answer_node")

    # If already resolved via cache, pass through
    if state.get("cache_hit") and state.get("current_evaluation"):
        return {
            "active_node": "evaluate_answer_node",
            "graph_history": history,
        }

    client = get_llm_client()
    q = state.get("current_question", "")
    ans = state.get("candidate_answer", "")
    topic = state.get("current_topic", "")
    grounding = "\n\n".join([d.get("text", "") for d in state.get("grounding_docs", [])])

    prompt = f"""
    You are an expert technical interviewer. Evaluate the candidate's answer based on the reference documentation.
    
    TOPIC: {topic}
    QUESTION: {q}
    REFERENCE DOCUMENTATION:
    {grounding}
    
    CANDIDATE ANSWER:
    {ans}
    
    Return JSON with:
    - score: (float between 0.0 and 10.0)
    - correct_points: (list of strings)
    - missing_concepts: (list of strings)
    - incorrect_points: (list of strings)
    - feedback: (string with actionable advice)
    - ideal_answer: (string summarizing the ideal response)
    """

    res = client.generate_json(prompt=prompt, model_tier="flash")
    data = res.get("data", {})
    confidence = data.get("confidence", 0.90)
    data["model_used"] = "flash"
    data["cache_hit"] = False
    data["ttft_ms"] = res.get("ttft_ms", 210.0)
    data["generation_latency_ms"] = res.get("generation_latency_ms", 280.0)
    data["latency_ms"] = res.get("latency_ms", 490.0)

    return {
        "current_evaluation": data,
        "flash_draft_evaluation": data,
        "quality_gate_confidence": confidence,
        "model_used": "flash",
        "ttft_ms": res.get("ttft_ms", 210.0),
        "generation_latency_ms": res.get("generation_latency_ms", 280.0),
        "active_node": "evaluate_answer_node",
        "graph_history": history,
        "total_input_tokens": state.get("total_input_tokens", 0) + res.get("input_tokens", 220),
        "total_output_tokens": state.get("total_output_tokens", 0) + res.get("output_tokens", 200),
    }
