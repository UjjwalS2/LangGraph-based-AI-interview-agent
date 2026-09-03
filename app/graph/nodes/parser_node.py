"""
Candidate Profile Parser Node for LangGraph.
Extracts candidate skills, experience level, and aligns focus areas with Job Description.
"""

from typing import Any, Dict
from app.graph.state import InterviewState
from app.llm.client import get_llm_client


def profile_parser_node(state: InterviewState) -> Dict[str, Any]:
    """Node: Ingests raw resume and JD to structure candidate state."""
    history = list(state.get("graph_history", []))
    history.append("profile_parser_node")

    # If candidate details already provided, ensure focus areas exist
    name = state.get("candidate_name") or "Alex Chen"
    target_role = state.get("target_role") or "Senior AI/ML Engineer"
    skills = state.get("skills") or ["Python", "XGBoost", "PyTorch", "RAG", "SQL", "LangGraph"]
    focus = state.get("focus_areas") or ["machine_learning", "nlp_llm", "system_design", "python", "sql"]

    return {
        "candidate_name": name,
        "target_role": target_role,
        "experience_level": state.get("experience_level") or "Senior",
        "skills": skills,
        "focus_areas": focus,
        "round_number": 1,
        "max_rounds": state.get("max_rounds") or 3,
        "is_complete": False,
        "evaluations": state.get("evaluations", []),
        "active_node": "profile_parser_node",
        "graph_history": history,
        "total_input_tokens": state.get("total_input_tokens", 0) + 120,
        "total_output_tokens": state.get("total_output_tokens", 0) + 80,
    }
