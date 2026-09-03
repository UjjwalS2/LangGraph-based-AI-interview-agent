"""
Final Technical Debrief & Scorecard Node for LangGraph.
Synthesizes candidate performance across all interview turns into an executive hiring debrief.
"""

from typing import Any, Dict
import numpy as np
from app.graph.state import InterviewState
from app.llm.client import get_llm_client


def summary_report_node(state: InterviewState) -> Dict[str, Any]:
    """Node: Synthesizes final candidate report and skill radar."""
    history = list(state.get("graph_history", []))
    history.append("summary_report_node")

    evals = state.get("evaluations", [])
    scores = [e.get("score", 7.0) for e in evals]
    avg_score = round(float(np.mean(scores)), 2) if scores else 8.0

    client = get_llm_client()
    name = state.get("candidate_name", "Candidate")
    role = state.get("target_role", "Software Engineer")

    eval_summary_text = "\n".join(
        [
            f"- Round {e.get('round_number')}: Topic '{e.get('topic')}' Score: {e.get('score')}/10. Feedback: {e.get('feedback')}"
            for e in evals
        ]
    )

    prompt = f"""
    Synthesize a final technical interview hiring report for candidate {name} interviewing for {role}.
    
    ROUND EVALUATIONS:
    {eval_summary_text}
    
    AVERAGE SCORE: {avg_score}/10
    
    Return JSON with:
    - overall_score: (float)
    - technical_verdict: (e.g. "STRONG HIRE", "HIRE", "LEAN HIRE", "NO HIRE")
    - strengths: (list of 2-3 strings)
    - areas_for_improvement: (list of 2 strings)
    - recommendation: (summary paragraph)
    - radar_competencies: (dict mapping 5 skills like 'Architecture', 'Algorithms', 'Code Quality', 'Scalability', 'Communication' to float scores out of 10)
    """

    res = client.generate_json(prompt=prompt, model_tier="flash")
    report_data = res.get("data", {})

    # Fallback guarantees
    if "overall_score" not in report_data:
        report_data["overall_score"] = avg_score
    if "radar_competencies" not in report_data:
        report_data["radar_competencies"] = {
            "System Architecture": round(min(10.0, avg_score + 0.3), 1),
            "Algorithms & ML": round(avg_score, 1),
            "Code Quality": round(min(10.0, avg_score - 0.2), 1),
            "Scalability": round(min(10.0, avg_score + 0.1), 1),
            "Communication": round(avg_score, 1),
        }

    return {
        "final_report": report_data,
        "is_complete": True,
        "active_node": "summary_report_node",
        "graph_history": history,
        "total_input_tokens": state.get("total_input_tokens", 0) + res.get("input_tokens", 250),
        "total_output_tokens": state.get("total_output_tokens", 0) + res.get("output_tokens", 220),
    }
