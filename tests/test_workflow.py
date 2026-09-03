"""
End-to-End Workflow Tests for LangGraph Pipeline.
"""

from app.graph.workflow import build_interview_graph
from app.graph.state import InterviewState


def test_compiled_workflow_run():
    # Build graph without checkpointer for fast in-memory execution test
    graph = build_interview_graph(with_checkpointer=False)

    initial_state: InterviewState = {
        "candidate_name": "Priya Sharma",
        "target_role": "NLP Engineer",
        "skills": ["Python", "Transformers", "RAG", "BGE-M3"],
        "experience_level": "Senior",
        "focus_areas": ["nlp_llm", "machine_learning"],
        "candidate_answer": "Random Forest builds multiple de-correlated decision trees using bootstrap aggregation and random feature sub-sampling to lower overall ensemble variance.",
        "max_rounds": 1,
    }

    final_state = graph.invoke(initial_state)

    assert final_state is not None
    assert final_state.get("is_complete") is True
    assert len(final_state.get("evaluations", [])) >= 1
    assert "summary_report_node" in final_state.get("graph_history", [])
    assert "final_report" in final_state
    assert final_state["final_report"].get("overall_score") > 0
