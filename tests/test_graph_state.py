"""
Tests for LangGraph State and Node Execution.
"""

from app.graph.state import InterviewState
from app.graph.nodes.parser_node import profile_parser_node
from app.graph.nodes.retrieval_node import retrieval_node
from app.graph.nodes.question_node import question_generator_node
from app.graph.nodes.cache_node import semantic_cache_check_node
from app.graph.nodes.evaluation_node import evaluate_answer_node
from app.graph.nodes.quality_gate import quality_gate_decision, cache_writeback_node


def test_profile_parser_node():
    state: InterviewState = {
        "candidate_name": "Test User",
        "target_role": "AI Engineer",
        "skills": ["Python", "PyTorch"],
    }
    res = profile_parser_node(state)
    assert res["candidate_name"] == "Test User"
    assert res["round_number"] == 1
    assert "profile_parser_node" in res["graph_history"]


def test_retrieval_node():
    state: InterviewState = {
        "round_number": 1,
        "focus_areas": ["machine_learning", "nlp_llm"],
    }
    res = retrieval_node(state)
    assert res["current_topic"] == "machine_learning"
    assert len(res["grounding_docs"]) > 0


def test_question_generator_node():
    state: InterviewState = {
        "current_topic": "machine_learning",
        "experience_level": "Senior",
        "grounding_docs": [{"text": "XGBoost uses gradient boosting with regularized objective."}],
    }
    res = question_generator_node(state)
    assert "current_question" in res
    assert len(res["expected_concepts"]) > 0


def test_quality_gate_routing():
    # Valid evaluation -> cache_writeback
    valid_state: InterviewState = {
        "cache_hit": False,
        "current_evaluation": {
            "score": 8.5,
            "feedback": "Comprehensive technical answer with good architecture intuition.",
        },
    }
    decision = quality_gate_decision(valid_state)
    assert decision == "cache_writeback_node"

    # Malformed evaluation -> pro_escalator
    bad_state: InterviewState = {
        "cache_hit": False,
        "current_evaluation": {},
    }
    bad_decision = quality_gate_decision(bad_state)
    assert bad_decision == "pro_escalator_node"
