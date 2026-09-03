"""
State definition for LangGraph Agentic Interview Platform.
Defines TypedDict and Pydantic models for graph execution and checkpointing.
"""

from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict
from pydantic import BaseModel, Field


class EvaluationDetail(BaseModel):
    round_number: int
    topic: str
    question: str
    candidate_answer: str
    score: float
    correct_points: List[str] = Field(default_factory=list)
    missing_concepts: List[str] = Field(default_factory=list)
    incorrect_points: List[str] = Field(default_factory=list)
    feedback: str = ""
    ideal_answer: str = ""
    model_used: str = "flash"
    escalated: bool = False
    escalation_reason: str = ""
    cache_hit: bool = False
    latency_ms: float = 0.0


class FinalReport(BaseModel):
    overall_score: float
    strengths: List[str]
    areas_for_improvement: List[str]
    technical_verdict: str
    recommendation: str
    radar_competencies: Dict[str, float]


class InterviewState(TypedDict, total=False):
    # Candidate & Setup Info
    session_id: str
    candidate_name: str
    target_role: str
    experience_level: str
    skills: List[str]
    job_description: str
    focus_areas: List[str]
    difficulty: str

    # Round Management
    round_number: int
    max_rounds: int
    is_complete: bool

    # Active Turn Details
    current_topic: str
    current_question: str
    question_id: str
    rubric_version: str
    expected_concepts: List[str]
    grounding_docs: List[Dict[str, Any]]
    candidate_answer: str

    # Evaluation & Routing State
    cache_hit: bool
    cache_bypassed: bool
    cache_bypass_reason: str
    model_used: str
    escalated: bool
    escalation_reason: str
    quality_gate_confidence: float
    flash_draft_evaluation: Dict[str, Any]
    current_evaluation: Dict[str, Any]
    evaluations: List[Dict[str, Any]]

    # Final Debrief
    final_report: Dict[str, Any]

    # Granular Latency Telemetry (per-turn in milliseconds)
    cache_lookup_latency_ms: float
    retrieval_latency_ms: float
    ttft_ms: float
    generation_latency_ms: float
    end_to_end_turn_latency_ms: float
    turn_telemetry: Dict[str, float]

    # Graph Tracing & Telemetry
    active_node: str
    graph_history: List[str]
    total_input_tokens: int
    total_output_tokens: int
    total_cost_usd: float
    baseline_cost_usd: float
