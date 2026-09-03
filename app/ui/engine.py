from __future__ import annotations

import os
import time
import uuid
from typing import Iterator, Optional

from app.ui.mock_engine import PipelineEvent, MockInterviewEngine, TOPICS

# IMPORTANT: keep the Streamlit startup path lightweight.  The real graph imports
# the RAG/embedding stack and can be expensive or fragile on hosted environments.
# It is therefore loaded only when explicitly enabled.
_graph = None
_import_error: Optional[str] = None


def _try_load_graph():
    """Load the real LangGraph backend on demand."""
    global _graph, _import_error
    if _graph is not None:
        return True
    try:
        from app.graph.workflow import get_compiled_graph
        _graph = get_compiled_graph()
        _import_error = None
        return True
    except Exception as e:
        _import_error = f"{type(e).__name__}: {e}"
        return False


class LiveEngine:
    def __init__(self, graph):
        self.graph = graph

    def start_session(self, candidate_name: str, target_role: str, track: str, num_rounds: int) -> dict:
        session_id = str(uuid.uuid4())[:8]
        return {
            "session_id": session_id,
            "thread_id": session_id,
            "candidate_name": candidate_name,
            "target_role": target_role,
            "track": track,
            "round": 0,
            "num_rounds": num_rounds,
            "transcript": [],
            "topic_scores": {t: [] for t in TOPICS},
            "cache_hits": 0,
            "cache_misses": 0,
            "escalations": 0,
            "cost_usd": 0.0,
            "cost_all_pro_baseline_usd": 0.0,
            "current_question": None,
            "current_question_id": None,
            "current_topic": None,
            "candidate_profile": {"name": candidate_name, "target_role": target_role},
            "track_id": track,
        }

    def next_question(self, state: dict) -> dict:
        config = {"configurable": {"thread_id": state["thread_id"]}}
        try:
            result = self.graph.invoke(state, config=config)
            if isinstance(result, dict):
                state.update(result)
        except Exception as e:
            state["_engine_error"] = f"graph.invoke failed: {e}"
        state["current_question"] = state.get("current_question") or state.get("question") or "⚠️ No question returned"
        state["current_question_id"] = state.get("current_question_id") or state.get("question_id")
        state["current_topic"] = state.get("current_topic") or state.get("topic") or "General"
        return state

    def run_round(self, state: dict, answer: str) -> Iterator[PipelineEvent]:
        config = {"configurable": {"thread_id": state["thread_id"]}}
        input_state = {**state, "candidate_answer": answer}
        topic = state.get("current_topic", "General")
        try:
            stream = self.graph.stream(input_state, config=config)
        except Exception as e:
            yield PipelineEvent("evaluate_answer_node", "active")
            yield PipelineEvent("evaluate_answer_node", "done", {"error": str(e)})
            return

        for event in stream:
            if not isinstance(event, dict):
                continue
            for node_name, output in event.items():
                yield PipelineEvent(node_name, "active")
                time.sleep(0.08)
                output = output or {}
                current_eval = output.get("current_evaluation", {}) if isinstance(output, dict) else {}
                detail = {
                    "hit": output.get("cache_hit") if isinstance(output, dict) else None,
                    "confidence": current_eval.get("confidence", output.get("confidence")) if isinstance(output, dict) else None,
                    "latency_ms": output.get("latency_ms") if isinstance(output, dict) else None,
                }
                yield PipelineEvent(node_name, "done", detail)
                if isinstance(output, dict):
                    state.update(output)

        current_eval = state.get("current_evaluation", {}) or {}
        is_hit = bool(state.get("cache_hit", False))
        confidence = float(current_eval.get("confidence", state.get("confidence", 0.75)) or 0.75)
        score = float(current_eval.get("score", state.get("score", confidence * 10)) or confidence * 10)
        model_used = state.get("model_used") or current_eval.get("model_used", "flash")
        escalated = bool(state.get("escalated", False) or model_used == "pro")

        state.setdefault("topic_scores", {t: [] for t in TOPICS})
        state["topic_scores"].setdefault(topic, []).append(score)
        if is_hit:
            state["cache_hits"] = state.get("cache_hits", 0) + 1
        else:
            state["cache_misses"] = state.get("cache_misses", 0) + 1
        if escalated and not is_hit:
            state["escalations"] = state.get("escalations", 0) + 1

        state["transcript"].append({
            "round": state.get("round", 0) + 1,
            "topic": topic,
            "question": state.get("current_question"),
            "answer": answer,
            "cache_hit": is_hit,
            "model": model_used,
            "confidence": confidence,
            "score": score,
        })
        state["round"] = state.get("round", 0) + 1

    def build_summary(self, state: dict) -> dict:
        report = state.get("final_report", {}) or {}
        competency = report.get("radar_competencies") or state.get("competency_scores")
        if not competency:
            competency = {
                t: round(sum(s) / len(s), 1)
                for t, s in state.get("topic_scores", {}).items() if s
            }
        overall = report.get("overall_score", state.get("overall_score"))
        if overall is None:
            overall = round(sum(competency.values()) / len(competency), 1) if competency else 0.0
        recommendation = report.get("recommendation") or state.get("recommendation") or "See competency breakdown"
        total_turns = state.get("cache_hits", 0) + state.get("cache_misses", 0)
        hit_rate = state.get("cache_hits", 0) / total_turns if total_turns else 0.0
        return {
            "overall_score": float(overall),
            "competency": competency,
            "recommendation": recommendation,
            "cache_hit_rate": hit_rate,
            "escalations": state.get("escalations", 0),
            "cost_usd": state.get("cost_usd", 0.0),
            "cost_all_pro_baseline_usd": state.get("cost_all_pro_baseline_usd", 0.0),
        }


def get_engine():
    """Return a fast offline engine by default; opt into the live graph explicitly.

    Set INTERVIEW_LIVE_BACKEND=1 in Streamlit secrets/environment when the full
    LangGraph/RAG backend should be used.  This keeps hosted UI startup reliable.
    """
    use_live = os.getenv("INTERVIEW_LIVE_BACKEND", "0").strip().lower() in {
        "1", "true", "yes", "on"
    }
    if use_live:
        if _try_load_graph() and _graph is not None:
            return LiveEngine(_graph), "live", None
        return MockInterviewEngine(), "offline", _import_error
    return MockInterviewEngine(), "offline", None
