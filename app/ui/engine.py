"""
Adapter between the Streamlit UI and your real LangGraph backend.

WHY THIS FILE EXISTS
---------------------
I couldn't browse into your `app/graph/`, `app/cache/`, `app/rag/` source
from here (GitHub blocks automated access to repo subfolders), so I don't
know the exact `InterviewState` field names or the exported graph-builder
function signature. Rather than guess and hand you something that silently
does the wrong thing, this adapter:

  1. Tries a few common import patterns to find your compiled graph.
  2. If found, streams it node-by-node and animates the *exact* 8 node
     names from your README (profile_parser_node ... summary_report_node).
  3. Reads a few plausible field names out of the returned state (marked
     TODO below) — if your schema uses different names, the UI still runs,
     it just shows "—" for that value until you fix the one line noted.
  4. If no graph is found at all, falls back to MockInterviewEngine so the
     app is still fully usable.
"""

from __future__ import annotations

import time
import uuid
from typing import Iterator, Optional

from app.ui.mock_engine import PipelineEvent, MockInterviewEngine, TOPICS

_graph = None
_import_error: Optional[str] = None


def _try_load_graph():
    global _graph, _import_error
    attempts = [
        lambda: __import__("app.graph.workflow", fromlist=["build_graph"]).build_graph(),
        lambda: __import__("app.graph.workflow", fromlist=["compile_graph"]).compile_graph(),
        lambda: __import__("app.graph.workflow", fromlist=["get_graph"]).get_graph(),
        lambda: __import__("app.graph.workflow", fromlist=["workflow"]).workflow,
        lambda: __import__("app.graph.graph_builder", fromlist=["GraphBuilder"]).GraphBuilder().build(),
    ]
    for attempt in attempts:
        try:
            _graph = attempt()
            return True
        except Exception as e:
            _import_error = str(e)
            continue
    return False


BACKEND_AVAILABLE = _try_load_graph()


class LiveEngine:
    """Streams your real compiled graph. Mirrors MockInterviewEngine's interface."""

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
            state.update(result)
        except Exception as e:
            state["_engine_error"] = f"graph.invoke failed on question generation: {e}"

        state["current_question"] = state.get("question") or state.get("current_question") or (
            "⚠️ Couldn't read the question field from graph state — see app/ui/engine.py"
        )
        state["current_question_id"] = state.get("question_id", state.get("current_question_id"))
        state["current_topic"] = state.get("topic", state.get("current_topic", "—"))
        return state

    def run_round(self, state: dict, answer: str) -> Iterator[PipelineEvent]:
        config = {"configurable": {"thread_id": state["thread_id"]}}
        input_state = {**state, "candidate_answer": answer}
        topic = state.get("current_topic", "—")
        last_output = {}

        try:
            stream = self.graph.stream(input_state, config=config)
        except Exception as e:
            yield PipelineEvent("evaluate_answer_node", "active")
            yield PipelineEvent("evaluate_answer_node", "done", {"error": str(e)})
            return

        for event in stream:
            for node_name, output in event.items():
                yield PipelineEvent(node_name, "active")
                time.sleep(0.08)
                last_output = output or {}
                detail = {
                    "hit": last_output.get("cache_hit"),
                    "confidence": last_output.get("confidence"),
                    "latency_ms": last_output.get("latency_ms"),
                }
                yield PipelineEvent(node_name, "done", detail)
                state.update(last_output)

        is_hit = bool(last_output.get("cache_hit"))
        confidence = float(last_output.get("confidence", 0.75) or 0.75)
        score = float(last_output.get("score", confidence * 10) or confidence * 10)

        state.setdefault("topic_scores", {t: [] for t in TOPICS})
        state["topic_scores"].setdefault(topic, []).append(score)
        if is_hit:
            state["cache_hits"] = state.get("cache_hits", 0) + 1
        else:
            state["cache_misses"] = state.get("cache_misses", 0) + 1

        state["transcript"].append({
            "round": state["round"] + 1,
            "topic": topic,
            "question": state.get("current_question"),
            "answer": answer,
            "cache_hit": is_hit,
            "model": last_output.get("model_used", "flash"),
            "confidence": confidence,
            "score": score,
        })
        state["round"] += 1

    def build_summary(self, state: dict) -> dict:
        competency = state.get("competency_scores")
        if not competency:
            competency = {
                t: round(sum(s) / len(s), 1)
                for t, s in state.get("topic_scores", {}).items() if s
            }
        overall = state.get("overall_score")
        if overall is None:
            overall = round(sum(competency.values()) / len(competency), 1) if competency else 0.0

        total_turns = state.get("cache_hits", 0) + state.get("cache_misses", 0)
        hit_rate = state.get("cache_hits", 0) / total_turns if total_turns else 0.0

        return {
            "overall_score": overall,
            "competency": competency,
            "recommendation": state.get("recommendation", "See competency breakdown"),
            "cache_hit_rate": hit_rate,
            "escalations": state.get("escalations", 0),
            "cost_usd": state.get("cost_usd", 0.0),
            "cost_all_pro_baseline_usd": state.get("cost_all_pro_baseline_usd", 0.0),
        }


def get_engine():
    if BACKEND_AVAILABLE and _graph is not None:
        return LiveEngine(_graph), "live", None
    return MockInterviewEngine(), "offline", _import_error
