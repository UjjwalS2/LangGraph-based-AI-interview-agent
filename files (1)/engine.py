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

TO WIRE THIS UP FOR REAL
-------------------------
Open the three TODOs below:
  TODO 1 — confirm the import path / function name for your compiled graph.
  TODO 2 — confirm your InterviewState's field names for the current
           question, cache hit flag, confidence, and latency.
  TODO 3 — confirm the field names your summary_report_node writes
           (competency scores per topic, overall score, recommendation).
"""

from __future__ import annotations

import time
import uuid
from typing import Iterator, Optional

from app.ui.mock_engine import PipelineEvent, MockInterviewEngine, TOPICS

_graph = None
_import_error: Optional[str] = None


def _try_load_graph():
    """TODO 1: adjust these import attempts to match your actual
    app/graph/workflow.py export. Common patterns are listed in order
    of likelihood based on your README ("StateGraph compilation &
    conditional edge wiring")."""
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
        except Exception as e:  # noqa: BLE001 - intentionally broad, this is a probe
            _import_error = str(e)
            continue
    return False


BACKEND_AVAILABLE = _try_load_graph()


class LiveEngine:
    """Streams your real compiled graph. Mirrors MockInterviewEngine's
    interface so the UI layer never has to know which one it's using."""

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
            # TODO 2: your InterviewState probably wants keys like these on
            # the first invoke — adjust to match app/graph/state.py.
            "candidate_profile": {"name": candidate_name, "target_role": target_role},
            "track_id": track,
        }

    def next_question(self, state: dict) -> dict:
        """Runs the graph up to (and including) question_generator_node for
        this round by invoking with no candidate_answer yet, then reads the
        question back out of state. TODO 2: swap "question" for your real
        field name if this shows blank questions."""
        config = {"configurable": {"thread_id": state["thread_id"]}}
        try:
            result = self.graph.invoke(state, config=config)
            state.update(result)
        except Exception as e:  # noqa: BLE001
            state["_engine_error"] = f"graph.invoke failed on question generation: {e}"

        state["current_question"] = state.get("question") or state.get("current_question") or (
            "⚠️ Couldn't read the question field from graph state — "
            "see TODO 2 in app/ui/engine.py"
        )
        state["current_question_id"] = state.get("question_id", state.get("current_question_id"))
        state["current_topic"] = state.get("topic", state.get("current_topic", "—"))
        return state

    def run_round(self, state: dict, answer: str) -> Iterator[PipelineEvent]:
        """Streams the graph and animates it against the real node names
        as each one completes. Falls back gracefully if .stream() isn't
        available on your compiled graph object."""
        config = {"configurable": {"thread_id": state["thread_id"]}}
        input_state = {**state, "candidate_answer": answer}  # TODO 2: rename if needed

        topic = state.get("current_topic", "—")
        last_output = {}

        try:
            stream = self.graph.stream(input_state, config=config)
        except Exception as e:  # noqa: BLE001
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
        """TODO 3: point these at whatever summary_report_node actually
        writes to state. Falls back to aggregating the transcript client-
        side if those fields aren't present."""
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
    """Returns (engine, backend_label, error_detail_or_none)."""
    if BACKEND_AVAILABLE and _graph is not None:
        return LiveEngine(_graph), "live", None
    return MockInterviewEngine(), "offline", _import_error
