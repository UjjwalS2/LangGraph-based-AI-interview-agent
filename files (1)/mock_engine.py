"""
Self-contained offline demo engine.

Mirrors the node pipeline and benchmark numbers documented in the
README (cache hit p50 ~1.8ms, Flash p50 ~1.48s, Pro-escalated p50
~4.9s, ~45.9% blended hit rate, <0.72 confidence gate) so the UI feels
representative of the real system even with zero API keys configured.

This also doubles as a reference for what LiveEngine (engine.py) needs
to produce at each step, since both implement the same interface:
    start_session(...) -> dict
    run_round(state, answer) -> Iterator[PipelineEvent]
    build_summary(state) -> dict
"""

from __future__ import annotations

import random
import time
import uuid
from dataclasses import dataclass, field
from typing import Iterator

TOPICS = [
    "Data Structures & Algorithms",
    "System Design",
    "Databases & SQL",
    "Cloud & DevOps",
    "Software Architecture",
]

# A handful of "core" questions per topic that repeat often across
# candidates (the 38% standardized head of Zipfian traffic in the
# README). Everything else is treated as novel long-tail and misses.
CORE_QUESTIONS = {
    "Data Structures & Algorithms": [
        "Walk me through how you'd detect a cycle in a linked list.",
        "How would you design a rate limiter using a sliding window?",
    ],
    "System Design": [
        "How would you shard a multi-tenant Postgres database?",
        "Design a URL shortener that handles 10k writes/sec.",
    ],
    "Databases & SQL": [
        "When would you reach for a composite index over two single-column indexes?",
        "Explain the difference between optimistic and pessimistic locking.",
    ],
    "Cloud & DevOps": [
        "How would you roll out a breaking API change with zero downtime?",
        "Describe your approach to autoscaling a bursty workload.",
    ],
    "Software Architecture": [
        "When does an event-driven architecture add more complexity than it removes?",
        "How do you decide between a monolith and microservices for a new team?",
    ],
}

NOVEL_SUFFIXES = [
    "Can you also account for concurrent writers?",
    "What changes if this needs to run at 100x current scale?",
    "How would you validate this decision with production data?",
    "What's the failure mode if your assumption here is wrong?",
]


@dataclass
class PipelineEvent:
    node: str
    status: str  # "active" | "done"
    detail: dict = field(default_factory=dict)


class MockInterviewEngine:
    """Offline simulation engine — no network calls."""

    def __init__(self) -> None:
        # Persisted for the life of the Streamlit session so repeat
        # "core" questions genuinely hit cache within a demo run.
        # Pre-warm one of the two core questions per topic, standing in
        # for "already asked by an earlier candidate" — otherwise a
        # short demo session would show an unrealistic 0% hit rate,
        # since a true repeat is unlikely inside 5-8 rounds.
        self._warm_cache: set[str] = {
            f"{topic}::{questions[0]}" for topic, questions in CORE_QUESTIONS.items()
        }

    # ---- lifecycle -----------------------------------------------------

    def start_session(self, candidate_name: str, target_role: str, track: str, num_rounds: int) -> dict:
        return {
            "session_id": str(uuid.uuid4())[:8],
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
        }

    def next_question(self, state: dict) -> dict:
        topic = TOPICS[state["round"] % len(TOPICS)] if state["track"] == "Mixed" else state["track"]
        bank = CORE_QUESTIONS[topic]
        # Weighted toward core questions to reflect the documented head/tail split.
        if random.random() < 0.55:
            q_text = random.choice(bank)
            q_id = f"{topic}::{q_text}"
        else:
            q_text = f"{random.choice(bank)} {random.choice(NOVEL_SUFFIXES)}"
            q_id = f"{topic}::novel::{uuid.uuid4().hex[:6]}"

        state["current_topic"] = topic
        state["current_question"] = q_text
        state["current_question_id"] = q_id
        return state

    # ---- turn execution --------------------------------------------------

    def run_round(self, state: dict, answer: str) -> Iterator[PipelineEvent]:
        """Generator that walks the 8-node pipeline for one answer,
        yielding an event per step so the UI can animate it live."""
        q_id = state["current_question_id"]
        topic = state["current_topic"]

        yield PipelineEvent("profile_parser_node", "active")
        time.sleep(0.12)
        yield PipelineEvent("profile_parser_node", "done")

        yield PipelineEvent("question_generator_node", "active")
        time.sleep(0.12)
        yield PipelineEvent("question_generator_node", "done")

        yield PipelineEvent("semantic_cache_check_node", "active")
        time.sleep(0.15)
        is_hit = q_id in self._warm_cache
        cache_latency_ms = random.lognormvariate(0.55, 0.15) if is_hit else random.lognormvariate(4.3, 0.25)
        cache_latency_ms = max(cache_latency_ms, 1.2)
        yield PipelineEvent("semantic_cache_check_node", "done", {"hit": is_hit, "latency_ms": cache_latency_ms})

        model_used = "flash"
        confidence = round(random.uniform(0.68, 0.97), 2)
        escalated = False

        if not is_hit:
            yield PipelineEvent("retrieval_node", "active")
            time.sleep(0.15)
            yield PipelineEvent("retrieval_node", "done")

            yield PipelineEvent("evaluate_answer_node", "active")
            eval_latency_ms = max(random.lognormvariate(7.3, 0.12), 900)
            time.sleep(0.2)
            yield PipelineEvent("evaluate_answer_node", "done", {
                "confidence": confidence, "latency_ms": eval_latency_ms,
            })

            if confidence < 0.72 or random.random() < 0.03:
                escalated = True
                model_used = "pro"
                yield PipelineEvent("pro_escalator_node", "active")
                pro_latency_ms = max(random.lognormvariate(8.5, 0.1), 3800)
                time.sleep(0.2)
                confidence = round(min(confidence + random.uniform(0.1, 0.25), 0.98), 2)
                yield PipelineEvent("pro_escalator_node", "done", {
                    "confidence": confidence, "latency_ms": pro_latency_ms,
                })

        yield PipelineEvent("cache_writeback_node", "active")
        time.sleep(0.1)
        self._warm_cache.add(q_id)
        yield PipelineEvent("cache_writeback_node", "done")

        # ---- bookkeeping ----
        score = round(min(10, max(0, confidence * 10 + random.uniform(-1.2, 0.8))), 1)
        state["topic_scores"][topic].append(score)
        if is_hit:
            state["cache_hits"] += 1
            state["cost_usd"] += 0.00002
        else:
            state["cache_misses"] += 1
            state["cost_usd"] += 0.0091 if escalated else 0.0008
            if escalated:
                state["escalations"] += 1
        state["cost_all_pro_baseline_usd"] += 0.0091

        state["transcript"].append({
            "round": state["round"] + 1,
            "topic": topic,
            "question": state["current_question"],
            "answer": answer,
            "cache_hit": is_hit,
            "model": model_used,
            "confidence": confidence,
            "score": score,
        })
        state["round"] += 1

    # ---- summary -----------------------------------------------------

    def build_summary(self, state: dict) -> dict:
        competency = {}
        for topic, scores in state["topic_scores"].items():
            if scores:
                competency[topic] = round(sum(scores) / len(scores), 1)
        overall = round(sum(competency.values()) / len(competency), 1) if competency else 0.0

        if overall >= 8:
            recommendation = "Strong hire"
        elif overall >= 6.5:
            recommendation = "Hire"
        elif overall >= 5:
            recommendation = "Borderline — recommend additional round"
        else:
            recommendation = "No hire"

        total_turns = state["cache_hits"] + state["cache_misses"]
        hit_rate = state["cache_hits"] / total_turns if total_turns else 0.0

        return {
            "overall_score": overall,
            "competency": competency,
            "recommendation": recommendation,
            "cache_hit_rate": hit_rate,
            "escalations": state["escalations"],
            "cost_usd": state["cost_usd"],
            "cost_all_pro_baseline_usd": state["cost_all_pro_baseline_usd"],
        }
