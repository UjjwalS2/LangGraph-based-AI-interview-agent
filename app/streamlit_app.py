"""
Enhanced Streamlit UI — Agentic AI Interview & Assessment Platform

Run:
    streamlit run app/streamlit_app.py

This drop-in replaces the previous app/streamlit_app.py. It auto-detects
your real compiled LangGraph workflow (see app/ui/engine.py); if it can't
find/import it, it runs on a self-contained offline demo engine so the UI is
fully explorable either way. Search app/ui/engine.py for "TODO" to wire
in your actual InterviewState field names.
"""

from __future__ import annotations

import json

import streamlit as st

from app.ui.engine import get_engine
from app.ui.theme import inject_css
from app.ui.widgets import (
    PIPELINE_NODES,
    cache_badge,
    competency_radar,
    confidence_badge,
    confidence_gauge,
    cost_bar,
    model_badge,
    render_answer_turn,
    render_metric_row,
    render_pipeline,
    render_question_turn,
)
from app.ui.mock_engine import TOPICS

st.set_page_config(
    page_title="Interview Console",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()


# ---------------------------------------------------------------------
# session bootstrap
# ---------------------------------------------------------------------

if "engine" not in st.session_state:
    engine, backend_label, backend_error = get_engine()
    st.session_state.engine = engine
    st.session_state.backend_label = backend_label
    st.session_state.backend_error = backend_error

if "stage" not in st.session_state:
    st.session_state.stage = "setup"  # setup -> interview -> summary
if "session" not in st.session_state:
    st.session_state.session = None
if "pending_events" not in st.session_state:
    st.session_state.pending_events = []

engine = st.session_state.engine


def reset_session():
    st.session_state.stage = "setup"
    st.session_state.session = None


# ---------------------------------------------------------------------
# sidebar
# ---------------------------------------------------------------------

with st.sidebar:
    st.markdown(
        """
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:2px;">
            <div style="width:9px;height:9px;border-radius:2px;background:#45D6C4;"></div>
            <span style="font-weight:800;font-size:1.05rem;">Interview Console</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    mode_cls = "badge-mode-live" if st.session_state.backend_label == "live" else "badge-mode-offline"
    mode_text = "● live backend" if st.session_state.backend_label == "live" else "○ offline simulation"
    st.markdown(f'<span class="badge {mode_cls}">{mode_text}</span>', unsafe_allow_html=True)

    if st.session_state.backend_label == "offline" and st.session_state.backend_error:
        with st.expander("why offline mode?", expanded=False):
            st.caption(
                "Couldn't import a compiled graph from `app.graph.workflow` / "
                "`app.graph.graph_builder`. Last import error:"
            )
            st.code(st.session_state.backend_error, language="text")
            st.caption("Adjust the TODOs in `app/ui/engine.py` to match your real module.")

    st.divider()

    if st.session_state.stage == "setup":
        candidate_name = st.text_input("Candidate name", value="Jordan Rivera")
        target_role = st.text_input("Target role", value="Senior Backend Engineer")
        track = st.selectbox("Track", ["Mixed"] + TOPICS, index=0)
        num_rounds = st.slider("Interview rounds", min_value=3, max_value=8, value=5)

        if st.button("Start interview", use_container_width=True):
            session = engine.start_session(candidate_name, target_role, track, num_rounds)
            session = engine.next_question(session)
            st.session_state.session = session
            st.session_state.stage = "interview"
            st.rerun()
    else:
        s = st.session_state.session
        st.markdown(f"**{s['candidate_name']}**")
        st.caption(s["target_role"])
        st.progress(min(s["round"] / s["num_rounds"], 1.0), text=f"round {s['round']} / {s['num_rounds']}")
        if st.button("End & reset session", use_container_width=True):
            reset_session()
            st.rerun()

    st.divider()
    with st.expander("pipeline reference", expanded=False):
        for key, label in PIPELINE_NODES:
            st.markdown(f'<span style="font-family:JetBrains Mono;font-size:0.75rem;color:#8C99A8;">{key}</span>', unsafe_allow_html=True)


# ---------------------------------------------------------------------
# header
# ---------------------------------------------------------------------

st.markdown(
    """
    <div class="console-hero">
        <span class="mark"></span>
        <p class="console-title">Agentic Interview Console</p>
    </div>
    <p class="console-subtitle">
        Composite-namespaced semantic cache, hybrid Qdrant + BM25 retrieval, and a
        confidence-gated Flash → Pro escalation path — visualized live as each answer
        moves through the graph.
    </p>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------
# setup empty state
# ---------------------------------------------------------------------

if st.session_state.stage == "setup":
    st.markdown(
        """
        <div class="card" style="max-width:560px;">
            <p style="font-weight:600;margin-bottom:6px;">Configure a session to begin</p>
            <p style="color:#8C99A8;font-size:0.9rem;line-height:1.6;margin:0;">
                Set the candidate, target role, and track in the sidebar, then start
                the interview. Each answer streams through the full node pipeline —
                cache lookup, retrieval-on-miss, evaluation, and speculative escalation
                — with real timing.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------
# interview stage
# ---------------------------------------------------------------------

elif st.session_state.stage == "interview":
    s = st.session_state.session
    tab_interview, tab_metrics = st.tabs(["Interview", "Live metrics"])

    with tab_interview:
        pipeline_ph = st.empty()
        done_keys = {k for k in st.session_state.pending_events}
        with pipeline_ph.container():
            render_pipeline(active_key=None, done_keys=set(), skipped_keys=set())

        # past turns
        for turn in s["transcript"]:
            render_question_turn(turn["round"], turn["topic"], turn["question"])
            badges = (
                cache_badge(turn["cache_hit"], 1.8 if turn["cache_hit"] else 900)
                + " " + model_badge(turn["model"])
                + " " + confidence_badge(turn["confidence"])
            )
            render_answer_turn(turn["answer"], badges)

        if s["round"] < s["num_rounds"]:
            render_question_turn(s["round"] + 1, s.get("current_topic", "—"), s.get("current_question", ""))

            answer = st.text_area(
                "Your answer",
                key=f"answer_{s['round']}",
                height=140,
                placeholder="Type the candidate's response…",
                label_visibility="collapsed",
            )
            c1, c2 = st.columns([1, 5])
            with c1:
                submit = st.button("Submit answer", type="primary")
            with c2:
                skip = st.button("Skip round")

            if submit or skip:
                answer_text = answer.strip() if submit and answer.strip() else "(no answer provided)"
                done = set()
                status_overrides = {}
                for ev in engine.run_round(s, answer_text):
                    if ev.status == "active":
                        with pipeline_ph.container():
                            render_pipeline(active_key=ev.node, done_keys=done, skipped_keys=set(), status_keys=status_overrides)
                    else:
                        done.add(ev.node)
                        if ev.node == "semantic_cache_check_node" and ev.detail.get("hit") is False:
                            status_overrides[ev.node] = "miss"
                        if ev.node == "pro_escalator_node":
                            status_overrides[ev.node] = "escalate"
                        with pipeline_ph.container():
                            render_pipeline(active_key=None, done_keys=done, skipped_keys=set(), status_keys=status_overrides)

                if s["round"] < s["num_rounds"]:
                    s = engine.next_question(s)
                st.session_state.session = s

                if s["round"] >= s["num_rounds"]:
                    st.session_state.stage = "summary"
                st.rerun()
        else:
            st.session_state.stage = "summary"
            st.rerun()

    with tab_metrics:
        total = s["cache_hits"] + s["cache_misses"]
        hit_rate = (s["cache_hits"] / total * 100) if total else 0.0
        render_metric_row([
            ("cache hit rate", f"{hit_rate:.1f}%"),
            ("cache hits", str(s["cache_hits"])),
            ("cache misses", str(s["cache_misses"])),
            ("escalations", str(s["escalations"])),
        ])
        st.write("")
        st.markdown('<p style="color:#8C99A8;font-size:0.85rem;">estimated cost vs. all-Pro baseline</p>', unsafe_allow_html=True)
        st.plotly_chart(
            cost_bar(0, s["cost_usd"], 0, s["cost_all_pro_baseline_usd"] or 0.0001),
            use_container_width=True,
            config={"displayModeBar": False},
        )


# ---------------------------------------------------------------------
# summary stage
# ---------------------------------------------------------------------

elif st.session_state.stage == "summary":
    s = st.session_state.session
    summary = engine.build_summary(s)

    col_left, col_right = st.columns([1, 1.3], gap="large")

    with col_left:
        st.markdown(
            f"""
            <div class="card">
                <div class="turn-label">overall competency</div>
                <p style="font-family:'JetBrains Mono';font-size:2.6rem;font-weight:600;margin:2px 0 4px 0;">
                    {summary['overall_score']:.1f}<span style="font-size:1.1rem;color:#8C99A8;">/10</span>
                </p>
                <p style="color:#45D6C4;font-weight:600;margin:0;">{summary['recommendation']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write("")
        render_metric_row([
            ("cache hit rate", f"{summary['cache_hit_rate']*100:.1f}%"),
            ("pro escalations", str(summary["escalations"])),
        ])
        st.write("")
        render_metric_row([
            ("session cost", f"${summary['cost_usd']:.4f}"),
            ("all-Pro baseline", f"${summary['cost_all_pro_baseline_usd']:.4f}"),
        ])

        report = {
            "candidate": s["candidate_name"],
            "target_role": s["target_role"],
            **summary,
        }
        st.write("")
        st.download_button(
            "Download report (JSON)",
            data=json.dumps(report, indent=2),
            file_name=f"interview_report_{s['session_id']}.json",
            mime="application/json",
            use_container_width=True,
        )
        if st.button("Start new interview", use_container_width=True):
            reset_session()
            st.rerun()

    with col_right:
        if summary["competency"]:
            st.plotly_chart(
                competency_radar(summary["competency"]),
                use_container_width=True,
                config={"displayModeBar": False},
            )
        else:
            st.info("No per-topic scores were available from this session.")

    st.divider()
    st.markdown('<p style="font-weight:600;margin-bottom:10px;">Full transcript</p>', unsafe_allow_html=True)
    for turn in s["transcript"]:
        render_question_turn(turn["round"], turn["topic"], turn["question"])
        badges = (
            cache_badge(turn["cache_hit"], 1.8 if turn["cache_hit"] else 900)
            + " " + model_badge(turn["model"])
            + " " + confidence_badge(turn["confidence"])
        )
        render_answer_turn(turn["answer"], badges)
