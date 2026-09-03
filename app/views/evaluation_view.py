"""
Evaluation & Debrief View for LangGraph Platform.
"""

import streamlit as st
from app.components.cards import score_hero_banner, metric_card
from app.components.charts import create_radar_competency_chart


def render_evaluation_view():
    state = st.session_state.get("graph_state")

    if not state or not state.get("evaluations"):
        st.info("💡 No evaluations completed yet. Start an interview in the **Live Interview Room** to see evaluations.")
        return

    evals = state.get("evaluations", [])
    report = state.get("final_report", {})
    is_complete = state.get("is_complete", False)

    # Top Hero Scorecard if completed
    if is_complete and report:
        verdict = report.get("technical_verdict", "HIRE")
        score = report.get("overall_score", 8.0)
        rec = report.get("recommendation", "")
        score_hero_banner(score, verdict, rec)

        col_l, col_r = st.columns([1.1, 0.9])
        with col_l:
            st.markdown("##### 🌟 Candidate Strengths")
            for s in report.get("strengths", []):
                st.markdown(f"- ✅ **{s}**")

            st.markdown("##### 🎯 Areas for Growth")
            for a in report.get("areas_for_improvement", []):
                st.markdown(f"- ⚠️ *{a}*")

        with col_r:
            st.markdown("##### 🕸️ Competency Radar")
            comps = report.get("radar_competencies", {})
            if comps:
                st.plotly_chart(create_radar_competency_chart(comps), use_container_width=True)

        st.markdown("---")

    # Round-by-Round Breakdown
    st.markdown(f"### 📋 Round-by-Round Rubric Breakdown ({len(evals)} Rounds Evaluated)")

    for idx, e in enumerate(evals, start=1):
        round_score = e.get("score", 7.5)
        badge_color = "#10b981" if round_score >= 8.0 else "#f59e0b"
        model_str = f"Cache HIT (0ms)" if e.get("cache_hit") else f"Gemini {e.get('model_used', 'flash').capitalize()}"
        if e.get("escalated"):
            model_str += " ➔ Pro Escalated"

        with st.expander(f"📍 Round {idx}: {e.get('topic', 'General').upper()} — Score: {round_score:.1f}/10 ({model_str})", expanded=(idx == len(evals))):
            st.markdown(f"**Question:** *{e.get('question')}*")
            st.caption(f"**Candidate Answer:** {e.get('candidate_answer')}")

            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("##### ✅ Correct Points")
                for cp in e.get("correct_points", []):
                    st.markdown(f"- {cp}")
            with c2:
                st.markdown("##### ⚠️ Missing Concepts")
                for mc in e.get("missing_concepts", []):
                    st.markdown(f"- {mc}")
            with c3:
                st.markdown("##### ❌ Technical Inaccuracies")
                inc = e.get("incorrect_points", [])
                if inc:
                    for ip in inc:
                        st.markdown(f"- {ip}")
                else:
                    st.markdown("*None detected.*")

            st.markdown(f"**💬 Detailed Feedback:** {e.get('feedback')}")
            if e.get("ideal_answer"):
                st.markdown(f"**💡 Benchmark Ideal Answer:** *{e.get('ideal_answer')}*")
