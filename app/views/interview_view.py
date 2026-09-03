"""
Live Interview Room View for LangGraph Platform.
"""

import streamlit as st
import time
from app.graph.nodes.cache_node import semantic_cache_check_node
from app.graph.nodes.evaluation_node import evaluate_answer_node
from app.graph.nodes.quality_gate import quality_gate_decision, pro_escalator_node, cache_writeback_node
from app.graph.nodes.retrieval_node import retrieval_node
from app.graph.nodes.question_node import question_generator_node
from app.graph.nodes.summary_node import summary_report_node

SAMPLE_ANSWERS = {
    "machine_learning": "Random Forest lowers ensemble variance through bagging (bootstrap sample aggregation) and random feature sub-sampling at every tree split. This de-correlates individual decision trees so their averaged prediction has lower variance without increasing bias.",
    "nlp_llm": "Reciprocal Rank Fusion (RRF) combines dense vector and BM25 lexical search by taking the reciprocal of document ranks (1 / (k + rank)) rather than raw scores. This avoids scale mismatch between unbounded BM25 scores and cosine distances.",
    "system_design": "Consistent Hashing maps both servers and keys onto a circular 2^32-1 hash ring. By introducing virtual nodes per physical machine, it balances load uniformly and ensures that when nodes join or leave, only K/N keys require migration.",
    "python": "CPython uses reference counting for immediate memory reclamation and generational cyclic garbage collection (Gen 0, 1, 2) to detect circular object references. The GIL prevents multiple native CPU threads from executing Python bytecode simultaneously.",
    "sql": "ROW_NUMBER assigns strict sequential integers without ties, RANK assigns the same rank to ties and skips subsequent positions, and DENSE_RANK assigns the same rank to ties without skipping subsequent positions.",
}


def render_interview_view():
    state = st.session_state.get("graph_state")

    if not state or not st.session_state.get("initialized"):
        st.warning("⚠️ Please configure candidate details in the **Candidate Setup** tab first.")
        return

    round_num = state.get("round_number", 1)
    max_rounds = state.get("max_rounds", 3)
    topic = state.get("current_topic", "machine_learning")
    question = state.get("current_question", "Loading question...")
    is_done = state.get("is_complete", False)

    if is_done:
        st.success("🎉 **Interview Complete!** All rounds have been evaluated by the LangGraph agent.")
        st.info("Navigate to the **📊 Assessment & Debrief** tab to view your final scorecard and radar debrief.")
        return

    # Header Progress
    c1, c2 = st.columns([3, 1])
    with c1:
        st.markdown(f"### 🎙️ Technical Interview — Round {round_num} of {max_rounds}")
        st.caption(f"Candidate: **{state.get('candidate_name')}** ({state.get('experience_level')}) | Domain: `{topic.upper()}`")
    with c2:
        st.progress(round_num / max_rounds, text=f"Round {round_num}/{max_rounds}")

    # Question Container Card
    st.markdown(
        f"""
        <div style="background: rgba(30, 41, 59, 0.8); border: 1px solid rgba(16, 185, 129, 0.4); border-radius: 14px; padding: 22px 26px; margin: 16px 0 20px 0; box-shadow: 0 4px 20px rgba(0,0,0,0.25);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <span style="background: rgba(16, 185, 129, 0.15); border: 1px solid rgba(16, 185, 129, 0.3); color: #34d399; font-size: 0.8rem; font-weight: 700; padding: 3px 8px; border-radius: 4px; text-transform: uppercase;">ROUND {round_num} • {topic.replace('_', ' ').upper()}</span>
                <span style="color: #94a3b8; font-size: 0.8rem;">Difficulty: <b style="color: #f8fafc;">{state.get('difficulty', 'medium').capitalize()}</b></span>
            </div>
            <h3 style="color: #f8fafc; font-size: 1.35rem; font-weight: 600; line-height: 1.4; margin: 6px 0;">{question}</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # RAG Grounding Expander
    with st.expander("📚 Authoritative RAG Grounding Reference (Retrieved by Hybrid Node)", expanded=False):
        docs = state.get("grounding_docs", [])
        if docs:
            for d in docs:
                st.markdown(f"**[{d.get('doc_id')}]** *(Relevance Score: {d.get('score')})*")
                st.code(d.get("text", "")[:350] + "...", language="markdown")
        else:
            st.caption("No grounding docs loaded for this round.")

    # Candidate Answer Box
    default_ans = SAMPLE_ANSWERS.get(topic, SAMPLE_ANSWERS["machine_learning"])

    c_tool1, c_tool2 = st.columns([3, 1])
    with c_tool2:
        if st.button("📝 Auto-Fill Sample Answer", use_container_width=True):
            st.session_state["user_ans_input"] = default_ans
            st.rerun()

    ans_text = st.text_area(
        "Your Technical Answer:",
        value=st.session_state.get("user_ans_input", ""),
        height=160,
        placeholder="Provide your in-depth technical explanation, referencing algorithms, tradeoffs, and failure modes...",
    )

    # Submit Button
    if st.button("🚀 Submit Answer & Execute LangGraph Step", type="primary", use_container_width=True):
        if not ans_text.strip():
            st.warning("Please provide an answer before submitting.")
            return

        with st.spinner("🤖 LangGraph DAG is executing: Cache Check ➔ Flash Evaluator ➔ Quality Gate..."):
            # Update state with submitted answer
            state["candidate_answer"] = ans_text

            # 1. Cache check node
            s_cache = semantic_cache_check_node(state)
            state.update(s_cache)

            # 2. Evaluation node
            s_eval = evaluate_answer_node(state)
            state.update(s_eval)

            # 3. Quality gate conditional routing
            gate_decision = quality_gate_decision(state)
            if gate_decision == "pro_escalator_node":
                s_pro = pro_escalator_node(state)
                state.update(s_pro)

            # 4. Cache writeback node
            s_wb = cache_writeback_node(state)
            state.update(s_wb)

            # 5. Check if finished or prepare next round
            if state.get("is_complete"):
                s_sum = summary_report_node(state)
                state.update(s_sum)
            else:
                # Prepare next round question
                s_ret = retrieval_node(state)
                state.update(s_ret)
                s_q = question_generator_node(state)
                state.update(s_q)

            st.session_state["graph_state"] = state
            st.session_state["user_ans_input"] = ""
            st.toast("Round evaluated successfully!", icon="✅")
            st.rerun()
