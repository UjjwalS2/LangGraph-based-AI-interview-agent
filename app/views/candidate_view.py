"""
Candidate Profile Setup View for LangGraph Platform.
"""

import streamlit as st
from app.graph.workflow import get_compiled_graph
from app.graph.state import InterviewState

PRESETS = {
    "Alex Chen (ML & Distributed Systems)": {
        "candidate_name": "Alex Chen",
        "target_role": "Senior AI/ML Engineer",
        "experience_level": "Senior",
        "skills": ["Python", "XGBoost", "PyTorch", "RAG", "Qdrant", "SQL", "LangGraph"],
        "focus_areas": ["machine_learning", "nlp_llm", "system_design"],
        "difficulty": "medium",
    },
    "Priya Sharma (NLP & RAG Systems)": {
        "candidate_name": "Priya Sharma",
        "target_role": "Staff NLP / LLM Engineer",
        "experience_level": "Staff",
        "skills": ["Transformers", "BGE-M3", "RRF", "Cross-Encoders", "Semantic Cache", "LangChain"],
        "focus_areas": ["nlp_llm", "deep_learning", "system_design"],
        "difficulty": "hard",
    },
    "David Miller (Backend & Distributed Data)": {
        "candidate_name": "David Miller",
        "target_role": "Lead Backend Data Engineer",
        "experience_level": "Lead",
        "skills": ["Python", "Asyncio", "PyMalloc", "PostgreSQL", "Kafka", "Consistent Hashing"],
        "focus_areas": ["python", "sql", "system_design"],
        "difficulty": "medium",
    },
}


def render_candidate_view():
    st.markdown("### 👤 Candidate Setup & Job Alignment")
    st.caption("Configure candidate profile or select a 1-click preset to initialize the LangGraph StateGraph.")

    # 1-Click Preset Bar
    st.markdown("##### ⚡ Quick Profile Presets")
    cols = st.columns(3)
    for idx, (p_name, p_data) in enumerate(PRESETS.items()):
        with cols[idx]:
            if st.button(f"🚀 {p_name.split(' (')[0]}", use_container_width=True, help=p_name):
                st.session_state["graph_state"] = {
                    "candidate_name": p_data["candidate_name"],
                    "target_role": p_data["target_role"],
                    "experience_level": p_data["experience_level"],
                    "skills": p_data["skills"],
                    "focus_areas": p_data["focus_areas"],
                    "difficulty": p_data["difficulty"],
                    "round_number": 1,
                    "max_rounds": 3,
                    "is_complete": False,
                    "evaluations": [],
                    "graph_history": ["START"],
                }
                st.session_state["initialized"] = True
                st.toast(f"Loaded preset for {p_data['candidate_name']}!", icon="✅")
                st.rerun()

    st.markdown("---")

    col_left, col_right = st.columns([1.1, 0.9])

    with col_left:
        st.markdown("##### 📝 Candidate Profile Details")
        cur_state = st.session_state.get("graph_state", {})

        name = st.text_input("Full Name", value=cur_state.get("candidate_name", "Alex Chen"))
        role = st.text_input("Target Role", value=cur_state.get("target_role", "Senior AI/ML Engineer"))
        
        c1, c2 = st.columns(2)
        with c1:
            level = st.selectbox(
                "Experience Level",
                ["Junior", "Mid-Level", "Senior", "Staff", "Principal"],
                index=2,
            )
        with c2:
            difficulty = st.selectbox(
                "Interview Difficulty",
                ["easy", "medium", "hard"],
                index=1,
            )

        skills_str = st.text_area(
            "Skills (comma-separated)",
            value=", ".join(cur_state.get("skills", ["Python", "XGBoost", "PyTorch", "RAG", "SQL", "LangGraph"])),
            height=80,
        )

        max_rounds = st.slider("Interview Rounds", min_value=1, max_value=5, value=3)

    with col_right:
        st.markdown("##### 🎯 Target Technical Focus Areas")
        all_topics = ["machine_learning", "nlp_llm", "deep_learning", "python", "sql", "system_design"]
        selected_topics = st.multiselect(
            "Focus Domains",
            all_topics,
            default=cur_state.get("focus_areas", ["machine_learning", "nlp_llm", "system_design"]),
        )

        st.markdown("##### 📄 Job Description Alignment (Optional)")
        jd_text = st.text_area(
            "Paste Target Job Description",
            value="Seeking a Senior AI/ML engineer with deep experience in RAG systems, vector embeddings, cost optimization, and scalable distributed architectures.",
            height=130,
        )

    if st.button("🚀 Initialize LangGraph Interview Session", type="primary", use_container_width=True):
        parsed_skills = [s.strip() for s in skills_str.split(",") if s.strip()]
        
        # Initialize graph state
        initial_state: InterviewState = {
            "session_id": "sess_live_01",
            "candidate_name": name,
            "target_role": role,
            "experience_level": level,
            "skills": parsed_skills,
            "job_description": jd_text,
            "focus_areas": selected_topics if selected_topics else ["machine_learning", "nlp_llm"],
            "difficulty": difficulty,
            "round_number": 1,
            "max_rounds": max_rounds,
            "is_complete": False,
            "evaluations": [],
            "graph_history": ["START"],
        }

        # Step 1: Run through parser and retrieval and question generator nodes
        from app.graph.nodes.parser_node import profile_parser_node
        from app.graph.nodes.retrieval_node import retrieval_node
        from app.graph.nodes.question_node import question_generator_node

        s1 = profile_parser_node(initial_state)
        merged = {**initial_state, **s1}
        s2 = retrieval_node(merged)
        merged = {**merged, **s2}
        s3 = question_generator_node(merged)
        merged = {**merged, **s3}

        st.session_state["graph_state"] = merged
        st.session_state["initialized"] = True
        st.session_state["current_tab"] = "🎙️ Live Interview Room"
        st.rerun()
