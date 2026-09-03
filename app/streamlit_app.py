"""
Main Streamlit Application for LangGraph Agentic Interview Platform.
"""

import streamlit as st
from pathlib import Path
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.views.candidate_view import render_candidate_view
from app.views.interview_view import render_interview_view
from app.views.evaluation_view import render_evaluation_view
from app.views.graph_inspector import render_graph_inspector
from app.views.analytics_view import render_analytics_view

st.set_page_config(
    page_title="LangGraph AI Interview Agent",
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Responsive Glassmorphism Styling
st.markdown(
    """
    <style>
    /* Global Styling */
    .stApp {
        background-color: #0e1117;
        color: #f8fafc;
    }
    
    /* Hide default Streamlit multi-page sidebar links */
    [data-testid="stSidebarNav"] {
        display: none !important;
    }
    
    /* Top Header Branding */
    .brand-header {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 10px 0 20px 0;
    }
    
    /* Buttons */
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s ease-in-out;
    }
    .stButton>button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize Session State
if "graph_state" not in st.session_state:
    st.session_state["graph_state"] = {
        "candidate_name": "Alex Chen",
        "target_role": "Senior AI/ML Engineer",
        "experience_level": "Senior",
        "skills": ["Python", "XGBoost", "PyTorch", "RAG", "SQL", "LangGraph"],
        "focus_areas": ["machine_learning", "nlp_llm", "system_design"],
        "difficulty": "medium",
        "round_number": 1,
        "max_rounds": 3,
        "is_complete": False,
        "evaluations": [],
        "graph_history": ["START"],
    }
    st.session_state["initialized"] = False

# Sidebar Navigation
with st.sidebar:
    st.markdown(
        """
        <div style="padding: 10px 0;">
            <h2 style="margin: 0; font-size: 1.4rem; color: #10b981;">🕸️ LangGraph Agent</h2>
            <div style="color: #94a3b8; font-size: 0.8rem; font-weight: 500;">Stateful Multi-Agent Assessment</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")

    selected_view = st.radio(
        "Navigation",
        [
            "👤 Candidate Setup",
            "🎙️ Live Interview Room",
            "📋 Assessment & Debrief",
            "🕸️ LangGraph DAG Inspector",
            "📈 Telemetry & Analytics",
        ],
        index=0 if not st.session_state.get("initialized") else 1,
    )

    st.markdown("---")
    
    # Active Session Info Widget
    cur_state = st.session_state.get("graph_state", {})
    st.markdown("##### 📌 Active Session")
    st.markdown(f"**Candidate:** `{cur_state.get('candidate_name', 'Not Set')}`")
    st.markdown(f"**Target Role:** `{cur_state.get('target_role', 'Not Set')}`")
    st.markdown(f"**Round:** `{cur_state.get('round_number', 1)} / {cur_state.get('max_rounds', 3)}`")
    st.markdown(f"**Active Node:** `{cur_state.get('active_node', 'START')}`")

    if st.button("🔄 Reset Full Graph State", use_container_width=True):
        st.session_state.clear()
        st.rerun()

# Main Router
if selected_view == "👤 Candidate Setup":
    render_candidate_view()
elif selected_view == "🎙️ Live Interview Room":
    render_interview_view()
elif selected_view == "📋 Assessment & Debrief":
    render_evaluation_view()
elif selected_view == "🕸️ LangGraph DAG Inspector":
    render_graph_inspector()
elif selected_view == "📈 Telemetry & Analytics":
    render_analytics_view()
