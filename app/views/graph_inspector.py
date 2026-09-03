"""
LangGraph DAG Inspector & State Visualizer View.
"""

import streamlit as st
import json
from app.components.cards import node_status_badge


def render_graph_inspector():
    st.markdown("### 🕸️ LangGraph DAG & State Inspector")
    st.caption("Live execution visualization of the LangGraph StateGraph, active node highlights, and checkpoint state.")

    state = st.session_state.get("graph_state", {})
    history = state.get("graph_history", ["START"])
    active_node = state.get("active_node", "START")

    # 1. Mermaid Graph Diagram
    st.markdown("##### 🗺️ LangGraph Topology")
    mermaid_code = f"""
    graph LR
        START([START]) --> P[profile_parser_node]
        P --> R[retrieval_node]
        R --> Q[question_generator_node]
        Q --> C[semantic_cache_check_node]
        C --> E[evaluate_answer_node]
        E -->|Quality Gate| G{{quality_gate}}
        G -->|Validation Failure / Hard| PRO[pro_escalator_node]
        G -->|Valid Response / Cache Hit| WB[cache_writeback_node]
        PRO --> WB
        WB -->|More Rounds| R
        WB -->|All Rounds Complete| S[summary_report_node]
        S --> END([END])

        style P fill:#1e293b,stroke:#64748b,stroke-width:2px
        style R fill:#1e293b,stroke:#64748b,stroke-width:2px
        style Q fill:#1e293b,stroke:#64748b,stroke-width:2px
        style C fill:#1e293b,stroke:#64748b,stroke-width:2px
        style E fill:#1e293b,stroke:#64748b,stroke-width:2px
        style G fill:#0f172a,stroke:#3b82f6,stroke-width:2px
        style PRO fill:#7f1d1d,stroke:#ef4444,stroke-width:2px
        style WB fill:#064e3b,stroke:#10b981,stroke-width:2px
        style S fill:#1e293b,stroke:#64748b,stroke-width:2px
    """
    st.markdown(f"```mermaid\n{mermaid_code}\n```")

    st.markdown("---")

    # 2. Execution Trace
    st.markdown("##### 📜 Node Execution Sequence (Turn Trace)")
    if history:
        for idx, node in enumerate(history):
            is_active = (idx == len(history) - 1)
            node_status_badge(f"{idx+1}. {node}", is_active=is_active)
    else:
        st.caption("No graph execution trace recorded yet.")

    st.markdown("---")

    # 3. Live State JSON Inspector
    st.markdown("##### 🔍 Live Checkpoint State (InterviewState)")
    with st.expander("View Full TypedDict Checkpoint State JSON", expanded=True):
        st.json(state)
