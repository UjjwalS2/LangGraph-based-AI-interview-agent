"""
Plotly Dark Theme Charts for LangGraph Platform.
"""

import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List


def create_radar_competency_chart(competencies: Dict[str, float]) -> go.Figure:
    categories = list(competencies.keys())
    values = list(competencies.values())
    categories.append(categories[0])
    values.append(values[0])

    fig = go.Figure(
        data=go.Scatterpolar(
            r=values,
            theta=categories,
            fill="toself",
            fillcolor="rgba(16, 185, 129, 0.2)",
            line=dict(color="#10b981", width=2.5),
            marker=dict(size=6, color="#34d399"),
        )
    )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 10], color="#64748b", gridcolor="rgba(255,255,255,0.08)"),
            angularaxis=dict(color="#94a3b8", gridcolor="rgba(255,255,255,0.08)"),
            bgcolor="rgba(0,0,0,0)",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=40, t=30, b=30),
        height=320,
        showlegend=False,
    )
    return fig


def create_cost_comparison_chart(baseline_cost: float, optimized_cost: float) -> go.Figure:
    fig = go.Figure(
        data=[
            go.Bar(
                x=["All-Pro Baseline", "LangGraph Optimized"],
                y=[baseline_cost, optimized_cost],
                marker_color=["#ef4444", "#10b981"],
                text=[f"${baseline_cost:.4f}", f"${optimized_cost:.4f}"],
                textposition="outside",
                width=0.45,
            )
        ]
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=30, b=20),
        height=280,
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.08)", color="#94a3b8"),
        xaxis=dict(color="#94a3b8"),
    )
    return fig
