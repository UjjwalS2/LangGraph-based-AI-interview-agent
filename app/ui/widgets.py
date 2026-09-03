"""
Reusable render helpers for the interview console.
Kept framework-light (plain HTML strings + plotly) so they're easy to
restyle without hunting through app logic.
"""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from app.ui.theme import COLORS, FONT_MONO, FONT_UI

PIPELINE_NODES = [
    ("profile_parser_node", "parse"),
    ("question_generator_node", "question"),
    ("semantic_cache_check_node", "cache"),
    ("retrieval_node", "retrieve"),
    ("evaluate_answer_node", "evaluate"),
    ("pro_escalator_node", "escalate"),
    ("cache_writeback_node", "writeback"),
    ("summary_report_node", "summary"),
]


def render_pipeline(active_key: str | None, done_keys: set[str], skipped_keys: set[str], status_keys: dict[str, str] | None = None) -> None:
    status_keys = status_keys or {}
    chips = []
    for key, label in PIPELINE_NODES:
        if key in skipped_keys:
            continue
        if key == active_key:
            cls = "active"
        elif key in done_keys:
            cls = status_keys.get(key, "done")
        else:
            cls = "pending"
        chips.append(f'<div class="node-chip {cls}">{label}</div>')
    st.markdown(f'<div class="pipeline-row">{"".join(chips)}</div>', unsafe_allow_html=True)


def cache_badge(hit: bool, latency_ms: float) -> str:
    if hit:
        return f'<span class="badge badge-hit">◆ cache hit · {latency_ms:.1f} ms</span>'
    return f'<span class="badge badge-miss">◇ cache miss · {latency_ms:.0f} ms</span>'


def model_badge(model: str) -> str:
    if model == "pro":
        return '<span class="badge badge-pro">⇡ escalated to Pro</span>'
    return '<span class="badge badge-flash">Flash</span>'


def confidence_badge(confidence: float) -> str:
    pct = confidence * 100
    color_cls = "badge-hit" if confidence >= 0.72 else "badge-miss"
    return f'<span class="badge {color_cls}">confidence {pct:.0f}%</span>'


def render_question_turn(round_no: int, topic: str, question: str) -> None:
    st.markdown(
        f"""
        <div class="turn-question">
            <div class="turn-label">round {round_no} · {topic}</div>
            {question}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_answer_turn(answer: str, badges_html: str) -> None:
    st.markdown(
        f"""
        <div class="turn-answer">
            <div class="turn-label">candidate response</div>
            {answer}
            <div class="turn-meta">{badges_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_tile(label: str, value: str) -> str:
    return f"""
        <div class="metric-tile">
            <div class="metric-value">{value}</div>
            <div class="metric-label">{label}</div>
        </div>
    """


def render_metric_row(metrics: list[tuple[str, str]]) -> None:
    cols = st.columns(len(metrics))
    for col, (label, value) in zip(cols, metrics):
        with col:
            st.markdown(metric_tile(label, value), unsafe_allow_html=True)


def confidence_gauge(value: float) -> go.Figure:
    color = COLORS["accent"] if value >= 0.72 else COLORS["red"]
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value * 100,
            number={"suffix": "%", "font": {"family": FONT_MONO, "size": 26, "color": COLORS["text"]}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": COLORS["text_faint"], "tickfont": {"size": 9}},
                "bar": {"color": color, "thickness": 0.35},
                "bgcolor": COLORS["surface_2"],
                "borderwidth": 0,
                "threshold": {"line": {"color": COLORS["amber"], "width": 2}, "thickness": 0.85, "value": 72},
                "steps": [{"range": [0, 100], "color": COLORS["surface_2"]}],
            },
        )
    )
    fig.update_layout(height=140, margin=dict(l=18, r=18, t=8, b=8), paper_bgcolor="rgba(0,0,0,0)", font={"family": FONT_UI, "color": COLORS["text"]})
    return fig


def competency_radar(scores: dict[str, float]) -> go.Figure:
    categories = list(scores.keys())
    values = list(scores.values())
    fig = go.Figure()
    if categories:
        fig.add_trace(
            go.Scatterpolar(
                r=values + [values[0]],
                theta=categories + [categories[0]],
                fill="toself",
                fillcolor="rgba(69, 214, 196, 0.18)",
                line=dict(color=COLORS["accent"], width=2),
                marker=dict(size=5, color=COLORS["accent"]),
            )
        )
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0, 10], gridcolor=COLORS["border"], tickfont={"size": 9, "color": COLORS["text_faint"]}),
            angularaxis=dict(gridcolor=COLORS["border"], tickfont={"family": FONT_UI, "size": 12, "color": COLORS["text"]}),
        ),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=40, t=30, b=20),
        height=380,
        font={"family": FONT_UI, "color": COLORS["text"]},
    )
    return fig


def cost_bar(cache_hit_cost: float, flash_cost: float, pro_cost: float, all_pro_baseline: float) -> go.Figure:
    actual = cache_hit_cost + flash_cost + pro_cost
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=["This session", "All-Pro baseline"],
            x=[actual, all_pro_baseline],
            orientation="h",
            marker_color=[COLORS["accent"], COLORS["border"]],
            text=[f"${actual:.4f}", f"${all_pro_baseline:.4f}"],
            textposition="outside",
            textfont=dict(family=FONT_MONO, color=COLORS["text"]),
        )
    )
    fig.update_layout(
        height=140,
        margin=dict(l=8, r=40, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, showticklabels=False),
        yaxis=dict(tickfont={"family": FONT_UI, "color": COLORS["text"], "size": 12}),
        font={"family": FONT_UI, "color": COLORS["text"]},
    )
    return fig
