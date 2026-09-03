"""
Reusable Glassmorphism UI Cards for LangGraph Platform.
"""

import streamlit as st
from typing import List, Optional


def metric_card(label: str, value: str, subtext: str = "", delta: Optional[str] = None):
    delta_html = f'<span style="color: #10b981; font-size: 0.85rem; font-weight: 600; margin-left: 8px;">{delta}</span>' if delta else ""
    st.markdown(
        f"""
        <div style="background: rgba(30, 41, 59, 0.7); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 18px 20px; box-shadow: 0 4px 20px rgba(0,0,0,0.2); backdrop-filter: blur(10px);">
            <div style="color: #94a3b8; font-size: 0.85rem; font-weight: 500; text-transform: uppercase; letter-spacing: 0.5px;">{label}</div>
            <div style="color: #f8fafc; font-size: 1.8rem; font-weight: 700; margin: 6px 0 2px 0;">{value} {delta_html}</div>
            <div style="color: #64748b; font-size: 0.8rem;">{subtext}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def node_status_badge(node_name: str, is_active: bool = False):
    color = "#10b981" if is_active else "#64748b"
    bg = "rgba(16, 185, 129, 0.15)" if is_active else "rgba(100, 116, 139, 0.15)"
    border = "rgba(16, 185, 129, 0.4)" if is_active else "rgba(255, 255, 255, 0.05)"
    
    st.markdown(
        f"""
        <span style="display: inline-block; background: {bg}; border: 1px solid {border}; color: {color}; padding: 4px 10px; border-radius: 6px; font-size: 0.8rem; font-family: monospace; font-weight: 600; margin-right: 6px; margin-bottom: 6px;">
            {'▶ ' if is_active else '✓ '} {node_name}
        </span>
        """,
        unsafe_allow_html=True,
    )


def score_hero_banner(score: float, verdict: str, subtitle: str = ""):
    color = "#10b981" if score >= 8.0 else "#f59e0b" if score >= 6.5 else "#ef4444"
    st.markdown(
        f"""
        <div style="background: linear-gradient(135deg, rgba(30, 41, 59, 0.9) 0%, rgba(15, 23, 42, 0.9) 100%); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 16px; padding: 24px 28px; margin-bottom: 24px; box-shadow: 0 10px 30px rgba(0,0,0,0.3);">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <span style="background: rgba(255,255,255,0.08); padding: 4px 10px; border-radius: 6px; color: #94a3b8; font-size: 0.85rem; font-weight: 600;">LANGGRAPH ASSESSMENT VERDICT</span>
                    <h2 style="color: #f8fafc; font-size: 1.8rem; margin: 8px 0 4px 0;">{verdict}</h2>
                    <p style="color: #94a3b8; margin: 0; font-size: 0.95rem;">{subtitle}</p>
                </div>
                <div style="text-align: right; background: rgba(0,0,0,0.3); padding: 14px 24px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.05);">
                    <div style="font-size: 2.6rem; font-weight: 800; color: {color}; line-height: 1;">{score:.1f}<span style="font-size: 1.2rem; color: #64748b;">/10</span></div>
                    <div style="color: #94a3b8; font-size: 0.8rem; font-weight: 600; text-transform: uppercase;">Composite Score</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
