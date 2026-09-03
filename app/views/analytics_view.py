"""
Analytics & Token Telemetry View for LangGraph Platform.
"""

import streamlit as st
from app.components.cards import metric_card
from app.components.charts import create_cost_comparison_chart
from app.cache.semantic_cache import get_semantic_cache
from app.config import config


def render_analytics_view():
    st.markdown("### 📈 Cost Optimization & Performance Telemetry")
    st.caption("Live financial telemetry, token consumption, and semantic caching efficiency across graph executions.")

    cache = get_semantic_cache()
    c_stats = cache.get_stats()
    state = st.session_state.get("graph_state", {})

    evals = state.get("evaluations", [])
    in_tokens = state.get("total_input_tokens", len(evals) * 450 + 200)
    out_tokens = state.get("total_output_tokens", len(evals) * 350 + 150)

    # Cost calculations
    flash_cost_per_m_in = config.pricing.flash.input_per_1m_tokens
    flash_cost_per_m_out = config.pricing.flash.output_per_1m_tokens
    pro_cost_per_m_in = config.pricing.pro.input_per_1m_tokens
    pro_cost_per_m_out = config.pricing.pro.output_per_1m_tokens

    # Compute actual vs baseline
    actual_cost = (in_tokens / 1e6) * flash_cost_per_m_in + (out_tokens / 1e6) * flash_cost_per_m_out
    baseline_cost = (in_tokens / 1e6) * pro_cost_per_m_in + (out_tokens / 1e6) * pro_cost_per_m_out
    reduction_pct = ((baseline_cost - actual_cost) / baseline_cost * 100.0) if baseline_cost > 0 else 0.0

    # KPI Summary Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Cache Hit Rate", f"{c_stats.get('hit_rate_pct', 0.0):.1f}%", f"{c_stats.get('hits', 0)} hits / {c_stats.get('total', 0)} lookups")
    with c2:
        metric_card("Cost Reduction", f"{reduction_pct:.1f}%", "vs All-Pro baseline", delta=f"-{reduction_pct:.1f}%")
    with c3:
        metric_card("Total Tokens", f"{(in_tokens + out_tokens):,}", f"{in_tokens:,} in / {out_tokens:,} out")
    with c4:
        metric_card("Optimized Cost", f"${actual_cost:.5f}", f"Baseline: ${baseline_cost:.5f}")

    st.markdown("---")

    col_l, col_r = st.columns([1.1, 0.9])
    with col_l:
        st.markdown("##### 💰 Model Spend Comparison")
        st.plotly_chart(create_cost_comparison_chart(baseline_cost, actual_cost), use_container_width=True)

    with col_r:
        st.markdown("##### ⚡ Semantic Cache Partition Health")
        st.markdown(
            f"""
            - **Similarity Threshold:** `Cosine >= {config.cache.similarity_threshold}`
            - **Max Cache Capacity:** `{config.cache.max_entries}` entries
            - **Active Cache Entries:** `{c_stats.get('cached_count', 0)}` items
            - **Cache Hits:** `{c_stats.get('hits', 0)}`
            - **Cache Misses:** `{c_stats.get('misses', 0)}`
            """
        )
        if st.button("🧹 Flush Semantic Cache", use_container_width=True):
            cache.clear()
            st.toast("Cache flushed successfully!", icon="🧹")
            st.rerun()
