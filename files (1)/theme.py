"""
Visual theme for the interview console.

Design concept: this isn't a chatbot skin — it's an engineering console for
the interview pipeline itself. Cache hits/misses, node execution, and
confidence gates are surfaced as first-class UI content, because that's
what this system actually does (see README benchmark section). Dark
slate base with a signal-teal accent for "live/pass", amber for
"escalate/borderline", and a mono type for anything numeric or a node
name, since that content is genuinely tabular/technical.
"""

import streamlit as st

COLORS = {
    "bg": "#0B0F14",
    "bg_alt": "#10151C",
    "surface": "#161C24",
    "surface_2": "#1C2530",
    "border": "#2A3441",
    "border_soft": "#212A35",
    "text": "#E8EEF4",
    "text_muted": "#8C99A8",
    "text_faint": "#4E5964",
    "accent": "#45D6C4",
    "accent_soft": "#173C38",
    "amber": "#F2A93B",
    "amber_soft": "#3D3013",
    "red": "#E5626B",
    "red_soft": "#3A1B20",
    "violet": "#9B8CE8",
    "violet_soft": "#241F3D",
}

FONT_UI = "'Manrope', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"
FONT_MONO = "'JetBrains Mono', 'SFMono-Regular', Consolas, monospace"


def inject_css() -> None:
    c = COLORS
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

        html, body, [class*="css"] {{
            font-family: {FONT_UI};
        }}

        .stApp {{
            background: radial-gradient(ellipse 1200px 600px at 50% -10%, {c['bg_alt']} 0%, {c['bg']} 55%);
            color: {c['text']};
        }}

        section[data-testid="stSidebar"] {{
            background: {c['bg_alt']};
            border-right: 1px solid {c['border_soft']};
        }}

        #MainMenu, footer, header[data-testid="stHeader"] {{
            background: transparent;
        }}

        h1, h2, h3, h4 {{
            font-family: {FONT_UI};
            font-weight: 700;
            letter-spacing: -0.01em;
            color: {c['text']};
        }}

        p, li, span, label {{
            color: {c['text']};
        }}

        /* ---------- hero ---------- */
        .console-hero {{
            display: flex;
            align-items: baseline;
            gap: 14px;
            margin-bottom: 2px;
        }}
        .console-hero .mark {{
            width: 10px; height: 10px; border-radius: 2px;
            background: {c['accent']};
            box-shadow: 0 0 12px {c['accent']}88;
            display: inline-block;
        }}
        .console-title {{
            font-size: 1.9rem;
            font-weight: 800;
            margin: 0;
        }}
        .console-subtitle {{
            color: {c['text_muted']};
            font-size: 0.95rem;
            margin-top: 2px;
            margin-bottom: 22px;
            line-height: 1.5;
            max-width: 640px;
        }}

        /* ---------- generic card ---------- */
        .card {{
            background: {c['surface']};
            border: 1px solid {c['border']};
            border-radius: 10px;
            padding: 18px 20px;
            animation: fadeIn 0.35s ease-out;
        }}
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(4px); }}
            to   {{ opacity: 1; transform: translateY(0); }}
        }}

        /* ---------- pipeline chips ---------- */
        .pipeline-row {{
            display: flex;
            gap: 6px;
            overflow-x: auto;
            padding: 4px 2px 14px 2px;
            margin-bottom: 6px;
        }}
        .node-chip {{
            font-family: {FONT_MONO};
            font-size: 0.68rem;
            font-weight: 500;
            padding: 6px 10px;
            border-radius: 6px;
            border: 1px solid {c['border']};
            background: {c['surface']};
            color: {c['text_faint']};
            white-space: nowrap;
            transition: all 0.25s ease;
        }}
        .node-chip.pending {{ opacity: 0.5; }}
        .node-chip.active {{
            background: {c['accent_soft']};
            border-color: {c['accent']};
            color: {c['accent']};
            box-shadow: 0 0 0 1px {c['accent']}55;
        }}
        .node-chip.done {{
            background: {c['surface_2']};
            border-color: {c['border']};
            color: {c['text_muted']};
        }}
        .node-chip.escalate {{
            background: {c['amber_soft']};
            border-color: {c['amber']};
            color: {c['amber']};
        }}
        .node-chip.miss {{
            background: {c['red_soft']};
            border-color: {c['red']};
            color: {c['red']};
        }}

        /* ---------- badges ---------- */
        .badge {{
            display: inline-flex;
            align-items: center;
            gap: 5px;
            font-family: {FONT_MONO};
            font-size: 0.72rem;
            font-weight: 500;
            padding: 3px 9px;
            border-radius: 5px;
            border: 1px solid transparent;
        }}
        .badge-hit {{ background: {c['accent_soft']}; color: {c['accent']}; border-color: {c['accent']}44; }}
        .badge-miss {{ background: {c['red_soft']}; color: {c['red']}; border-color: {c['red']}44; }}
        .badge-pro {{ background: {c['violet_soft']}; color: {c['violet']}; border-color: {c['violet']}44; }}
        .badge-flash {{ background: {c['surface_2']}; color: {c['text_muted']}; border-color: {c['border']}; }}
        .badge-mode-live {{ background: {c['accent_soft']}; color: {c['accent']}; border-color: {c['accent']}44; }}
        .badge-mode-offline {{ background: {c['surface_2']}; color: {c['text_muted']}; border-color: {c['border']}; }}

        /* ---------- transcript ---------- */
        .turn-question {{
            background: {c['surface']};
            border: 1px solid {c['border']};
            border-left: 2px solid {c['accent']};
            border-radius: 8px;
            padding: 14px 18px;
            margin-bottom: 10px;
            animation: fadeIn 0.3s ease-out;
        }}
        .turn-answer {{
            background: {c['bg_alt']};
            border: 1px solid {c['border_soft']};
            border-left: 2px solid {c['text_faint']};
            border-radius: 8px;
            padding: 14px 18px;
            margin-bottom: 6px;
            animation: fadeIn 0.3s ease-out;
        }}
        .turn-label {{
            font-family: {FONT_MONO};
            font-size: 0.68rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: {c['text_faint']};
            margin-bottom: 6px;
        }}
        .turn-meta {{
            display: flex; gap: 8px; margin-top: 10px; flex-wrap: wrap;
        }}

        /* ---------- metric tiles ---------- */
        .metric-tile {{
            background: {c['surface']};
            border: 1px solid {c['border']};
            border-radius: 10px;
            padding: 14px 16px;
        }}
        .metric-value {{
            font-family: {FONT_MONO};
            font-size: 1.5rem;
            font-weight: 600;
            color: {c['text']};
            line-height: 1.1;
        }}
        .metric-label {{
            font-size: 0.76rem;
            color: {c['text_muted']};
            margin-top: 4px;
        }}

        /* ---------- buttons ---------- */
        .stButton > button {{
            background: {c['accent']};
            color: #06211E;
            border: none;
            border-radius: 7px;
            font-weight: 700;
            padding: 0.5rem 1.1rem;
            transition: filter 0.15s ease;
        }}
        .stButton > button:hover {{
            filter: brightness(1.08);
        }}
        .stButton > button:disabled {{
            background: {c['surface_2']};
            color: {c['text_faint']};
        }}

        div[data-testid="stTextArea"] textarea, div[data-testid="stTextInput"] input {{
            background: {c['bg_alt']} !important;
            border: 1px solid {c['border']} !important;
            color: {c['text']} !important;
            border-radius: 8px !important;
        }}

        .stProgress > div > div > div > div {{
            background: {c['accent']};
        }}

        /* scrollbar */
        ::-webkit-scrollbar {{ height: 6px; width: 6px; }}
        ::-webkit-scrollbar-thumb {{ background: {c['border']}; border-radius: 4px; }}
        </style>
        """,
        unsafe_allow_html=True,
    )
