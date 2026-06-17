"""ASTraM Command Center — Theme Engine.

Injects the full Palantir-grade dark design system into Streamlit via CSS.
NO glow. NO neon. NO glassmorphism. Industrial, restrained, executive.
"""

import streamlit as st

def apply_theme():
    """Inject the complete design system CSS into the Streamlit app."""
    st.markdown("""
    <style>
    /* ── Google Fonts ─────────────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Inter+Tight:wght@500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    /* ── Root Tokens ──────────────────────────────────────────────── */
    :root {
        --bg:           #0A0D0C;
        --bg-secondary: #121715;
        --bg-tertiary:  #1B2220;
        --border:       #232A28;
        --text:         #F3F2EE;
        --text-sec:     #B5B8B1;
        --muted:        #7D857F;
        --success:      #2EA66F;
        --warning:      #B8833B;
        --critical:     #B04A4A;
        --info:         #2F5D9F;
        --radius:       6px;
        --transition:   150ms ease;
    }

    /* ── Global ───────────────────────────────────────────────────── */
    .stApp {
        background-color: var(--bg) !important;
        color: var(--text) !important;
        font-family: 'Inter', -apple-system, sans-serif !important;
    }

    .stApp > header { background-color: var(--bg) !important; }

    /* ── Sidebar ──────────────────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background-color: var(--bg-secondary) !important;
        border-right: 1px solid var(--border) !important;
    }
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown span {
        color: var(--text-sec) !important;
        font-size: 0.85rem;
    }
    /* Lock sidebar open and hide collapse arrows */
    [data-testid="collapsedControl"] { display: none !important; }
    [data-testid="stSidebarCollapseButton"] { display: none !important; }
    
    /* Disable Streamlit rerun dimming/fading and force wide layout */
    [data-testid="stAppViewBlockContainer"] { 
        opacity: 1 !important; 
        filter: none !important; 
        transition: none !important; 
        max-width: 100% !important; 
        padding-left: 5% !important; 
        padding-right: 5% !important; 
    }
    [data-testid="stStatusWidget"] { visibility: hidden !important; display: none !important; }

    /* ── Typography ───────────────────────────────────────────────── */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter Tight', 'Inter', sans-serif !important;
        color: var(--text) !important;
        font-weight: 600 !important;
        letter-spacing: -0.02em;
    }
    h1 { font-size: 1.75rem !important; }
    h2 { font-size: 1.35rem !important; }
    h3 { font-size: 1.1rem !important; }
    p, li, label, .stMarkdown {
        font-family: 'Inter', sans-serif !important;
        color: var(--text-sec) !important;
    }
    code, pre, .stCode {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* ── Buttons ──────────────────────────────────────────────────── */
    .stButton > button {
        background-color: var(--bg-tertiary) !important;
        color: var(--text) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 500 !important;
        font-size: 0.85rem !important;
        padding: 0.45rem 1.1rem !important;
        transition: all var(--transition) !important;
    }
    .stButton > button:hover {
        background-color: var(--border) !important;
        border-color: var(--muted) !important;
    }
    .stButton > button[kind="primary"] {
        background-color: #1C7C54 !important;
        border-color: #1C7C54 !important;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #2EA66F !important;
    }

    /* ── Inputs ───────────────────────────────────────────────────── */
    .stSelectbox [data-baseweb="select"],
    .stMultiSelect [data-baseweb="select"],
    .stTextInput input,
    .stNumberInput input,
    .stDateInput input {
        background-color: var(--bg-tertiary) !important;
        color: var(--text) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
        font-family: 'Inter', sans-serif !important;
    }

    /* ── Tabs ─────────────────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0px;
        background-color: var(--bg-secondary);
        border-radius: var(--radius);
        padding: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        color: var(--muted);
        border-radius: var(--radius);
        font-family: 'Inter', sans-serif !important;
        font-size: 0.85rem;
        font-weight: 500;
        padding: 8px 16px;
    }
    .stTabs [aria-selected="true"] {
        background-color: var(--bg-tertiary) !important;
        color: var(--text) !important;
    }

    /* ── Expander ─────────────────────────────────────────────────── */
    .streamlit-expanderHeader {
        background-color: var(--bg-secondary) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
        color: var(--text) !important;
        font-weight: 500 !important;
    }
    .streamlit-expanderContent {
        background-color: var(--bg-secondary) !important;
        border: 1px solid var(--border) !important;
        border-top: none !important;
    }

    /* ── Dataframes ───────────────────────────────────────────────── */
    .stDataFrame {
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
    }

    /* ── Metric ───────────────────────────────────────────────────── */
    [data-testid="stMetric"] {
        background-color: var(--bg-secondary);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 12px 16px;
    }
    [data-testid="stMetricLabel"] {
        color: var(--muted) !important;
        font-size: 0.75rem !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    [data-testid="stMetricValue"] {
        color: var(--text) !important;
        font-family: 'Inter Tight', sans-serif !important;
        font-weight: 600 !important;
    }

    /* ── Divider ──────────────────────────────────────────────────── */
    hr { border-color: var(--border) !important; opacity: 0.5; }

    /* ── Scrollbar ────────────────────────────────────────────────── */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: var(--bg); }
    ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--muted); }

    /* ── Hide Streamlit branding ──────────────────────────────────── */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header[data-testid="stHeader"] { background-color: var(--bg) !important; }

    </style>
    """, unsafe_allow_html=True)


def render_footer():
    """Render the persistent footer."""
    st.markdown("""
    <div style="
        position: fixed; bottom: 0; left: 0; right: 0;
        background: #0A0D0C; border-top: 1px solid #232A28;
        padding: 8px 24px; z-index: 999;
        display: flex; justify-content: space-between; align-items: center;
    ">
        <span style="color: #7D857F; font-size: 0.7rem; font-family: 'Inter', sans-serif;">
            ASTraM Command Center &nbsp;|&nbsp; Flipkart Gridlock 2.0 &nbsp;|&nbsp; Built for Traffic Intelligence
        </span>
        <span style="color: #7D857F; font-size: 0.7rem; font-family: 'JetBrains Mono', monospace;">
            v0.1.0
        </span>
    </div>
    """, unsafe_allow_html=True)
