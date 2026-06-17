"""ASTraM Command Center — Streamlit Entrypoint.

Multi-page app with dark Palantir-grade theme.
"""

import sys
import os

# Ensure command_center root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
from frontend.components.theme import apply_theme, render_footer
from backend.services.model_adapter import PlaceholderModel
from backend.services.data_service import DataService
from frontend.components.ui import render_sidebar_health


# ── Page Config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ASTraM Command Center",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme ─────────────────────────────────────────────────────────────────
apply_theme()

# ── Init Session State ───────────────────────────────────────────────────
if "model" not in st.session_state:
    st.session_state.model = PlaceholderModel()
    st.session_state.model.load_model()

if "data_service" not in st.session_state:
    st.session_state.data_service = DataService()
    st.session_state.data_service.load_data()

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

# ── Sidebar ───────────────────────────────────────────────────────────────
st.sidebar.markdown("""
<div style="padding:8px 0 16px 0;">
    <div style="color:#F3F2EE;font-size:1.15rem;font-weight:700;
        font-family:'Inter Tight',sans-serif;letter-spacing:-0.02em;">
        🚦 ASTraM
    </div>
    <div style="color:#7D857F;font-size:0.7rem;font-family:'Inter',sans-serif;margin-top:2px;">
        Traffic Incident Command Center
    </div>
</div>
""", unsafe_allow_html=True)

model_info = st.session_state.model.get_model_metadata()
data_rows = len(st.session_state.data_service.df)
num_corridors = st.session_state.data_service.df["corridor"].nunique()
num_modules = len([f for f in os.listdir(os.path.join(os.path.dirname(__file__), "pages")) if f.endswith(".py") and not f.startswith("_")])
render_sidebar_health(model_info, data_rows)

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="color:#7D857F;font-size:0.65rem;padding:8px 0;">
    Flipkart Gridlock 2.0<br>
    Bengaluru Traffic Police
</div>
""", unsafe_allow_html=True)

# ── Landing ───────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:80px 0 40px 0;">
    <div style="font-size:3rem;margin-bottom:8px;">🚦</div>
    <h1 style="color:#F3F2EE;font-size:2rem;font-weight:700;margin-bottom:8px;">
        ASTraM Command Center
    </h1>
    <p style="color:#7D857F;font-size:1rem;max-width:600px;margin:0 auto;">
        Real-time Traffic Incident Intelligence Platform for Bengaluru
    </p>
    <div style="margin-top:32px;display:flex;justify-content:center;gap:16px;flex-wrap:wrap;">
        <div style="background:#121715;border:1px solid #232A28;border-radius:6px;
            padding:16px 24px;min-width:140px;">
            <div style="color:#C2A878;font-size:1.8rem;font-weight:700;">{0:,}</div>
            <div style="color:#7D857F;font-size:0.7rem;text-transform:uppercase;">Incidents</div>
        </div>
        <div style="background:#121715;border:1px solid #232A28;border-radius:6px;
            padding:16px 24px;min-width:140px;">
            <div style="color:#1C7C54;font-size:1.8rem;font-weight:700;">{1}</div>
            <div style="color:#7D857F;font-size:0.7rem;text-transform:uppercase;">Corridors</div>
        </div>
        <div style="background:#121715;border:1px solid #232A28;border-radius:6px;
            padding:16px 24px;min-width:140px;">
            <div style="color:#2F5D9F;font-size:1.8rem;font-weight:700;">{2}</div>
            <div style="color:#7D857F;font-size:0.7rem;text-transform:uppercase;">Modules</div>
        </div>
    </div>
    <p style="color:#7D857F;font-size:0.8rem;margin-top:32px;">
        Select a module from the sidebar to begin
    </p>
</div>
""".format(data_rows, num_corridors, num_modules), unsafe_allow_html=True)

render_footer()
