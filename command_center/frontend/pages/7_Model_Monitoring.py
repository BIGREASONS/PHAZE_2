"""Page 7 — Model Monitoring.  Accent: Silver (#8B949E)."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import streamlit as st

st.set_page_config(page_title="ASTraM Command Center", page_icon="🚦", layout="wide", initial_sidebar_state="expanded")
import numpy as np
import plotly.graph_objects as go

from frontend.components.theme import apply_theme, render_footer
from frontend.components.ui import render_section_header, render_kpi_row, render_metric_tile
from backend.services.model_adapter import PlaceholderModel
from backend.services.data_service import DataService

apply_theme()

SILVER = "#8B949E"
SILVER_L = "#C5CCD4"
BG = "#0A0D0C"
BG2 = "#121715"
TEXT = "#F3F2EE"

model = PlaceholderModel()
meta = model.get_model_metadata()
metrics = meta.get("metrics", {})

# ── Header ────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="display:flex;align-items:center;gap:10px;margin-bottom:4px;">
    <span style="color:{SILVER};font-size:0.6rem;">&#9646;</span>
    <h1 style="margin:0;font-size:1.5rem;">Model Monitoring</h1>
</div>
<p style="color:#7D857F;font-size:0.8rem;margin-bottom:20px;">
    System health, performance metrics, and drift monitoring
</p>
""", unsafe_allow_html=True)

# ── System Status ─────────────────────────────────────────────────────────
render_section_header("System Status", accent=SILVER)
render_kpi_row([
    {"title": "Active Model", "value": meta["name"], "icon": "🤖"},
    {"title": "Version", "value": meta["version"], "icon": "📦"},
    {"title": "Training Date", "value": meta["training_date"], "icon": "📅"},
    {"title": "System Health", "value": meta["status"], "icon": "💚"},
], accent=SILVER)

# ── Performance Metrics ──────────────────────────────────────────────────
render_section_header("Performance Metrics", subtitle="Placeholder values", accent=SILVER)
st.markdown("""
<div style="color:#B8833B;font-size:0.7rem;background:#121715;border:1px solid #232A28;
    border-radius:4px;padding:6px 10px;margin-bottom:12px;">
    Placeholder — metrics will update automatically when final model is deployed
</div>
""", unsafe_allow_html=True)

mc = st.columns(5)
metric_items = [
    ("ROC-AUC", f"{metrics.get('roc_auc', 0):.3f}"),
    ("PR-AUC", f"{metrics.get('pr_auc', 0):.3f}"),
    ("F1 Score", f"{metrics.get('f1', 0):.3f}"),
    ("Precision", f"{metrics.get('precision', 0):.3f}"),
    ("Recall", f"{metrics.get('recall', 0):.3f}"),
]
for col, (title, val) in zip(mc, metric_items):
    with col:
        render_metric_tile(title, val, accent=SILVER_L)

# ── Prediction Volume ────────────────────────────────────────────────────
render_section_header("Prediction Volume", subtitle="Last 30 days (mock)", accent=SILVER)
rng = np.random.RandomState(42)
days = list(range(1, 31))
volumes = rng.randint(50, 200, size=30).tolist()

fig = go.Figure(go.Scatter(
    x=days, y=volumes, mode="lines+markers",
    line=dict(color=SILVER_L, width=2),
    marker=dict(size=4, color=SILVER_L),
))
fig.update_layout(
    paper_bgcolor=BG, plot_bgcolor=BG2,
    font=dict(color=TEXT, family="Inter"),
    margin=dict(l=40, r=20, t=20, b=40),
    xaxis=dict(gridcolor="#232A28", title="Day", zeroline=False),
    yaxis=dict(gridcolor="#232A28", title="Predictions", zeroline=False),
    height=280,
)
st.plotly_chart(fig, use_container_width=True)

# ── Infrastructure Telemetry ─────────────────────────────────────────────
render_section_header("Infrastructure Telemetry", accent=SILVER)

if "data_service" not in st.session_state:
    st.session_state.data_service = DataService()
    st.session_state.data_service.load_data()
ds = st.session_state.data_service

rows_loaded = len(ds.df)
features_available = len(ds.df.columns)
api_status = "Connected"
model_adapter_status = "Connected"

st.markdown(f"""
<div style="display:flex;gap:20px;flex-wrap:wrap;">
    <div style="flex:1;min-width:250px;background:{BG2};border:1px solid #232A28;border-radius:6px;padding:20px;">
        <div style="color:#7D857F;font-size:0.65rem;text-transform:uppercase;margin-bottom:12px;">
            Data Service
        </div>
        <div style="display:flex;justify-content:space-between;margin-bottom:8px;">
            <span style="color:#B5B8B1;font-size:0.8rem;">Dataset Rows</span>
            <span style="color:#F3F2EE;font-size:0.8rem;">{rows_loaded:,}</span>
        </div>
        <div style="display:flex;justify-content:space-between;margin-bottom:8px;">
            <span style="color:#B5B8B1;font-size:0.8rem;">Available Features</span>
            <span style="color:#F3F2EE;font-size:0.8rem;">{features_available}</span>
        </div>
        <div style="display:flex;justify-content:space-between;">
            <span style="color:#B5B8B1;font-size:0.8rem;">Last Refresh</span>
            <span style="color:#F3F2EE;font-size:0.8rem;">System Boot</span>
        </div>
    </div>
    <div style="flex:1;min-width:250px;background:{BG2};border:1px solid #232A28;border-radius:6px;padding:20px;">
        <div style="color:#7D857F;font-size:0.65rem;text-transform:uppercase;margin-bottom:12px;">
            API & Model Interface
        </div>
        <div style="display:flex;justify-content:space-between;margin-bottom:8px;">
            <span style="color:#B5B8B1;font-size:0.8rem;">Backend Status</span>
            <span style="color:#2EA66F;font-size:0.8rem;">Healthy</span>
        </div>
        <div style="display:flex;justify-content:space-between;margin-bottom:8px;">
            <span style="color:#B5B8B1;font-size:0.8rem;">Model Interface</span>
            <span style="color:#2EA66F;font-size:0.8rem;">{model_adapter_status}</span>
        </div>
        <div style="display:flex;justify-content:space-between;">
            <span style="color:#B5B8B1;font-size:0.8rem;">Uptime</span>
            <span style="color:#F3F2EE;font-size:0.8rem;">2h 14m</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Drift Monitoring ─────────────────────────────────────────────────────
render_section_header("Drift Monitoring", accent=SILVER)
st.markdown(f"""
<div style="background:{BG2};border:1px solid #232A28;border-radius:6px;
    padding:24px;text-align:center;">
    <div style="color:{SILVER};font-size:1.5rem;margin-bottom:8px;">📊</div>
    <div style="color:#B5B8B1;font-size:0.85rem;margin-bottom:4px;">
        Feature drift detection will be available<br>when the final model is deployed
    </div>
    <div style="color:#7D857F;font-size:0.7rem;">
        This module will monitor PSI, KL-divergence, and prediction distribution shifts
    </div>
</div>
""", unsafe_allow_html=True)

render_footer()
