"""Page 4 — Explainability Center.  Accent: Copper (#B87333)."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import streamlit as st

st.set_page_config(page_title="GridSight AI Command Center", page_icon="🚦", layout="wide", initial_sidebar_state="expanded")
import plotly.graph_objects as go

from frontend.components.theme import apply_theme, render_footer
from frontend.components.ui import render_section_header, render_kpi_row
from backend.services.data_service import DataService
from backend.services.model_adapter import get_model

apply_theme()

COPPER = "#B87333"
COPPER_L = "#D08C4A"
BG = "#0A0D0C"
BG2 = "#121715"
TEXT = "#F3F2EE"

@st.cache_resource
def get_ds():
    ds = DataService()
    ds.load_data()
    return ds

ds = get_ds()
df = ds.df
model = get_model()
is_placeholder = "placeholder" in model.get_model_metadata()["name"].lower()

# ── Header ────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="display:flex;align-items:center;gap:10px;margin-bottom:4px;">
    <span style="color:{COPPER};font-size:0.6rem;">&#9646;</span>
    <h1 style="margin:0;font-size:1.5rem;">Explainability Center</h1>
</div>
<p style="color:#7D857F;font-size:0.8rem;margin-bottom:4px;">
    Model interpretability and feature analysis
</p>
""", unsafe_allow_html=True)

_explain_note = (
    "Fallback model active — values below are illustrative until production artifacts load"
    if is_placeholder else
    "Global importances from the frozen ensemble's tree members; local attribution by "
    "single-feature ablation over the full 7-model calibrated ensemble"
)
st.markdown(f"""
<div style="color:{COPPER};font-size:0.7rem;background:#121715;border:1px solid #232A28;
    border-radius:4px;padding:8px 12px;margin-bottom:20px;">
    {_explain_note}
</div>
""", unsafe_allow_html=True)

# ── Global Feature Importance ─────────────────────────────────────────────
render_section_header("Global Feature Importance", accent=COPPER)

importances = model.get_global_importances()
sorted_imp = sorted(importances.items(), key=lambda x: x[1])
fig = go.Figure(go.Bar(
    y=[i[0] for i in sorted_imp],
    x=[i[1] for i in sorted_imp],
    orientation="h", marker_color=COPPER,
))
fig.update_layout(
    paper_bgcolor=BG, plot_bgcolor=BG2,
    font=dict(color=TEXT, family="Inter"), height=320,
    margin=dict(l=120, r=20, t=20, b=20),
    xaxis=dict(gridcolor="#232A28", title="Importance"),
    yaxis=dict(gridcolor="#232A28"),
)
st.plotly_chart(fig, use_container_width=True)

# ── Local Explanation ─────────────────────────────────────────────────────
render_section_header("Local Explanation", subtitle="Select an incident", accent=COPPER)

sample_ids = df["id"].head(50).tolist()
selected_id = st.selectbox("Incident", sample_ids, key="exp_sel")

if selected_id:
    incident = df[df["id"] == selected_id].iloc[0]
    features = {
        "event_type": incident.get("event_type", ""),
        "event_cause": incident.get("event_cause", ""),
        "corridor": incident.get("corridor", ""),
        "police_station": incident.get("police_station", ""),
        "zone": incident.get("zone", ""),
        "latitude": incident.get("latitude", 0),
        "longitude": incident.get("longitude", 0),
        "hour": incident.get("hour", 0),
        "weekday": incident.get("weekday", 0),
    }
    explanation = model.explain(features)
    contribs = explanation["feature_contributions"]
    sorted_c = sorted(contribs.items(), key=lambda x: x[1])

    # Waterfall-style chart
    feat_names = [c[0] for c in sorted_c]
    feat_vals = [c[1] for c in sorted_c]
    colors = [COPPER if v >= 0 else "#2F5D9F" for v in feat_vals]

    fig = go.Figure(go.Bar(
        y=feat_names, x=feat_vals, orientation="h",
        marker_color=colors,
    ))
    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG2,
        font=dict(color=TEXT, family="Inter"), height=300,
        margin=dict(l=120, r=20, t=20, b=20),
        xaxis=dict(gridcolor="#232A28", zeroline=True, zerolinecolor="#232A28",
                   title="Contribution"),
        yaxis=dict(gridcolor="#232A28"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Drivers Cards ─────────────────────────────────────────────────────
    c1, c2 = st.columns(2)
    with c1:
        render_section_header("Top Positive Drivers", accent=COPPER)
        for d in explanation["top_positive_drivers"]:
            st.markdown(f"""
            <div style="background:{BG2};border:1px solid #232A28;border-radius:4px;
                padding:10px 12px;margin-bottom:6px;border-left:3px solid {COPPER};">
                <div style="color:#F3F2EE;font-size:0.82rem;font-weight:500;">{d['feature']}</div>
                <div style="color:{COPPER};font-size:0.75rem;">+{d['contribution']:.4f}</div>
            </div>
            """, unsafe_allow_html=True)

    with c2:
        render_section_header("Top Negative Drivers", accent="#2F5D9F")
        for d in explanation["top_negative_drivers"]:
            st.markdown(f"""
            <div style="background:{BG2};border:1px solid #232A28;border-radius:4px;
                padding:10px 12px;margin-bottom:6px;border-left:3px solid #2F5D9F;">
                <div style="color:#F3F2EE;font-size:0.82rem;font-weight:500;">{d['feature']}</div>
                <div style="color:#4C7CC0;font-size:0.75rem;">{d['contribution']:.4f}</div>
            </div>
            """, unsafe_allow_html=True)

# ── Confidence Distribution ───────────────────────────────────────────────
render_section_header("Confidence Distribution", subtitle="Illustrative distribution", accent=COPPER)
fig = go.Figure(go.Pie(
    labels=["High", "Medium", "Low", "Very Low"],
    values=[45, 30, 20, 5],
    hole=0.55,
    marker_colors=[COPPER, COPPER_L, "#8B949E", "#7D857F"],
    textfont=dict(color=TEXT),
))
fig.update_layout(
    paper_bgcolor=BG, font=dict(color=TEXT, family="Inter"),
    height=300, margin=dict(l=20, r=20, t=20, b=20),
    legend=dict(font=dict(color=TEXT)),
)
st.plotly_chart(fig, use_container_width=True)

render_footer()
