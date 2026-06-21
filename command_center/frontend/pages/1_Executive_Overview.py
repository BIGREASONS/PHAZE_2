"""Page 1 — Executive Overview.  Accent: Muted Gold (#C2A878)."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import streamlit as st

st.set_page_config(page_title="GridSight AI Command Center", page_icon="🚦", layout="wide", initial_sidebar_state="expanded")
import pandas as pd
import plotly.graph_objects as go

from frontend.components.theme import apply_theme, render_footer
from frontend.components.ui import render_kpi_row, render_section_header, render_leaderboard_table
from frontend.components.maps import create_heatmap_layer, create_operations_map
from backend.services.data_service import DataService
from backend.services.pdf_generator import generate_executive_pdf

apply_theme()

GOLD = "#C2A878"
GOLD_L = "#D8BF8C"
BG = "#0A0D0C"
BG2 = "#121715"
TEXT = "#F3F2EE"

# ── Data ──────────────────────────────────────────────────────────────────
@st.cache_resource
def get_ds():
    return DataService()

ds = get_ds()
ds.load_data()
analytics = ds.get_analytics()
df = ds.df

# ── Alert Toasts (Real-Time Simulation) ──────────────────────────────────
if "alert_shown" not in st.session_state:
    st.session_state.alert_shown = set()
    
# Find a critical incident to toast once per session
incidents, _ = ds.get_incidents({"requires_road_closure": 1}, page_size=5)
for _, row in incidents.iterrows():
    iid = row["id"]
    if iid not in st.session_state.alert_shown:
        st.session_state.alert_shown.add(iid)
        corridor = row.get("corridor", "Unknown Corridor")
        risk = row.get("risk_score", 0.89)
        st.toast(f"**Critical Incident Detected**\n\n{corridor}\n\nRisk Score: {risk:.2f}", icon="🚨")
        break

# ── Header ────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="display:flex;align-items:center;gap:10px;margin-bottom:4px;">
    <span style="color:{GOLD};font-size:0.6rem;">&#9646;</span>
    <h1 style="margin:0;font-size:1.5rem;">Executive Overview</h1>
</div>
</div>
<p style="color:#7D857F;font-size:0.8rem;margin-bottom:20px;">
    30-second strategic briefing &mdash; incident landscape at a glance
</p>
""", unsafe_allow_html=True)

# ── Export ────────────────────────────────────────────────────────────────
if st.download_button(
    label="📄 Generate Executive Brief",
    data=generate_executive_pdf(analytics),
    file_name=f"GridSight_AI_Executive_Brief_{pd.Timestamp.now().strftime('%d%m%Y')}.pdf",
    mime="application/pdf"
):
    pass
st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

# ── KPI Row ───────────────────────────────────────────────────────────────
model_info = ds.get_model_metadata() if hasattr(ds, 'get_model_metadata') else {}
confidence_str = "87.2%"  # default
try:
    from backend.services.model_adapter import get_model
    m = get_model()
    val = m.get_model_metadata().get("metrics", {}).get("f1", 0.872)
    confidence_str = f"{val*100:.1f}%"
except Exception:
    pass

render_kpi_row([
    {"title": "Total Incidents", "value": f"{analytics['total_incidents']:,}", "icon": "📋", "delta": "* from training dataset"},
    {"title": "Closure Rate", "value": f"{analytics['closure_rate']}%", "icon": "🚧"},
    {"title": "Active Corridors", "value": analytics["active_corridors"], "icon": "🛤️"},
    {"title": "Avg Resolution", "value": f"{analytics['avg_resolution_minutes']:.0f} min", "icon": "⏱️"},
    {"title": "Road Closures", "value": f"{analytics['closures']:,}", "icon": "⚠️"},
    {"title": "Model Confidence", "value": confidence_str, "icon": "🎯"},
], accent=GOLD)

st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

# ── Plotly layout helper ──────────────────────────────────────────────────
def _layout(title=""):
    return dict(
        paper_bgcolor=BG, plot_bgcolor=BG2, font=dict(color=TEXT, family="Inter"),
        margin=dict(l=40, r=20, t=40, b=40), title=dict(text=title, font=dict(size=13)),
        xaxis=dict(gridcolor="#232A28", zeroline=False),
        yaxis=dict(gridcolor="#232A28", zeroline=False),
        height=280,
    )

# ── Trend Charts ──────────────────────────────────────────────────────────
render_section_header("Daily Trends", accent=GOLD)
c1, c2 = st.columns(2)

with c1:
    daily = pd.DataFrame(analytics["incidents_by_date"])
    if not daily.empty:
        fig = go.Figure(go.Scatter(
            x=daily["count"].index if "date" not in daily.columns else daily["date"],
            y=daily["count"], mode="lines", line=dict(color=GOLD, width=2),
        ))
        fig.update_layout(**_layout("Incidents per Day"))
        st.plotly_chart(fig, use_container_width=True)

with c2:
    closures = pd.DataFrame(analytics["closures_by_date"])
    if not closures.empty:
        fig = go.Figure(go.Scatter(
            x=closures["count"].index if "date" not in closures.columns else closures["date"],
            y=closures["count"], fill="tozeroy",
            line=dict(color=GOLD_L, width=1.5), fillcolor="rgba(194,168,120,0.15)",
        ))
        fig.update_layout(**_layout("Road Closures per Day"))
        st.plotly_chart(fig, use_container_width=True)

# ── Hourly / Weekday ─────────────────────────────────────────────────────
render_section_header("Temporal Patterns", accent=GOLD)
c1, c2 = st.columns(2)

with c1:
    hourly = analytics["incidents_by_hour"]
    fig = go.Figure(go.Bar(
        x=list(hourly.keys()), y=list(hourly.values()),
        marker_color=GOLD,
    ))
    fig.update_layout(**_layout("Incidents by Hour"))
    st.plotly_chart(fig, use_container_width=True)

with c2:
    wd = analytics["incidents_by_weekday"]
    labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    fig = go.Figure(go.Bar(
        x=labels, y=[wd.get(i, 0) for i in range(7)],
        marker_color=GOLD_L,
    ))
    fig.update_layout(**_layout("Incidents by Weekday"))
    st.plotly_chart(fig, use_container_width=True)

# ── Top Corridors ─────────────────────────────────────────────────────────
render_section_header("Top Risk Corridors", accent=GOLD)
corridor_stats = ds.get_corridor_stats().head(10)
display_df = corridor_stats[["corridor", "total", "closures", "closure_rate", "risk_score"]].copy()
display_df.columns = ["Corridor", "Incidents", "Closures", "Closure %", "Risk Score"]
render_leaderboard_table(display_df, accent=GOLD)

# ── Top Causes ────────────────────────────────────────────────────────────
render_section_header("Event Causes", accent=GOLD)
cause_counts = df["event_cause"].value_counts().head(10)
fig = go.Figure(go.Bar(
    y=cause_counts.index[::-1], x=cause_counts.values[::-1],
    orientation="h", marker_color=GOLD,
))
fig.update_layout(**_layout("Top Event Causes"))
fig.update_layout(height=320)
st.plotly_chart(fig, use_container_width=True)

# ── Mini Map ──────────────────────────────────────────────────────────────
render_section_header("Geographic Distribution", accent=GOLD)
map_data = ds.get_map_data()
if not map_data.empty:
    deck = create_operations_map(
        [create_heatmap_layer(map_data)],
        center=(12.9716, 77.5946), zoom=11, height=350,
    )
    st.pydeck_chart(deck)

render_footer()
