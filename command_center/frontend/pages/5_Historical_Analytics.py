"""Page 5 — Historical Analytics.  Accent: Burgundy (#6E1F28)."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from frontend.components.theme import apply_theme, render_footer
from frontend.components.ui import render_section_header, render_leaderboard_table
from backend.services.data_service import DataService

apply_theme()

BURG = "#6E1F28"
BURG_L = "#8B2635"
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

def _layout(title="", h=300):
    return dict(
        paper_bgcolor=BG, plot_bgcolor=BG2, font=dict(color=TEXT, family="Inter"),
        margin=dict(l=40, r=20, t=40, b=40), title=dict(text=title, font=dict(size=13)),
        xaxis=dict(gridcolor="#232A28", zeroline=False),
        yaxis=dict(gridcolor="#232A28", zeroline=False),
        height=h,
    )

# ── Header ────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="display:flex;align-items:center;gap:10px;margin-bottom:4px;">
    <span style="color:{BURG_L};font-size:0.6rem;">&#9646;</span>
    <h1 style="margin:0;font-size:1.5rem;">Historical Analytics</h1>
</div>
<p style="color:#7D857F;font-size:0.8rem;margin-bottom:20px;">
    Deep-dive into incident patterns and correlations
</p>
""", unsafe_allow_html=True)

# ── Temporal Analysis ─────────────────────────────────────────────────────
render_section_header("Temporal Analysis", accent=BURG_L)
tab1, tab2, tab3 = st.tabs(["Hourly", "Daily", "Monthly"])

with tab1:
    hourly = df.groupby("hour").size()
    fig = go.Figure(go.Bar(x=hourly.index, y=hourly.values, marker_color=BURG_L))
    fig.update_layout(**_layout("Incidents by Hour"))
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    wd = df.groupby("weekday").size()
    fig = go.Figure(go.Bar(x=labels, y=[wd.get(i, 0) for i in range(7)], marker_color=BURG_L))
    fig.update_layout(**_layout("Incidents by Weekday"))
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    monthly = df.groupby("month").size()
    fig = go.Figure(go.Bar(x=monthly.index, y=monthly.values, marker_color=BURG))
    fig.update_layout(**_layout("Incidents by Month"))
    st.plotly_chart(fig, use_container_width=True)

# ── Event Cause Analysis ─────────────────────────────────────────────────
render_section_header("Event Cause Analysis", accent=BURG_L)
cause_closure = pd.crosstab(df["event_cause"], df["requires_road_closure"])
cause_closure.columns = ["No Closure", "Closure"]
cause_closure = cause_closure.sort_values("Closure", ascending=True).tail(12)

fig = go.Figure()
fig.add_trace(go.Bar(y=cause_closure.index, x=cause_closure["No Closure"],
                     name="No Closure", orientation="h", marker_color="#7D857F"))
fig.add_trace(go.Bar(y=cause_closure.index, x=cause_closure["Closure"],
                     name="Closure", orientation="h", marker_color=BURG_L))
fig.update_layout(**_layout("Event Causes by Closure Status", h=380))
fig.update_layout(barmode="stack", legend=dict(font=dict(color=TEXT)))
st.plotly_chart(fig, use_container_width=True)

# ── Corridor Statistics ───────────────────────────────────────────────────
render_section_header("Corridor Statistics", accent=BURG_L)
corr_stats = ds.get_corridor_stats()
display = corr_stats[["corridor", "total", "closures", "closure_rate"]].copy()
display.columns = ["Corridor", "Total", "Closures", "Rate %"]
render_leaderboard_table(display, accent=BURG_L)

# ── Police Station Activity ──────────────────────────────────────────────
render_section_header("Police Station Activity", accent=BURG_L)
station = df["police_station"].value_counts().head(15)
fig = go.Figure(go.Bar(
    y=station.index[::-1], x=station.values[::-1],
    orientation="h", marker_color=BURG,
))
fig.update_layout(**_layout("Incidents per Station", h=380))
st.plotly_chart(fig, use_container_width=True)

# ── Correlation Matrix ───────────────────────────────────────────────────
render_section_header("Feature Correlations", accent=BURG_L)
num_cols = ["hour", "weekday", "latitude", "longitude", "requires_road_closure"]
available = [c for c in num_cols if c in df.columns]
corr = df[available].corr()
fig = go.Figure(go.Heatmap(
    z=corr.values, x=corr.columns, y=corr.index,
    colorscale=[[0, "#2F5D9F"], [0.5, BG2], [1, BURG_L]],
    text=np.round(corr.values, 2), texttemplate="%{text}",
    textfont=dict(size=11, color=TEXT),
))
fig.update_layout(**_layout("Correlation Matrix", h=350))
st.plotly_chart(fig, use_container_width=True)

# ── Pivot Table ───────────────────────────────────────────────────────────
render_section_header("Pivot Table", accent=BURG_L)
pivot = pd.crosstab(df["event_type"], df["event_cause"])
st.dataframe(pivot, use_container_width=True)

# ── Export ────────────────────────────────────────────────────────────────
render_section_header("Export Data", accent=BURG_L)
c1, c2 = st.columns(2)
with c1:
    csv_data = ds.export_data("csv")
    st.download_button("Download CSV", csv_data, "astram_incidents.csv", "text/csv",
                       use_container_width=True)
with c2:
    json_data = ds.export_data("json")
    st.download_button("Download JSON", json_data, "astram_incidents.json", "application/json",
                       use_container_width=True)

render_footer()
