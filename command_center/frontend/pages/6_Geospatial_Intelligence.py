"""Page 6 — Geospatial Intelligence.  Accent: Teal (#0F766E)."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import streamlit as st

st.set_page_config(page_title="ASTraM Command Center", page_icon="🚦", layout="wide", initial_sidebar_state="expanded")
import pandas as pd
import numpy as np

from frontend.components.theme import apply_theme, render_footer
from frontend.components.ui import render_section_header, render_leaderboard_table
from frontend.components.maps import (
    create_heatmap_layer, create_cluster_layer, create_incident_layer,
    create_operations_map, get_risk_color,
)
from backend.services.data_service import DataService

apply_theme()

TEAL = "#0F766E"
TEAL_L = "#14B8A6"
BG2 = "#121715"

@st.cache_resource
def get_ds():
    ds = DataService()
    ds.load_data()
    return ds

ds = get_ds()
df = ds.df

# ── Header ────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="display:flex;align-items:center;gap:10px;margin-bottom:4px;">
    <span style="color:{TEAL_L};font-size:0.6rem;">&#9646;</span>
    <h1 style="margin:0;font-size:1.5rem;">Geospatial Intelligence</h1>
</div>
<p style="color:#7D857F;font-size:0.8rem;margin-bottom:20px;">
    Spatial analysis, hotspots, and corridor intelligence
</p>
""", unsafe_allow_html=True)

map_data = ds.get_map_data()

# ── Interactive Time Scrubbing ──────────────────────────────────────────
st.markdown('<div style="margin-bottom:5px;color:#B5B8B1;font-size:0.85rem;font-weight:600;letter-spacing:0.05em;">TEMPORAL FILTER (TIME MACHINE)</div>', unsafe_allow_html=True)
hours = [f"{str(i).zfill(2)}:00" for i in range(24)]
selected_time = st.select_slider("Time", options=hours, value="12:00", label_visibility="collapsed")
selected_hour = int(selected_time.split(":")[0])

filtered_map_data = map_data[map_data["hour"] == selected_hour]

st.markdown(f"<div style='color:#7D857F;font-size:0.75rem;margin-bottom:20px;'>Showing {len(filtered_map_data)} incidents occurring during {selected_time}</div>", unsafe_allow_html=True)

# ── Hotspot Analysis ──────────────────────────────────────────────────────
render_section_header("Hotspot Analysis", subtitle="Hexagonal density clusters", accent=TEAL_L)
if not filtered_map_data.empty:
    deck = create_operations_map([create_cluster_layer(filtered_map_data)], height=450)
    st.pydeck_chart(deck)

# ── Density Map ───────────────────────────────────────────────────────────
render_section_header("Density Heatmap", accent=TEAL_L)
if not filtered_map_data.empty:
    deck = create_operations_map([create_heatmap_layer(filtered_map_data)], height=400)
    st.pydeck_chart(deck)

# ── Corridor Intelligence ────────────────────────────────────────────────
render_section_header("Corridor Intelligence", accent=TEAL_L)
corridors = sorted(df["corridor"].dropna().unique().tolist())
sel_corr = st.selectbox("Select Corridor", corridors, key="geo_corr")

if sel_corr:
    corr_df = df[df["corridor"] == sel_corr]
    corr_map = corr_df.dropna(subset=["latitude", "longitude"]).copy()
    corr_map["color"] = corr_map["requires_road_closure"].apply(
        lambda x: get_risk_color(0.85 if x else 0.15)
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div style="background:{BG2};border:1px solid #232A28;border-radius:6px;padding:16px;text-align:center;">
            <div style="color:#7D857F;font-size:0.65rem;text-transform:uppercase;">Incidents</div>
            <div style="color:{TEAL_L};font-size:1.8rem;font-weight:700;">{len(corr_df)}</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        rate = corr_df["requires_road_closure"].mean() * 100 if len(corr_df) else 0
        st.markdown(f"""
        <div style="background:{BG2};border:1px solid #232A28;border-radius:6px;padding:16px;text-align:center;">
            <div style="color:#7D857F;font-size:0.65rem;text-transform:uppercase;">Closure Rate</div>
            <div style="color:{TEAL_L};font-size:1.8rem;font-weight:700;">{rate:.1f}%</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        stations = corr_df["police_station"].nunique()
        st.markdown(f"""
        <div style="background:{BG2};border:1px solid #232A28;border-radius:6px;padding:16px;text-align:center;">
            <div style="color:#7D857F;font-size:0.65rem;text-transform:uppercase;">Stations</div>
            <div style="color:{TEAL_L};font-size:1.8rem;font-weight:700;">{stations}</div>
        </div>""", unsafe_allow_html=True)

    if not corr_map.empty:
        center = (corr_map["latitude"].mean(), corr_map["longitude"].mean())
        deck = create_operations_map(
            [create_incident_layer(corr_map)], center=center, zoom=13, height=350,
        )
        st.pydeck_chart(deck)

# ── Radius Search ─────────────────────────────────────────────────────────
render_section_header("Radius Search", accent=TEAL_L)
rc1, rc2, rc3 = st.columns(3)
with rc1:
    r_lat = st.number_input("Latitude", value=12.9716, format="%.6f", key="geo_rlat")
with rc2:
    r_lon = st.number_input("Longitude", value=77.5946, format="%.6f", key="geo_rlon")
with rc3:
    radius = st.slider("Radius (km)", 0.5, 5.0, 2.0, 0.5, key="geo_rad")

if st.button("Search", key="geo_search"):
    nearby = ds.get_nearest_incidents(r_lat, r_lon, k=200)
    nearby = nearby[nearby["distance_km"] <= radius]
    st.markdown(f"<div style='color:#B5B8B1;font-size:0.8rem;margin:8px 0;'>"
                f"Found {len(nearby)} incidents within {radius} km</div>", unsafe_allow_html=True)

    if not nearby.empty:
        nearby_map = nearby.dropna(subset=["latitude", "longitude"]).copy()
        nearby_map["color"] = nearby_map["requires_road_closure"].apply(
            lambda x: get_risk_color(0.85 if x else 0.15)
        )
        deck = create_operations_map(
            [create_incident_layer(nearby_map)],
            center=(r_lat, r_lon), zoom=14, height=350,
        )
        st.pydeck_chart(deck)

# ── Geographic Summary ────────────────────────────────────────────────────
render_section_header("Zone Summary", accent=TEAL_L)
zone_stats = (
    df.groupby("zone")
    .agg(incidents=("id", "size"), closures=("requires_road_closure", "sum"))
    .assign(rate=lambda x: (x["closures"] / x["incidents"] * 100).round(1))
    .sort_values("incidents", ascending=False)
    .reset_index()
)
zone_stats.columns = ["Zone", "Incidents", "Closures", "Rate %"]
render_leaderboard_table(zone_stats, accent=TEAL_L)

render_footer()
