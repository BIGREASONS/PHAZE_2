"""Page 2 — Incident Command Center.  Accent: Jade (#1C7C54)."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import streamlit as st

st.set_page_config(page_title="GridSight AI Command Center", page_icon="🚦", layout="wide", initial_sidebar_state="expanded")
import pandas as pd
from datetime import datetime, timedelta

from frontend.components.theme import apply_theme, render_footer
from frontend.components.ui import render_section_header, render_status_badge
from frontend.components.maps import (
    create_incident_layer, create_heatmap_layer, create_cluster_layer,
    create_operations_map, get_risk_color,
)
from backend.services.data_service import DataService
from backend.services.model_adapter import get_model

apply_theme()

JADE = "#1C7C54"
JADE_L = "#2EA66F"

@st.cache_resource
def get_ds():
    ds = DataService()
    ds.load_data()
    return ds

ds = get_ds()
df = ds.df
model = get_model()

# ── Header ────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="display:flex;align-items:center;gap:10px;margin-bottom:4px;">
    <span style="color:{JADE};font-size:0.6rem;">&#9646;</span>
    <h1 style="margin:0;font-size:1.5rem;">Incident Command Center</h1>
</div>
<p style="color:#7D857F;font-size:0.8rem;margin-bottom:12px;">
    Traffic operations — live feed, map, intelligence
</p>
""", unsafe_allow_html=True)

# ── Three-Pane Layout ────────────────────────────────────────────────────
left, center, right = st.columns([3, 5, 2.5])

# ── LEFT: Incident Feed ──────────────────────────────────────────────────
with left:
    render_section_header("Incident Feed", accent=JADE)

    time_filter = st.radio(
        "Time Range", ["Last 24h", "Last 7d", "Last 30d", "All Time"],
        horizontal=True, key="icc_time", label_visibility="collapsed",
    )

    filters = {}
    if time_filter != "All Time":
        now = df["created_date"].max()
        if pd.notna(now):
            delta_map = {"Last 24h": 1, "Last 7d": 7, "Last 30d": 30}
            cutoff = now - timedelta(days=delta_map.get(time_filter, 9999))
            filters["date_start"] = str(cutoff)

    with st.expander("Filter", expanded=False):
        sel_type = st.multiselect("Type", sorted(df["event_type"].unique()), key="icc_ft")
        if sel_type:
            filters["event_type"] = sel_type
        sel_cause = st.multiselect("Cause", sorted(df["event_cause"].unique()), key="icc_fc")
        if sel_cause:
            filters["event_cause"] = sel_cause
        sel_corr = st.multiselect("Corridor", sorted(df["corridor"].unique()), key="icc_co")
        if sel_corr:
            filters["corridor"] = sel_corr

    filtered, total = ds.get_incidents(filters, page=1, page_size=100)

    st.markdown(f"<div style='color:#7D857F;font-size:0.7rem;margin-bottom:8px;'>"
                f"Showing {len(filtered)} of {total} incidents</div>", unsafe_allow_html=True)

    incident_ids = filtered["id"].tolist()
    selected_id = st.selectbox("Search Incident ID", incident_ids, key="icc_sel",
                               label_visibility="visible") if incident_ids else None

    # Render incident cards
    for _, row in filtered.head(20).iterrows():
        is_closure = row.get("requires_road_closure", 0)
        badge_color = "#B04A4A" if is_closure else JADE
        badge_label = "CLOSURE" if is_closure else "OPEN"
        st.markdown(f"""
        <div style="background:#121715;border:1px solid #232A28;border-radius:4px;
            padding:10px 12px;margin-bottom:6px;border-left:3px solid {badge_color};
            cursor:pointer;transition:background 150ms;">
            <div style="display:flex;justify-content:space-between;align-items:center;">
                <span style="color:#F3F2EE;font-size:0.78rem;font-weight:500;">
                    {row.get('id','--')[:12]}
                </span>
                {render_status_badge(badge_label, badge_color)}
            </div>
            <div style="color:#B5B8B1;font-size:0.7rem;margin-top:4px;">
                {row.get('event_cause','--')} &middot; {row.get('corridor','--')}
            </div>
            <div style="color:#7D857F;font-size:0.65rem;margin-top:2px;">
                {str(row.get('created_date',''))[:16]}
            </div>
        </div>
        """, unsafe_allow_html=True)

# ── CENTER: Operations Map ───────────────────────────────────────────────
with center:
    render_section_header("Operations Map", accent=JADE)

    mc1, mc2, mc3 = st.columns(3)
    show_incidents = mc1.checkbox("Incidents", True, key="icc_mi")
    show_heatmap = mc2.checkbox("Heatmap", False, key="icc_mh")
    show_clusters = mc3.checkbox("Clusters", False, key="icc_mc")

    map_data = ds.get_map_data(filters)
    if not map_data.empty:
        map_data["color"] = map_data["requires_road_closure"].apply(
            lambda x: get_risk_color(0.85 if x else 0.15)
        )
        layers = []
        if show_incidents:
            layers.append(create_incident_layer(map_data))
        if show_heatmap:
            layers.append(create_heatmap_layer(map_data))
        if show_clusters:
            layers.append(create_cluster_layer(map_data))

        if layers:
            deck = create_operations_map(layers, height=520)
            st.pydeck_chart(deck)
        else:
            st.info("Toggle a layer to display the map.")
    else:
        st.info("No geo data available for current filters.")

# ── RIGHT: Intelligence Panel ────────────────────────────────────────────
with right:
    render_section_header("Intelligence", accent=JADE)

    if selected_id:
        incident = df[df["id"] == selected_id]
        if not incident.empty:
            row = incident.iloc[0]
            pred = model.predict({
                "event_type": row.get("event_type", ""),
                "event_cause": row.get("event_cause", ""),
                "corridor": row.get("corridor", ""),
                "police_station": row.get("police_station", ""),
                "zone": row.get("zone", ""),
                "latitude": row.get("latitude", 0),
                "longitude": row.get("longitude", 0),
                "hour": row.get("hour", 0),
                "weekday": row.get("weekday", 0),
            })

            sev_color = {"LOW": JADE, "MEDIUM": "#B8833B", "HIGH": "#D08C4A", "CRITICAL": "#B04A4A"}

            st.markdown(f"""
            <div style="background:#121715;border:1px solid #232A28;border-radius:6px;padding:16px;margin-bottom:12px;">
                <div style="color:#7D857F;font-size:0.6rem;text-transform:uppercase;letter-spacing:0.08em;">
                    Incident Detail
                </div>
                <div style="color:#F3F2EE;font-size:0.95rem;font-weight:600;margin-top:6px;">
                    {row.get('id','--')}
                </div>
                <div style="margin-top:12px;">
                    <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
                        <span style="color:#7D857F;font-size:0.75rem;">Type</span>
                        <span style="color:#B5B8B1;font-size:0.75rem;">{row.get('event_type','--')}</span>
                    </div>
                    <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
                        <span style="color:#7D857F;font-size:0.75rem;">Cause</span>
                        <span style="color:#B5B8B1;font-size:0.75rem;">{row.get('event_cause','--')}</span>
                    </div>
                    <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
                        <span style="color:#7D857F;font-size:0.75rem;">Corridor</span>
                        <span style="color:#B5B8B1;font-size:0.75rem;">{row.get('corridor','--')}</span>
                    </div>
                    <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
                        <span style="color:#7D857F;font-size:0.75rem;">Station</span>
                        <span style="color:#B5B8B1;font-size:0.75rem;">{row.get('police_station','--')}</span>
                    </div>
                    <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
                        <span style="color:#7D857F;font-size:0.75rem;">Status</span>
                        <span style="color:#B5B8B1;font-size:0.75rem;">{row.get('status','--')}</span>
                    </div>
                </div>
            </div>

            <div style="background:#121715;border:1px solid #232A28;border-radius:6px;padding:16px;">
                <div style="color:#7D857F;font-size:0.6rem;text-transform:uppercase;letter-spacing:0.08em;">
                    Risk Assessment
                </div>
                <div style="color:{sev_color.get(pred['severity'], JADE)};font-size:2rem;
                    font-weight:700;font-family:'Inter Tight',sans-serif;margin-top:6px;">
                    {pred['probability']:.1%}
                </div>
                <div style="margin-top:4px;">
                    {render_status_badge(pred['severity'], sev_color.get(pred['severity'], JADE))}
                </div>
                <div style="color:#7D857F;font-size:0.7rem;margin-top:10px;">
                    {pred['recommended_action']}
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background:#121715;border:1px solid #232A28;border-radius:6px;
            padding:32px;text-align:center;">
            <div style="color:#7D857F;font-size:0.85rem;">
                Select an incident from the feed<br>to view intelligence
            </div>
        </div>
        """, unsafe_allow_html=True)

render_footer()
