"""Page 8 — Competition Showcase.  Accent: Champagne (#D9C7A3)."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import streamlit as st

from frontend.components.theme import apply_theme, render_footer
from frontend.components.ui import render_section_header, render_status_badge
from frontend.components.maps import create_incident_layer, create_operations_map, get_risk_color
from backend.services.model_adapter import PlaceholderModel
from shared.constants import SCENARIOS, REPLAY_STATES
import pandas as pd

apply_theme()

CHAMP = "#D9C7A3"
CHAMP_L = "#E7DBC2"
BG2 = "#121715"

model = PlaceholderModel()

if "showcase_results" not in st.session_state:
    st.session_state.showcase_results = {}

# ── Header ────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="display:flex;align-items:center;gap:10px;margin-bottom:4px;">
    <span style="color:{CHAMP};font-size:0.6rem;">&#9646;</span>
    <h1 style="margin:0;font-size:1.5rem;">Competition Showcase</h1>
</div>
<p style="color:#7D857F;font-size:0.8rem;margin-bottom:20px;">
    One-click scenario demonstrations for judges &mdash; test the system in under 30 seconds
</p>
""", unsafe_allow_html=True)

# ── Scenario Cards ────────────────────────────────────────────────────────
render_section_header("Scenario Library", accent=CHAMP)

for row_start in range(0, len(SCENARIOS), 3):
    cols = st.columns(3)
    for i, col in enumerate(cols):
        idx = row_start + i
        if idx >= len(SCENARIOS):
            break
        sc = SCENARIOS[idx]
        with col:
            st.markdown(f"""
            <div style="background:{BG2};border:1px solid #232A28;border-radius:6px;
                padding:16px;min-height:120px;border-top:2px solid {CHAMP};">
                <div style="font-size:1.5rem;margin-bottom:4px;">{sc['icon']}</div>
                <div style="color:#F3F2EE;font-size:0.95rem;font-weight:600;">{sc['name']}</div>
                <div style="color:#7D857F;font-size:0.72rem;margin-top:4px;">{sc['desc']}</div>
                <div style="color:#B5B8B1;font-size:0.65rem;margin-top:6px;">
                    {sc['features']['corridor']} &middot; {sc['features']['event_cause']}
                </div>
            </div>
            """, unsafe_allow_html=True)

            if st.button(f"Run Scenario", key=f"sc_{idx}", use_container_width=True):
                result = model.predict(sc["features"])
                st.session_state.showcase_results[sc["name"]] = result

            # Show result if exists
            if sc["name"] in st.session_state.showcase_results:
                res = st.session_state.showcase_results[sc["name"]]
                sev_colors = {"LOW": "#2EA66F", "MEDIUM": "#B8833B",
                              "HIGH": "#D08C4A", "CRITICAL": "#B04A4A"}
                sev_c = sev_colors.get(res["severity"], CHAMP)
                st.markdown(f"""
                <div style="background:#0A0D0C;border:1px solid {sev_c}40;border-radius:4px;
                    padding:10px;margin-top:4px;">
                    <div style="display:flex;justify-content:space-between;align-items:center;">
                        <span style="color:{sev_c};font-size:1.4rem;font-weight:700;
                            font-family:'Inter Tight',sans-serif;">{res['probability']:.1%}</span>
                        {render_status_badge(res['severity'], sev_c)}
                    </div>
                    <div style="color:#7D857F;font-size:0.65rem;margin-top:4px;">
                        Confidence: {res['confidence']:.1%}
                    </div>
                </div>
                """, unsafe_allow_html=True)

# ── Incident Replay System ───────────────────────────────────────────────
st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
render_section_header("Incident Replay System", subtitle="Control-room simulation", accent=CHAMP)

st.markdown(f"""
<div style="color:{CHAMP};font-size:0.7rem;background:{BG2};border:1px solid #232A28;
    border-radius:4px;padding:8px 12px;margin-bottom:12px;">
    Simulate how an incident evolves from initial report to resolution
</div>
""", unsafe_allow_html=True)

time_point = st.select_slider(
    "Timeline",
    options=[0, 5, 10, 20, 30],
    format_func=lambda x: f"T+{x} min",
    value=0,
    key="replay_time",
)

state = REPLAY_STATES.get(time_point, REPLAY_STATES[0])

# Use a scenario for the replay
replay_sc = SCENARIOS[0]  # Major Accident
replay_lat = replay_sc["features"]["latitude"]
replay_lon = replay_sc["features"]["longitude"]

rc1, rc2 = st.columns([3, 2])

with rc1:
    # Map showing the incident
    map_df = pd.DataFrame([{
        "latitude": replay_lat,
        "longitude": replay_lon,
        "color": get_risk_color(0.85 if time_point < 30 else 0.15),
    }])
    deck = create_operations_map(
        [create_incident_layer(map_df)],
        center=(replay_lat, replay_lon), zoom=14, height=350,
    )
    st.pydeck_chart(deck)

with rc2:
    status_colors = {
        "Reported": "#B04A4A",
        "Acknowledged": "#B8833B",
        "In Progress": "#D08C4A",
        "Mitigation": "#2F5D9F",
        "Resolved": "#2EA66F",
    }
    s_color = status_colors.get(state["status"], CHAMP)

    st.markdown(f"""
    <div style="background:{BG2};border:1px solid #232A28;border-radius:6px;padding:20px;">
        <div style="color:#7D857F;font-size:0.6rem;text-transform:uppercase;
            letter-spacing:0.08em;margin-bottom:12px;">Incident State</div>

        <div style="margin-bottom:16px;">
            <div style="color:#7D857F;font-size:0.7rem;">Status</div>
            <div style="margin-top:4px;">
                {render_status_badge(state['status'], s_color)}
            </div>
        </div>

        <div style="margin-bottom:16px;">
            <div style="color:#7D857F;font-size:0.7rem;">Elapsed</div>
            <div style="color:#F3F2EE;font-size:1.2rem;font-weight:600;
                font-family:'Inter Tight',sans-serif;">T+{time_point} min</div>
        </div>

        <div style="margin-bottom:16px;">
            <div style="color:#7D857F;font-size:0.7rem;">Resources Deployed</div>
            <div style="color:{CHAMP};font-size:1.6rem;font-weight:700;
                font-family:'Inter Tight',sans-serif;">{state['resources']}</div>
        </div>

        <div>
            <div style="color:#7D857F;font-size:0.7rem;">Current Action</div>
            <div style="color:#B5B8B1;font-size:0.8rem;margin-top:4px;">
                {state['action']}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Timeline dots
    st.markdown("<div style='margin-top:12px;'>", unsafe_allow_html=True)
    for tp in [0, 5, 10, 20, 30]:
        is_active = tp <= time_point
        dot_color = "#2EA66F" if is_active else "#232A28"
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">
            <div style="width:8px;height:8px;border-radius:50%;background:{dot_color};"></div>
            <span style="color:{'#B5B8B1' if is_active else '#7D857F'};font-size:0.7rem;">
                T+{tp} — {REPLAY_STATES[tp]['status']}
            </span>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ── Quick Stats ──────────────────────────────────────────────────────────
if st.session_state.showcase_results:
    render_section_header("Demo Summary", accent=CHAMP)
    summary_data = []
    for name, res in st.session_state.showcase_results.items():
        summary_data.append({
            "Scenario": name,
            "Probability": f"{res['probability']:.1%}",
            "Severity": res["severity"],
            "Confidence": f"{res['confidence']:.1%}",
        })
    st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)

render_footer()
