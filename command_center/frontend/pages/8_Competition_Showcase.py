"""Page 8 — Competition Showcase.  Accent: Champagne (#D9C7A3)."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import streamlit as st

st.set_page_config(page_title="ASTraM Command Center", page_icon="🚦", layout="wide", initial_sidebar_state="expanded")

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
    Emergency Response Simulator &mdash; observe system telemetry and intelligence as an incident unfolds
</p>
""", unsafe_allow_html=True)

import time

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

            if st.button(f"Deploy Simulator", key=f"sc_{idx}", use_container_width=True):
                st.session_state.active_scenario = sc
                st.session_state.run_sim = True
                # st.rerun() # Let it fall through to render the simulator below

# ── Emergency Response Simulator ─────────────────────────────────────────
st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
render_section_header("Emergency Response Simulator", subtitle="Live telemetry feed", accent=CHAMP)

if "active_scenario" in st.session_state and st.session_state.get("run_sim"):
    sc = st.session_state.active_scenario
    
    # Placeholders for progressive reveal
    map_ph = st.empty()
    risk_ph = st.empty()
    pred_ph = st.empty()
    timeline_ph = st.empty()
    
    # 1. Map Fly-To Animation
    map_df = pd.DataFrame([{
        "latitude": sc["features"]["latitude"],
        "longitude": sc["features"]["longitude"],
        "color": get_risk_color(0.85),
    }])
    deck = create_operations_map([create_incident_layer(map_df)], center=(sc["features"]["latitude"], sc["features"]["longitude"]), zoom=14, height=350)
    map_ph.pydeck_chart(deck)
    time.sleep(0.8)
    
    # 2. Risk Panel
    risk_ph.markdown(f"""
    <div style="background:{BG2};border:1px solid #B04A4A;border-left:4px solid #B04A4A;border-radius:6px;padding:16px;margin-bottom:16px;">
        <div style="color:#B04A4A;font-weight:700;font-size:1.1rem;margin-bottom:4px;">🚨 INCIDENT DETECTED</div>
        <div style="color:#F3F2EE;font-size:0.9rem;">{sc['name']}</div>
        <div style="color:#B5B8B1;font-size:0.8rem;margin-top:4px;">Location: {sc['features']['corridor']}</div>
    </div>
    """, unsafe_allow_html=True)
    time.sleep(1.0)
    
    # 3. Prediction Appears
    res = model.predict(sc["features"])
    sev_colors = {"LOW": "#2EA66F", "MEDIUM": "#B8833B", "HIGH": "#D08C4A", "CRITICAL": "#B04A4A"}
    sev_c = sev_colors.get(res["severity"], CHAMP)
    pred_ph.markdown(f"""
    <div style="display:flex;gap:16px;margin-bottom:16px;">
        <div style="flex:1;background:#0A0D0C;border:1px solid {sev_c}40;border-radius:6px;padding:16px;">
            <div style="color:#7D857F;font-size:0.65rem;text-transform:uppercase;">AI Risk Prediction</div>
            <div style="color:{sev_c};font-size:1.8rem;font-weight:700;margin:4px 0;">{res['probability']:.1%}</div>
            {render_status_badge(res['severity'], sev_c)}
        </div>
        <div style="flex:1;background:#0A0D0C;border:1px solid #232A28;border-radius:6px;padding:16px;">
            <div style="color:#7D857F;font-size:0.65rem;text-transform:uppercase;">Key Risk Drivers</div>
            <div style="color:#B5B8B1;font-size:0.8rem;margin-top:8px;">1. {list(res['feature_contributions'].keys())[0]} (+{list(res['feature_contributions'].values())[0]:.2f})</div>
            <div style="color:#B5B8B1;font-size:0.8rem;margin-top:4px;">2. {list(res['feature_contributions'].keys())[1]} (+{list(res['feature_contributions'].values())[1]:.2f})</div>
            <div style="color:#B5B8B1;font-size:0.8rem;margin-top:4px;">3. {list(res['feature_contributions'].keys())[2]} (+{list(res['feature_contributions'].values())[2]:.2f})</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    time.sleep(1.2)
    
    # 4. Timeline Replay
    for tp in [0, 5, 10, 20, 30]:
        state = REPLAY_STATES.get(tp, REPLAY_STATES[0])
        status_colors = {"Reported": "#B04A4A", "Acknowledged": "#B8833B", "In Progress": "#D08C4A", "Mitigation": "#2F5D9F", "Resolved": "#2EA66F"}
        s_color = status_colors.get(state["status"], CHAMP)
        
        timeline_ph.markdown(f"""
        <div style="background:{BG2};border:1px solid #232A28;border-radius:6px;padding:20px;">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:16px;">
                <div style="color:#F3F2EE;font-size:1.2rem;font-weight:600;">Timeline: T+{tp} min</div>
                {render_status_badge(state['status'], s_color)}
            </div>
            <div style="display:flex;gap:32px;">
                <div>
                    <div style="color:#7D857F;font-size:0.7rem;">Resources Deployed</div>
                    <div style="color:{CHAMP};font-size:1.4rem;font-weight:700;">{state['resources']}</div>
                </div>
                <div>
                    <div style="color:#7D857F;font-size:0.7rem;">Current Action</div>
                    <div style="color:#B5B8B1;font-size:0.9rem;margin-top:4px;">{state['action']}</div>
                </div>
            </div>
            <div style="margin-top:16px;height:4px;background:#0A0D0C;border-radius:2px;overflow:hidden;">
                <div style="height:100%;width:{(tp/30)*100}%;background:{s_color};transition:width 0.5s ease;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        time.sleep(1.0 if tp < 30 else 0)
        
    st.session_state.run_sim = False

elif "active_scenario" not in st.session_state:
    st.markdown(f"""
    <div style="background:{BG2};border:1px solid #232A28;border-radius:6px;
        padding:32px;text-align:center;">
        <div style="color:#7D857F;font-size:0.85rem;">
            Select a scenario from the library above to deploy the simulator
        </div>
    </div>
    """, unsafe_allow_html=True)

render_footer()
