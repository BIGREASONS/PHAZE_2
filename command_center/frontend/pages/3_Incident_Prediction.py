"""Page 3 — Incident Prediction.  Accent: Deep Sapphire (#2F5D9F)."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import streamlit as st

st.set_page_config(page_title="GridSight AI Command Center", page_icon="🚦", layout="wide", initial_sidebar_state="expanded")
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, date, time

from frontend.components.theme import apply_theme, render_footer
from frontend.components.ui import render_section_header, render_status_badge
from backend.services.data_service import DataService
from backend.services.model_adapter import get_model

apply_theme()

SAPPHIRE = "#2F5D9F"
SAPPHIRE_L = "#4C7CC0"
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

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

# ── Header ────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="display:flex;align-items:center;gap:10px;margin-bottom:4px;">
    <span style="color:{SAPPHIRE};font-size:0.6rem;">&#9646;</span>
    <h1 style="margin:0;font-size:1.5rem;">Incident Prediction</h1>
</div>
<p style="color:#7D857F;font-size:0.8rem;margin-bottom:20px;">
    Predict road closure probability for a hypothetical incident
</p>
""", unsafe_allow_html=True)

# ── Input Form ────────────────────────────────────────────────────────────
render_section_header("Input Parameters <span style='color:#B04A4A'>*</span>", accent=SAPPHIRE)
c1, c2 = st.columns(2)

with c1:
    event_type = st.selectbox("Event Type", sorted(df["event_type"].unique()), key="p_et")
    event_cause = st.selectbox("Event Cause", sorted(df["event_cause"].unique()), key="p_ec")
    veh_type = st.selectbox("Vehicle Type", sorted(df["veh_type"].unique()), key="p_vt")
    corridor = st.selectbox("Corridor", sorted(df["corridor"].unique()), key="p_co")
    police_station = st.selectbox("Police Station", sorted(df["police_station"].unique()), key="p_ps")
    zone = st.selectbox("Zone", sorted(df["zone"].dropna().unique()), key="p_zo")

with c2:
    lat = st.number_input("Latitude", value=12.9716, format="%.6f", key="p_lat")
    lon = st.number_input("Longitude", value=77.5946, format="%.6f", key="p_lon")
    d = st.date_input("Date", value=date.today(), key="p_date", format="DD/MM/YYYY")
    t = st.time_input("Time", value=time(12, 0), key="p_time")

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    inference_mode_raw = st.radio("Inference Mode", ["Production (Recommended)", "Research (Includes TabPFN)"], key="p_mode")
    mode_str = "Production" if "Production" in inference_mode_raw else "Research"
    
    if mode_str == "Research" and not st.session_state.get("research_mode_confirmed", False):
        st.warning("Research Mode includes TabPFN foundation models. The first prediction may require significant model loading time. Subsequent predictions will be substantially faster. Continue?")
        rc1, rc2 = st.columns(2)
        if rc1.button("Enable Research Mode"):
            st.session_state.research_mode_confirmed = True
            st.rerun()
        if rc2.button("Cancel"):
            st.session_state.p_mode = "Production (Recommended)"
            st.rerun()
        st.stop()

    latency_str = "< 1 second" if mode_str == "Production" else "30s–240s"
    
    st.markdown(f"""
    <div style="background:{BG2};border:1px solid #232A28;border-radius:6px;padding:12px;margin-bottom:16px;">
        <div style="color:#7D857F;font-size:0.75rem;margin-bottom:4px;">Current Mode: <span style="color:{TEXT};font-weight:600;">{mode_str}</span></div>
        <div style="color:#7D857F;font-size:0.75rem;">Expected Latency: <span style="color:{SAPPHIRE_L};">{latency_str}</span></div>
    </div>
    """, unsafe_allow_html=True)
    
    predict_clicked = st.button("Predict Closure Risk", type="primary", use_container_width=True)

# ── Prediction Output ────────────────────────────────────────────────────
if predict_clicked:
    if mode_str == "Research":
        with st.status("Loading Research Components", expanded=True) as status:
            st.write("✓ Loading Production Ensemble")
            st.write("● Initializing TabPFN")
            st.write("○ Loading Calibration Assets")
            st.write("○ Preparing Inference Engine")
            st.write("*This may take several minutes depending on available hardware.*")
            
            res = model.ensure_tabpfn_loaded()
            if res.get("status") == "unsupported":
                status.update(label="Research Mode Unavailable", state="error", expanded=True)
                st.error(res.get("reason", "Research Mode unavailable on current deployment tier. Production Mode remains fully functional."))
                st.info("For full TabPFN experimentation, run locally.")
                st.stop()
            elif res.get("status") == "error":
                status.update(label="Error loading TabPFN", state="error", expanded=True)
                st.error(f"Error loading TabPFN: {res.get('reason')}")
                st.stop()
            else:
                status.update(label="Research Models Ready", state="complete", expanded=False)

    features = {
        "event_type": event_type, "event_cause": event_cause,
        "veh_type": veh_type, "corridor": corridor,
        "police_station": police_station, "zone": zone,
        "latitude": lat, "longitude": lon,
        "hour": t.hour, "weekday": d.weekday(),
        "inference_mode": mode_str
    }
    result = model.predict(features)

    # Store in history
    st.session_state.prediction_history.append({
        "time": datetime.now().strftime("%H:%M:%S"),
        "corridor": corridor, "cause": event_cause,
        "probability": result["probability"],
        "severity": result["severity"],
    })

    sev_colors = {"LOW": "#2EA66F", "MEDIUM": "#B8833B", "HIGH": "#D08C4A", "CRITICAL": "#B04A4A"}
    sev_c = sev_colors.get(result["severity"], SAPPHIRE)

    render_section_header("Prediction Result", accent=SAPPHIRE)
    r1, r2, r3, r4 = st.columns(4)

    with r1:
        st.markdown(f"""
        <div style="background:{BG2};border:1px solid #232A28;border-radius:6px;
            padding:20px;text-align:center;">
            <div style="color:#7D857F;font-size:0.65rem;text-transform:uppercase;">Probability</div>
            <div style="color:{sev_c};font-size:2.8rem;font-weight:700;
                font-family:'Inter Tight',sans-serif;">{result['probability']:.1%}</div>
        </div>""", unsafe_allow_html=True)
    with r2:
        st.markdown(f"""
        <div style="background:{BG2};border:1px solid #232A28;border-radius:6px;
            padding:20px;text-align:center;">
            <div style="color:#7D857F;font-size:0.65rem;text-transform:uppercase;">Severity</div>
            <div style="margin-top:12px;">{render_status_badge(result['severity'], sev_c)}</div>
        </div>""", unsafe_allow_html=True)
    with r3:
        st.markdown(f"""
        <div style="background:{BG2};border:1px solid #232A28;border-radius:6px;
            padding:20px;text-align:center;">
            <div style="color:#7D857F;font-size:0.65rem;text-transform:uppercase;">Confidence</div>
            <div style="color:{SAPPHIRE_L};font-size:2rem;font-weight:700;
                font-family:'Inter Tight',sans-serif;margin-top:4px;">{result['confidence']:.1%}</div>
        </div>""", unsafe_allow_html=True)
    with r4:
        st.markdown(f"""
        <div style="background:{BG2};border:1px solid #232A28;border-radius:6px;
            padding:20px;">
            <div style="color:#7D857F;font-size:0.65rem;text-transform:uppercase;">Action</div>
            <div style="color:#B5B8B1;font-size:0.78rem;margin-top:8px;">
                {result['recommended_action']}
            </div>
        </div>""", unsafe_allow_html=True)

    # ── Explanation ───────────────────────────────────────────────────────
    render_section_header("Feature Contributions", accent=SAPPHIRE)
    explanation = model.explain(features)
    contribs = explanation["feature_contributions"]
    sorted_feats = sorted(contribs.items(), key=lambda x: x[1])
    feat_names = [f[0] for f in sorted_feats]
    feat_vals = [f[1] for f in sorted_feats]
    colors = [SAPPHIRE if v >= 0 else "#B04A4A" for v in feat_vals]

    fig = go.Figure(go.Bar(
        y=feat_names, x=feat_vals, orientation="h",
        marker_color=colors,
    ))
    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG2,
        font=dict(color=TEXT, family="Inter"), height=300,
        margin=dict(l=100, r=20, t=20, b=20),
        xaxis=dict(gridcolor="#232A28", zeroline=True, zerolinecolor="#232A28"),
        yaxis=dict(gridcolor="#232A28"),
    )
    st.plotly_chart(fig, use_container_width=True)

    _contrib_note = (
        "Fallback model active — illustrative contributions until production artifacts load"
        if is_placeholder else
        "Contributions via single-feature ablation over the full 7-model calibrated ensemble"
    )
    st.markdown(f"""
    <div style="color:#7D857F;font-size:0.7rem;padding:8px;background:{BG2};
        border:1px solid #232A28;border-radius:4px;margin-top:8px;">
        {_contrib_note}
    </div>
    """, unsafe_allow_html=True)

# ── Prediction History ───────────────────────────────────────────────────
if st.session_state.prediction_history:
    render_section_header("Prediction History", accent=SAPPHIRE)
    hist_df = pd.DataFrame(st.session_state.prediction_history)
    st.dataframe(hist_df, use_container_width=True, hide_index=True)

render_footer()
