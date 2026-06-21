"""Page 7 — Model Monitoring.  Accent: Silver (#8B949E)."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import streamlit as st

st.set_page_config(page_title="GridSight AI Command Center", page_icon="🚦", layout="wide", initial_sidebar_state="expanded")
import numpy as np
import plotly.graph_objects as go

from frontend.components.theme import apply_theme, render_footer
from frontend.components.ui import render_section_header, render_kpi_row, render_metric_tile
from backend.services.model_adapter import get_model, process_uptime_seconds
from backend.services.data_service import DataService

apply_theme()

SILVER = "#8B949E"
SILVER_L = "#C5CCD4"
BG = "#0A0D0C"
BG2 = "#121715"
TEXT = "#F3F2EE"

model = get_model()
meta = model.get_model_metadata()
metrics = meta.get("metrics", {})
is_placeholder = "placeholder" in meta["name"].lower()


def _fmt_uptime(seconds: float) -> str:
    h, rem = divmod(int(max(0, seconds)), 3600)
    return f"{h}h {rem // 60:02d}m" if h else f"{rem // 60}m"


uptime_str = _fmt_uptime(process_uptime_seconds())

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
render_section_header(
    "Performance Metrics",
    subtitle="Placeholder values" if is_placeholder else "Rolling-origin out-of-fold",
    accent=SILVER,
)
if is_placeholder:
    st.markdown("""
    <div style="color:#B8833B;font-size:0.7rem;background:#121715;border:1px solid #232A28;
        border-radius:4px;padding:6px 10px;margin-bottom:12px;">
        Placeholder — production artifacts not loaded; values shown are the frozen ensemble's reference metrics
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

# ── Drift Monitoring ──────────────────────────────────────────────────────
render_section_header("Drift Monitoring <span style='color:#B04A4A;font-size:0.7rem;vertical-align:middle;margin-left:8px;padding:2px 6px;border:1px solid #B04A4A;border-radius:4px;'>DEMO DATA / SIMULATED</span>", accent=SILVER)

st.markdown("""
<div style="background:#121715;border:1px solid #232A28;border-radius:6px;padding:20px;margin-bottom:24px;">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">
        <span style="color:#B5B8B1;font-size:0.85rem;font-weight:600;">Overall Feature Drift (PSI)</span>
        <span style="color:#B04A4A;font-size:0.75rem;font-weight:600;">SIMULATED</span>
    </div>
    <div style="display:flex;align-items:flex-end;gap:12px;">
        <span style="color:#C2A878;font-size:2rem;font-weight:700;">0.18</span>
        <span style="color:#7D857F;font-size:0.8rem;margin-bottom:6px;">Warning Threshold: 0.20</span>
    </div>
    <div style="color:#7D857F;font-size:0.75rem;margin-top:8px;">
        Drift detected in weather APIs and congestion volume patterns over the last 14 simulated days.
    </div>
</div>
""", unsafe_allow_html=True)

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
backend_state = "Healthy" if not is_placeholder else "Fallback"
backend_color = "#2EA66F" if not is_placeholder else "#B8833B"
model_state = meta.get("status", "Unknown")

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
            <span style="color:{backend_color};font-size:0.8rem;">{backend_state}</span>
        </div>
        <div style="display:flex;justify-content:space-between;margin-bottom:8px;">
            <span style="color:#B5B8B1;font-size:0.8rem;">Model Interface</span>
            <span style="color:#2EA66F;font-size:0.8rem;">{model_state}</span>
        </div>
        <div style="display:flex;justify-content:space-between;">
            <span style="color:#B5B8B1;font-size:0.8rem;">Uptime</span>
            <span style="color:#F3F2EE;font-size:0.8rem;">{uptime_str}</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Profiling Telemetry (Live) ─────────────────────────────────────────────
render_section_header("Live Performance Telemetry <span style='color:#B04A4A;font-size:0.7rem;vertical-align:middle;margin-left:8px;padding:2px 6px;border:1px solid #B04A4A;border-radius:4px;'>SIMULATED</span>", accent="#B8833B")

try:
    import json
    profile_path = os.path.join(os.path.dirname(__file__), "..", "..", "cache", "profiling_results.json")
    if os.path.exists(profile_path) and os.environ.get("GRIDSIGHT_PROFILE") == "1":
        with open(profile_path, "r") as f:
            prof_data = json.load(f)
    else:
        prof_data = {
            "timings": {
                "backend.services.model_adapter._calibrated_proba": {"count": 142, "total_time": 0.08, "calls": [0.0005, 0.0007, 0.0004]},
                "frontend.components.maps.create_cluster_layer": {"count": 15, "total_time": 1.2, "calls": [0.08]},
            },
            "caches": {
                "ModelAdapter.predict": {"hits": 890, "misses": 142},
                "DataService.get_incidents": {"hits": 450, "misses": 50}
            }
        }
        
        timings = prof_data.get("timings", {})
        caches = prof_data.get("caches", {})
        
        st.markdown(f"""
        <div style="background:{BG2};border:1px solid #232A28;border-radius:6px;padding:20px;margin-bottom:16px;">
            <div style="color:#B8833B;font-size:0.8rem;font-weight:600;margin-bottom:12px;">
                Instrumented Function Latencies
            </div>
            <table style="width:100%;color:#B5B8B1;font-size:0.8rem;text-align:left;border-collapse:collapse;">
                <tr style="border-bottom:1px solid #232A28;color:#7D857F;font-size:0.7rem;text-transform:uppercase;">
                    <th style="padding:8px 4px;">Function</th>
                    <th style="padding:8px 4px;">Calls</th>
                    <th style="padding:8px 4px;">Mean Time</th>
                    <th style="padding:8px 4px;">Max Time</th>
                    <th style="padding:8px 4px;">Total Time</th>
                </tr>
        """, unsafe_allow_html=True)
        
        table_rows = []
        for func, stats in timings.items():
            calls = stats["count"]
            if calls > 0:
                mean_ms = (stats["total_time"] / calls) * 1000
                max_ms = max(stats["calls"]) * 1000
                tot_s = stats["total_time"]
                table_rows.append(f"""
                <tr style="border-bottom:1px solid #1A211E;">
                    <td style="padding:8px 4px;font-family:monospace;color:{TEXT}">{func}</td>
                    <td style="padding:8px 4px;">{calls}</td>
                    <td style="padding:8px 4px;">{mean_ms:.1f} ms</td>
                    <td style="padding:8px 4px;color:#D08C4A;">{max_ms:.1f} ms</td>
                    <td style="padding:8px 4px;">{tot_s:.2f} s</td>
                </tr>
                """)
                
        if not table_rows:
            table_rows.append("<tr><td colspan='5' style='padding:8px;text-align:center;'>No profiling data recorded yet. Interact with the app!</td></tr>")
            
        st.markdown("".join(table_rows) + "</table></div>", unsafe_allow_html=True)
        
        if caches:
            st.markdown(f"""
            <div style="background:{BG2};border:1px solid #232A28;border-radius:6px;padding:20px;">
                <div style="color:#2EA66F;font-size:0.8rem;font-weight:600;margin-bottom:12px;">
                    Cache Hit Rates
                </div>
                <div style="display:flex;gap:16px;flex-wrap:wrap;">
            """, unsafe_allow_html=True)
            
            for cname, cstats in caches.items():
                hits = cstats.get("hits", 0)
                misses = cstats.get("misses", 0)
                total = hits + misses
                rate = (hits / total * 100) if total > 0 else 0
                st.markdown(f"""
                <div style="background:#0A0D0C;border:1px solid #1A211E;border-radius:4px;padding:12px;min-width:140px;">
                    <div style="color:#7D857F;font-size:0.65rem;text-transform:uppercase;">{cname}</div>
                    <div style="color:{TEXT};font-size:1.2rem;font-weight:600;">{rate:.1f}%</div>
                    <div style="color:#B5B8B1;font-size:0.7rem;">{hits} hits / {total} reqs</div>
                </div>
                """, unsafe_allow_html=True)
                
            st.markdown("</div></div>", unsafe_allow_html=True)

except Exception as e:
    st.error(f"Error reading profile data: {e}")

render_footer()
