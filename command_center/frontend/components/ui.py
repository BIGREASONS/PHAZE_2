"""GridSight AI Command Center — Reusable UI Components.

Bento-style KPI cards, leaderboard tables, status badges, section headers.
All rendered via st.markdown with inline CSS — no external stylesheets.
"""

from typing import Any, Dict, List, Optional
import streamlit as st
import pandas as pd


def render_kpi_card(
    title: str,
    value: Any,
    delta: Optional[str] = None,
    icon: str = "",
    accent: str = "#C2A878",
) -> None:
    delta_html = ""
    if delta:
        delta_html = f'<div style="color:#B5B8B1;font-size:0.7rem;margin-top:2px;">{delta}</div>'

    st.markdown(f"""
    <div style="
        background:#121715; border:1px solid #232A28; border-radius:6px;
        padding:16px 18px; border-top:2px solid {accent};
    ">
        <div style="color:#7D857F;font-size:0.65rem;text-transform:uppercase;
            letter-spacing:0.08em;font-family:'Inter',sans-serif;margin-bottom:6px;">
            {icon} {title}
        </div>
        <div style="color:#F3F2EE;font-size:1.55rem;font-weight:700;
            font-family:'Inter Tight',sans-serif;line-height:1.1;">
            {value}
        </div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


def render_kpi_row(kpis: List[Dict[str, Any]], accent: str = "#C2A878") -> None:
    cols = st.columns(len(kpis))
    for col, kpi in zip(cols, kpis):
        with col:
            render_kpi_card(
                title=kpi["title"],
                value=kpi["value"],
                delta=kpi.get("delta"),
                icon=kpi.get("icon", ""),
                accent=accent,
            )


def render_section_header(
    title: str,
    subtitle: Optional[str] = None,
    accent: str = "#C2A878",
) -> None:
    sub_html = ""
    if subtitle:
        sub_html = f'<span style="color:#7D857F;font-size:0.8rem;margin-left:12px;">{subtitle}</span>'
    st.markdown(f"""
    <div style="display:flex;align-items:baseline;margin:24px 0 12px 0;
        padding-bottom:8px;border-bottom:1px solid #232A28;">
        <span style="color:{accent};font-size:0.55rem;margin-right:8px;">&#9646;</span>
        <h3 style="margin:0;color:#F3F2EE;font-size:1.05rem;">{title}</h3>
        {sub_html}
    </div>
    """, unsafe_allow_html=True)


def render_leaderboard_table(
    df: pd.DataFrame,
    title: str = "",
    accent: str = "#C2A878",
) -> None:
    if title:
        render_section_header(title, accent=accent)

    header = "".join(
        f'<th style="padding:8px 12px;text-align:left;color:#7D857F;font-size:0.7rem;'
        f'text-transform:uppercase;letter-spacing:0.06em;border-bottom:1px solid #232A28;">{c}</th>'
        for c in df.columns
    )
    rows = ""
    for _, row in df.iterrows():
        cells = "".join(
            f'<td style="padding:8px 12px;color:#B5B8B1;font-size:0.82rem;'
            f'border-bottom:1px solid #1B2220;">{v}</td>'
            for v in row
        )
        rows += f'<tr style="transition:background 150ms;">{cells}</tr>'

    st.markdown(f"""
    <div style="border:1px solid #232A28;border-radius:6px;overflow:hidden;margin-bottom:16px;">
        <table style="width:100%;border-collapse:collapse;background:#121715;">
            <thead><tr>{header}</tr></thead>
            <tbody>{rows}</tbody>
        </table>
    </div>
    """, unsafe_allow_html=True)


def render_status_badge(label: str, color: str = "#2EA66F") -> str:
    return (
        f'<span style="display:inline-block;padding:2px 10px;border-radius:10px;'
        f'font-size:0.7rem;font-weight:600;color:{color};background:{color}18;'
        f'border:1px solid {color}33;font-family:Inter,sans-serif;">{label}</span>'
    )


def render_metric_tile(
    title: str,
    value: Any,
    subtitle: str = "",
    accent: str = "#C2A878",
) -> None:
    sub_html = ""
    if subtitle:
        sub_html = f'<div style="color:#7D857F;font-size:0.7rem;margin-top:4px;">{subtitle}</div>'
    st.markdown(f"""
    <div style="background:#121715;border:1px solid #232A28;border-radius:6px;
        padding:20px;text-align:center;">
        <div style="color:#7D857F;font-size:0.65rem;text-transform:uppercase;
            letter-spacing:0.08em;margin-bottom:8px;">{title}</div>
        <div style="color:{accent};font-size:2.2rem;font-weight:700;
            font-family:'Inter Tight',sans-serif;">{value}</div>
        {sub_html}
    </div>
    """, unsafe_allow_html=True)


def render_loading_skeleton() -> None:
    st.markdown("""
    <div style="background:#121715;border:1px solid #232A28;border-radius:6px;
        padding:24px;text-align:center;">
        <div style="color:#7D857F;font-size:0.85rem;">Loading...</div>
    </div>
    """, unsafe_allow_html=True)


def render_empty_state(message: str = "No data available") -> None:
    st.markdown(f"""
    <div style="background:#121715;border:1px solid #232A28;border-radius:6px;
        padding:40px;text-align:center;">
        <div style="color:#7D857F;font-size:1.5rem;margin-bottom:8px;">--</div>
        <div style="color:#7D857F;font-size:0.85rem;">{message}</div>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar_health(model_info: Dict[str, Any], data_rows: int) -> None:
    st.sidebar.markdown(f"""
    <div style="background:#1B2220;border:1px solid #232A28;border-radius:6px;
        padding:12px;margin-top:12px;">
        <div style="color:#7D857F;font-size:0.6rem;text-transform:uppercase;
            letter-spacing:0.08em;margin-bottom:8px;">System Health</div>
        <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
            <span style="color:#B5B8B1;font-size:0.75rem;">Model</span>
            <span style="color:#F3F2EE;font-size:0.75rem;font-weight:500;">{model_info.get('name','--')}</span>
        </div>
        <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
            <span style="color:#B5B8B1;font-size:0.75rem;">Status</span>
            {render_status_badge(model_info.get('status','--'), '#2EA66F')}
        </div>
        <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
            <span style="color:#B5B8B1;font-size:0.75rem;">Incidents</span>
            <span style="color:#F3F2EE;font-size:0.75rem;font-weight:500;">{data_rows:,}</span>
        </div>
        <div style="display:flex;justify-content:space-between;">
            <span style="color:#B5B8B1;font-size:0.75rem;">Version</span>
            <span style="color:#7D857F;font-size:0.75rem;font-family:'JetBrains Mono',monospace;">
                {model_info.get('version','--')}
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)
