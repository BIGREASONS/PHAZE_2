"""ASTraM Command Center — Global Filter System.

Renders multi-select filters in the sidebar or inline.
Filter state persists in st.session_state across pages.
"""

from typing import Any, Dict, Optional
import pandas as pd
import streamlit as st


def render_global_filters(df: pd.DataFrame, key_prefix: str = "gf") -> Dict[str, Any]:
    """Render filter widgets and return a filter dict."""
    filters: Dict[str, Any] = {}

    with st.expander("Filters", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            corridors = sorted(df["corridor"].dropna().unique().tolist())
            sel_corr = st.multiselect("Corridor", corridors, key=f"{key_prefix}_corridor")
            if sel_corr:
                filters["corridor"] = sel_corr

            causes = sorted(df["event_cause"].dropna().unique().tolist())
            sel_cause = st.multiselect("Event Cause", causes, key=f"{key_prefix}_cause")
            if sel_cause:
                filters["event_cause"] = sel_cause

        with c2:
            stations = sorted(df["police_station"].dropna().unique().tolist())
            sel_station = st.multiselect("Police Station", stations, key=f"{key_prefix}_station")
            if sel_station:
                filters["police_station"] = sel_station

            types = sorted(df["event_type"].dropna().unique().tolist())
            sel_type = st.multiselect("Event Type", types, key=f"{key_prefix}_type")
            if sel_type:
                filters["event_type"] = sel_type

        # Date range
        if "created_date" in df.columns:
            min_date = df["created_date"].min()
            max_date = df["created_date"].max()
            if pd.notna(min_date) and pd.notna(max_date):
                d1, d2 = st.columns(2)
                with d1:
                    start = st.date_input("From", value=min_date.date(), key=f"{key_prefix}_ds")
                    filters["date_start"] = str(start)
                with d2:
                    end = st.date_input("To", value=max_date.date(), key=f"{key_prefix}_de")
                    filters["date_end"] = str(end)

    return filters


def apply_filters(df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
    """Apply a filter dict to a DataFrame."""
    if not filters:
        return df
    out = df.copy()
    for col in ["event_type", "event_cause", "corridor", "police_station", "zone", "status"]:
        if col in filters and filters[col]:
            out = out[out[col].isin(filters[col])]
    if filters.get("date_start"):
        out = out[out["created_date"] >= pd.Timestamp(filters["date_start"])]
    if filters.get("date_end"):
        out = out[out["created_date"] <= pd.Timestamp(filters["date_end"])]
    return out
