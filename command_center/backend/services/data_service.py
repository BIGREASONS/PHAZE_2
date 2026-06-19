"""GridSight AI Command Center — Data Service.

Loads the ASTraM dataset, caches it, and exposes filtering / aggregation /
analytics methods consumed by both the FastAPI backend and the Streamlit
frontend.  Contains ZERO model logic.
"""

from __future__ import annotations
import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from backend.config import DATA_PATH


class DataService:
    """Singleton-style data service.  Call ``load_data()`` once at startup."""

    _df: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def load_data(self, path: Optional[str] = None) -> pd.DataFrame:
        if self._df is not None:
            return self._df

        src = path or DATA_PATH
        df = pd.read_csv(src, low_memory=False)

        # Parse dates
        df["created_date"] = pd.to_datetime(df["created_date"], errors="coerce")
        df["start_datetime"] = pd.to_datetime(df["start_datetime"], errors="coerce")
        df["end_datetime"] = pd.to_datetime(df.get("end_datetime"), errors="coerce")
        df["closed_datetime"] = pd.to_datetime(df.get("closed_datetime"), errors="coerce")

        # Sort by time
        df = df.sort_values("created_date").reset_index(drop=True)

        # Derived fields
        df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
        df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
        df["hour"] = df["created_date"].dt.hour.fillna(0).astype(int)
        df["weekday"] = df["created_date"].dt.weekday.fillna(0).astype(int)
        df["month"] = df["created_date"].dt.month.fillna(0).astype(int)
        df["date"] = df["created_date"].dt.date

        # Ensure target is int
        df["requires_road_closure"] = (
            pd.to_numeric(df["requires_road_closure"], errors="coerce")
            .fillna(0).astype(int)
        )

        # Fill categorical NAs
        cat_cols = ["event_type", "event_cause", "veh_type", "corridor",
                    "police_station", "zone", "status"]
        for c in cat_cols:
            if c in df.columns:
                df[c] = df[c].astype(str).fillna("NA")

        # Resolution time (minutes)
        if "closed_datetime" in df.columns:
            df["resolution_minutes"] = (
                (df["closed_datetime"] - df["created_date"])
                .dt.total_seconds()
                .div(60)
                .clip(lower=0)
            )
        else:
            df["resolution_minutes"] = np.nan

        self._df = df
        return df

    @property
    def df(self) -> pd.DataFrame:
        if self._df is None:
            self.load_data()
        return self._df  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------
    def get_incidents(
        self,
        filters: Optional[Dict[str, Any]] = None,
        page: int = 1,
        page_size: int = 50,
    ) -> Tuple[pd.DataFrame, int]:
        data = self._apply_filters(self.df, filters)
        total = len(data)
        start = (page - 1) * page_size
        return data.iloc[start: start + page_size], total

    def _apply_filters(
        self, data: pd.DataFrame, filters: Optional[Dict[str, Any]]
    ) -> pd.DataFrame:
        if not filters:
            return data
        df = data.copy()
        if filters.get("event_type"):
            df = df[df["event_type"].isin(filters["event_type"])]
        if filters.get("event_cause"):
            df = df[df["event_cause"].isin(filters["event_cause"])]
        if filters.get("corridor"):
            df = df[df["corridor"].isin(filters["corridor"])]
        if filters.get("police_station"):
            df = df[df["police_station"].isin(filters["police_station"])]
        if filters.get("zone"):
            df = df[df["zone"].isin(filters["zone"])]
        if filters.get("status"):
            df = df[df["status"].isin(filters["status"])]
        if filters.get("date_start"):
            df = df[df["created_date"] >= pd.Timestamp(filters["date_start"])]
        if filters.get("date_end"):
            df = df[df["created_date"] <= pd.Timestamp(filters["date_end"])]
        return df

    # ------------------------------------------------------------------
    # Analytics
    # ------------------------------------------------------------------
    def get_analytics(self) -> Dict[str, Any]:
        df = self.df
        total = len(df)
        closures = int(df["requires_road_closure"].sum())
        closure_rate = round(closures / total * 100, 2) if total else 0
        corridors = int(df["corridor"].nunique())
        avg_res = round(df["resolution_minutes"].mean(), 1) if "resolution_minutes" in df.columns else 0

        return {
            "total_incidents": total,
            "closures": closures,
            "closure_rate": closure_rate,
            "active_corridors": corridors,
            "avg_resolution_minutes": avg_res if not np.isnan(avg_res) else 0,
            "incidents_by_hour": df.groupby("hour").size().to_dict(),
            "incidents_by_weekday": df.groupby("weekday").size().to_dict(),
            "incidents_by_month": df.groupby("month").size().to_dict(),
            "incidents_by_date": df.groupby("date").size().reset_index(name="count").to_dict(orient="records"),
            "closures_by_date": df[df["requires_road_closure"] == 1].groupby("date").size().reset_index(name="count").to_dict(orient="records"),
            "top_corridors": (
                df.groupby("corridor")
                .agg(total=("id", "size"), closures=("requires_road_closure", "sum"))
                .assign(closure_rate=lambda x: (x["closures"] / x["total"] * 100).round(1))
                .assign(risk_score=lambda x: (x["closure_rate"] * np.log1p(x["total"])).round(2))
                .sort_values("risk_score", ascending=False)
                .head(15)
                .reset_index()
                .to_dict(orient="records")
            ),
            "top_causes": (
                df["event_cause"]
                .value_counts()
                .head(10)
                .reset_index()
                .rename(columns={"index": "cause", "event_cause": "cause", "count": "incidents"})
                .to_dict(orient="records")
            ),
        }

    def get_corridor_stats(self) -> pd.DataFrame:
        df = self.df
        stats = (
            df.groupby("corridor")
            .agg(
                total=("id", "size"),
                closures=("requires_road_closure", "sum"),
            )
            .assign(closure_rate=lambda x: (x["closures"] / x["total"] * 100).round(1))
            .assign(risk_score=lambda x: (x["closure_rate"] * np.log1p(x["total"])).round(2))
            .sort_values("total", ascending=False)
            .reset_index()
        )
        return stats

    def get_temporal_stats(self) -> Dict[str, Any]:
        df = self.df
        return {
            "hourly": df.groupby("hour").size().to_dict(),
            "weekday": df.groupby("weekday").size().to_dict(),
            "monthly": df.groupby("month").size().to_dict(),
            "daily": df.groupby("date").size().reset_index(name="count").to_dict(orient="records"),
        }

    def get_map_data(self, filters: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        df = self._apply_filters(self.df, filters)
        cols = ["id", "latitude", "longitude", "event_type", "event_cause",
                "corridor", "police_station", "requires_road_closure",
                "status", "created_date", "address", "hour"]
        available = [c for c in cols if c in df.columns]
        out = df[available].dropna(subset=["latitude", "longitude"]).copy()
        return out

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    def search_incidents(self, query: str, limit: int = 50) -> pd.DataFrame:
        df = self.df
        q = query.lower()
        mask = (
            df["address"].astype(str).str.lower().str.contains(q, na=False)
            | df["description"].astype(str).str.lower().str.contains(q, na=False)
            | df["corridor"].astype(str).str.lower().str.contains(q, na=False)
            | df["id"].astype(str).str.lower().str.contains(q, na=False)
        )
        return df[mask].head(limit)

    # ------------------------------------------------------------------
    # Nearest Incidents
    # ------------------------------------------------------------------
    @staticmethod
    def _haversine(lat1: float, lon1: float, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
        R = 6371.0
        rlat1, rlon1 = np.radians(lat1), np.radians(lon1)
        rlat2, rlon2 = np.radians(lat2), np.radians(lon2)
        dlat = rlat2 - rlat1
        dlon = rlon2 - rlon1
        a = np.sin(dlat / 2) ** 2 + np.cos(rlat1) * np.cos(rlat2) * np.sin(dlon / 2) ** 2
        return R * 2 * np.arcsin(np.sqrt(a))

    def get_nearest_incidents(self, lat: float, lon: float, k: int = 10) -> pd.DataFrame:
        df = self.df.dropna(subset=["latitude", "longitude"]).copy()
        df["distance_km"] = self._haversine(
            lat, lon, df["latitude"].values, df["longitude"].values,
        )
        return df.nsmallest(k, "distance_km")

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    def export_data(self, fmt: str = "csv", filters: Optional[Dict[str, Any]] = None) -> bytes:
        df = self._apply_filters(self.df, filters)
        if fmt == "json":
            return df.to_json(orient="records", date_format="iso").encode()
        elif fmt == "excel":
            buf = io.BytesIO()
            df.to_excel(buf, index=False, engine="openpyxl")
            return buf.getvalue()
        else:
            return df.to_csv(index=False).encode()
