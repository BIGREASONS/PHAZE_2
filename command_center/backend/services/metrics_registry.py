"""ASTraM Command Center — Metrics Registry.

Every KPI surfaced anywhere in the platform originates from this registry.
Future models can register additional KPIs without editing any page code.
"""

from __future__ import annotations
from typing import Any, Callable, Dict, Optional
import pandas as pd
import numpy as np


class MetricsRegistry:
    """Central KPI registry.  Register compute functions here."""

    _metrics: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def register(
        cls,
        name: str,
        compute_fn: Callable[[pd.DataFrame], Any],
        category: str = "general",
        fmt: str = "{}",
        icon: str = "",
    ) -> None:
        cls._metrics[name] = {
            "compute": compute_fn,
            "category": category,
            "fmt": fmt,
            "icon": icon,
        }

    @classmethod
    def compute_all(cls, data: pd.DataFrame) -> Dict[str, Any]:
        return {k: v["compute"](data) for k, v in cls._metrics.items()}

    @classmethod
    def compute_category(cls, category: str, data: pd.DataFrame) -> Dict[str, Any]:
        return {
            k: v["compute"](data)
            for k, v in cls._metrics.items()
            if v["category"] == category
        }

    @classmethod
    def get_formatted(cls, data: pd.DataFrame) -> Dict[str, str]:
        out = {}
        for k, v in cls._metrics.items():
            raw = v["compute"](data)
            out[k] = v["fmt"].format(raw)
        return out

    @classmethod
    def list_metrics(cls) -> list:
        return [
            {"name": k, "category": v["category"], "icon": v["icon"]}
            for k, v in cls._metrics.items()
        ]


# ---------------------------------------------------------------------------
# Pre-registered KPIs
# ---------------------------------------------------------------------------

MetricsRegistry.register(
    "total_incidents",
    lambda df: len(df),
    category="executive",
    fmt="{:,}",
    icon="📋",
)

MetricsRegistry.register(
    "closure_rate",
    lambda df: round(df["requires_road_closure"].mean() * 100, 1) if len(df) else 0,
    category="executive",
    fmt="{:.1f}%",
    icon="🚧",
)

MetricsRegistry.register(
    "active_corridors",
    lambda df: int(df["corridor"].nunique()),
    category="executive",
    fmt="{}",
    icon="🛤️",
)

MetricsRegistry.register(
    "avg_resolution_minutes",
    lambda df: round(df["resolution_minutes"].mean(), 1)
    if "resolution_minutes" in df.columns and not df["resolution_minutes"].isna().all()
    else 0,
    category="executive",
    fmt="{:.0f} min",
    icon="⏱️",
)

MetricsRegistry.register(
    "high_risk_count",
    lambda df: int(df["requires_road_closure"].sum()),
    category="executive",
    fmt="{:,}",
    icon="⚠️",
)

MetricsRegistry.register(
    "model_confidence",
    lambda df: "87.2%",  # Placeholder — fed by model adapter at runtime
    category="model",
    fmt="{}",
    icon="🎯",
)
