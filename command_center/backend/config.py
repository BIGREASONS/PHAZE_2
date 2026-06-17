"""ASTraM Command Center — Application Configuration."""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = os.getenv(
    "DATA_PATH",
    str(BASE_DIR.parent / "astram_review_bundle_v2" / "review_bundle"
        / "Astram event data_anonymized - Astram event data_anonymizedb40ac87.csv"),
)

# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
STREAMLIT_PORT = int(os.getenv("STREAMLIT_PORT", "8501"))
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

# ---------------------------------------------------------------------------
# Model (placeholder until Claude provides the final model)
# ---------------------------------------------------------------------------
MODEL_NAME = os.getenv("MODEL_NAME", "PlaceholderModel")

# ---------------------------------------------------------------------------
# Design System — Color Tokens
# ---------------------------------------------------------------------------

# Core palette
COLOR_BG          = "#0A0D0C"
COLOR_SECONDARY   = "#121715"
COLOR_TERTIARY    = "#1B2220"
COLOR_BORDER      = "#232A28"

# Typography
COLOR_TEXT         = "#F3F2EE"
COLOR_TEXT_SEC     = "#B5B8B1"
COLOR_MUTED        = "#7D857F"

# Page accents
ACCENT_EXECUTIVE   = "#C2A878"
ACCENT_EXECUTIVE_L = "#D8BF8C"
ACCENT_OPS         = "#1C7C54"
ACCENT_OPS_L       = "#2EA66F"
ACCENT_PREDICT     = "#2F5D9F"
ACCENT_PREDICT_L   = "#4C7CC0"
ACCENT_EXPLAIN     = "#B87333"
ACCENT_EXPLAIN_L   = "#D08C4A"
ACCENT_ANALYTICS   = "#6E1F28"
ACCENT_ANALYTICS_L = "#8B2635"
ACCENT_GEO         = "#0F766E"
ACCENT_GEO_L       = "#14B8A6"
ACCENT_MONITOR     = "#8B949E"
ACCENT_MONITOR_L   = "#C5CCD4"
ACCENT_SHOWCASE    = "#D9C7A3"
ACCENT_SHOWCASE_L  = "#E7DBC2"

# Status
STATUS_SUCCESS  = "#2EA66F"
STATUS_WARNING  = "#B8833B"
STATUS_CRITICAL = "#B04A4A"
STATUS_INFO     = "#2F5D9F"
STATUS_NEUTRAL  = "#8B949E"
