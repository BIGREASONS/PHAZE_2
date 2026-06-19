"""GridSight AI Command Center — Shared Constants."""

from pathlib import Path
import os

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = os.getenv(
    "DATA_PATH",
    str(BASE_DIR.parent
        / "Astram event data_anonymized - Astram event data_anonymizedb40ac87.csv"),
)

# ---------------------------------------------------------------------------
# Map
# ---------------------------------------------------------------------------
MAP_STYLE = "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json"
MAP_CENTER_LAT = 12.9716
MAP_CENTER_LON = 77.5946
MAP_DEFAULT_ZOOM = 11

# ---------------------------------------------------------------------------
# Risk Thresholds
# ---------------------------------------------------------------------------
RISK_LOW = 0.25
RISK_MEDIUM = 0.50
RISK_HIGH = 0.75

def risk_label(p: float) -> str:
    if p < RISK_LOW:
        return "LOW"
    if p < RISK_MEDIUM:
        return "MEDIUM"
    if p < RISK_HIGH:
        return "HIGH"
    return "CRITICAL"

def risk_color(p: float) -> str:
    if p < RISK_LOW:
        return "#2EA66F"
    if p < RISK_MEDIUM:
        return "#B8833B"
    if p < RISK_HIGH:
        return "#D08C4A"
    return "#B04A4A"

# ---------------------------------------------------------------------------
# Design System — Color Tokens
# ---------------------------------------------------------------------------
COLOR_BG          = "#0A0D0C"
COLOR_SECONDARY   = "#121715"
COLOR_TERTIARY    = "#1B2220"
COLOR_BORDER      = "#232A28"
COLOR_TEXT         = "#F3F2EE"
COLOR_TEXT_SEC     = "#B5B8B1"
COLOR_MUTED        = "#7D857F"

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

STATUS_SUCCESS  = "#2EA66F"
STATUS_WARNING  = "#B8833B"
STATUS_CRITICAL = "#B04A4A"
STATUS_INFO     = "#2F5D9F"
STATUS_NEUTRAL  = "#8B949E"

# ---------------------------------------------------------------------------
# Page Names
# ---------------------------------------------------------------------------
PAGES = {
    "executive":  "Executive Overview",
    "command":    "Incident Command Center",
    "predict":    "Incident Prediction",
    "explain":    "Explainability Center",
    "analytics":  "Historical Analytics",
    "geo":        "Geospatial Intelligence",
    "monitor":    "Model Monitoring",
    "showcase":   "Competition Showcase",
}

# ---------------------------------------------------------------------------
# Competition Showcase — Pre-loaded Scenarios
# ---------------------------------------------------------------------------
SCENARIOS = [
    {
        "name": "Major Accident",
        "icon": "🚗",
        "desc": "Multi-vehicle collision on arterial road",
        "features": {
            "event_type": "unplanned", "event_cause": "accident",
            "veh_type": "heavy_vehicle", "corridor": "Hosur Road",
            "police_station": "Madiwala", "zone": "South",
            "latitude": 12.9166, "longitude": 77.6101,
            "hour": 9, "weekday": 1,
        },
    },
    {
        "name": "Tree Fall",
        "icon": "🌳",
        "desc": "Large tree blocking road after storm",
        "features": {
            "event_type": "unplanned", "event_cause": "tree_fall",
            "veh_type": "NA", "corridor": "MG Road",
            "police_station": "Cubbon Park", "zone": "Central",
            "latitude": 12.9758, "longitude": 77.6066,
            "hour": 15, "weekday": 3,
        },
    },
    {
        "name": "Heavy Rain",
        "icon": "🌧️",
        "desc": "Flash flooding causing traffic disruption",
        "features": {
            "event_type": "unplanned", "event_cause": "water_logging",
            "veh_type": "NA", "corridor": "ORR East 1",
            "police_station": "Marathahalli", "zone": "East",
            "latitude": 12.9352, "longitude": 77.6245,
            "hour": 17, "weekday": 4,
        },
    },
    {
        "name": "Water Logging",
        "icon": "💧",
        "desc": "Persistent waterlogging on industrial corridor",
        "features": {
            "event_type": "unplanned", "event_cause": "water_logging",
            "veh_type": "NA", "corridor": "Tumkur Road",
            "police_station": "Peenya", "zone": "North",
            "latitude": 13.0400, "longitude": 77.5180,
            "hour": 8, "weekday": 0,
        },
    },
    {
        "name": "VIP Movement",
        "icon": "🚔",
        "desc": "Dignitary convoy requiring road closure",
        "features": {
            "event_type": "planned", "event_cause": "vip_movement",
            "veh_type": "NA", "corridor": "Airport Road",
            "police_station": "HAL", "zone": "East",
            "latitude": 12.9698, "longitude": 77.6140,
            "hour": 11, "weekday": 2,
        },
    },
    {
        "name": "Road Closure",
        "icon": "🚧",
        "desc": "Scheduled road maintenance and repair",
        "features": {
            "event_type": "planned", "event_cause": "road_work",
            "veh_type": "NA", "corridor": "Outer Ring Road",
            "police_station": "Bellandur", "zone": "South-East",
            "latitude": 12.9279, "longitude": 77.6271,
            "hour": 22, "weekday": 5,
        },
    },
]

# ---------------------------------------------------------------------------
# Replay Timeline States
# ---------------------------------------------------------------------------
REPLAY_STATES = {
    0:  {"status": "Reported",        "resources": 0, "action": "Incident logged by patrol unit"},
    5:  {"status": "Acknowledged",    "resources": 1, "action": "First responder dispatched"},
    10: {"status": "In Progress",     "resources": 2, "action": "Assessment complete, backup requested"},
    20: {"status": "Mitigation",      "resources": 3, "action": "Road diversion active, crew on site"},
    30: {"status": "Resolved",        "resources": 1, "action": "Road cleared, traffic normalising"},
}
