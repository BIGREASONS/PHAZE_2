"""GridSight AI Command Center — PyDeck Map Layers.

All map rendering uses PyDeck with CARTO Dark Matter basemap.
No Folium. No Mapbox API key required.
"""

from typing import List, Optional, Tuple
import pandas as pd
import pydeck as pdk

from shared.profiler import profile_time

MAP_STYLE = "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json"
DEFAULT_CENTER = (12.9716, 77.5946)
DEFAULT_ZOOM = 11


def get_risk_color(prob: float) -> List[int]:
    """Return [R, G, B, A] based on risk probability."""
    if prob < 0.25:
        return [46, 166, 111, 180]   # green
    elif prob < 0.50:
        return [184, 131, 59, 200]   # amber
    elif prob < 0.75:
        return [208, 140, 74, 220]   # orange
    else:
        return [176, 74, 74, 240]    # red


def _positions(data: pd.DataFrame, extra: Optional[List[str]] = None) -> pd.DataFrame:
    """JSON-safe frame for PyDeck.

    PyDeck's ``to_json`` cannot serialize pandas ``Timestamp`` / numpy ``int64``
    cells (they have no ``__dict__``), so we keep only float lon/lat plus any
    explicitly requested passthrough columns (e.g. the RGBA ``color`` list).
    """
    out = pd.DataFrame({
        "longitude": pd.to_numeric(data["longitude"], errors="coerce").astype(float),
        "latitude": pd.to_numeric(data["latitude"], errors="coerce").astype(float),
    })
    for col in (extra or []):
        if col in data.columns:
            out[col] = list(data[col])
    return out.dropna(subset=["longitude", "latitude"]).reset_index(drop=True)


def create_incident_layer(data: pd.DataFrame, color_col: str = "color") -> pdk.Layer:
    """ScatterplotLayer coloured by risk."""
    df = _positions(data, [color_col])
    if color_col not in df.columns:
        df[color_col] = [[46, 166, 111, 180]] * len(df)
    return pdk.Layer(
        "ScatterplotLayer",
        data=df,
        get_position=["longitude", "latitude"],
        get_color=color_col,
        get_radius=120,
        radius_min_pixels=3,
        radius_max_pixels=12,
        pickable=True,
        auto_highlight=True,
    )


def create_heatmap_layer(data: pd.DataFrame) -> pdk.Layer:
    """HeatmapLayer for incident density."""
    return pdk.Layer(
        "HeatmapLayer",
        data=_positions(data),
        get_position=["longitude", "latitude"],
        get_weight=1,
        radiusPixels=40,
        intensity=1.2,
        threshold=0.05,
        color_range=[
            [46, 166, 111, 60],
            [184, 131, 59, 120],
            [208, 140, 74, 180],
            [176, 74, 74, 220],
        ],
    )


def create_cluster_layer(data: pd.DataFrame) -> pdk.Layer:
    """HexagonLayer for clustering."""
    return pdk.Layer(
        "HexagonLayer",
        data=_positions(data),
        get_position=["longitude", "latitude"],
        radius=300,
        elevation_scale=4,
        elevation_range=[0, 1000],
        extruded=True,
        pickable=True,
        auto_highlight=True,
        color_range=[
            [46, 166, 111],
            [184, 131, 59],
            [208, 140, 74],
            [176, 74, 74],
        ],
    )


def create_station_layer(data: pd.DataFrame) -> pdk.Layer:
    """ScatterplotLayer sized by incident count per station."""
    df = _positions(data, ["radius"])
    if "radius" in df.columns:
        df["radius"] = pd.to_numeric(df["radius"], errors="coerce").fillna(0).astype(float)
    return pdk.Layer(
        "ScatterplotLayer",
        data=df,
        get_position=["longitude", "latitude"],
        get_radius="radius",
        get_fill_color=[15, 118, 110, 160],
        pickable=True,
        auto_highlight=True,
    )


@profile_time("frontend.create_operations_map")
def create_operations_map(
    layers: List[pdk.Layer],
    center: Tuple[float, float] = DEFAULT_CENTER,
    zoom: int = DEFAULT_ZOOM,
    height: int = 500,
) -> pdk.Deck:
    """Assemble a full PyDeck Deck with dark basemap."""
    view = pdk.ViewState(
        latitude=center[0],
        longitude=center[1],
        zoom=zoom,
        pitch=35,
        bearing=0,
    )
    return pdk.Deck(
        layers=layers,
        initial_view_state=view,
        map_style=MAP_STYLE,
        height=height,
    )
