"""ASTraM Command Center — FastAPI Backend."""

from __future__ import annotations
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.services.data_service import DataService
from backend.services.model_adapter import PlaceholderModel, ModelInterface
from backend.config import API_HOST, API_PORT

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------
data_service = DataService()
model: ModelInterface = PlaceholderModel()


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    data_service.load_data()
    model.load_model()
    yield


app = FastAPI(
    title="ASTraM Command Center API",
    version="0.1.0",
    description="Traffic Incident Intelligence — Backend Services",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    event_type: str = "unplanned"
    event_cause: str = "accident"
    veh_type: str = "NA"
    corridor: str = "Unknown"
    police_station: str = "Unknown"
    zone: str = "Unknown"
    latitude: float = 12.9716
    longitude: float = 77.5946
    hour: int = 12
    weekday: int = 2


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": True,
        "data_rows": len(data_service.df),
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/incidents")
async def get_incidents(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    event_type: Optional[str] = None,
    event_cause: Optional[str] = None,
    corridor: Optional[str] = None,
    police_station: Optional[str] = None,
):
    filters: Dict[str, Any] = {}
    if event_type:
        filters["event_type"] = [event_type]
    if event_cause:
        filters["event_cause"] = [event_cause]
    if corridor:
        filters["corridor"] = [corridor]
    if police_station:
        filters["police_station"] = [police_station]

    df, total = data_service.get_incidents(filters, page, page_size)
    records = df.to_dict(orient="records")
    # Convert timestamps to strings for JSON serialization
    for r in records:
        for k, v in r.items():
            if hasattr(v, "isoformat"):
                r[k] = v.isoformat()
    return {"incidents": records, "total": total, "page": page, "page_size": page_size}


@app.get("/analytics")
async def get_analytics():
    analytics = data_service.get_analytics()
    return analytics


@app.post("/predict")
async def predict(req: PredictRequest):
    features = req.model_dump()
    result = model.predict(features)
    return result


@app.post("/explain")
async def explain(req: PredictRequest):
    features = req.model_dump()
    result = model.explain(features)
    return result


@app.get("/model-info")
async def model_info():
    return model.get_model_metadata()


# ---------------------------------------------------------------------------
# WebSocket — Telemetry Heartbeat
# ---------------------------------------------------------------------------
import asyncio

@app.websocket("/stream/incidents")
async def stream_incidents(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            await asyncio.sleep(5)
            await ws.send_json({
                "timestamp": datetime.utcnow().isoformat(),
                "active_incidents": len(data_service.df),
                "high_risk_incidents": int(data_service.df["requires_road_closure"].sum()),
                "system_health": "healthy",
                "model_status": "placeholder",
                "data_freshness": "loaded"
            })
    except WebSocketDisconnect:
        pass


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host=API_HOST, port=API_PORT, reload=True)
