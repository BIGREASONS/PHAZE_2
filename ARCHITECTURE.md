# Architecture

GridSight AI is designed to separate concerns across three major pillars: the Streamlit Frontend, the FastAPI Backend Service, and the Frozen ML Pipeline.

## 1. Frontend Architecture
**Framework:** Streamlit
**Design:** Palantir-grade dark theme (custom CSS via `components/theme.py`)
- The UI handles zero heavy computation. All inference logic, data crunching, and external API polling is offloaded to the backend.
- **Mapping:** Built primarily on **PyDeck** utilizing the `CARTO Dark Matter` basemap for high-performance, WebGL-accelerated rendering.

## 2. Backend Architecture
**Framework:** FastAPI + Uvicorn
The backend serves as the proxy and computational bridge:
- **`model_adapter.py`**: A unified interface utilizing the `ModelInterface` contract. It loads the `ProductionEnsembleModel` and strictly parses the frozen `manifest.json`.
- **`mapmyindia.py`**: A secure, isolated API proxy resolving Reverse Geocoding and Place Intelligence via MapmyIndia. Designed with offline file caching (`mapmyindia_cache.json`) to guarantee stable demos and prevent quota exhaustion.

## 3. Model Architecture
**Ensemble Type:** Equal-weight probability average.
**Base Models:** CatBoost, LightGBM, XGBoost, RandomForest, ExtraTrees, Logistic Regression, TabPFN.
- Features are categorically encoded and scaled identically to the frozen state recorded in `final_validation/artifacts/encoder.pkl`.
- Predictions are mapped against the target operating thresholds (`thresholds.json`) following rigorous isotonic calibration.

## 4. Artifact Flow
1. `final_validation/export_production_model.py` bundles all weights, encoders, and config into `artifacts/`.
2. `Dockerfile` copies `final_validation/artifacts/` into the image context.
3. FastAPI's `ProductionEnsembleModel` parses `manifest.json` on boot, initializing the models into memory.
4. If artifacts are inaccessible, a deterministic `PlaceholderModel` is initialized to preserve the UI.

## 5. MapmyIndia Integration Flow
1. Incident selection triggers a backend `GET` request to `/location/reverse-geocode` and `/location/nearby`.
2. The `MapmyIndiaService` checks its local JSON cache.
3. On a cache miss, it contacts the MapmyIndia REST API using the stored `MAPMYINDIA_API_KEY`.
4. If successful, results are cached and formatted strings are returned. If an error occurs, empty arrays are returned.
5. The UI conditionally renders the Place Intelligence components based on the backend response.
