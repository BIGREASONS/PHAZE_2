# ASTraM Command Center

**Real-time Traffic Incident Intelligence Platform for Bengaluru**

Built for Flipkart Gridlock 2.0 — Bengaluru Traffic Police

---

## Architecture

```
command_center/
├── backend/
│   ├── main.py                     # FastAPI application
│   ├── config.py                   # Application configuration
│   └── services/
│       ├── model_adapter.py        # ModelInterface + PlaceholderModel
│       ├── data_service.py         # Dataset loading & analytics
│       ├── metrics_registry.py     # KPI registry
│       └── auth.py                 # Role-based auth
├── frontend/
│   ├── app.py                      # Streamlit entrypoint
│   ├── components/
│   │   ├── theme.py                # Dark design system CSS
│   │   ├── ui.py                   # Bento KPI cards, tables
│   │   ├── maps.py                 # PyDeck dark map layers
│   │   └── filters.py             # Global filter system
│   └── pages/
│       ├── 1_Executive_Overview.py
│       ├── 2_Incident_Command_Center.py
│       ├── 3_Incident_Prediction.py
│       ├── 4_Explainability_Center.py
│       ├── 5_Historical_Analytics.py
│       ├── 6_Geospatial_Intelligence.py
│       ├── 7_Model_Monitoring.py
│       └── 8_Competition_Showcase.py
├── shared/
│   └── constants.py                # Colors, scenarios, risk thresholds
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

## Quick Start

### Manual (Development)

```bash
cd command_center

# Install dependencies
pip install -r requirements.txt

# Start FastAPI backend (port 8000)
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

# Start Streamlit frontend (port 8501) — in a separate terminal
streamlit run frontend/app.py --server.port 8501
```

### Docker

```bash
cd command_center
docker-compose up --build
```

- **Dashboard**: http://localhost:8501
- **API Docs**: http://localhost:8000/docs

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | System health check |
| GET | `/incidents` | Paginated, filtered incident list |
| GET | `/analytics` | Full analytics payload |
| POST | `/predict` | Predict road closure probability |
| POST | `/explain` | Feature contribution explanation |
| GET | `/model-info` | Model metadata |
| WS | `/stream/incidents` | WebSocket streaming (placeholder) |

## Model Integration Guide

The system is designed to accept **any** future model. To integrate:

1. Implement `ModelInterface` in `backend/services/model_adapter.py`
2. Replace `PlaceholderModel` instantiation in `backend/main.py`
3. No page code changes required

```python
class YourModel(ModelInterface):
    def load_model(self):
        # Load your weights
    def predict(self, features):
        # Return {probability, confidence, severity, recommended_action}
    def predict_batch(self, features_list):
        # Batch prediction
    def explain(self, features):
        # Return {feature_contributions, top_positive_drivers, top_negative_drivers}
    def get_model_metadata(self):
        # Return {name, version, training_date, metrics, status}
```

## Design System

- **Theme**: Palantir-grade dark industrial
- **No glow / neon / glassmorphism**
- **Each page has a unique accent color** (Gold, Jade, Sapphire, Copper, Burgundy, Teal, Silver, Champagne)
- **Typography**: Inter, Inter Tight, JetBrains Mono
- **Maps**: PyDeck with CARTO Dark Matter basemap
