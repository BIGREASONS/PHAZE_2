# GridSight AI

**GridSight AI combines a rigorously validated road-closure prediction ensemble with MapmyIndia-powered geospatial intelligence and an operational command-center interface for traffic incident response.**

Built for Flipkart Gridlock 2.0 — Bengaluru Traffic Police

## Overview
GridSight AI is an end-to-end traffic incident intelligence platform. By combining a highly optimized machine learning ensemble with robust spatial visualization tools, the platform identifies high-risk corridors, predicts severe closures before they cascade, and enables operations teams to act preemptively.

## Features
- **Traffic Incident Command Center**: Live monitoring of real-time incoming traffic anomalies.
- **Ensemble Prediction Pipeline**: A frozen 7-model equal-weight probability ensemble (CatBoost, LightGBM, XGBoost, RandomForest, ExtraTrees, Logistic, TabPFN) predicting closure risks.
- **Geospatial Intelligence**: MapmyIndia-powered location intelligence overlaid on performant PyDeck/CARTO rendering, featuring hotspot and corridor analysis.
- **Explainability Center**: Real-time SHAP and single-feature ablation providing trust and transparency for every prediction.

## Command Center UI

![Executive Overview](docs/assets/executive_overview.png)
![Incident Command Center](docs/assets/incident_command_center.png)
![Incident Prediction](docs/assets/incident_prediction.png)
![Explainability Center](docs/assets/explainability_center.png)
![Geospatial Intelligence](docs/assets/geospatial_intelligence.png)
![Model Monitoring](docs/assets/model_monitoring.png)
![Competition Showcase](docs/assets/competition_showcase.png)

## Demo & Architecture
- **Demo Video**: [docs/assets/demo_video.mp4](docs/assets/demo_video.mp4)
- **Architecture Diagram**: ![Architecture Diagram](docs/assets/architecture_diagram.png)

## Research Mode

GridSight AI includes an optional TabPFN-powered Research Mode for experimentation and model comparison workflows.

Due to the computational requirements of TabPFN (memory allocation and initialization time), the public demo executes only the validated Production Ensemble. Users wishing to explore Research Mode can run the application locally using the provided setup instructions.

| Capability              | Hosted Demo | Local Research |
| ----------------------- | ----------- | -------------- |
| Production Ensemble     | ✓           | ✓              |
| Explainability          | ✓           | ✓              |
| MapmyIndia Intelligence | ✓           | ✓              |
| Historical Analytics    | ✓           | ✓              |
| TabPFN Research Mode    | ✗           | ✓              |
| Deployment Diagnostics  | ✓           | ✓              |

## Quick Start (Docker)
The easiest way to launch the entire stack is via Docker Compose:
```bash
cd command_center

# Optional: Add your MapmyIndia Key for location enrichment
export MAPMYINDIA_API_KEY="your-static-key-here"

docker-compose up --build
```
- **Dashboard**: http://localhost:8501
- **API Docs**: http://localhost:8000/docs

## Architecture
See [ARCHITECTURE.md](ARCHITECTURE.md) for details on the ML and Engineering topology.
See [RELEASE_NOTES_v1.0.md](RELEASE_NOTES_v1.0.md) for details on validation and the freezing criteria.
