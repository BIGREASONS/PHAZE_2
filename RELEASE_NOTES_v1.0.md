# GridSight AI v1.0 Release Notes

**GridSight AI combines a rigorously validated road-closure prediction ensemble with MapmyIndia-powered geospatial intelligence and an operational command-center interface for traffic incident response.**

## Release Overview
Version 1.0 represents the final, production-ready release candidate for the Flipkart Gridlock 2.0 hackathon. 

### Frozen Ensemble Summary
The core predictive engine is a **7-model equal-weight probability ensemble**:
- CatBoost
- LightGBM
- XGBoost
- RandomForest
- ExtraTrees
- Logistic Regression
- TabPFN

The ensemble relies on isotonic calibration to ensure reliable probability mapping against the strict operational thresholds. All model weights, inference pipelines, and calibration bounds are firmly **frozen**. 

### Validation Methodology
The ensemble survived a brutal validation framework prior to this freeze:
- Pure random K-Fold cross-validation was rejected to avoid leakage.
- Rolling-origin temporal validation was implemented (train on past, test on future).
- Spatial corridor blocking tests were employed to measure true generalizability to unseen corridors.
- *Final Incumbent Status*: The equal-weight strategy consistently dominated single-model and stacked approaches across all rigorous gates.

### MapmyIndia Integration
GridSight AI v1.0 features an integrated location enrichment layer powered by **MapmyIndia (Mappls)**:
- **Reverse Geocoding**: Translates raw lat/lon signals into human-readable locations.
- **Place Intelligence**: Dynamically lists nearby landmarks inside the Incident Command Center.
- *Zero-Risk Fallback*: The integration operates over a robust caching mechanism. If the external MapmyIndia API limits are reached, the system gracefully degrades to raw coordinates while continuing to operate flawlessly over the PyDeck/CARTO rendering engine.

### Command Center Features
The frontend interface features a fully functional Palantir-style dashboard:
- Executive Overview (KPIs, Charts, PDF Briefing generation)
- Live Incident Command Center
- Explainability Center (Feature ablations & global driver insights)
- Model Monitoring
- Competition Showcase

### Known Limitations
- **TabPFN CPU Latency**: Due to its transformer-based context window loading, TabPFN may take a few minutes to initialize on CPU during the first `predict()` or `load_model()` operation. This is expected behavior for the frozen ensemble configuration.
- **Docker Fallback Context**: If `ASTRAM_ARTIFACT_DIR` is missing or unreachable at boot, the API intentionally falls back to a deterministic `PlaceholderModel` with simulated weights to ensure the UI stays up for judging.
