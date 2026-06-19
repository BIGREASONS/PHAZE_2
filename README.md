# GridSight AI
## Traffic Intelligence & Road Closure Prediction Platform

GridSight AI is an operational traffic intelligence platform built on the ASTraM traffic incident dataset, combining machine learning, geospatial analytics, explainable AI, and command-center decision support.
![Command Center Mockup](docs/assets/command_center_preview.png) *(Preview snippet)*

## The Problem
Given a reported traffic incident in Bengaluru, predict whether it **requires a road closure** at the time of reporting.
The dataset is highly imbalanced (8.3% positive) and exhibits **severe temporal distribution shift**.

## Distribution Shift Discovered
An adversarial classifier easily separates the early incidents (training data) from later incidents (test data) with an **AUC ≈ 0.87**.
Because *where* and *when* incidents happen shifts drastically over time, standard random cross-validation heavily overestimates model performance. 

## Validation Strategy
We implemented a **rolling-origin expanding-window CV** with 4 temporal folds. 
To prevent data leakage, all categorical encoders and scalers are strictly fit on the training portion of each fold.
Any model upgrade was subjected to a rigid adoption gate: it had to surpass the fold-to-fold standard deviation (noise band). Stacking meta-learners, rank averaging, and drift-aware TabPFN proxies were rigorously tested and **rejected** by this gate.

## Final Ensemble
The accepted, frozen solution is an **equal-weight probability-averaged 7-model ensemble**:
- CatBoost
- LightGBM
- XGBoost
- RandomForest
- ExtraTrees
- Logistic Regression
- TabPFN

The raw probability outputs are passed through an out-of-fold isotonic calibrator to yield true risk probabilities.

## Command Center
The system is deployed via a FastAPI backend and a Streamlit frontend ("GridSight AI Command Center").
It serves real-time calibrated predictions alongside validated operating thresholds, mapping raw closure probabilities to actionable operational postures (LOW, MEDIUM, HIGH, CRITICAL). 
Local explainability is handled via single-feature ablations against a dynamic dataset baseline.

## Results
Out-of-time (rolling-origin) performance:
- **PR-AUC:** 0.3641 ± 0.0550 (~4.4× the 8.3% base rate)
- **ROC-AUC:** 0.7887 ± 0.0391
- **Max-F1 operating point:** F1 0.440 / Precision 0.468 / Recall 0.415

GridSight AI reaches the model ceiling for the given data under its severe temporal shift. The solution has been frozen, audited, and deployed.

### Read More
- [Technical Report](docs/TECHNICAL_REPORT.md)
- [Portfolio Showcase](docs/PORTFOLIO_SHOWCASE.md)
- [Reproducibility Guide](docs/REPRODUCIBILITY.md)
- [Deployment Checklist](docs/DEPLOYMENT_CHECKLIST.md)
