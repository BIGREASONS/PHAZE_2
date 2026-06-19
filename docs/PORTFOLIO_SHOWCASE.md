# GridSight AI — Shift-Aware Road-Closure Prediction & Command Center

> A traffic-incident ML system that knows the difference between a real improvement and
> noise — and ships the part that survives.

**One line:** Built, validated, calibrated, and deployed a 7-model ensemble that predicts
whether a Bengaluru traffic incident will require a road closure, under **severe temporal
distribution shift (adversarial AUC ≈ 0.87)** — with a validation harness strict enough to
**reject three of my own upgrades** and freeze the honest baseline.

---

## The 30-second version

- **Problem:** 8,173 incidents, 8.3 % require a closure. Predict it at report time.
- **Twist:** the data drifts hard over time — train and test months barely overlap; an
  adversarial classifier separates early from late records at **AUC 0.87**.
- **What I built:** a rolling-origin temporal validation harness, a 7-model calibrated
  ensemble, and a production dashboard ("Command Center") wired to the live model.
- **What makes it credible:** I held every candidate to a noise-band gate. Rank averaging,
  temporal-decay weighting, and a drift-aware TabPFN variant **all failed the gate** — and
  I froze the simple ensemble instead of shipping a flattering-but-fake gain.
- **Result:** **PR-AUC 0.364 ± 0.055 / ROC-AUC 0.789 ± 0.039** out-of-time, calibrated,
  with operating thresholds and a one-line model swap into the app.

---

## Why this is worth your attention

Most small-data ML projects quietly overfit their own validation. This one was built to
**resist** that. The headline deliverable isn't a leaderboard number — it's a **decision
process** that produced a trustworthy number and a clean refusal to chase noise.

### 1. Rigorous, shift-aware validation
- **Rolling-origin expanding-window CV** (train on the past, predict the future), 4 folds,
  with **every encoder and scaler fit on train rows only, per fold** — zero future leakage.
- A **pre-registered adoption gate**: a candidate is adopted only if its mean PR-AUC margin
  is positive, it wins a majority of folds, **and** the gain exceeds fold-to-fold std
  (≈ 0.055). Ties go to the incumbent.
- Calibration and operating thresholds are fit on **out-of-fold** predictions, so they're
  honest, not in-sample.

### 2. I quantified the shift before trusting any score
- **Adversarial validation:** trained classifiers to tell early vs. late records apart →
  **AUC ≈ 0.87**. The top drift drivers were *where* and *when* (zone, weekday, lat/lon).
- **Statistical confirmation:** χ²(calendar month) ≈ 5,118 (p ≈ 0); test half is 61 %
  March / 39 % April with **no** winter months. The label rate, though, is stable
  (0.082 vs 0.086) — so it's **covariate shift, not concept drift**, which dictated the
  whole strategy.

### 3. The most valuable result is a negative one
A stacking meta-learner **won under a random split (+0.026 AP)** but **lost under a
temporal split (−0.013 AP)**. That single contrast is the thesis: gains a random split
rewards are exactly the ones that evaporate out-of-time. Equal-weight probability averaging
carries no fitted structure to overfit — so it transports, and it's what I shipped.

### 4. Disciplined ensemble design
- 7 diverse members: **CatBoost, LightGBM, XGBoost, RandomForest, ExtraTrees, Logistic,
  TabPFN** — gradient boosting + bagging + linear + in-context transformer.
- Combined by **equal-weight probability averaging** (no learned weights to overfit),
  then **isotonic-calibrated**.
- Self-describing artifacts: a `manifest.json` declares each member's input routing, so the
  serving adapter reads the ensemble definition as *data*, not hard-coded logic.

### 5. Deployment readiness — not a notebook
- A FastAPI backend and an 8-page Streamlit "Command Center" both consume **one
  `ModelInterface`**; swapping the placeholder for the real ensemble was a one-line
  `get_model()` change, already wired across every page.
- The production adapter loads the frozen artifacts, refits TabPFN once and keeps it warm,
  serves calibrated probabilities + severity bands tied to validated thresholds, and
  **falls back gracefully** so the system always boots.
- I **measured** the real serving cost (TabPFN CPU ≈ 4 min/prediction, independent of batch
  size) and engineered around it: single-call batched explanations, a prediction cache, and
  a documented GPU path — rather than hiding the latency.

---

## Results at a glance

| | Out-of-time (rolling-origin) |
|---|---|
| **PR-AUC** | **0.3641 ± 0.0550** (~4.4× the 8.3 % base rate) |
| **ROC-AUC** | **0.7887 ± 0.0391** |
| Max-F1 operating point | F1 0.440 · precision 0.468 · recall 0.415 @ 0.333 |
| Adversarial shift (train vs test) | AUC ≈ 0.87 |
| Candidates tested against incumbent | 3 |
| Candidates that beat the noise band | **0** → freeze |

---

## Tech stack

`Python 3.12` · `scikit-learn` · `CatBoost` · `LightGBM` · `XGBoost` · `TabPFN` (PyTorch) ·
`FastAPI` · `Streamlit` · `Plotly` / `PyDeck` · isotonic calibration · rolling-origin
temporal CV · adversarial validation.

---

## What I'd want a reviewer to take away

- I can **diagnose** a dataset's real difficulty (shift), not just fit models to it.
- I build **validation I can't fool myself with**, and I **act on it** — including killing
  my own ideas when the evidence says noise.
- I think about **calibration, thresholds, and operating posture**, not just AUC.
- I take a model **all the way to a serving boundary** with measured latency, failure
  modes, and a clean integration contract.

---

### Read more
- **`docs/TECHNICAL_REPORT.md`** — full writeup (shift discovery, adversarial validation,
  failed experiments, leakage, architecture, metrics).
- **`docs/DEPLOYMENT_CHECKLIST.md`** — required files, deps, startup, failure modes,
  health checks.
- **`docs/REPRODUCIBILITY.md`** — clone → train → validate → export → launch, end to end.
- **`final_validation/results/final_recommendation_report.md`** — the machine-generated
  freeze verdict.
