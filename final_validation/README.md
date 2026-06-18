# final_validation/ — ASTraM last-chance-before-freeze validation

Isolated, end-to-end harness that ruthlessly tests the three surviving audit ideas
against the frozen incumbent under **honest rolling-origin temporal CV**, then exports
the winning ensemble. Touches nothing in `command_center/` or `phase1–phase7`.

## Incumbent
Equal-weight 7-model **probability-averaged** ensemble:
CatBoost · LightGBM · XGBoost · RandomForest · ExtraTrees · Logistic · TabPFN.

## Method (why results are trustworthy)
- **Rolling-origin expanding window**, 4 folds. Fold *k* trains on `[0:cut_k)` and
  validates on the next contiguous future block `[cut_k:cut_{k+1})`.
- Every encoder / scaler / model is **fit on train rows only, per fold** — no future
  leakage. We deliberately do **not** reuse the repo's `TimeSeriesSplit(5)` OOF caches
  (different folds → silent apples-to-oranges).
- Base predictions are cached per fold so Phases B/C/D reuse them.
- **Adoption policy:** a candidate is adopted ONLY if its mean PR-AUC margin > 0 AND it
  wins a majority of folds AND the margin exceeds the fold-to-fold std (noise band).
  Ties go to the incumbent.

## Run order
```bash
python final_ensemble_validation.py   # Phase A: framework + incumbent baseline (slow: trains 7×4)
python rank_vs_probability.py          # Phase B: rank vs probability averaging
python temporal_decay.py               # Phase C: fixed-half-life decay weighting
python dr_tabpfn.py                    # Phase D: DR-TabPFN feasibility + recency proxy
python final_gate.py                   # Phase E: aggregate + final_recommendation_report.md
python export_production_model.py      # Phase F: export blessed ensemble -> artifacts/
```

## Outputs
- `results/final_ensemble_validation.json` — incumbent baseline + per-fold geometry
- `results/rank_vs_probability_results.json`
- `results/temporal_decay_results.json`
- `results/dr_tabpfn_results.json`
- `results/final_recommendation_report.md` — the human-readable verdict
- `results/final_decision.json` — machine-readable choice consumed by Phase F
- `artifacts/` — encoder.pkl, per-member models, tabpfn_refit.npz, calibrator.pkl,
  thresholds.json, manifest.json, metadata.json
  (heavy `*.pkl/*.cbm/*.npz` are git-ignored; regenerate with Phase F)

## Result (this run)
Incumbent PR-AUC = **0.3641 ± 0.0550**. All three candidates landed **inside the noise
band** → **FREEZE the incumbent**. See `results/final_recommendation_report.md`.
