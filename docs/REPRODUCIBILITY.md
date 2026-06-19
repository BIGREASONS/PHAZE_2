# ASTraM — Reproducibility Guide

End-to-end, copy-pasteable. A new user can go from a clean machine to a running Command
Center serving the frozen ensemble, **reproducing the validation numbers along the way**,
without any further instructions.

**Pipeline:** clone → install → (train + validate) → export artifacts → launch.

> The model is **frozen**. This guide reproduces the accepted solution; it does not
> explore new models, features, or experiments.

---

## 0. Prerequisites

- **Python 3.12.x** (the exported pickles were produced on 3.12.0; match the minor version
  to avoid scikit-learn unpickling warnings).
- **Git**, ~2 GB free RAM per process, ~1.5 GB disk for the regenerated model artifacts.
- OS-agnostic (developed on Windows + Git Bash; commands below are POSIX shell).
- A CPU is enough to reproduce everything. A CUDA GPU is **optional** and only speeds up
  TabPFN at serving time.

---

## 1. Clone

```bash
git clone https://github.com/BIGREASONS/PHAZE_2.git
cd PHAZE_2
```

**What you get vs. what you must rebuild.** The repo tracks all source, the dataset, and
the three small JSON artifacts (`manifest/metadata/thresholds`). It does **not** track the
heavy model binaries (`*.pkl`, `*.cbm`, `*.npz` — ~270 MB, git-ignored). You regenerate
those in step 4.

```bash
# the dataset ships with the repo at the project root:
ls "Astram event data_anonymized - Astram event data_anonymizedb40ac87.csv"
```

---

## 2. Environment & dependencies

```bash
python -m venv .venv
# Linux/macOS:        source .venv/bin/activate
# Windows (Git Bash): source .venv/Scripts/activate

python -m pip install --upgrade pip

# (a) model stack — pinned to the export environment (train/validate/export/serve)
python -m pip install -r requirements-model.txt

# (b) dashboard stack
#  NOTE: the last line of command_center/requirements.txt is byte-corrupted
#  ("f p d f 2 ..."). Fix it to "fpdf2>=2.7.5" before installing, or fpdf2 is skipped.
python -m pip install -r command_center/requirements.txt
```

**Sanity check the environment matches the artifacts:**
```bash
python - <<'PY'
import numpy, pandas, sklearn, catboost, lightgbm, xgboost, tabpfn, torch
print("numpy", numpy.__version__, "| sklearn", sklearn.__version__,
      "| catboost", catboost.__version__, "| tabpfn", tabpfn.__version__,
      "| torch", torch.__version__)
# expect numpy 2.2.3 | sklearn 1.6.1 | catboost 1.2.10 | tabpfn 2.0.9
PY
```

---

## 3. Reproduce the validation (train + measure)

Run the phases **from inside `final_validation/`** (the scripts import sibling modules and
resolve the dataset at the project root). Phase A trains 7 models × 4 folds and is the slow
step; later phases reuse its cached out-of-fold predictions (`final_validation/cache/`).

```bash
cd final_validation

python final_ensemble_validation.py   # Phase A: rolling-origin folds + incumbent baseline (SLOW)
python rank_vs_probability.py          # Phase B: rank vs probability averaging
python temporal_decay.py               # Phase C: fixed-half-life decay weighting
python dr_tabpfn.py                    # Phase D: DR-TabPFN feasibility + recency proxy
python final_gate.py                   # Phase E: aggregate verdict + recommendation report
```

> ⏱️ **TabPFN on CPU is slow.** Phases A and D refit/score TabPFN over a 4000-row
> in-context set; expect Phase A to take a while on CPU. The boosted-tree members are fast.

**Reproduced outputs (compare against the committed copies):**

| File | Expected headline |
|---|---|
| `results/final_ensemble_validation.json` | incumbent **PR-AUC 0.3641 ± 0.0550**, ROC-AUC 0.7887 ± 0.0391 |
| `results/rank_vs_probability_results.json` | rank-avg margin **−0.0079** → reject |
| `results/temporal_decay_results.json` | decay margin **+0.0009** (< noise) → reject |
| `results/dr_tabpfn_results.json` | proxy margin **+0.0036** (< noise) → reject |
| `results/final_decision.json` | `"final_choice": "incumbent"`, `"adopt_candidate": false` |
| `results/final_recommendation_report.md` | **FREEZE** the 7-model ensemble |

**Quick verification:**
```bash
python - <<'PY'
import json
d = json.load(open("results/final_decision.json"))
assert d["final_choice"] == "incumbent" and d["adopt_candidate"] is False
print("FREEZE confirmed · incumbent PR-AUC =", round(d["incumbent_pr_auc"], 4),
      "±", round(d["incumbent_pr_auc_std"], 4))
PY
```

> **Determinism note.** All learners use fixed seeds (`SEED = 0`) and the boosted-tree /
> linear members reproduce to many decimals. TabPFN on CPU/GPU can vary at the ~1e-3 level;
> this is far below the ±0.055 fold-to-fold noise band, so the **FREEZE verdict is stable**
> even if a TabPFN-touched digit differs.

---

## 4. Export the production artifacts (Phase F)

Still inside `final_validation/`:

```bash
python export_production_model.py
```

This refits every member on **all** rows, stores the fixed TabPFN in-context set, fits the
isotonic calibrator and operating thresholds on **honest pooled out-of-fold** predictions,
and writes everything to `final_validation/artifacts/`.

**Verify the export (expect 12 files):**
```bash
ls -1 artifacts/ | sort
# CatBoost.cbm  ExtraTrees.pkl  LightGBM.pkl  Logistic.pkl  RandomForest.pkl
# XGBoost.pkl  calibrator.pkl  encoder.pkl  manifest.json  metadata.json
# tabpfn_refit.npz  thresholds.json

python - <<'PY'
import json
m = json.load(open("artifacts/manifest.json"))
print("members:", m["members"])           # 7 members
print("combiner:", m["combiner"])          # prob_mean
PY
cd ..   # back to project root
```

---

## 5. Validate the served model (adapter contract)

Before launching the UI, confirm the adapter loads the artifacts and honours the contract
(allow ~4 min on CPU for the TabPFN-backed prediction):

```bash
python - <<'PY'
import sys; sys.path.insert(0, "command_center")
from backend.services.model_adapter import ProductionEnsembleModel
m = ProductionEnsembleModel(); m.load_model()
p = m.predict({"event_type":"unplanned","event_cause":"accident","veh_type":"car",
               "corridor":"ORR","police_station":"Madiwala","zone":"South",
               "latitude":12.9716,"longitude":77.5946,"hour":18,"weekday":2})
assert {"probability","severity","confidence","recommended_action","feature_contributions"} <= set(p)
md = m.get_model_metadata()
assert md["name"] == "ASTraM 7-Model Equal-Weight Ensemble"
print("OK ·", md["name"], "· P(closure)=", p["probability"], "· severity=", p["severity"])
PY
```

---

## 6. Launch the Command Center

Two processes. From the project root:

```bash
# Terminal 1 — FastAPI backend (loads the model once at startup)
cd command_center
uvicorn backend.main:app --host 0.0.0.0 --port 8000

# Terminal 2 — Streamlit dashboard
cd command_center
streamlit run frontend/app.py --server.port 8501
```

- **Dashboard:** http://localhost:8501
- **API docs:** http://localhost:8000/docs

**Confirm the real model is live (not the fallback):**
```bash
curl -s localhost:8000/model-info | python -m json.tool
# "name": "ASTraM 7-Model Equal-Weight Ensemble", "status": "ONLINE", 7 members
```
If it says `PlaceholderModel`, the artifacts or ML deps are missing — re-check steps 2–4
and read the backend log line `ProductionEnsembleModel unavailable (<reason>)`.

> **Fast UI demo without heavy inference:** `ASTRAM_USE_PLACEHOLDER=1` before launching
> serves the mock model instantly (useful for front-end work; not real predictions).

---

## 7. One-shot reproduction script

```bash
#!/usr/bin/env bash
set -euo pipefail
git clone https://github.com/BIGREASONS/PHAZE_2.git && cd PHAZE_2
python -m venv .venv && source .venv/Scripts/activate   # or .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-model.txt
sed -i 's/^f p d f.*/fpdf2>=2.7.5/' command_center/requirements.txt || true
python -m pip install -r command_center/requirements.txt
cd final_validation
python final_ensemble_validation.py
python rank_vs_probability.py
python temporal_decay.py
python dr_tabpfn.py
python final_gate.py
python export_production_model.py
cd ..
echo "Done. Launch: (t1) cd command_center && uvicorn backend.main:app --port 8000"
echo "             (t2) cd command_center && streamlit run frontend/app.py --server.port 8501"
```

---

## 8. Troubleshooting

| Symptom | Fix |
|---|---|
| `sklearn` `InconsistentVersionWarning` | Use scikit-learn **1.6.1** (`requirements-model.txt`); re-run Phase F to re-export if you must change versions |
| `FileNotFoundError: artifacts/ExtraTrees.pkl` | You skipped Phase F (step 4) — the heavy binaries are git-ignored and must be regenerated |
| `/model-info` shows `PlaceholderModel` | ML stack not installed or artifacts missing; see backend log reason; reinstall (step 2) + re-export (step 4) |
| `pip` fails on `fpdf2` | Repair the corrupted line to `fpdf2>=2.7.5` |
| Predictions take minutes | Expected: TabPFN on CPU (~4 min/call). Use a GPU torch build, or `ASTRAM_USE_PLACEHOLDER=1` for UI work |
| `ModuleNotFoundError` running phase scripts | Run them **from inside `final_validation/`**, not the project root |
| Out of memory at load | Provision ≥ 2 GB RAM/process (ExtraTrees + RandomForest are large) |

---

## 9. What "reproduced" means here

You should land on **PR-AUC 0.3641 ± 0.0550** for the incumbent, **all three candidates
rejected inside the ±0.055 noise band**, and a `final_decision.json` that says
`incumbent` / `adopt_candidate: false`. Boosted-tree and linear members reproduce to many
decimals; TabPFN may wobble at ~1e-3, which never changes the verdict. That convergence —
not a single lucky score — is the reproducible result.
