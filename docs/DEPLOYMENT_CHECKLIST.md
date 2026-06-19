# GridSight AI — Production Deployment Checklist

**Model:** Frozen equal-weight 7-model probability-averaged ensemble (incumbent).
**Status:** FROZEN — validation gate complete, no further model changes.
**Adapter:** `command_center/backend/services/model_adapter.py → ProductionEnsembleModel`
**Served via:** `get_model()` (process-wide singleton, graceful fallback to `PlaceholderModel`).

This is an operational checklist. Work top to bottom; every box has a concrete
verification command or expected output.

---

## 0. TL;DR for the on-call engineer

1. Artifacts must exist in `final_validation/artifacts/` (they are **git-ignored** — see §1).
2. The serving environment needs the **ML stack** (catboost/lightgbm/xgboost/tabpfn/torch),
   not just the dashboard requirements (§2).
3. **TabPFN on CPU is ~4 min per prediction** (measured). This is a property of the
   frozen model, not a bug. Serve on GPU for interactive latency, or rely on the
   built-in prediction cache + async usage (§4, §5).
4. If anything is missing, the system **still boots** on `PlaceholderModel` and the
   `/health` + `/model-info` endpoints will show it. Never assume "it started" means
   "the real model loaded" — verify with §5.

---

## 1. Required files

### 1a. Model artifacts — `final_validation/artifacts/`

| File | Size | Tracked in git? | Required for inference |
|---|---|---|---|
| `manifest.json` | ~1.4 KB | ✅ yes | ✅ ensemble definition / input routing |
| `metadata.json` | ~1.2 KB | ✅ yes | ✅ provenance, versions, metrics |
| `thresholds.json` | ~0.5 KB | ✅ yes | ✅ calibrated operating points |
| `encoder.pkl` | ~3.7 KB | ❌ **git-ignored** | ✅ shared OrdinalEncoder + feature spec |
| `calibrator.pkl` | ~1.2 KB | ❌ **git-ignored** | ✅ isotonic calibrator |
| `CatBoost.cbm` | ~0.85 MB | ❌ **git-ignored** | ✅ member |
| `LightGBM.pkl` | ~1.4 MB | ❌ **git-ignored** | ✅ member |
| `XGBoost.pkl` | ~1.1 MB | ❌ **git-ignored** | ✅ member |
| `RandomForest.pkl` | ~66 MB | ❌ **git-ignored** | ✅ member |
| `ExtraTrees.pkl` | ~206 MB | ❌ **git-ignored** | ✅ member |
| `Logistic.pkl` | ~7.5 KB | ❌ **git-ignored** | ✅ member |
| `tabpfn_refit.npz` | ~85 KB | ❌ **git-ignored** | ✅ TabPFN in-context training set |

> ⚠️ **The heavy binaries are NOT in the repository.** `final_validation/.gitignore`
> excludes `artifacts/*.pkl`, `*.cbm`, `*.npz` (ExtraTrees + RandomForest alone are
> ~270 MB, over GitHub's limit). A fresh clone has only the three JSON files.
> **Regenerate the binaries before deploying** (see §3, step 0, and `REPRODUCIBILITY.md`).

**Verify presence:**
```bash
ls -1 final_validation/artifacts/ | sort
# expect 12 files: 3 JSON + encoder.pkl + calibrator.pkl + 6 model files + tabpfn_refit.npz
```

### 1b. Application code

- `command_center/backend/` — FastAPI app + adapter + data service
- `command_center/frontend/` — Streamlit multi-page dashboard
- Source dataset CSV at the path in `command_center/backend/config.py:DATA_PATH`
  (default: `astram_review_bundle_v2/review_bundle/Astram event data_anonymized…csv`).
  Used by the **dashboard's DataService**, not by the model adapter.

---

## 2. Runtime dependencies

### 2a. Model-serving stack (pinned to the export environment — `metadata.json`)

The adapter loads pickles produced by these exact versions. **Match them** — sklearn
in particular will warn or fail to unpickle across minor versions.

```
python        == 3.12.0
numpy         == 2.2.3
pandas        == 2.2.3
scikit-learn  == 1.6.1
catboost      == 1.2.10
lightgbm      == 4.6.0
xgboost       == 3.1.2
tabpfn        == 2.0.9
torch         >= 2.9   (CPU build is sufficient; GPU build strongly recommended — see §4)
joblib        >= 1.5
```

### 2b. Dashboard stack — `command_center/requirements.txt`

`fastapi`, `uvicorn`, `streamlit`, `pandas`, `numpy`, `plotly`, `pydeck`, `scipy`,
`openpyxl`, `python-multipart`, `websockets`, `fpdf2`.

> 🐞 **Known issue:** the last line of `command_center/requirements.txt` is byte-corrupted
> (`f p d f 2 > = 2 . 7 . 5` — spaced characters from a bad encoding append). `pip install -r`
> may fail or silently skip `fpdf2` (used only by the PDF export in `pdf_generator.py`).
> Fix the line to `fpdf2>=2.7.5` before installing. Tracked separately from the model freeze.

> ⚠️ **The dashboard requirements do NOT include the model-serving stack (§2a).**
> Installing only `command_center/requirements.txt` yields a system that boots but
> **silently falls back to `PlaceholderModel`** (import of catboost/tabpfn fails inside
> `load_model`). Install both requirement sets to serve the real ensemble.

### 2c. Docker note

`command_center/Dockerfile` uses `python:3.11-slim` and installs only the dashboard
requirements. As written it will (a) run on Python 3.11 not 3.12, and (b) ship without
the ML stack and without the `final_validation/artifacts/` binaries (build context is
`command_center/`). To serve the real model in a container you must:
- base on `python:3.12-slim` to match the pickles,
- install the §2a stack,
- copy `final_validation/artifacts/` into the image (or mount a volume / set
  `ASTRAM_ARTIFACT_DIR`).
Until then, the container serves `PlaceholderModel` by design (it still boots).

---

## 3. Startup sequence

```
0. (fresh clone only) Regenerate artifacts:
     cd final_validation
     python final_ensemble_validation.py     # rebuilds rolling-OOF cache (slow)
     python export_production_model.py        # writes artifacts/  (full-data refit)

1. Install deps:
     pip install -r requirements-model.txt    # §2a, pinned
     pip install -r command_center/requirements.txt   # §2b (fix fpdf2 line first)

2. Start backend (loads the model once at lifespan startup):
     cd command_center
     uvicorn backend.main:app --host 0.0.0.0 --port 8000

3. Start frontend (separate process; shares the singleton within its own process):
     streamlit run frontend/app.py --server.port 8501

4. Validate the load (see §5) BEFORE announcing readiness.
```

**Load-order facts:**
- `get_model()` builds the model **once per process** and caches it (module-level
  singleton). The FastAPI backend and the Streamlit server are **separate processes**,
  so each pays the load cost (~270 MB read + TabPFN refit ≈ 0.3 s) once.
- `load_model()` is **idempotent** — calling it again is a no-op.
- First load reads ~270 MB from disk. Provision RAM accordingly (§4).

---

## 4. Resource & latency profile (measured on CPU)

| Operation | Cost | Notes |
|---|---|---|
| Artifact load (all 7 members) | a few seconds | ~270 MB read; ExtraTrees dominates |
| TabPFN refit at load | **0.3 s** | from `tabpfn_refit.npz` (4000×10 context) |
| **`predict()` single incident** | **~235–242 s** | dominated by TabPFN CPU forward pass |
| `predict()` warm (same incident) | **~0 s** | served from the adapter's predict cache |
| `explain()` single incident | **~230 s** | 11-row ablation **batched into one call** |
| `predict_batch([N])` | ≈ one `predict()` | TabPFN cost is per-call, not per-row |

**Key engineering facts (verified, not estimated):**
- TabPFN cost is **independent of query batch size** — predicting 1 row and 11 rows both
  take ~230 s on CPU. This is why `explain()` batches the actual incident + one
  baselined variant per feature into a *single* ensemble call, and why `predict_batch`
  scores all rows in one shot.
- Warm latency ≈ cold latency: the ~4 min is the in-context transformer pass over the
  4000-row support set × `n_estimators=3`, **not** a startup artifact.
- This latency is a property of the **frozen** model. It is **not** tunable without
  changing predictions (context size and `n_estimators` are part of the frozen spec).

**Memory:** budget **≥ 2 GB RAM per process** (ExtraTrees + RandomForest in memory,
plus TabPFN + torch). Two processes (API + dashboard) ⇒ ≥ 4 GB.

**Latency mitigation (operational, model unchanged):**
- **GPU** is the intended TabPFN deployment — predictions drop to ~1–2 s. Install a
  CUDA torch build; the adapter passes `device="cpu"` today, so set up GPU serving
  deliberately if required.
- The adapter **caches identical single-incident predictions** (bounded LRU, 512
  entries) — re-selecting the same incident in the dashboard is instant.
- For a **fast UI smoke test without real inference**, set `ASTRAM_USE_PLACEHOLDER=1`.

---

## 5. Model-loading validation (do this every deploy)

**5a. Health endpoint**
```bash
curl -s localhost:8000/health
# {"status":"healthy","model_loaded":true,"data_rows":8173,"timestamp":...}
```

**5b. Confirm the REAL model loaded (not the fallback)**
```bash
curl -s localhost:8000/model-info
```
Expect:
- `"name": "GridSight AI 7-Model Equal-Weight Ensemble"`  ← real model
- `"status": "ONLINE"`
- `"members": ["CatBoost","LightGBM","XGBoost","RandomForest","ExtraTrees","Logistic","TabPFN"]`
- `"metrics": {"roc_auc":0.7887,"pr_auc":0.3641,...}`

> If you instead see `"name": "PlaceholderModel"`, the real model **did not load**.
> Check the backend log for `ProductionEnsembleModel unavailable (...)` — the bracketed
> reason is the exact exception (missing artifact, missing dependency, version mismatch).

**5c. End-to-end smoke prediction** (allow ~4 min on CPU)
```bash
curl -s -X POST localhost:8000/predict -H 'content-type: application/json' \
  -d '{"event_type":"unplanned","event_cause":"accident","veh_type":"car",
       "corridor":"ORR","police_station":"Madiwala","zone":"South",
       "latitude":12.9716,"longitude":77.5946,"hour":18,"weekday":2}'
# expect: {"probability":<0..1>,"confidence":...,"severity":"LOW|MEDIUM|HIGH|CRITICAL",
#          "recommended_action":...,"feature_contributions":{...},
#          "model":"ProductionEnsembleModel",...}
```

**5d. Offline contract check** (no server; prints PASS/FAIL):
```bash
python - <<'PY'
import sys; sys.path.insert(0, "command_center")
from backend.services.model_adapter import ProductionEnsembleModel
m = ProductionEnsembleModel(); m.load_model()
p = m.predict({"event_type":"unplanned","event_cause":"accident","veh_type":"car",
               "corridor":"ORR","police_station":"Madiwala","zone":"South",
               "latitude":12.97,"longitude":77.59,"hour":18,"weekday":2})
assert {"probability","severity","confidence","recommended_action","feature_contributions"} <= set(p)
assert 0 <= p["probability"] <= 1 and p["severity"] in ("LOW","MEDIUM","HIGH","CRITICAL")
print("PASS", p["probability"], p["severity"])
PY
```

---

## 6. Failure modes & responses

| Symptom | Likely cause | Response |
|---|---|---|
| `/model-info` shows `PlaceholderModel` | Artifacts missing OR ML deps not installed OR load exception | Read backend log line `ProductionEnsembleModel unavailable (<reason>)`; fix per the reason; restart |
| `FileNotFoundError: …artifacts/ExtraTrees.pkl` | Heavy binaries never regenerated after clone | Run §3 step 0 (`export_production_model.py`) |
| `InconsistentVersionWarning` / unpickfrom sklearn | Serving sklearn ≠ 1.6.1 | Pin sklearn to 1.6.1 (§2a); re-export if you must move versions |
| `predict()` "hangs" for minutes | TabPFN CPU forward pass (expected, §4) | Not a fault. Use GPU, the cache, or `ASTRAM_USE_PLACEHOLDER=1` for demos |
| `ModuleNotFoundError: catboost/tabpfn/torch` | Only dashboard reqs installed | Install §2a stack |
| OOM at load | < 2 GB RAM/process | Increase memory; ExtraTrees+RF need headroom |
| `pip install` fails on `fpdf2` | Corrupted requirements line (§2b) | Repair to `fpdf2>=2.7.5` |
| Unknown category at inference (e.g. new corridor) | Covariate not seen in training | **Handled** — OrdinalEncoder maps unknowns to code `-1`; no crash, prediction proceeds |
| Missing categorical key in payload (e.g. page-2 omits `veh_type`) | Partial payload | **Handled** — adapter fills missing cats with `"NA"`, missing nums with `0` |

---

## 7. Pre-flight sign-off

- [ ] All 12 artifact files present in `final_validation/artifacts/` (§1a)
- [ ] `metadata.json` package versions match the serving environment (§2a)
- [ ] `command_center/requirements.txt` `fpdf2` line repaired (§2b)
- [ ] Backend starts; `/health` → `model_loaded: true` (§5a)
- [ ] `/model-info` → name = **GridSight AI 7-Model Equal-Weight Ensemble**, status **ONLINE** (§5b)
- [ ] Smoke `/predict` returns a valid schema (§5c) — operator briefed on ~4 min CPU latency
- [ ] RAM ≥ 2 GB/process provisioned (§4); GPU decision made for interactive use
- [ ] Decision recorded: serve real model (default) **or** `ASTRAM_USE_PLACEHOLDER=1` (demo only)
```
