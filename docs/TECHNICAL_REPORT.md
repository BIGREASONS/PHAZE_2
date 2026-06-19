# GridSight AI: Shift-Aware Road-Closure Prediction under Severe Temporal Distribution Shift

**A technical report on the modelling, validation, and freeze decision.**

Bengaluru Traffic Police · Flipkart Gridlock 2.0 · GridSight AI road-closure prediction.

---

## Abstract

Given a reported traffic incident, predict whether it **requires a road closure**
(`requires_road_closure ∈ {0,1}`). The dataset is small (8,173 incidents, 8.3 %
positive) and exhibits **severe covariate shift over time**: an adversarial classifier
separates early from late records at **AUC ≈ 0.87**, and the calendar months of the
train and test halves barely overlap (χ² ≈ 5,118, p ≈ 0). Under these conditions, the
question is not "which clever architecture wins" but "which measured improvement is
*real* versus fold-to-fold noise."

We built a **rolling-origin temporal validation harness** (expanding window, 4 folds,
train-only encoders) and adopted a strict gate: a candidate is accepted only if its mean
PR-AUC margin is positive **and** it wins a majority of folds **and** the margin exceeds
the fold-to-fold standard deviation. Under that gate, an **equal-weight 7-model
probability-averaged ensemble** (CatBoost, LightGBM, XGBoost, RandomForest, ExtraTrees,
Logistic Regression, TabPFN) scores **PR-AUC 0.3641 ± 0.0550 / ROC-AUC 0.7887 ± 0.0391**.
Three plausible upgrades — rank averaging, temporal-decay weighting, and a drift-aware
TabPFN proxy as an 8th member — were each tested and **rejected**: every measured gain
fell inside the ±0.055 noise band. Stacking, which won under a random split, **collapsed
under shift**. The ensemble is **frozen** and exported for production, with isotonic
calibration and honest operating thresholds derived from out-of-fold predictions.

---

## 1. Problem statement

- **Task.** Binary classification: does an incident require a road closure?
- **Operational use.** Feed a real-time incident command dashboard (the "GridSight AI Command
  Center") so dispatchers can triage closures, pre-stage diversions, and prioritise
  response. Predictions must be **calibrated** (a "30 % closure risk" should mean 30 %)
  and accompanied by **operating thresholds** for different recall postures.
- **Why it is hard.**
  1. **Class imbalance** — 676 positives in 8,173 rows (8.27 %). ROC-AUC is optimistic
     here; **PR-AUC is the headline metric** and average-precision variance is large.
  2. **Small data** — 8k rows is far too little for bespoke deep architectures; the
     project policy explicitly bans training foundation/graph models on this data.
  3. **Distribution shift** — the dominant difficulty; see §3.
- **Headline metric.** PR-AUC (average precision), reported as mean ± std across temporal
  folds. ROC-AUC reported as a secondary, lower-variance sanity signal.

---

## 2. Dataset description

| Property | Value |
|---|---|
| Rows | **8,173** incidents |
| Positives (`requires_road_closure = 1`) | **676** (**8.27 %**) |
| Time span | **2023-09-29 → 2024-04-08** (~6 months) |
| Raw columns | 46 (IDs, geocoords, timestamps, free text, outcome fields, …) |
| Final modelled features | **10** (6 categorical + 4 numeric) |

**Final feature set (10):**
- **Categorical (6):** `event_type`, `event_cause`, `veh_type`, `corridor`,
  `police_station`, `zone`
- **Numeric (4):** `lat`, `lon`, `hour` (of day), `weekday`

These are the fields **knowable at the moment an incident is reported**, before any
closure decision. The remaining 36 raw columns were dropped as IDs, redundant geometry,
post-outcome fields, or leaky free text (§7).

**Signal sanity check (event cause → closure rate).** The structured `event_cause` field
is legitimately predictive and matches operational intuition:

| event_cause | n | closure rate |
|---|---|---|
| vip_movement | 20 | **0.80** |
| public_event | 84 | 0.46 |
| protest | 15 | 0.40 |
| tree_fall | 284 | 0.39 |
| procession | 72 | 0.26 |
| construction | 480 | 0.26 |
| vehicle_breakdown | 4,896 | 0.04 |
| accident | 365 | 0.03 |
| pot_holes | 537 | 0.02 |

The plurality cause (`vehicle_breakdown`, 60 % of rows) almost never closes a road, while
rare civic events almost always do — a hallmark of an imbalanced, signal-sparse problem.

---

## 3. Distribution-shift discovery

The central empirical finding of the project. Splitting the time-sorted data at its
midpoint and characterising train-vs-test (Phase 1 — `phase1_distribution_discovery.py`):

- **Calendar months are nearly disjoint.** Train half spans months {11, 12, 1, 2} with a
  sliver of 3; the test half is **61 % March, 39 % April** with **zero** Nov–Feb.
  χ²(month) ≈ **5,118**, p ≈ 0. The split is effectively "predict spring from winter."
- **Spatial / categorical drift.** `corridor` χ² ≈ 60.5 (p ≈ 1.9e-5); `event_cause`
  χ² ≈ 424.7 with **3 cause categories present only in the training half**;
  `event_type` χ² ≈ 8.8 (p ≈ 0.003).
- **The label rate is stable.** Train closure rate 0.0818 vs test 0.0862 (Δ = 0.0044).

> **This is covariate shift, not concept drift.** *P(x)* moves substantially while
> *P(y|x)* stays roughly fixed. That distinction drove every later decision: methods that
> assume a stationary feature distribution (e.g. a stacking meta-learner fit on global OOF)
> are exactly the ones expected to degrade out-of-time — and they did (§6).

---

## 4. Adversarial validation findings

To quantify the shift, we trained classifiers to distinguish **train rows from test
rows** (label = "is this a test-half row?"). A high AUC means the two periods are easily
separable, i.e. strong shift.

| Adversarial classifier | AUC (5-fold) |
|---|---|
| RandomForest | 0.8716 ± 0.0050 |
| LightGBM | 0.8669 ± 0.0057 |
| CatBoost | 0.8649 ± 0.0062 |
| **Overall** | **≈ 0.868** |

**Top shift drivers** (features the adversary leaned on most): `zone` (dominant for the
tree models), then `weekday`, `lat`/`lon`, `event_cause`, `police_station`, and the
date-derived fields. In other words, *where* and *when* incidents occur moved the most
between the two periods.

**Implications adopted for the rest of the project:**
1. A random k-fold CV would be **optimistic** — folds would leak future-period structure
   into training. All headline numbers therefore use **rolling-origin temporal CV** (§9).
2. Improvements that look good on a random split must be **re-checked under shift** before
   being believed (§6, stacking).
3. The fold-to-fold variance under temporal CV (PR-AUC std ≈ 0.055) becomes the **noise
   band** that any candidate must beat to be adopted.

---

## 5. Model evolution

The work progressed through scripted phases (`phase1…phase7`, then a dedicated
`final_validation/` gate). The arc:

1. **Single-model baselines.** CatBoost / LightGBM / XGBoost / RandomForest / ExtraTrees /
   Logistic / TabPFN, each tuned lightly and compared. No single family dominates; tree
   ensembles and TabPFN cluster together.
2. **Seed bagging & light Optuna.** Marginal, within-noise movements — recorded, not
   adopted.
3. **Stacking (L2/L3 meta-learners).** Promising under a *random* split; **fails under
   shift** (§6).
4. **Equal-weight probability averaging of all 7 members.** Simple, robust, lowest-variance
   on the temporal folds. Becomes the **incumbent**.
5. **Final gate (`final_validation/`).** Three surviving ideas tested ruthlessly against
   the incumbent under identical rolling-origin folds (§6). All rejected → **FREEZE**.

**Standalone members under rolling-origin temporal CV** (the honest comparison; mean
PR-AUC ± std over 4 folds):

| Member | ROC-AUC | PR-AUC |
|---|---|---|
| CatBoost | 0.7951 ± 0.044 | **0.3711 ± 0.048** |
| TabPFN | 0.7905 ± 0.035 | 0.3631 ± 0.051 |
| RandomForest | 0.7751 ± 0.036 | 0.3322 ± 0.054 |
| Logistic | 0.7639 ± 0.046 | 0.3193 ± 0.040 |
| XGBoost | 0.7645 ± 0.046 | 0.3184 ± 0.049 |
| LightGBM | 0.7569 ± 0.050 | 0.3136 ± 0.051 |
| ExtraTrees | 0.7562 ± 0.032 | 0.3128 ± 0.055 |
| **Equal-weight ensemble** | **0.7887 ± 0.039** | **0.3641 ± 0.055** |

The ensemble matches the best single members (CatBoost, TabPFN) on the mean while
diversifying away their idiosyncratic per-fold failures — the reason it is the incumbent
even though it does not strictly top the table on the point estimate.

---

## 6. Failed experiments (and why "failed" is the point)

Every candidate below was tested on the **same** rolling-origin folds and the **same**
members as the incumbent. The adoption gate: **mean PR-AUC margin > 0 AND majority-fold
win AND margin > noise band (std ≈ 0.055)**. Ties go to the incumbent.

| Candidate | Cand. PR-AUC | Margin vs incumbent | Fold wins | Verdict |
|---|---|---|---|---|
| **B. Rank averaging** | 0.3562 ± 0.0555 | **−0.0079** | prob 4 / rank 0 | ❌ reject (worse, loses every fold) |
| **C. Temporal-decay weighting** (half-life 64 d, fixed a-priori) | 0.3650 ± 0.0557 | +0.0009 | decay 3 / inc 1 | ❌ reject (gain ≪ noise band) |
| **D. DR-TabPFN proxy** (recency-context TabPFN as 8th member) | 0.3678 ± 0.0554 | +0.0036 | 8-mem 4 / 7-mem 0 | ❌ reject (gain ≪ noise band) |

Notes that make these rejections trustworthy rather than convenient:
- **No post-hoc tuning.** The decay half-life (64 d ≈ ⅓ of the time span) was **locked
  before any metric was observed**. Tuning it to the folds would have manufactured a win.
- **Honest about feasibility.** The genuine **Drift-Resilient TabPFN** (Helli et al., 2024)
  is a research fork with a separate checkpoint and is **not pip-installable**
  (`drift-tabpfn` / `dr-tabpfn` absent from PyPI; `tabpfn_extensions` not importable). We
  therefore tested only a clearly-labelled **recency-context proxy** and reported it as
  such — we did **not** claim to have run the published method.

### Stacking collapses under shift — the most instructive negative result

A stacking meta-learner (Logistic/Ridge/LightGBM over base OOF) was evaluated under three
split regimes (`run_stack_shift.py` → `stack_shift_summary.json`):

| Split regime | Best standalone (AP) | Best stack (AP) | Stacking verdict |
|---|---|---|---|
| **Random** (i.i.d.) | RandomForest 0.4230 | Ridge **0.4486** | ✅ helps (+0.026) |
| **Temporal** (out-of-time) | CatBoost 0.3347 | Ridge 0.3215 | ❌ **hurts** (−0.013) |
| **Corridor** (spatial holdout) | XGBoost 0.2917 | Logistic 0.3026 | ~ marginal, high variance |

This is the project's thesis in one table: **the gains that a random split rewards are
precisely the gains that disappear (or invert) under realistic shift.** A meta-learner
overfits the global OOF correlation structure, which does not transport to a future period
whose covariates have moved (AV-AUC 0.87). Equal-weight averaging carries no such fitted
structure and is therefore more robust out-of-time.

---

## 7. Leakage investigation

Two classes of leakage were identified and removed; this is why the model uses only 10
pre-decision structured features.

**7a. Outcome / future-information columns.** The raw schema contains fields populated
*after* a closure is decided or resolved: `resolved_at_*`, `closed_by_id`,
`closed_datetime`, `resolved_*`, `end_datetime`, `status`. Using any of these to predict
the closure decision is target leakage from the future. All were excluded.

**7b. Free-text target keyword leakage (`description`, `comment`).** Operator free-text
notes frequently **state the outcome verbatim** — e.g. *"tree fallen on main road
closed"*, *"service road is closed regarding metro work"*, *"need to be closure"*. A
keyword detector over `description` trivially correlates with the label
(`text_leakage_examples.csv` catalogues confirmed cases, including rows where the text
says "closed" for both positive and *negative* labels — so the text is both leaky **and**
noisy). Any TF-IDF / embedding feature built on this text would inflate offline scores
while learning the dispatcher's own words, not a predictive signal. **All free-text
features were dropped.**

The result is a feature set that is defensible for deployment: every input is available at
incident-report time and none encodes the answer.

---

## 8. Final ensemble architecture

```
incident payload (6 categoricals + lat/lon/hour/weekday)
        │
        ├── raw frame ───────────────► CatBoost (native categorical)        ─┐
        │                              Logistic (one-hot + scaler pipeline)  │
        │                                                                    │
        └── shared OrdinalEncoder ───► LightGBM ┐                            │
            (encoder.pkl) → ordinal    XGBoost   │ ordinal matrix            ├─► P(closure) per member
                            matrix     RandomForest                          │
                                       ExtraTrees                            │
                                       TabPFN (refit on fixed 4000-row       │
                                              in-context support set)        ─┘
                                                                    │
                                       equal-weight mean of 7 probs (prob_mean)
                                                                    │
                                       isotonic calibrator (calibrator.pkl)
                                                                    │
                                       calibrated P(closure) → thresholds.json
```

- **Members (7):** CatBoost, LightGBM, XGBoost, RandomForest, ExtraTrees, Logistic, TabPFN.
- **Combiner:** equal-weight probability average (`prob_mean`), weight = 1/7 each.
- **Calibration:** isotonic regression, fit on **honest pooled rolling-OOF** scores
  (4,087 OOF rows, 369 positives) — never in-sample.
- **Input routing:** CatBoost and Logistic consume the raw frame (their own encoders);
  the four tree members and TabPFN consume a shared ordinal-encoded matrix; TabPFN is
  refit at load on a fixed, class-balanced 4000-row in-context set stored in the artifact.
- **Why equal weight, not learned weight:** learned weights are a stacking variant, and §6
  shows stacking does not transport across the shift. Equal weighting has no fitted
  parameters to overfit the OOF.

**Operating thresholds** (calibrated, from out-of-fold — `thresholds.json`):

| Policy | Threshold | Precision | Recall | F1 |
|---|---|---|---|---|
| Max-F1 | 0.333 | 0.468 | 0.415 | **0.440** |
| Recall ≥ 0.50 | 0.190 | — | 0.50 | — |
| Recall ≥ 0.70 | 0.082 | — | 0.70 | — |
| Recall ≥ 0.90 | 0.033 | — | 0.90 | — |

The dashboard maps these to severity bands (LOW < 0.033 ≤ MEDIUM < 0.082 ≤ HIGH < 0.333 ≤
CRITICAL), so the UI's risk language is tied directly to validated operating points rather
than arbitrary cutoffs.

---

## 9. Validation methodology

**Rolling-origin expanding-window CV** (`final_validation/common.py`):

- Data sorted by `created_date`. Fold *k* trains on the contiguous prefix `[0 : cut_k)`
  and validates on the next contiguous future block `[cut_k : cut_{k+1})`. First cut at
  50 % of the timeline; 4 folds.
- **Train-only encoders.** Every OrdinalEncoder / OneHotEncoder / scaler / model is fit on
  **train rows only, per fold**, then applied to the held-out future block. No target, no
  future row, ever touches `fit()`.
- **No reuse of the repo's random/`TimeSeriesSplit` OOF caches** — those folds differ, and
  mixing them would be a silent apples-to-oranges comparison.

| Fold | n_train | n_val | pos_val |
|---|---|---|---|
| 0 | 4,086 | 1,021 | 88 |
| 1 | 5,107 | 1,022 | 100 |
| 2 | 6,129 | 1,022 | 91 |
| 3 | 7,151 | 1,022 | 90 |

**Adoption policy.** Adopt a candidate **only if** mean PR-AUC margin > 0 **and** it wins a
majority of folds **and** the margin exceeds the fold-to-fold std (noise band ≈ 0.055).
Otherwise reject; ties go to the incumbent. The threshold/calibration fitting uses pooled
OOF predictions, so neither is optimistic.

---

## 10. Final metrics

**Frozen incumbent — equal-weight 7-model probability average:**

- **PR-AUC = 0.3641 ± 0.0550**  (≈ 4.4× the 0.083 base rate)
- **ROC-AUC = 0.7887 ± 0.0391**
- Max-F1 operating point: **F1 0.440 / precision 0.468 / recall 0.415** @ threshold 0.333

| Fold | ROC-AUC | PR-AUC |
|---|---|---|
| 0 | 0.7599 | 0.3886 |
| 1 | 0.7498 | 0.2777 |
| 2 | 0.7951 | 0.3629 |
| 3 | 0.8499 | 0.4275 |

The clear upward trend across folds (later, larger training prefixes score better) is
consistent with the shift story: more recent training data resembles the validation
period more closely. Fold 1 is the hardest block.

**Interpretation.** Under AV-AUC ≈ 0.87 shift with only 676 positives, every candidate
improvement measured (−0.008 to +0.004) sits inside the ±0.055 fold-to-fold band. The
honest conclusion is that we are at the **model ceiling for this dataset**: further gains
require *more or less-shifted data*, not a cleverer combiner. **Decision: FREEZE.**

---

## 11. Deployment architecture

```
┌─────────────────────────────┐        ┌──────────────────────────────┐
│  Streamlit dashboard         │        │  FastAPI backend             │
│  (8 pages, dark UI)          │        │  /health /predict /explain   │
│  imports get_model()         │        │  /model-info /incidents ...  │
└──────────────┬──────────────┘        └───────────────┬──────────────┘
               │  ModelInterface                        │  ModelInterface
               └────────────────┬───────────────────────┘
                                ▼
              backend/services/model_adapter.py
              ┌──────────────────────────────────────────────┐
              │ get_model()  → process-wide singleton         │
              │   ProductionEnsembleModel  (real, default)    │
              │   └ load: encoder, 6 model files, calibrator, │
              │     thresholds; refit TabPFN once (0.3 s)     │
              │   predict / predict_batch / explain / meta    │
              │   (graceful fallback → PlaceholderModel)      │
              └──────────────────────────────────────────────┘
                                │ reads
                                ▼
              final_validation/artifacts/  (Phase-F export)
```

- **One adapter, two consumers.** Both the FastAPI service and the Streamlit app talk to
  the same `ModelInterface`; neither knows which model is behind it. Swapping models is a
  one-line `get_model()` change, already wired.
- **Self-describing artifacts.** The adapter reads `manifest.json` for member list,
  combiner, and per-member input routing — the ensemble definition is data, not code.
- **Honest fallback.** If artifacts or the ML stack are absent, `get_model()` serves a
  clearly-labelled `PlaceholderModel` so the system still boots; `/model-info` reveals
  which model is live.
- **Calibrated outputs + operating thresholds** flow straight from the artifacts into the
  dashboard's severity bands.
- **Known operational property:** TabPFN on CPU costs ~4 min per prediction (independent of
  batch size — the cost is the 4000-row in-context pass), so `explain()` batches its
  ablation into a single call, `predict_batch` scores all rows at once, identical
  single-incident predictions are cached, and GPU serving is recommended for interactive
  latency. Full operational detail in `DEPLOYMENT_CHECKLIST.md`.

---

## 12. What we would and would not claim

- **We claim:** a rigorously validated, calibrated, shift-aware baseline that is honest
  about its ceiling; reproducible artifacts; and a deployment path with documented
  failure modes.
- **We do not claim:** a state-of-the-art score, a beaten leaderboard, or that any of the
  three rejected ideas "would have worked with more tuning." Under this much shift and this
  few positives, the disciplined result is to **stop, measure, and freeze** — which is what
  we did.

---

### Appendix — provenance

All numbers above are emitted by the scripts in `final_validation/` and `phase*` and
stored as JSON next to the report:
`final_validation/results/{final_ensemble_validation,rank_vs_probability_results,temporal_decay_results,dr_tabpfn_results,final_decision}.json`,
`phase1_results.json`, `phase7_tournament_results.json`, `stack_shift_summary.json`,
and the exported `final_validation/artifacts/{manifest,metadata,thresholds}.json`.
Export environment: Python 3.12.0, numpy 2.2.3, pandas 2.2.3, scikit-learn 1.6.1,
catboost 1.2.10, lightgbm 4.6.0, xgboost 3.1.2, tabpfn 2.0.9.
