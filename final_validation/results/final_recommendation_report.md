# ASTraM — Final Recommendation Report

_Last-chance-before-freeze validation. Rolling-origin expanding-window CV; all base learners refit per fold on train rows only (no future leakage); every candidate uses the identical folds and members as the incumbent._

## 0. Setup

- Dataset: **N=8173**, positives **676** (**0.0827**), dates 2023-09-29 → 2024-04-08
- Folds: **4** expanding-window, first cut at 50% of the timeline
- Members: CatBoost, LightGBM, XGBoost, RandomForest, ExtraTrees, Logistic, TabPFN
- Incumbent combiner: **equal-weight probability average**

**Decision policy** — adopt a candidate ONLY if mean PR-AUC margin > 0 AND it wins a majority of folds AND the margin exceeds the fold-to-fold std (noise band). Otherwise reject; ties go to the incumbent.

## 1. Frozen incumbent (baseline)

| fold | n_val | pos_val | ROC-AUC | PR-AUC |
|---|---|---|---|---|
| 0 | 1021 | 88 | 0.7599 | 0.3886 |
| 1 | 1022 | 100 | 0.7498 | 0.2777 |
| 2 | 1022 | 91 | 0.7951 | 0.3629 |
| 3 | 1022 | 90 | 0.8499 | 0.4275 |

- **Incumbent PR-AUC = 0.3641 ± 0.0550**, ROC-AUC = 0.7887 ± 0.0391
- Fold-to-fold PR-AUC noise band (std) = **0.0550** — the bar every candidate must clear.

## 2. Candidate experiments

### B — Rank averaging vs probability averaging

**Incumbent per-fold**

| fold | n_val | pos_val | ROC-AUC | PR-AUC |
|---|---|---|---|---|
| 0 | 1021 | 88 | 0.7599 | 0.3886 |
| 1 | 1022 | 100 | 0.7498 | 0.2777 |
| 2 | 1022 | 91 | 0.7951 | 0.3629 |
| 3 | 1022 | 90 | 0.8499 | 0.4275 |

**Candidate per-fold**

| fold | n_val | pos_val | ROC-AUC | PR-AUC |
|---|---|---|---|---|
| 0 | 1021 | 88 | 0.7683 | 0.3826 |
| 1 | 1022 | 100 | 0.7556 | 0.2742 |
| 2 | 1022 | 91 | 0.7920 | 0.3433 |
| 3 | 1022 | 90 | 0.8539 | 0.4249 |

- Incumbent PR-AUC: **0.3641 ± 0.0550**  (ROC-AUC 0.7887)
- Candidate PR-AUC: **0.3562 ± 0.0555**  (ROC-AUC 0.7924)
- Mean PR-AUC margin: **-0.0079**  |  noise band (std): **0.0555**
- Fold wins: `{'prob': 4, 'rank': 0}`  |  majority winner: **prob**
- Robust improvement over incumbent: **False**
- Verdict: **KEEP prob-avg (incumbent) -- rank-avg gain within noise / not majority**

### C — Temporal decay weighting (half-life 64d, fixed)

Decay-weighted members: CatBoost, LightGBM, XGBoost; tuning: none -- half-life locked a-priori at ~1/3 of span before any metric seen

**Incumbent per-fold**

| fold | n_val | pos_val | ROC-AUC | PR-AUC |
|---|---|---|---|---|
| 0 | 1021 | 88 | 0.7599 | 0.3886 |
| 1 | 1022 | 100 | 0.7498 | 0.2777 |
| 2 | 1022 | 91 | 0.7951 | 0.3629 |
| 3 | 1022 | 90 | 0.8499 | 0.4275 |

**Candidate per-fold**

| fold | n_val | pos_val | ROC-AUC | PR-AUC |
|---|---|---|---|---|
| 0 | 1021 | 88 | 0.7625 | 0.3890 |
| 1 | 1022 | 100 | 0.7502 | 0.2809 |
| 2 | 1022 | 91 | 0.7922 | 0.3565 |
| 3 | 1022 | 90 | 0.8480 | 0.4336 |

- Incumbent PR-AUC: **0.3641 ± 0.0550**  (ROC-AUC 0.7887)
- Candidate PR-AUC: **0.3650 ± 0.0557**  (ROC-AUC 0.7882)
- Mean PR-AUC margin: **+0.0009**  |  noise band (std): **0.0557**
- Fold wins: `{'incumbent': 1, 'decay': 3}`  |  majority winner: **decay**
- Robust improvement over incumbent: **False**
- Verdict: **KEEP incumbent -- decay gain within noise / not majority**

### D — DR-TabPFN (proxy, as 8th member)

Genuine DR-TabPFN feasible: **False**. Blockers: PyPI: no installable distribution for 'drift-tabpfn'; PyPI: no installable distribution for 'dr-tabpfn'; python: 'tabpfn_extensions' (hosts drift utilities) not importable. Reported result uses a clearly-labelled recency-context TabPFN **proxy**.

**Incumbent per-fold**

| fold | n_val | pos_val | ROC-AUC | PR-AUC |
|---|---|---|---|---|
| 0 | 1021 | 88 | 0.7599 | 0.3886 |
| 1 | 1022 | 100 | 0.7498 | 0.2777 |
| 2 | 1022 | 91 | 0.7951 | 0.3629 |
| 3 | 1022 | 90 | 0.8499 | 0.4275 |

**Candidate per-fold**

| fold | n_val | pos_val | ROC-AUC | PR-AUC |
|---|---|---|---|---|
| 0 | 1021 | 88 | 0.7631 | 0.3968 |
| 1 | 1022 | 100 | 0.7512 | 0.2794 |
| 2 | 1022 | 91 | 0.7992 | 0.3667 |
| 3 | 1022 | 90 | 0.8498 | 0.4281 |

- Incumbent PR-AUC: **0.3641 ± 0.0550**  (ROC-AUC 0.7887)
- Candidate PR-AUC: **0.3678 ± 0.0554**  (ROC-AUC 0.7908)
- Mean PR-AUC margin: **+0.0036**  |  noise band (std): **0.0554**
- Fold wins: `{'7-member': 0, '8-member': 4}`  |  majority winner: **8-member**
- Robust improvement over incumbent: **False**
- Verdict: **KEEP 7-member incumbent -- proxy gain within noise / not majority**

## 3. Summary leaderboard

| Experiment | Cand. PR-AUC (mean±std) | Margin vs incumbent | Noise band | Majority winner | Robust? | Run verdict |
|---|---|---|---|---|---|---|
| B Rank avg | 0.3562±0.0555 | -0.0079 | 0.0555 | prob | False | KEEP prob-avg (incumbent) -- rank-avg gain within noise / not majority |
| C Temporal decay | 0.3650±0.0557 | +0.0009 | 0.0557 | decay | False | KEEP incumbent -- decay gain within noise / not majority |
| D DR-TabPFN proxy | 0.3678±0.0554 | +0.0036 | 0.0554 | 8-member | False | KEEP 7-member incumbent -- proxy gain within noise / not majority |

## 4. Final recommendation

**FREEZE the equal-weight 7-model probability-averaged ensemble.**

No candidate cleared the noise band on a majority of folds. Under AV-AUC≈0.87 shift and only 676 positives, every measured improvement is within fold-to-fold variance. Per policy, ties go to the incumbent. The three checks did their job: they confirm the freeze rather than overturn it. Phase F exports the incumbent.
