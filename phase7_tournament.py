"""
Phase 7 – FINAL TOURNAMENT
===========================
Loads ALL OOF artifacts from phases 2-6, scores each on ROC-AUC and PR-AUC,
produces a unified ranking, and recommends the best model for submission.

Saves: phase7_tournament_results.json
"""

import os, json, warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

# ── Ground truth ────────────────────────────────────────────────────────────
DATA = 'Astram event data_anonymized - Astram event data_anonymizedb40ac87.csv'
df = pd.read_csv(DATA, low_memory=False)
df['t'] = pd.to_datetime(df['created_date'], errors='coerce')
df = df.sort_values('t').reset_index(drop=True)
y = df['requires_road_closure'].astype(int).values
N = len(df)
print(f"Ground truth loaded: N={N}  pos_rate={y.mean():.4f}\n")

# ── OOF manifest ───────────────────────────────────────────────────────────
# (filename, display_name, phase)
OOF_MANIFEST = [
    # Phase 2 – seed-bagged
    ("oof_seedbag_CatBoost.npy",      "SeedBag CatBoost",      "P2-SeedBag"),
    ("oof_seedbag_LightGBM.npy",      "SeedBag LightGBM",      "P2-SeedBag"),
    ("oof_seedbag_XGBoost.npy",       "SeedBag XGBoost",       "P2-SeedBag"),
    ("oof_seedbag_RandomForest.npy",  "SeedBag RandomForest",  "P2-SeedBag"),
    ("oof_seedbag_ExtraTrees.npy",    "SeedBag ExtraTrees",    "P2-SeedBag"),
    # Phase 3 – stacking
    ("oof_stack_l2_Logistic.npy",     "Stack L2 Logistic",     "P3-Stack"),
    ("oof_stack_l2_Ridge.npy",        "Stack L2 Ridge",        "P3-Stack"),
    ("oof_stack_l2_LightGBM.npy",     "Stack L2 LightGBM",    "P3-Stack"),
    ("oof_stack_l3.npy",              "Stack L3",              "P3-Stack"),
    # Single models (random-seed CV)
    ("oof_random_CatBoost.npy",       "Single CatBoost",       "P1-Single"),
    ("oof_random_LightGBM.npy",       "Single LightGBM",       "P1-Single"),
    ("oof_random_XGBoost.npy",        "Single XGBoost",        "P1-Single"),
    ("oof_random_RandomForest.npy",   "Single RandomForest",   "P1-Single"),
    ("oof_random_ExtraTrees.npy",     "Single ExtraTrees",     "P1-Single"),
    ("oof_random_Logistic.npy",       "Single Logistic",       "P1-Single"),
    ("oof_tabpfn_random.npy",         "TabPFN",                "P1-Single"),
    # Phase 6 – Optuna-tuned
    ("oof_optuna_catboost.npy",       "Optuna CatBoost",       "P6-Optuna"),
    ("oof_optuna_lgbm.npy",           "Optuna LightGBM",       "P6-Optuna"),
    ("oof_optuna_xgb.npy",            "Optuna XGBoost",        "P6-Optuna"),
]

# ── Score every available OOF ───────────────────────────────────────────────
results = []
for fname, name, phase in OOF_MANIFEST:
    if not os.path.exists(fname):
        warnings.warn(f"[SKIP] {fname} not found")
        continue

    oof = np.load(fname)
    if oof.shape[0] != N:
        warnings.warn(f"[SKIP] {fname}: length {oof.shape[0]} != N={N}")
        continue

    # score only on non-NaN indices
    mask = ~np.isnan(oof)
    n_valid = int(mask.sum())
    coverage = n_valid / N

    if n_valid == 0:
        warnings.warn(f"[SKIP] {fname}: all NaN")
        continue

    y_sub = y[mask]
    p_sub = oof[mask]

    # need both classes present
    if len(np.unique(y_sub)) < 2:
        warnings.warn(f"[SKIP] {fname}: only one class in valid indices")
        continue

    roc = roc_auc_score(y_sub, p_sub)
    pr  = average_precision_score(y_sub, p_sub)

    results.append({
        "model":    name,
        "phase":    phase,
        "file":     fname,
        "roc_auc":  round(roc, 6),
        "pr_auc":   round(pr, 6),
        "coverage": round(coverage, 4),
        "n_valid":  n_valid,
    })
    print(f"  [OK] {name:<25s}  ROC={roc:.5f}  PR={pr:.5f}  cov={coverage:.2%}")

print(f"\n{'='*70}")
print(f"  Loaded {len(results)} / {len(OOF_MANIFEST)} OOF artifacts")
print(f"{'='*70}\n")

if len(results) == 0:
    print("WARNING: No OOF files found - nothing to rank.")
    raise SystemExit(1)

# ── Build ranking table ────────────────────────────────────────────────────
res_df = pd.DataFrame(results).sort_values("pr_auc", ascending=False).reset_index(drop=True)
res_df.index += 1
res_df.index.name = "Rank"

print("=" * 80)
print("  FINAL TOURNAMENT RANKING  (sorted by PR-AUC desc)")
print("=" * 80)
cols = ["model", "phase", "roc_auc", "pr_auc", "coverage"]
print(res_df[cols].to_string())
print()

# ── Recommendation ─────────────────────────────────────────────────────────
top3 = res_df.head(3)
best = res_df.iloc[0]

print("=" * 80)
print("  >>> RECOMMENDATION <<<")
print("=" * 80)
print()
print("  Top 3 models by PR-AUC:")
for i, (_, row) in enumerate(top3.iterrows(), 1):
    medal = ["#1", "#2", "#3"][i - 1]
    print(f"    {medal}  {row['model']:<25s}  PR-AUC={row['pr_auc']:.5f}  ROC-AUC={row['roc_auc']:.5f}  ({row['phase']})")
print()
print(f"  ==> If the leaderboard closed today, submit: {best['model']}")
print(f"     (PR-AUC = {best['pr_auc']:.5f},  ROC-AUC = {best['roc_auc']:.5f})")
print()

# ── Also show best by ROC-AUC ──────────────────────────────────────────────
roc_best = res_df.sort_values("roc_auc", ascending=False).iloc[0]
if roc_best['model'] != best['model']:
    print(f"  [NOTE] Best by ROC-AUC: {roc_best['model']}  (ROC={roc_best['roc_auc']:.5f})")
    print()

# ── Per-phase summary ──────────────────────────────────────────────────────
print("=" * 80)
print("  PER-PHASE BEST")
print("=" * 80)
for phase in res_df['phase'].unique():
    sub = res_df[res_df['phase'] == phase].iloc[0]
    print(f"    {phase:<12s}  {sub['model']:<25s}  PR={sub['pr_auc']:.5f}  ROC={sub['roc_auc']:.5f}")
print()

# ── Save JSON ──────────────────────────────────────────────────────────────
out = {
    "tournament_results": results,
    "ranking": res_df[cols].reset_index().to_dict(orient="records"),
    "recommendation": {
        "best_model":    best["model"],
        "best_pr_auc":   best["pr_auc"],
        "best_roc_auc":  best["roc_auc"],
        "best_phase":    best["phase"],
        "best_file":     best["file"],
    },
    "top3": [
        {"rank": i + 1, "model": row["model"], "pr_auc": row["pr_auc"], "roc_auc": row["roc_auc"]}
        for i, (_, row) in enumerate(top3.iterrows())
    ],
    "n_models_scored": len(results),
    "n_models_total":  len(OOF_MANIFEST),
    "N": N,
}

with open("phase7_tournament_results.json", "w") as f:
    json.dump(out, f, indent=2)

print("SAVED: phase7_tournament_results.json")
print("TOURNAMENT COMPLETE.")
