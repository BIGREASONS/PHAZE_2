"""
Phase 2 – Seed-Bagged Ensembles
5 model families × 20 seeds × 5-fold CV each
Produces averaged OOF predictions per family and a comparison table.
"""

import numpy as np
import pandas as pd
import json
import time
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

# ── Data Loading ─────────────────────────────────────────────────────────────
DATA = 'Astram event data_anonymized - Astram event data_anonymizedb40ac87.csv'
df = pd.read_csv(DATA, low_memory=False)
df['t'] = pd.to_datetime(df['created_date'], errors='coerce')
df = df.sort_values('t').reset_index(drop=True)
df['lat'] = pd.to_numeric(df['latitude'], errors='coerce')
df['lon'] = pd.to_numeric(df['longitude'], errors='coerce')
df['hour'] = df['t'].dt.hour.fillna(0).astype(int)
df['weekday'] = df['t'].dt.weekday.fillna(0).astype(int)
y = df['requires_road_closure'].astype(int).values
cats = ['event_type', 'event_cause', 'veh_type', 'corridor', 'police_station', 'zone']
nums = ['lat', 'lon', 'hour', 'weekday']
for c in cats:
    df[c] = df[c].astype(str).fillna('NA')

codes = OrdinalEncoder(
    handle_unknown='use_encoded_value', unknown_value=-1
).fit_transform(df[cats]).astype(int)
X = pd.DataFrame(codes, columns=cats)
for c in nums:
    X[c] = df[c].astype(float).values

cat_idx = list(range(len(cats)))
N = len(y)
N_SEEDS = 20
N_FOLDS = 5

print(f"Dataset: {N} rows | positive rate: {y.mean():.4f}")
print(f"Seeds: {N_SEEDS} | Folds: {N_FOLDS}")
print("=" * 80)

# ── Model Factory ────────────────────────────────────────────────────────────
def make_model(family, seed):
    if family == 'CatBoost':
        return CatBoostClassifier(
            iterations=400, depth=6, learning_rate=0.05,
            verbose=0, random_seed=seed,
            cat_features=cat_idx, thread_count=-1
        )
    elif family == 'LightGBM':
        return LGBMClassifier(
            n_estimators=400, learning_rate=0.05, num_leaves=31,
            random_state=seed, n_jobs=-1, verbose=-1
        )
    elif family == 'RandomForest':
        return RandomForestClassifier(
            n_estimators=500, n_jobs=-1,
            random_state=seed, class_weight='balanced'
        )
    elif family == 'ExtraTrees':
        return ExtraTreesClassifier(
            n_estimators=600, n_jobs=-1,
            random_state=seed, class_weight='balanced'
        )
    elif family == 'XGBoost':
        return XGBClassifier(
            n_estimators=400, learning_rate=0.05, max_depth=6,
            tree_method='hist', random_state=seed, n_jobs=-1,
            eval_metric='logloss', enable_categorical=False
        )
    else:
        raise ValueError(f"Unknown family: {family}")

# ── Single-seed CV runner ────────────────────────────────────────────────────
def run_one_seed(family, seed):
    """Return OOF probability array for one seed."""
    oof = np.zeros(N, dtype=np.float64)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        model = make_model(family, seed)

        if family == 'LightGBM':
            model.fit(X_tr, y_tr, categorical_feature=cat_idx)
        elif family == 'CatBoost':
            model.fit(X_tr, y_tr)          # cat_features set at init
        else:
            model.fit(X_tr, y_tr)

        oof[va_idx] = model.predict_proba(X_va)[:, 1]

    return oof

# ── Main loop ────────────────────────────────────────────────────────────────
families = ['CatBoost', 'LightGBM', 'RandomForest', 'ExtraTrees', 'XGBoost']
results = {}

for family in families:
    print(f"\n{'─' * 80}")
    print(f"  Track: {family}  ({N_SEEDS} seeds × {N_FOLDS} folds)")
    print(f"{'─' * 80}")

    seed_oofs = []
    seed_aucs = []
    seed_praucs = []
    t0 = time.time()

    for seed in range(N_SEEDS):
        oof = run_one_seed(family, seed)
        auc = roc_auc_score(y, oof)
        prauc = average_precision_score(y, oof)
        seed_oofs.append(oof)
        seed_aucs.append(auc)
        seed_praucs.append(prauc)
        print(f"  seed {seed:2d}  ROC-AUC={auc:.5f}  PR-AUC={prauc:.5f}")

    elapsed = time.time() - t0

    # Average across seeds
    avg_oof = np.mean(seed_oofs, axis=0)
    avg_auc = roc_auc_score(y, avg_oof)
    avg_prauc = average_precision_score(y, avg_oof)

    print(f"\n  ▸ Seed-bagged OOF  ROC-AUC = {avg_auc:.5f}")
    print(f"  ▸ Seed-bagged OOF  PR-AUC  = {avg_prauc:.5f}")
    print(f"  ▸ Individual AUC   mean={np.mean(seed_aucs):.5f}  "
          f"std={np.std(seed_aucs):.5f}")
    print(f"  ▸ Time: {elapsed:.1f}s")

    # Save averaged OOF
    np.save(f"oof_seedbag_{family}.npy", avg_oof)

    results[family] = {
        'avg_roc_auc': float(avg_auc),
        'avg_pr_auc': float(avg_prauc),
        'individual_roc_aucs': [float(a) for a in seed_aucs],
        'individual_pr_aucs': [float(a) for a in seed_praucs],
        'mean_individual_auc': float(np.mean(seed_aucs)),
        'std_individual_auc': float(np.std(seed_aucs)),
        'elapsed_seconds': float(elapsed),
    }

# ── Save results JSON ───────────────────────────────────────────────────────
with open('phase2_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\n✓ Saved phase2_results.json")

# ── Comparison Table ─────────────────────────────────────────────────────────
print(f"\n{'=' * 80}")
print(f"  SEED-BAGGED ENSEMBLE COMPARISON  ({N_SEEDS} seeds)")
print(f"{'=' * 80}")
header = f"{'Family':<15} {'ROC-AUC':>10} {'PR-AUC':>10} {'Ind.AUC μ':>10} {'Ind.AUC σ':>10} {'Time(s)':>8}"
print(header)
print("─" * len(header))

best_auc = max(r['avg_roc_auc'] for r in results.values())
for fam in families:
    r = results[fam]
    marker = " ★" if r['avg_roc_auc'] == best_auc else ""
    print(f"{fam:<15} {r['avg_roc_auc']:>10.5f} {r['avg_pr_auc']:>10.5f} "
          f"{r['mean_individual_auc']:>10.5f} {r['std_individual_auc']:>10.5f} "
          f"{r['elapsed_seconds']:>8.1f}{marker}")

print("─" * len(header))
print("★ = best seed-bagged ROC-AUC")
print(f"\nSaved files:")
for fam in families:
    print(f"  oof_seedbag_{fam}.npy")
print("  phase2_results.json")
print("\nDone.")
