"""
Phase 1 – Distribution Discovery
=================================
D1: Adversarial Validation  (CatBoost / LightGBM / RandomForest)
D2: Distribution Shift Report (chi-squared tests on categoricals + date bins)
"""

import json, warnings, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from scipy.stats import chi2_contingency

warnings.filterwarnings("ignore")
np.random.seed(42)

# ──────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────
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

print(f"Dataset: {len(df)} rows,  target-rate: {y.mean():.4f}")

# ──────────────────────────────────────────────
# Split boundary: first 80 % vs last 20 %
# ──────────────────────────────────────────────
split_idx = int(len(df) * 0.8)
adv_label = np.zeros(len(df), dtype=int)
adv_label[split_idx:] = 1
print(f"Split index: {split_idx}  |  train-like: {(adv_label==0).sum()}  |  test-like: {(adv_label==1).sum()}")

# ──────────────────────────────────────────────
# Feature matrix (OrdinalEncoder for cats)
# ──────────────────────────────────────────────
oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X_cats = oe.fit_transform(df[cats])
X_nums = df[nums].fillna(0).values
X = np.hstack([X_cats, X_nums])
feature_names = cats + nums

# Separate DataFrame for CatBoost (needs str categoricals, not floats)
X_cb = df[cats + nums].copy()
for c in cats:
    X_cb[c] = X_cb[c].astype(str)
for c in nums:
    X_cb[c] = X_cb[c].fillna(0).astype(float)

print(f"Feature matrix shape: {X.shape}")
print()

# ======================================================================
# EXPERIMENT D1 – Adversarial Validation
# ======================================================================
print("=" * 70)
print("  EXPERIMENT D1 – Adversarial Validation")
print("=" * 70)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

classifiers = {
    'CatBoost': lambda: CatBoostClassifier(
        iterations=300, depth=6, learning_rate=0.05,
        verbose=0, random_seed=42, eval_metric='AUC'
    ),
    'LightGBM': lambda: LGBMClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        random_state=42, verbose=-1, n_jobs=-1
    ),
    'RandomForest': lambda: RandomForestClassifier(
        n_estimators=300, max_depth=12, random_state=42, n_jobs=-1
    ),
}

d1_results = {}

for clf_name, clf_factory in classifiers.items():
    fold_aucs = []
    fold_importances = np.zeros(len(feature_names))

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, adv_label), 1):
        clf = clf_factory()

        if clf_name == 'CatBoost':
            clf.fit(X_cb.iloc[tr_idx], adv_label[tr_idx], cat_features=cats)
            preds = clf.predict_proba(X_cb.iloc[va_idx])[:, 1]
        else:
            clf.fit(X[tr_idx], adv_label[tr_idx])
            if hasattr(clf, 'predict_proba'):
                preds = clf.predict_proba(X[va_idx])[:, 1]
            else:
                preds = clf.predict(X[va_idx])

        auc = roc_auc_score(adv_label[va_idx], preds)
        fold_aucs.append(auc)

        # feature importance
        if hasattr(clf, 'feature_importances_'):
            fold_importances += clf.feature_importances_
        elif hasattr(clf, 'get_feature_importance'):
            fold_importances += clf.get_feature_importance()

    fold_importances /= 5
    mean_auc = np.mean(fold_aucs)
    std_auc = np.std(fold_aucs)

    # Top-10 features
    top10_idx = np.argsort(fold_importances)[::-1][:10]
    top10 = [(feature_names[i], round(float(fold_importances[i]), 4)) for i in top10_idx]

    d1_results[clf_name] = {
        'mean_auc': round(mean_auc, 4),
        'std_auc': round(std_auc, 4),
        'fold_aucs': [round(a, 4) for a in fold_aucs],
        'top10_features': top10,
    }

    print(f"\n{'─'*50}")
    print(f"  {clf_name}")
    print(f"{'─'*50}")
    print(f"  ROC-AUC:  {mean_auc:.4f} ± {std_auc:.4f}")
    print(f"  Fold AUCs: {[round(a,4) for a in fold_aucs]}")
    if mean_auc > 0.6:
        print(f"  ⚠  AUC > 0.6 → significant distribution shift detected!")
    else:
        print(f"  ✓  AUC ≤ 0.6 → distributions appear similar.")
    print(f"  Top-10 Feature Importances:")
    for rank, (fname, imp) in enumerate(top10, 1):
        print(f"    {rank:2d}. {fname:<20s}  {imp:.4f}")

# Overall adversarial verdict
avg_across = np.mean([d1_results[k]['mean_auc'] for k in d1_results])
print(f"\n{'='*70}")
print(f"  D1 OVERALL:  Average AUC across classifiers = {avg_across:.4f}")
if avg_across > 0.6:
    print("  ⚠  VERDICT: Train/test distributions DIFFER significantly.")
else:
    print("  ✓  VERDICT: Train/test distributions are reasonably similar.")
print(f"{'='*70}\n")


# ======================================================================
# EXPERIMENT D2 – Distribution Shift Report
# ======================================================================
print("=" * 70)
print("  EXPERIMENT D2 – Distribution Shift Report")
print("=" * 70)

df_train = df.iloc[:split_idx].copy()
df_test  = df.iloc[split_idx:].copy()

d2_results = {}

# ── D2a: Date distributions (month / quarter) ──
print(f"\n{'─'*50}")
print("  Date Distributions")
print(f"{'─'*50}")

for period_name, accessor in [('month', lambda s: s.dt.month), ('quarter', lambda s: s.dt.quarter)]:
    train_vals = accessor(df_train['t']).dropna().astype(int)
    test_vals  = accessor(df_test['t']).dropna().astype(int)

    all_periods = sorted(set(train_vals) | set(test_vals))
    train_counts = train_vals.value_counts().reindex(all_periods, fill_value=0)
    test_counts  = test_vals.value_counts().reindex(all_periods, fill_value=0)

    # Normalise to proportions
    train_pct = (train_counts / train_counts.sum() * 100).round(2)
    test_pct  = (test_counts / test_counts.sum() * 100).round(2)

    print(f"\n  {period_name.upper()} distribution (%):")
    header = f"  {'Period':<10s} {'Train%':>8s} {'Test%':>8s} {'Δ':>8s}"
    print(header)
    print(f"  {'─'*36}")
    for p in all_periods:
        delta = test_pct.get(p, 0) - train_pct.get(p, 0)
        print(f"  {p:<10d} {train_pct.get(p,0):>8.2f} {test_pct.get(p,0):>8.2f} {delta:>+8.2f}")

    # chi-squared on raw counts
    contingency = pd.DataFrame({'train': train_counts, 'test': test_counts})
    # Only run chi2 if we have at least 2 categories
    if len(all_periods) >= 2:
        chi2, pval, dof, _ = chi2_contingency(contingency.values.T)
    else:
        chi2, pval, dof = 0.0, 1.0, 0

    print(f"  χ² = {chi2:.2f},  p = {pval:.6f},  dof = {dof}")
    if pval < 0.05:
        print(f"  ⚠  Significant difference in {period_name} distribution (p < 0.05)")
    else:
        print(f"  ✓  No significant difference in {period_name} distribution")

    d2_results[f'date_{period_name}'] = {
        'chi2': round(chi2, 4),
        'p_value': round(pval, 6),
        'dof': int(dof),
        'significant': bool(pval < 0.05),
        'train_pct': {str(k): float(v) for k, v in train_pct.items()},
        'test_pct':  {str(k): float(v) for k, v in test_pct.items()},
    }

# ── D2b: Categorical distributions ──
for cat_col in ['corridor', 'event_type', 'event_cause']:
    print(f"\n{'─'*50}")
    print(f"  {cat_col} Distribution")
    print(f"{'─'*50}")

    train_vc = df_train[cat_col].value_counts()
    test_vc  = df_test[cat_col].value_counts()
    all_vals = sorted(set(train_vc.index) | set(test_vc.index))

    train_counts = train_vc.reindex(all_vals, fill_value=0)
    test_counts  = test_vc.reindex(all_vals, fill_value=0)

    train_pct = (train_counts / train_counts.sum() * 100).round(2)
    test_pct  = (test_counts / test_counts.sum() * 100).round(2)

    # Top-15 by overall frequency for printing
    total_counts = train_counts + test_counts
    top15 = total_counts.sort_values(ascending=False).head(15).index.tolist()

    print(f"  Showing top 15 categories (of {len(all_vals)} total):")
    header = f"  {'Category':<35s} {'Train%':>8s} {'Test%':>8s} {'Δ':>8s}"
    print(header)
    print(f"  {'─'*61}")
    for val in top15:
        delta = test_pct.get(val, 0) - train_pct.get(val, 0)
        label = str(val)[:33]
        print(f"  {label:<35s} {train_pct.get(val,0):>8.2f} {test_pct.get(val,0):>8.2f} {delta:>+8.2f}")

    # Chi-squared test
    contingency = pd.DataFrame({'train': train_counts, 'test': test_counts})
    # Filter out rows where both are zero
    contingency = contingency[(contingency > 0).any(axis=1)]
    if len(contingency) >= 2:
        chi2, pval, dof, _ = chi2_contingency(contingency.values.T)
    else:
        chi2, pval, dof = 0.0, 1.0, 0

    print(f"\n  χ² = {chi2:.2f},  p = {pval:.6f},  dof = {dof}")
    if pval < 0.05:
        print(f"  ⚠  Significant difference in {cat_col} distribution (p < 0.05)")
    else:
        print(f"  ✓  No significant difference in {cat_col} distribution")

    # Categories that appear only in one split
    only_train = set(train_vc.index) - set(test_vc.index)
    only_test  = set(test_vc.index) - set(train_vc.index)
    if only_train:
        print(f"  Categories only in train ({len(only_train)}): {list(only_train)[:5]}{'...' if len(only_train)>5 else ''}")
    if only_test:
        print(f"  Categories only in test  ({len(only_test)}): {list(only_test)[:5]}{'...' if len(only_test)>5 else ''}")

    d2_results[cat_col] = {
        'chi2': round(chi2, 4),
        'p_value': round(pval, 6),
        'dof': int(dof),
        'significant': bool(pval < 0.05),
        'n_categories_total': len(all_vals),
        'n_only_train': len(only_train),
        'n_only_test': len(only_test),
    }

# ── Target rate shift ──
print(f"\n{'─'*50}")
print("  Target Rate Comparison")
print(f"{'─'*50}")
train_rate = y[:split_idx].mean()
test_rate  = y[split_idx:].mean()
print(f"  Train target rate: {train_rate:.4f}  ({y[:split_idx].sum()}/{split_idx})")
print(f"  Test  target rate: {test_rate:.4f}  ({y[split_idx:].sum()}/{len(df)-split_idx})")
print(f"  Δ target rate:     {test_rate - train_rate:+.4f}")
d2_results['target_rate'] = {
    'train_rate': round(float(train_rate), 4),
    'test_rate': round(float(test_rate), 4),
    'delta': round(float(test_rate - train_rate), 4),
}

# ── Date range info ──
print(f"\n{'─'*50}")
print("  Date Range Info")
print(f"{'─'*50}")
print(f"  Train: {df_train['t'].min()} → {df_train['t'].max()}")
print(f"  Test:  {df_test['t'].min()} → {df_test['t'].max()}")
d2_results['date_range'] = {
    'train_start': str(df_train['t'].min()),
    'train_end':   str(df_train['t'].max()),
    'test_start':  str(df_test['t'].min()),
    'test_end':    str(df_test['t'].max()),
}

# ======================================================================
# Save all results
# ======================================================================
results = {
    'experiment': 'Phase 1 – Distribution Discovery',
    'dataset_size': len(df),
    'split_index': split_idx,
    'D1_adversarial_validation': d1_results,
    'D1_overall_avg_auc': round(float(avg_across), 4),
    'D1_shift_detected': bool(avg_across > 0.6),
    'D2_distribution_shift': d2_results,
}

with open('phase1_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n{'='*70}")
print(f"  Results saved to phase1_results.json")
print(f"{'='*70}")
print("Done.")
