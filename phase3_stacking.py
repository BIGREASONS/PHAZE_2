"""
Phase 3: 3-Level Stacking Ensemble + Pseudo-Labeling (Self-Training)
====================================================================
Part 1 – Build a 3-level stack:
  L1: CatBoost, LightGBM, XGBoost, RandomForest, ExtraTrees (+TabPFN if available)
  L2: LogisticRegression, Ridge, LGBM meta-learner (5-fold on L1 OOFs)
  L3: Weighted ensemble of L2 outputs (scipy optimised)
Part 2 – Pseudo-labeling via iterative self-training (3 rounds)
"""

import os, json, time, warnings, pathlib
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from scipy.optimize import minimize
from scipy.special import expit

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── Data loading ────────────────────────────────────────────────────────────
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

enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
codes = enc.fit_transform(df[cats]).astype(int)
X = pd.DataFrame(codes, columns=cats)
for c in nums:
    X[c] = df[c].astype(float).values
cat_idx = list(range(len(cats)))

N = len(y)
print(f"Dataset: {N} rows | positive rate: {y.mean():.4f}")

results = {}

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PART 1: 3-LEVEL STACKING                                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# ── Helper: generate fresh 5-fold OOF for a model ──────────────────────────
FOLDS = 5
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)
fold_indices = list(skf.split(X, y))


def generate_oof(model_fn, name, X_arr, y_arr, folds):
    """Train model_fn() per fold, return OOF probabilities."""
    oof = np.zeros(len(y_arr), dtype=np.float64)
    for i, (tr, va) in enumerate(folds):
        print(f"  {name} fold {i+1}/{FOLDS} ...", end=" ", flush=True)
        t0 = time.time()
        mdl = model_fn()
        mdl.fit(X_arr[tr], y_arr[tr])
        if hasattr(mdl, 'predict_proba'):
            oof[va] = mdl.predict_proba(X_arr[va])[:, 1]
        else:
            # RidgeClassifier → decision_function → sigmoid
            oof[va] = expit(mdl.decision_function(X_arr[va]))
        print(f"{time.time()-t0:.1f}s")
    return oof


# ── Level 1 base models ───────────────────────────────────────────────────
print("\n" + "="*72)
print("LEVEL 1: Base Models")
print("="*72)

X_np = X.values.astype(np.float32)

# Names of seed-bagged OOF files produced by phase2
phase2_oof_map = {
    'catboost':    'oof_catboost_seedbag.npy',
    'lightgbm':    'oof_lightgbm_seedbag.npy',
    'xgboost':     'oof_xgboost_seedbag.npy',
    'randomforest':'oof_rf_seedbag.npy',
    'extratrees':  'oof_et_seedbag.npy',
}

l1_oofs = {}
l1_names = []

# Try loading phase2 seed-bagged OOFs
for tag, fname in phase2_oof_map.items():
    if os.path.exists(fname):
        arr = np.load(fname)
        if len(arr) == N:
            print(f"  ✓ Loaded {fname} ({tag})")
            l1_oofs[tag] = arr
            l1_names.append(tag)
        else:
            print(f"  ✗ {fname} length mismatch ({len(arr)} vs {N}), will regenerate")
    else:
        print(f"  ✗ {fname} not found, will regenerate")

# Generate fresh OOFs for any missing base models
models_to_generate = {}

if 'catboost' not in l1_oofs:
    from catboost import CatBoostClassifier
    models_to_generate['catboost'] = lambda: CatBoostClassifier(
        iterations=1000, depth=6, learning_rate=0.05,
        l2_leaf_reg=3, auto_class_weights='Balanced',
        verbose=0, random_seed=42, cat_features=cat_idx
    )

if 'lightgbm' not in l1_oofs:
    import lightgbm as lgb
    models_to_generate['lightgbm'] = lambda: lgb.LGBMClassifier(
        n_estimators=1000, num_leaves=31, learning_rate=0.05,
        is_unbalance=True, verbose=-1, random_state=42
    )

if 'xgboost' not in l1_oofs:
    import xgboost as xgb
    pos = y.sum(); neg = len(y) - pos
    sw = neg / max(pos, 1)
    models_to_generate['xgboost'] = lambda: xgb.XGBClassifier(
        n_estimators=1000, max_depth=6, learning_rate=0.05,
        scale_pos_weight=sw, eval_metric='logloss',
        verbosity=0, random_state=42, use_label_encoder=False
    )

if 'randomforest' not in l1_oofs:
    models_to_generate['randomforest'] = lambda: RandomForestClassifier(
        n_estimators=500, max_depth=12, class_weight='balanced',
        random_state=42, n_jobs=-1
    )

if 'extratrees' not in l1_oofs:
    models_to_generate['extratrees'] = lambda: ExtraTreesClassifier(
        n_estimators=500, max_depth=12, class_weight='balanced',
        random_state=42, n_jobs=-1
    )

for tag, fn in models_to_generate.items():
    print(f"\n  Generating OOF for {tag} ...")
    oof = generate_oof(fn, tag, X_np, y, fold_indices)
    l1_oofs[tag] = oof
    l1_names.append(tag)
    np.save(f'oof_{tag}_fresh.npy', oof)

# Try loading TabPFN OOF
tabpfn_file = 'oof_tabpfn_random.npy'
if os.path.exists(tabpfn_file):
    arr = np.load(tabpfn_file)
    if len(arr) == N:
        print(f"  ✓ Loaded {tabpfn_file} (tabpfn)")
        l1_oofs['tabpfn'] = arr
        l1_names.append('tabpfn')
    else:
        print(f"  ✗ {tabpfn_file} length mismatch, skipping")
else:
    print(f"  ✗ {tabpfn_file} not found, skipping TabPFN")

# Report Level 1 metrics
print(f"\nLevel 1 models: {l1_names}")
l1_metrics = {}
for tag in l1_names:
    roc = roc_auc_score(y, l1_oofs[tag])
    pr  = average_precision_score(y, l1_oofs[tag])
    l1_metrics[tag] = {'roc_auc': round(roc, 6), 'pr_auc': round(pr, 6)}
    print(f"  {tag:15s}  ROC-AUC={roc:.6f}  PR-AUC={pr:.6f}")
results['level1'] = l1_metrics

# ── Level 2 meta-learners ─────────────────────────────────────────────────
print("\n" + "="*72)
print("LEVEL 2: Meta-Learners (5-fold CV on L1 OOF stack)")
print("="*72)

# Build L1 stacked matrix
l1_stack = np.column_stack([l1_oofs[t] for t in l1_names])
print(f"L1 stack shape: {l1_stack.shape}")

import lightgbm as lgb

l2_models = {
    'lr': lambda: LogisticRegression(max_iter=1000, class_weight='balanced',
                                      random_state=42),
    'ridge': lambda: RidgeClassifier(alpha=1.0, class_weight='balanced'),
    'lgbm_meta': lambda: lgb.LGBMClassifier(
        n_estimators=200, num_leaves=15, learning_rate=0.05,
        verbose=-1, random_state=42
    ),
}

skf2 = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=123)
l2_folds = list(skf2.split(l1_stack, y))

l2_oofs = {}
l2_metrics = {}

for tag, fn in l2_models.items():
    print(f"\n  Training L2 meta-learner: {tag}")
    oof = generate_oof(fn, tag, l1_stack, y, l2_folds)
    l2_oofs[tag] = oof
    np.save(f'oof_stack_l2_{tag}.npy', oof)

    roc = roc_auc_score(y, oof)
    pr  = average_precision_score(y, oof)
    l2_metrics[tag] = {'roc_auc': round(roc, 6), 'pr_auc': round(pr, 6)}
    print(f"  {tag:15s}  ROC-AUC={roc:.6f}  PR-AUC={pr:.6f}")

results['level2'] = l2_metrics

# ── Level 3: Optimised weighted ensemble ───────────────────────────────────
print("\n" + "="*72)
print("LEVEL 3: Optimised Weighted Ensemble of L2 Outputs")
print("="*72)

l2_names = list(l2_oofs.keys())
l2_stack = np.column_stack([l2_oofs[t] for t in l2_names])


def neg_pr_auc(w):
    """Objective: negative PR-AUC for minimisation."""
    w_norm = np.array(w) / (np.sum(w) + 1e-12)
    blend = l2_stack @ w_norm
    return -average_precision_score(y, blend)


n_l2 = len(l2_names)
best_result = None

# Multi-start Nelder-Mead for robustness
np.random.seed(42)
for trial in range(30):
    if trial == 0:
        w0 = np.ones(n_l2) / n_l2
    else:
        w0 = np.random.dirichlet(np.ones(n_l2))
    res = minimize(neg_pr_auc, w0, method='Nelder-Mead',
                   options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10})
    # Clip to [0, 1]
    w_opt = np.clip(res.x, 0, 1)
    w_opt /= w_opt.sum() + 1e-12
    score = -neg_pr_auc(w_opt)
    if best_result is None or score > best_result[1]:
        best_result = (w_opt, score)

w_final = best_result[0]
l3_blend = l2_stack @ w_final

roc3 = roc_auc_score(y, l3_blend)
pr3  = average_precision_score(y, l3_blend)

print(f"\nOptimal L3 weights:")
for i, name in enumerate(l2_names):
    print(f"  {name:15s}: {w_final[i]:.6f}")
print(f"\nLevel 3 Ensemble   ROC-AUC={roc3:.6f}  PR-AUC={pr3:.6f}")

np.save('oof_stack_l3.npy', l3_blend)
results['level3'] = {
    'weights': {n: round(float(w_final[i]), 6) for i, n in enumerate(l2_names)},
    'roc_auc': round(roc3, 6),
    'pr_auc': round(pr3, 6),
}


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PART 2: PSEUDO-LABELING (SELF-TRAINING)                              ║
# ╚══════════════════════════════════════════════════════════════════════════╝

print("\n" + "="*72)
print("PART 2: Pseudo-Labeling via Iterative Self-Training")
print("="*72)


def cv_evaluate(X_data, y_data, model_fn, folds, desc=""):
    """Run stratified k-fold CV and return (roc_auc, pr_auc)."""
    oof = np.zeros(len(y_data), dtype=np.float64)
    for i, (tr, va) in enumerate(folds):
        mdl = model_fn()
        mdl.fit(X_data[tr], y_data[tr])
        if hasattr(mdl, 'predict_proba'):
            oof[va] = mdl.predict_proba(X_data[va])[:, 1]
        else:
            oof[va] = expit(mdl.decision_function(X_data[va]))
    roc = roc_auc_score(y_data, oof)
    pr  = average_precision_score(y_data, oof)
    return roc, pr


def lgbm_factory():
    return lgb.LGBMClassifier(
        n_estimators=1000, num_leaves=31, learning_rate=0.05,
        is_unbalance=True, verbose=-1, random_state=42
    )


# Baseline: standard 5-fold CV (no pseudo-labeling)
skf_pl = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)
folds_pl = list(skf_pl.split(X_np, y))

roc_base, pr_base = cv_evaluate(X_np, y, lgbm_factory, folds_pl, "baseline")
print(f"\nBaseline (no pseudo-labeling):")
print(f"  ROC-AUC={roc_base:.6f}  PR-AUC={pr_base:.6f}")

pseudo_results = {
    'baseline': {'roc_auc': round(roc_base, 6), 'pr_auc': round(pr_base, 6)}
}

# ── Iterative self-training ────────────────────────────────────────────────
HIGH_CONF_POS = 0.9
HIGH_CONF_NEG = 0.1
NUM_ROUNDS = 3

print(f"\nSelf-training: {NUM_ROUNDS} rounds, thresholds P>{HIGH_CONF_POS} or P<{HIGH_CONF_NEG}")

X_train_curr = X_np.copy()
y_train_curr = y.copy()

for rnd in range(1, NUM_ROUNDS + 1):
    print(f"\n--- Round {rnd} ---")

    # Train on current augmented training set, predict on the ORIGINAL data
    mdl = lgbm_factory()
    mdl.fit(X_train_curr, y_train_curr)
    preds = mdl.predict_proba(X_np)[:, 1]  # predict on original rows

    # Identify high-confidence pseudo-labels from original data
    high_pos = preds > HIGH_CONF_POS
    high_neg = preds < HIGH_CONF_NEG
    high_conf_mask = high_pos | high_neg
    n_pseudo = high_conf_mask.sum()
    n_pseudo_pos = high_pos.sum()
    n_pseudo_neg = high_neg.sum()

    print(f"  High-confidence samples: {n_pseudo} "
          f"(pos={n_pseudo_pos}, neg={n_pseudo_neg})")

    if n_pseudo == 0:
        print("  No high-confidence samples found, stopping early.")
        break

    # Create pseudo-labeled rows
    pseudo_X = X_np[high_conf_mask]
    pseudo_y = (preds[high_conf_mask] > 0.5).astype(int)

    # Augment: original training data + pseudo-labeled rows
    X_train_curr = np.vstack([X_np, pseudo_X])
    y_train_curr = np.concatenate([y, pseudo_y])

    print(f"  Augmented training set: {len(y_train_curr)} rows "
          f"(original {N} + pseudo {n_pseudo})")

    # Evaluate via 5-fold CV on ORIGINAL data with augmented model
    # We evaluate the benefit by doing CV on original data,
    # but in each fold we add the pseudo-labels from the training portion
    oof_pl = np.zeros(N, dtype=np.float64)
    for fi, (tr_idx, va_idx) in enumerate(folds_pl):
        # Train fold model on original train split
        mdl_fold = lgbm_factory()
        mdl_fold.fit(X_np[tr_idx], y[tr_idx])

        # Predict on full original set to find high-confidence pseudo-labels
        preds_fold = mdl_fold.predict_proba(X_np[tr_idx])[:, 1]
        hc_pos = preds_fold > HIGH_CONF_POS
        hc_neg = preds_fold < HIGH_CONF_NEG
        hc_mask = hc_pos | hc_neg

        # Augment training fold with pseudo-labels
        pseudo_X_fold = X_np[tr_idx][hc_mask]
        pseudo_y_fold = (preds_fold[hc_mask] > 0.5).astype(int)

        X_aug = np.vstack([X_np[tr_idx], pseudo_X_fold])
        y_aug = np.concatenate([y[tr_idx], pseudo_y_fold])

        # Retrain on augmented fold data
        mdl_aug = lgbm_factory()
        mdl_aug.fit(X_aug, y_aug)
        oof_pl[va_idx] = mdl_aug.predict_proba(X_np[va_idx])[:, 1]

    roc_r = roc_auc_score(y, oof_pl)
    pr_r  = average_precision_score(y, oof_pl)
    print(f"  CV ROC-AUC={roc_r:.6f}  PR-AUC={pr_r:.6f}")

    pseudo_results[f'round_{rnd}'] = {
        'roc_auc': round(roc_r, 6),
        'pr_auc': round(pr_r, 6),
        'n_pseudo_pos': int(n_pseudo_pos),
        'n_pseudo_neg': int(n_pseudo_neg),
        'augmented_size': int(len(y_train_curr)),
    }

results['pseudo_labeling'] = pseudo_results

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SAVE RESULTS                                                         ║
# ╚══════════════════════════════════════════════════════════════════════════╝

out_file = 'phase3_stacking_results.json'
with open(out_file, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n{'='*72}")
print(f"Results saved to {out_file}")
print(f"OOF files: oof_stack_l2_*.npy, oof_stack_l3.npy")

# ── Pretty summary ─────────────────────────────────────────────────────────
print(f"\n{'='*72}")
print("SUMMARY")
print(f"{'='*72}")
print(f"\n{'Level':<25s} {'ROC-AUC':>10s} {'PR-AUC':>10s}")
print("-" * 47)
for tag in l1_names:
    m = l1_metrics[tag]
    print(f"L1 {tag:<21s} {m['roc_auc']:10.6f} {m['pr_auc']:10.6f}")
print("-" * 47)
for tag in l2_names:
    m = l2_metrics[tag]
    print(f"L2 {tag:<21s} {m['roc_auc']:10.6f} {m['pr_auc']:10.6f}")
print("-" * 47)
print(f"L3 ensemble            {roc3:10.6f} {pr3:10.6f}")
print("-" * 47)
print(f"\nPseudo-labeling:")
for k, v in pseudo_results.items():
    print(f"  {k:<15s}  ROC-AUC={v['roc_auc']:.6f}  PR-AUC={v['pr_auc']:.6f}")
print(f"\n{'='*72}")
print("Phase 3 complete.")
