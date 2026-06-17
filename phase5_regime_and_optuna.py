"""
Phase 5 – Event-Regime Splitting  &  Phase 6 – Optuna AP Optimisation
=====================================================================
Part 1: Split planned vs unplanned events, train per-regime CatBoost,
        blend OOF, compare with unified model.
Part 2: Optuna 100-trial hyper-parameter search (AP objective) for
        CatBoost, LightGBM, XGBoost. Save best OOF arrays + results.
"""

import json, warnings, pathlib
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from catboost import CatBoostClassifier
import lightgbm as lgb
import xgboost as xgb
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

SEED = 42
N_SPLITS = 5
OPTUNA_TRIALS = 100

# ── data loading ────────────────────────────────────────────────────
DATA = "Astram event data_anonymized - Astram event data_anonymizedb40ac87.csv"
df = pd.read_csv(DATA, low_memory=False)
df["t"] = pd.to_datetime(df["created_date"], errors="coerce")
df = df.sort_values("t").reset_index(drop=True)
df["lat"] = pd.to_numeric(df["latitude"], errors="coerce")
df["lon"] = pd.to_numeric(df["longitude"], errors="coerce")
df["hour"] = df["t"].dt.hour.fillna(0).astype(int)
df["weekday"] = df["t"].dt.weekday.fillna(0).astype(int)
y = df["requires_road_closure"].astype(int).values
cats = ["event_type", "event_cause", "veh_type", "corridor", "police_station", "zone"]
nums = ["lat", "lon", "hour", "weekday"]
for c in cats:
    df[c] = df[c].astype(str).fillna("NA")
codes = OrdinalEncoder(
    handle_unknown="use_encoded_value", unknown_value=-1
).fit_transform(df[cats]).astype(int)
X = pd.DataFrame(codes, columns=cats)
for c in nums:
    X[c] = df[c].astype(float).values
cat_idx = list(range(len(cats)))
N = len(df)
print(f"Rows: {N}   Positive rate: {y.mean():.4f}")

# =====================================================================
# PART 1 – Event Regime Splitting (Phase 5)
# =====================================================================
print("\n" + "=" * 70)
print("PART 1 — EVENT REGIME SPLITTING")
print("=" * 70)

# ── classify planned vs unplanned ───────────────────────────────────
planned_mask = df["event_type"].str.lower().str.contains(
    "planned|scheduled", na=False
)
print(f"Planned rows : {planned_mask.sum()}")
print(f"Unplanned rows: {(~planned_mask).sum()}")

# ── helper: 5-fold CatBoost OOF on a subset ────────────────────────
def catboost_oof(X_sub, y_sub, cat_idx, seed=SEED):
    """Return OOF probability array for a subset."""
    oof = np.zeros(len(y_sub), dtype=np.float64)
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    for fold, (tr, va) in enumerate(skf.split(X_sub, y_sub)):
        cb = CatBoostClassifier(
            iterations=600,
            depth=6,
            learning_rate=0.08,
            l2_leaf_reg=3,
            auto_class_weights="Balanced",
            cat_features=cat_idx,
            random_seed=seed,
            verbose=0,
        )
        cb.fit(X_sub.iloc[tr], y_sub[tr], eval_set=(X_sub.iloc[va], y_sub[va]),
               early_stopping_rounds=50, verbose=0)
        oof[va] = cb.predict_proba(X_sub.iloc[va])[:, 1]
    return oof

# ── unified model OOF ──────────────────────────────────────────────
print("\nTraining unified CatBoost (5-fold) …")
oof_unified = catboost_oof(X, y, cat_idx)

# ── regime-specific models ──────────────────────────────────────────
oof_regime = np.zeros(N, dtype=np.float64)

idx_planned = np.where(planned_mask)[0]
idx_unplanned = np.where(~planned_mask)[0]

for tag, idx in [("Planned", idx_planned), ("Unplanned", idx_unplanned)]:
    X_sub = X.iloc[idx].reset_index(drop=True)
    y_sub = y[idx]
    n_pos = y_sub.sum()
    print(f"\n  {tag}: N={len(idx)}, pos={n_pos} ({n_pos/len(idx):.4f})")
    if n_pos < N_SPLITS:
        # too few positives for stratified CV – fall back to unified prediction
        print(f"    ⚠ Too few positives for stratified CV – using unified OOF")
        oof_regime[idx] = oof_unified[idx]
    else:
        oof_sub = catboost_oof(X_sub, y_sub, cat_idx)
        oof_regime[idx] = oof_sub

# ── metrics ─────────────────────────────────────────────────────────
def calc_metrics(y_true, y_prob, label=""):
    roc = roc_auc_score(y_true, y_prob)
    pr  = average_precision_score(y_true, y_prob)
    print(f"  {label:20s}  ROC-AUC={roc:.5f}   PR-AUC={pr:.5f}")
    return {"roc_auc": round(roc, 6), "pr_auc": round(pr, 6)}

print("\n── Unified vs Regime-Split comparison ─────────────────────")
m_uni = calc_metrics(y, oof_unified, "Unified")
m_reg = calc_metrics(y, oof_regime,  "Regime-Split")

phase5 = {
    "planned_rows": int(planned_mask.sum()),
    "unplanned_rows": int((~planned_mask).sum()),
    "unified": m_uni,
    "regime_split": m_reg,
}
with open("phase5_regime_results.json", "w") as f:
    json.dump(phase5, f, indent=2)
print("\n✓ Saved phase5_regime_results.json")

# =====================================================================
# PART 2 – Optuna AP Optimisation (Phase 6)
# =====================================================================
print("\n" + "=" * 70)
print("PART 2 — OPTUNA AP OPTIMISATION  (100 trials × 3 models)")
print("=" * 70)

skf_optuna = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

# ── CatBoost ────────────────────────────────────────────────────────
def objective_catboost(trial):
    params = dict(
        iterations      = trial.suggest_int("iterations", 200, 1000),
        depth           = trial.suggest_int("depth", 4, 10),
        learning_rate   = trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        l2_leaf_reg     = trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        border_count    = trial.suggest_int("border_count", 32, 255),
        bagging_temperature = trial.suggest_float("bagging_temperature", 0.0, 1.0),
        auto_class_weights  = "Balanced",
        cat_features    = cat_idx,
        random_seed     = SEED,
        verbose         = 0,
    )
    oof = np.zeros(N, dtype=np.float64)
    for tr, va in skf_optuna.split(X, y):
        m = CatBoostClassifier(**params)
        m.fit(X.iloc[tr], y[tr], eval_set=(X.iloc[va], y[va]),
              early_stopping_rounds=50, verbose=0)
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
    return average_precision_score(y, oof)

print("\n▸ CatBoost Optuna (100 trials) …")
study_cb = optuna.create_study(direction="maximize",
                               sampler=optuna.samplers.TPESampler(seed=SEED))
study_cb.optimize(objective_catboost, n_trials=OPTUNA_TRIALS, show_progress_bar=True)
print(f"  Best AP = {study_cb.best_value:.5f}")
print(f"  Best params: {study_cb.best_params}")

# retrain with best params → OOF
bp = study_cb.best_params.copy()
bp.update(auto_class_weights="Balanced", cat_features=cat_idx,
          random_seed=SEED, verbose=0)
oof_cb = np.zeros(N, dtype=np.float64)
for tr, va in skf_optuna.split(X, y):
    m = CatBoostClassifier(**bp)
    m.fit(X.iloc[tr], y[tr], eval_set=(X.iloc[va], y[va]),
          early_stopping_rounds=50, verbose=0)
    oof_cb[va] = m.predict_proba(X.iloc[va])[:, 1]
np.save("oof_optuna_catboost.npy", oof_cb)
cb_ap = average_precision_score(y, oof_cb)
cb_roc = roc_auc_score(y, oof_cb)
print(f"  Retrained OOF — AP={cb_ap:.5f}  ROC-AUC={cb_roc:.5f}")

# ── LightGBM ───────────────────────────────────────────────────────
def objective_lgbm(trial):
    params = dict(
        n_estimators      = trial.suggest_int("n_estimators", 200, 1000),
        num_leaves        = trial.suggest_int("num_leaves", 15, 63),
        learning_rate     = trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        min_child_samples = trial.suggest_int("min_child_samples", 5, 50),
        reg_alpha         = trial.suggest_float("reg_alpha", 0.0, 10.0),
        reg_lambda        = trial.suggest_float("reg_lambda", 0.0, 10.0),
        subsample         = trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree  = trial.suggest_float("colsample_bytree", 0.6, 1.0),
        class_weight      = "balanced",
        random_state      = SEED,
        verbose           = -1,
    )
    oof = np.zeros(N, dtype=np.float64)
    for tr, va in skf_optuna.split(X, y):
        m = lgb.LGBMClassifier(**params)
        m.fit(X.iloc[tr], y[tr],
              eval_set=[(X.iloc[va], y[va])],
              callbacks=[lgb.early_stopping(50, verbose=False),
                         lgb.log_evaluation(period=0)])
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
    return average_precision_score(y, oof)

print("\n▸ LightGBM Optuna (100 trials) …")
study_lg = optuna.create_study(direction="maximize",
                               sampler=optuna.samplers.TPESampler(seed=SEED))
study_lg.optimize(objective_lgbm, n_trials=OPTUNA_TRIALS, show_progress_bar=True)
print(f"  Best AP = {study_lg.best_value:.5f}")
print(f"  Best params: {study_lg.best_params}")

bp = study_lg.best_params.copy()
bp.update(class_weight="balanced", random_state=SEED, verbose=-1)
oof_lg = np.zeros(N, dtype=np.float64)
for tr, va in skf_optuna.split(X, y):
    m = lgb.LGBMClassifier(**bp)
    m.fit(X.iloc[tr], y[tr],
          eval_set=[(X.iloc[va], y[va])],
          callbacks=[lgb.early_stopping(50, verbose=False),
                     lgb.log_evaluation(period=0)])
    oof_lg[va] = m.predict_proba(X.iloc[va])[:, 1]
np.save("oof_optuna_lgbm.npy", oof_lg)
lg_ap = average_precision_score(y, oof_lg)
lg_roc = roc_auc_score(y, oof_lg)
print(f"  Retrained OOF — AP={lg_ap:.5f}  ROC-AUC={lg_roc:.5f}")

# ── XGBoost ─────────────────────────────────────────────────────────
pos_w = (y == 0).sum() / max((y == 1).sum(), 1)

def objective_xgb(trial):
    params = dict(
        n_estimators     = trial.suggest_int("n_estimators", 200, 1000),
        max_depth        = trial.suggest_int("max_depth", 4, 10),
        learning_rate    = trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        min_child_weight = trial.suggest_int("min_child_weight", 1, 10),
        gamma            = trial.suggest_float("gamma", 0.0, 5.0),
        subsample        = trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.6, 1.0),
        reg_alpha        = trial.suggest_float("reg_alpha", 0.0, 10.0),
        reg_lambda       = trial.suggest_float("reg_lambda", 0.0, 10.0),
        scale_pos_weight = pos_w,
        use_label_encoder= False,
        eval_metric      = "logloss",
        random_state     = SEED,
        verbosity        = 0,
    )
    oof = np.zeros(N, dtype=np.float64)
    for tr, va in skf_optuna.split(X, y):
        m = xgb.XGBClassifier(**params)
        m.fit(X.iloc[tr], y[tr],
              eval_set=[(X.iloc[va], y[va])],
              verbose=False)
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
    return average_precision_score(y, oof)

print("\n▸ XGBoost Optuna (100 trials) …")
study_xg = optuna.create_study(direction="maximize",
                               sampler=optuna.samplers.TPESampler(seed=SEED))
study_xg.optimize(objective_xgb, n_trials=OPTUNA_TRIALS, show_progress_bar=True)
print(f"  Best AP = {study_xg.best_value:.5f}")
print(f"  Best params: {study_xg.best_params}")

bp = study_xg.best_params.copy()
bp.update(scale_pos_weight=pos_w, use_label_encoder=False,
          eval_metric="logloss", random_state=SEED, verbosity=0)
oof_xg = np.zeros(N, dtype=np.float64)
for tr, va in skf_optuna.split(X, y):
    m = xgb.XGBClassifier(**bp)
    m.fit(X.iloc[tr], y[tr],
          eval_set=[(X.iloc[va], y[va])],
          verbose=False)
    oof_xg[va] = m.predict_proba(X.iloc[va])[:, 1]
np.save("oof_optuna_xgb.npy", oof_xg)
xg_ap = average_precision_score(y, oof_xg)
xg_roc = roc_auc_score(y, oof_xg)
print(f"  Retrained OOF — AP={xg_ap:.5f}  ROC-AUC={xg_roc:.5f}")

# ── save phase6 results ────────────────────────────────────────────
phase6 = {
    "catboost": {
        "best_params": study_cb.best_params,
        "best_trial_ap": round(study_cb.best_value, 6),
        "retrained_ap": round(cb_ap, 6),
        "retrained_roc_auc": round(cb_roc, 6),
    },
    "lightgbm": {
        "best_params": study_lg.best_params,
        "best_trial_ap": round(study_lg.best_value, 6),
        "retrained_ap": round(lg_ap, 6),
        "retrained_roc_auc": round(lg_roc, 6),
    },
    "xgboost": {
        "best_params": study_xg.best_params,
        "best_trial_ap": round(study_xg.best_value, 6),
        "retrained_ap": round(xg_ap, 6),
        "retrained_roc_auc": round(xg_roc, 6),
    },
}
with open("phase6_optuna_results.json", "w") as f:
    json.dump(phase6, f, indent=2)
print("\n✓ Saved phase6_optuna_results.json")

# ── final summary ──────────────────────────────────────────────────
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)
print(f"{'Model':<20s} {'AP':>8s} {'ROC-AUC':>10s}")
print("-" * 40)
print(f"{'Unified CatBoost':<20s} {average_precision_score(y, oof_unified):>8.5f} "
      f"{roc_auc_score(y, oof_unified):>10.5f}")
print(f"{'Regime-Split':<20s} {average_precision_score(y, oof_regime):>8.5f} "
      f"{roc_auc_score(y, oof_regime):>10.5f}")
print(f"{'Optuna CatBoost':<20s} {cb_ap:>8.5f} {cb_roc:>10.5f}")
print(f"{'Optuna LightGBM':<20s} {lg_ap:>8.5f} {lg_roc:>10.5f}")
print(f"{'Optuna XGBoost':<20s} {xg_ap:>8.5f} {xg_roc:>10.5f}")
print("\nOOF arrays: oof_optuna_catboost.npy  oof_optuna_lgbm.npy  oof_optuna_xgb.npy")
print("Done ✓")
