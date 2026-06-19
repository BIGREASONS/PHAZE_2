"""
PHASE F - Production export of the Phase-E-blessed ensemble.

Reads results/final_decision.json (written by Phase E) and exports the chosen
ensemble. If no candidate cleared the gate, the incumbent (equal-weight 7-model
probability average) is exported -- the policy default.

Artifacts written to final_validation/artifacts/:
  encoder.pkl        fitted OrdinalEncoder (tree members) + feature spec
  <member>.*         one fitted artifact per base member (full-data fit)
  tabpfn_refit.npz   fixed in-context training set for TabPFN (exact reconstruction)
  calibrator.pkl     isotonic calibrator fit on HONEST pooled rolling-OOF (not in-sample)
  thresholds.json    operating points (max-F1 + fixed-recall) from calibrated OOF
  manifest.json      ensemble definition: members, combiner, weights, artifact map
  metadata.json      training provenance: data stats, versions, git commit, timestamp

Calibration / thresholds are derived from out-of-fold predictions so they are not
optimistic. The exported models themselves are fit on ALL rows for deployment.
"""
import os, json, pickle, platform, subprocess, datetime
import numpy as np
import pandas as pd
import joblib

import common as C
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import precision_recall_curve, f1_score


def _git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=C.ROOT,
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        return "unknown"


def _versions():
    import sklearn, catboost, lightgbm, xgboost, tabpfn
    return {"python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "scikit-learn": sklearn.__version__,
            "catboost": catboost.__version__, "lightgbm": lightgbm.__version__,
            "xgboost": xgboost.__version__, "tabpfn": tabpfn.__version__}


def load_decision():
    p = os.path.join(C.RESULTS_DIR, "final_decision.json")
    if os.path.exists(p):
        return json.load(open(p))
    print("[warn] final_decision.json not found -- defaulting to frozen incumbent.")
    return {"final_choice": "incumbent", "adopt_candidate": False,
            "export": {"members": C.BASE_ORDER, "combiner": "prob_mean",
                       "add_recency_tabpfn": False, "use_rank_avg": False,
                       "use_temporal_decay": False}}


def build_pooled_oof(exp):
    """Reconstruct honest pooled rolling-OOF combined scores + labels for calibration."""
    from final_ensemble_validation import get_plain_payload
    plain = get_plain_payload(verbose=False)
    members = list(C.BASE_ORDER)

    extra = None
    if exp.get("add_recency_tabpfn"):
        from dr_tabpfn import build_recency_tabpfn_oof
        extra = build_recency_tabpfn_oof()

    decay = None
    decay_members = []
    if exp.get("use_temporal_decay"):
        from temporal_decay import build_decay_payload, DECAY_MEMBERS
        decay = build_decay_payload()
        decay_members = DECAY_MEMBERS

    combiner = C.rank_mean if exp.get("use_rank_avg") else C.prob_mean
    scores, ys = [], []
    for k in range(len(plain["y_val"])):
        cols = []
        for m in members:
            if decay is not None and m in decay_members:
                cols.append(np.array(decay["preds"][m][k]))
            else:
                cols.append(np.array(plain["preds"][m][k]))
        if extra is not None:
            cols.append(np.array(extra["preds"][k]))
        scores.append(combiner(cols))
        ys.append(np.array(plain["y_val"][k]))
    return np.concatenate(scores), np.concatenate(ys)


def fit_full_members(exp):
    """Fit every base member on ALL rows; return dict name->artifact_spec and the encoder."""
    df, y, Xraw, cat_idx = C.load_data()
    factories = C.make_models()
    artifacts = {}

    # shared ordinal encoder (tree members)
    from sklearn.preprocessing import OrdinalEncoder
    oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1).fit(Xraw[C.CATS])
    codes = oe.transform(Xraw[C.CATS]).astype(float)
    Xtree = np.hstack([codes, Xraw[C.NUMS].values.astype(float)])
    joblib.dump({"ordinal_encoder": oe, "cats": C.CATS, "nums": C.NUMS},
                os.path.join(C.ARTIFACT_DIR, "encoder.pkl"))

    sw = None
    if exp.get("use_temporal_decay"):
        from temporal_decay import decay_weight_fn
        sw = decay_weight_fn(df["t"])

    # CatBoost (raw cats)
    cb = factories["CatBoost"]()
    cb.fit(Xraw[C.CATS + C.NUMS], y, cat_features=cat_idx,
           sample_weight=sw if exp.get("use_temporal_decay") else None)
    cb.save_model(os.path.join(C.ARTIFACT_DIR, "CatBoost.cbm"))
    artifacts["CatBoost"] = {"file": "CatBoost.cbm", "input": "raw_frame[cats+nums]"}

    # LightGBM
    lg = factories["LightGBM"]()
    lg.fit(Xtree, y, categorical_feature=cat_idx,
           sample_weight=sw if exp.get("use_temporal_decay") else None)
    joblib.dump(lg, os.path.join(C.ARTIFACT_DIR, "LightGBM.pkl"))
    artifacts["LightGBM"] = {"file": "LightGBM.pkl", "input": "ordinal_matrix"}

    # XGBoost
    xg = factories["XGBoost"]()
    xg.fit(Xtree, y, sample_weight=sw if exp.get("use_temporal_decay") else None)
    joblib.dump(xg, os.path.join(C.ARTIFACT_DIR, "XGBoost.pkl"))
    artifacts["XGBoost"] = {"file": "XGBoost.pkl", "input": "ordinal_matrix"}

    # RandomForest / ExtraTrees (plain -- not decay-weighted per Phase C spec)
    for name in ("RandomForest", "ExtraTrees"):
        m = factories[name]()
        m.fit(Xtree, y)
        joblib.dump(m, os.path.join(C.ARTIFACT_DIR, f"{name}.pkl"))
        artifacts[name] = {"file": f"{name}.pkl", "input": "ordinal_matrix"}

    # Logistic (own one-hot+scaler pipeline)
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    ct = ColumnTransformer([("c", OneHotEncoder(handle_unknown="ignore", min_frequency=10), C.CATS),
                            ("n", StandardScaler(), C.NUMS)])
    lr = Pipeline([("ct", ct), ("lr", LogisticRegression(max_iter=2000, class_weight="balanced"))])
    lr.fit(Xraw[C.CATS + C.NUMS], y)
    joblib.dump(lr, os.path.join(C.ARTIFACT_DIR, "Logistic.pkl"))
    artifacts["Logistic"] = {"file": "Logistic.pkl", "input": "raw_frame[cats+nums]"}

    # TabPFN: store the exact in-context training set (balanced cap=4000) for exact refit.
    idx = np.arange(len(y))
    if len(idx) > 4000:
        pos = idx[y == 1]; neg = idx[y == 0]
        negk = np.random.RandomState(0).choice(neg, 4000 - len(pos), replace=False)
        idx = np.sort(np.concatenate([pos, negk]))
    np.savez_compressed(os.path.join(C.ARTIFACT_DIR, "tabpfn_refit.npz"),
                        X=Xtree[idx], y=y[idx])
    artifacts["TabPFN"] = {"file": "tabpfn_refit.npz", "input": "ordinal_matrix",
                           "note": "in-context training set; refit TabPFNClassifier(cpu,n_estimators=3) at load"}

    if exp.get("add_recency_tabpfn"):
        pos = np.arange(len(y))[y == 1]
        neg_recent = np.arange(len(y))[y == 0][-(4000 - len(pos)):]
        sel = np.sort(np.concatenate([pos, neg_recent]))
        np.savez_compressed(os.path.join(C.ARTIFACT_DIR, "tabpfn_recency_refit.npz"),
                            X=Xtree[sel], y=y[sel])
        artifacts["TabPFN_recency"] = {"file": "tabpfn_recency_refit.npz",
                                       "input": "ordinal_matrix", "note": "recency-context proxy member"}
    return artifacts


def main():
    decision = load_decision()
    exp = decision["export"]
    print("=" * 84)
    print("PHASE F  -  PRODUCTION EXPORT")
    print("=" * 84)
    print(f"  exporting: {decision['final_choice']}  (adopt_candidate={decision['adopt_candidate']})")

    # 1. honest OOF for calibration + thresholds
    oof_scores, oof_y = build_pooled_oof(exp)
    cal = IsotonicRegression(out_of_bounds="clip").fit(oof_scores, oof_y)
    cal_p = cal.predict(oof_scores)
    joblib.dump(cal, os.path.join(C.ARTIFACT_DIR, "calibrator.pkl"))

    # thresholds from calibrated OOF
    prec, rec, thr = precision_recall_curve(oof_y, cal_p)
    f1s = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(prec), where=(prec + rec) > 0)
    best_i = int(np.nanargmax(f1s[:-1])) if len(thr) else 0
    def thr_at_recall(target):
        ok = np.where(rec[:-1] >= target)[0]
        return float(thr[ok[-1]]) if len(ok) else float(thr[0]) if len(thr) else 0.5
    thresholds = {
        "max_f1": {"threshold": float(thr[best_i]) if len(thr) else 0.5,
                   "f1": float(f1s[best_i]), "precision": float(prec[best_i]),
                   "recall": float(rec[best_i])},
        "recall_0.50": {"threshold": thr_at_recall(0.50)},
        "recall_0.70": {"threshold": thr_at_recall(0.70)},
        "recall_0.90": {"threshold": thr_at_recall(0.90)},
        "basis": "isotonic-calibrated pooled rolling-OOF (honest, not in-sample)",
        "oof_rows": int(len(oof_y)), "oof_pos": int(oof_y.sum()),
    }
    json.dump(thresholds, open(os.path.join(C.ARTIFACT_DIR, "thresholds.json"), "w"), indent=2)

    # 2. fit full-data members
    artifacts = fit_full_members(exp)

    # 3. manifest + metadata
    members = list(exp["members"])
    if exp.get("add_recency_tabpfn"):
        members = members + ["TabPFN_recency"]
    manifest = {
        "ensemble": exp["ensemble"],
        "combiner": "rank_mean" if exp.get("use_rank_avg") else "prob_mean",
        "members": members,
        "equal_weights": [round(1.0 / len(members), 6)] * len(members),
        "temporal_decay": bool(exp.get("use_temporal_decay")),
        "artifacts": artifacts,
        "calibrator": "calibrator.pkl",
        "encoder": "encoder.pkl",
        "thresholds": "thresholds.json",
        "inference_note": "Apply each member to its declared input; combine with the "
                          "combiner; pass combined score through calibrator.pkl; threshold "
                          "with thresholds.json.",
    }
    json.dump(manifest, open(os.path.join(C.ARTIFACT_DIR, "manifest.json"), "w"), indent=2)

    info = C.base_rate_info()
    metadata = {
        "project": "GridSight AI road-closure prediction",
        "exported_choice": decision["final_choice"],
        "adopted_candidate": decision["adopt_candidate"],
        "trained_on": "all rows (full dataset)",
        "dataset": info,
        "incumbent_oof_pr_auc": decision.get("incumbent_pr_auc"),
        "incumbent_oof_pr_auc_std": decision.get("incumbent_pr_auc_std"),
        "feature_spec": {"categoricals": C.CATS, "numerics": C.NUMS},
        "validation": "rolling-origin expanding window, 4 folds, train-only encoders",
        "git_commit": _git_commit(),
        "package_versions": _versions(),
        "exported_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    json.dump(metadata, open(os.path.join(C.ARTIFACT_DIR, "metadata.json"), "w"), indent=2)

    print(f"  OOF calibration rows={len(oof_y)} pos={int(oof_y.sum())}  "
          f"max-F1 thr={thresholds['max_f1']['threshold']:.4f} "
          f"(F1={thresholds['max_f1']['f1']:.3f})")
    print(f"  artifacts written to {os.path.relpath(C.ARTIFACT_DIR, C.ROOT)}/:")
    for f in sorted(os.listdir(C.ARTIFACT_DIR)):
        sz = os.path.getsize(os.path.join(C.ARTIFACT_DIR, f))
        print(f"    {f:28s} {sz/1024:8.1f} KB")
    print("\nPhase F complete.")


if __name__ == "__main__":
    main()
