"""
final_validation/common.py
==========================
Shared foundation for the GridSight AI "last chance before freeze" validation.

Design contract (read before touching anything):

  * Data loading is byte-identical to run_stack_shift.py / phase1: same 6 cats,
    4 nums, sorted by created_date. This keeps us comparable to every prior audit.

  * Rolling-origin EXPANDING-window folds. Fold k trains on a contiguous prefix
    [0 : cut_k) and validates on the next contiguous block [cut_k : cut_k+val).
    This is the ONLY split that mimics "train on past, predict the future" under
    the project's severe temporal shift. We do NOT reuse the cached TimeSeriesSplit
    OOF on disk because those folds differ from ours -- mixing them would be a
    silent apples-to-oranges comparison.

  * Per fold, every encoder / scaler / model is fit on TRAIN ROWS ONLY and applied
    to the held-out future block. No target, no future row, ever touches fit().
    Base predictions are cached per (fold, model) so Phases B/C/D reuse them.

  * The incumbent is the equal-weight 7-model ensemble averaged in PROBABILITY
    space (the current frozen default). Phase B will challenge that averaging rule.

Nothing here imports from command_center/ or the phase1-7 scripts. Self-contained.
"""
from __future__ import annotations
import os, sys, io, json, time, hashlib, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
except Exception:
    pass

from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

# ---------------------------------------------------------------- paths / config
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "Astram event data_anonymized - Astram event data_anonymizedb40ac87.csv")
CACHE_DIR = os.path.join(HERE, "cache")
RESULTS_DIR = os.path.join(HERE, "results")
ARTIFACT_DIR = os.path.join(HERE, "artifacts")
for d in (CACHE_DIR, RESULTS_DIR, ARTIFACT_DIR):
    os.makedirs(d, exist_ok=True)

SEED = 0
CATS = ["event_type", "event_cause", "veh_type", "corridor", "police_station", "zone"]
NUMS = ["lat", "lon", "hour", "weekday"]
BASE_ORDER = ["CatBoost", "LightGBM", "XGBoost", "RandomForest", "ExtraTrees", "Logistic", "TabPFN"]

# Rolling-origin config: 4 expanding folds over the time-sorted data.
N_FOLDS = 4
# First training cut at 50% of the data, then expand. Each validation block is the
# next contiguous slice up to the following cut. Last fold validates the final slice.
FIRST_CUT_FRAC = 0.50


# ---------------------------------------------------------------- data
def load_data():
    """Return (df, y, Xtree, Xraw, cat_idx, t). Identical preprocessing to prior audits."""
    df = pd.read_csv(DATA, low_memory=False)
    df["t"] = pd.to_datetime(df["created_date"], errors="coerce")
    df = df.sort_values("t").reset_index(drop=True)
    df["lat"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["lon"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["hour"] = df["t"].dt.hour.fillna(0).astype(int)
    df["weekday"] = df["t"].dt.weekday.fillna(0).astype(int)
    y = df["requires_road_closure"].astype(int).values
    for c in CATS:
        df[c] = df[c].astype(str).fillna("NA")
    for c in ["lat", "lon"]:
        df[c] = df[c].astype(float)
    Xraw = df[CATS + NUMS].copy()         # raw frame for per-fold encoders
    cat_idx = list(range(len(CATS)))
    return df, y, Xraw, cat_idx


def rolling_folds(n, n_folds=N_FOLDS, first_cut_frac=FIRST_CUT_FRAC):
    """
    Expanding-window folds over a time-sorted array of length n.
    Returns list of (train_idx, val_idx) with contiguous, forward-only blocks.

    fold k:  train = [0, cut_k),  val = [cut_k, cut_{k+1})
    cuts are evenly spaced from first_cut_frac*n .. n.
    """
    first_cut = int(round(first_cut_frac * n))
    cuts = np.linspace(first_cut, n, n_folds + 1).astype(int)
    cuts = np.unique(cuts)
    folds = []
    for k in range(len(cuts) - 1):
        tr = np.arange(0, cuts[k])
        va = np.arange(cuts[k], cuts[k + 1])
        if len(va) == 0 or va.sum() == 0:
            continue
        folds.append((tr, va))
    return folds


# ---------------------------------------------------------------- base models
def make_models():
    """Factories identical in spirit to run_stack_shift.py base config."""
    from catboost import CatBoostClassifier
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier
    return {
        "CatBoost": lambda: CatBoostClassifier(
            iterations=400, depth=6, learning_rate=0.05, verbose=0,
            random_seed=SEED, thread_count=-1),
        "LightGBM": lambda: LGBMClassifier(
            n_estimators=400, learning_rate=0.05, num_leaves=31,
            random_state=SEED, n_jobs=-1, verbose=-1),
        "XGBoost": lambda: XGBClassifier(
            n_estimators=400, learning_rate=0.05, max_depth=6,
            tree_method="hist", random_state=SEED, n_jobs=-1, eval_metric="logloss"),
        "RandomForest": lambda: RandomForestClassifier(
            n_estimators=500, n_jobs=-1, random_state=SEED, class_weight="balanced"),
        "ExtraTrees": lambda: ExtraTreesClassifier(
            n_estimators=600, n_jobs=-1, random_state=SEED, class_weight="balanced"),
    }


def _ordinal_fit_transform(Xraw_tr, Xraw_va):
    """Fit OrdinalEncoder on TRAIN cats only; transform both. Nums passed through."""
    oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    tr_codes = oe.fit_transform(Xraw_tr[CATS]).astype(float)
    va_codes = oe.transform(Xraw_va[CATS]).astype(float)
    Xtr = np.hstack([tr_codes, Xraw_tr[NUMS].values.astype(float)])
    Xva = np.hstack([va_codes, Xraw_va[NUMS].values.astype(float)])
    return Xtr, Xva


def _fit_predict_tree(name, factory, Xraw_tr, y_tr, Xraw_va, cat_idx, sample_weight=None):
    """Tree/GBM/LR base learner, fit on train block, predict val block. Train-only encoding."""
    if name == "CatBoost":
        from catboost import Pool
        m = factory()
        m.fit(Xraw_tr[CATS + NUMS], y_tr, cat_features=cat_idx, sample_weight=sample_weight)
        return m.predict_proba(Xraw_va[CATS + NUMS])[:, 1]

    if name == "Logistic":
        ct = ColumnTransformer([
            ("c", OneHotEncoder(handle_unknown="ignore", min_frequency=10), CATS),
            ("n", StandardScaler(), NUMS)])
        pipe = Pipeline([("ct", ct),
                         ("lr", LogisticRegression(max_iter=2000, class_weight="balanced"))])
        if sample_weight is not None:
            pipe.fit(Xraw_tr[CATS + NUMS], y_tr, lr__sample_weight=sample_weight)
        else:
            pipe.fit(Xraw_tr[CATS + NUMS], y_tr)
        return pipe.predict_proba(Xraw_va[CATS + NUMS])[:, 1]

    # LightGBM / XGBoost / RandomForest / ExtraTrees use ordinal codes
    Xtr, Xva = _ordinal_fit_transform(Xraw_tr, Xraw_va)
    m = factory()
    if name == "LightGBM":
        m.fit(Xtr, y_tr, categorical_feature=cat_idx, sample_weight=sample_weight)
    elif name in ("RandomForest", "ExtraTrees"):
        # class_weight already set; sample_weight composes on top if provided
        m.fit(Xtr, y_tr, sample_weight=sample_weight)
    else:  # XGBoost
        m.fit(Xtr, y_tr, sample_weight=sample_weight)
    return m.predict_proba(Xva)[:, 1]


# ---------------------------------------------------------------- TabPFN base
def _fit_predict_tabpfn(Xraw_tr, y_tr, Xraw_va, recency_context=False, t_tr=None,
                        cap=4000, seed=0):
    """
    CPU TabPFN base learner. Mirrors tabpfn_test.py capping logic.

    recency_context=False -> class-balanced random subsample of train (the standard
        member already in the frozen ensemble).
    recency_context=True  -> when capping, keep the MOST RECENT rows (drift-aware
        proxy used in Phase D). All positives kept; negatives filled by recency.
    """
    os.environ["TABPFN_ALLOW_CPU_LARGE_DATASET"] = "1"
    import torch
    from tabpfn import TabPFNClassifier
    try:
        torch.set_num_threads(min(16, os.cpu_count() or 4))
    except Exception:
        pass

    oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    tr_codes = oe.fit_transform(Xraw_tr[CATS]).astype(float)
    va_codes = oe.transform(Xraw_va[CATS]).astype(float)
    Xtr = np.hstack([tr_codes, Xraw_tr[NUMS].values.astype(float)])
    Xva = np.hstack([va_codes, Xraw_va[NUMS].values.astype(float)])

    idx = np.arange(len(y_tr))
    if len(idx) > cap:
        pos = idx[y_tr == 1]
        neg = idx[y_tr == 0]
        n_neg = max(cap - len(pos), 1)
        if recency_context:
            # keep most-recent negatives (idx is already time-sorted within the train prefix)
            neg_keep = neg[-n_neg:]
        else:
            neg_keep = np.random.RandomState(seed).choice(neg, min(n_neg, len(neg)), replace=False)
        sel = np.sort(np.concatenate([pos, neg_keep]))
        Xtr, y_tr = Xtr[sel], y_tr[sel]

    clf = TabPFNClassifier(device="cpu", ignore_pretraining_limits=True, n_estimators=3)
    clf.fit(Xtr, y_tr)
    return clf.predict_proba(Xva)[:, 1]


# ---------------------------------------------------------------- OOF orchestration
def _cache_path(tag):
    return os.path.join(CACHE_DIR, f"{tag}.npz")


def build_base_oof(weight_fn=None, tag="plain", include_tabpfn=True,
                   tabpfn_recency=False, verbose=True):
    """
    Generate per-fold base predictions for all members under the rolling folds.

    weight_fn: optional callable(t_train_series) -> sample_weight array, applied to
               the tree/LR members (Phase C). None = unweighted (incumbent).
    Returns dict:
        {
          'folds': [(tr, va), ...],
          'y_val': [y[va] for each fold],
          'preds': { model_name: [pred_va_fold0, pred_va_fold1, ...] },
        }
    Cached to cache/{tag}.npz keyed by a hash of the fold structure + members.
    """
    df, y, Xraw, cat_idx = load_data()
    n = len(df)
    folds = rolling_folds(n)
    members = list(BASE_ORDER) if include_tabpfn else [m for m in BASE_ORDER if m != "TabPFN"]

    key = hashlib.md5(
        (tag + "|" + ",".join(members) + "|" + str(folds[-1][1][-1]) + f"|rec{tabpfn_recency}").encode()
    ).hexdigest()[:8]
    cpath = os.path.join(CACHE_DIR, f"oof_{tag}_{key}.npz")
    if os.path.exists(cpath):
        if verbose:
            print(f"[cache] loading {os.path.basename(cpath)}", flush=True)
        z = np.load(cpath, allow_pickle=True)
        return z["payload"].item()

    factories = make_models()
    preds = {m: [] for m in members}
    y_val = []
    for k, (tr, va) in enumerate(folds):
        Xtr_raw, Xva_raw = Xraw.iloc[tr], Xraw.iloc[va]
        y_tr = y[tr]
        y_val.append(y[va])
        sw = None
        if weight_fn is not None:
            sw = weight_fn(df["t"].iloc[tr])
        t0 = time.time()
        for m in members:
            if m == "TabPFN":
                p = _fit_predict_tabpfn(Xtr_raw, y_tr, Xva_raw,
                                        recency_context=tabpfn_recency,
                                        cap=4000, seed=k)
            elif m == "Logistic":
                p = _fit_predict_tree("Logistic", None, Xtr_raw, y_tr, Xva_raw,
                                      cat_idx, sample_weight=sw)
            else:
                p = _fit_predict_tree(m, factories[m], Xtr_raw, y_tr, Xva_raw,
                                      cat_idx, sample_weight=sw)
            preds[m].append(np.asarray(p, dtype=float))
        if verbose:
            print(f"[{tag}] fold{k} train={len(tr)} val={len(va)} "
                  f"pos_val={int(y[va].sum())} ({time.time()-t0:.0f}s)", flush=True)

    payload = {"folds": [(tr.tolist(), va.tolist()) for tr, va in folds],
               "y_val": [yv.tolist() for yv in y_val],
               "preds": {m: [p.tolist() for p in preds[m]] for m in members},
               "members": members}
    np.savez_compressed(cpath, payload=np.array(payload, dtype=object))
    if verbose:
        print(f"[cache] saved {os.path.basename(cpath)}", flush=True)
    return payload


# ---------------------------------------------------------------- ensemble combiners
def prob_mean(pred_list):
    """Equal-weight probability average. pred_list: list of 1-D arrays (members)."""
    return np.mean(np.column_stack(pred_list), axis=1)


def rank_mean(pred_list):
    """Equal-weight rank average, normalised to [0,1]. Rank-based, shift-robust."""
    M = np.column_stack(pred_list)
    ranks = np.empty_like(M, dtype=float)
    for j in range(M.shape[1]):
        order = M[:, j].argsort()
        r = np.empty(len(order), dtype=float)
        r[order] = np.arange(len(order))
        # average ties
        ranks[:, j] = pd.Series(M[:, j]).rank(method="average").values - 1
    rk = ranks.mean(axis=1)
    denom = (len(rk) - 1) if len(rk) > 1 else 1
    return rk / denom


# ---------------------------------------------------------------- metrics
def score(y_true, p):
    y_true = np.asarray(y_true)
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return float("nan"), float("nan")
    return roc_auc_score(y_true, p), average_precision_score(y_true, p)


def evaluate_combiner(payload, members, combiner):
    """
    Apply `combiner` to the chosen members per fold; return per-fold and aggregate.
    """
    per_fold = []
    for k in range(len(payload["y_val"])):
        yv = np.array(payload["y_val"][k])
        pl = [np.array(payload["preds"][m][k]) for m in members]
        p = combiner(pl)
        auc, ap = score(yv, p)
        per_fold.append({"fold": k, "n_val": int(len(yv)),
                         "pos_val": int(yv.sum()), "roc_auc": auc, "pr_auc": ap})
    aucs = np.array([f["roc_auc"] for f in per_fold], float)
    aps = np.array([f["pr_auc"] for f in per_fold], float)
    return {
        "per_fold": per_fold,
        "roc_auc_mean": float(np.nanmean(aucs)), "roc_auc_std": float(np.nanstd(aucs)),
        "pr_auc_mean": float(np.nanmean(aps)), "pr_auc_std": float(np.nanstd(aps)),
    }


def base_rate_info():
    df, y, _, _ = load_data()
    return {"n": int(len(y)), "pos": int(y.sum()), "pos_rate": float(y.mean()),
            "date_min": str(df["t"].min()), "date_max": str(df["t"].max())}


def fmt_table(rows, headers):
    """Plain monospace table -> string."""
    widths = [max(len(str(h)), *(len(str(r[i])) for r in rows)) if rows else len(str(h))
              for i, h in enumerate(headers)]
    line = " | ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers))
    sep = "-+-".join("-" * widths[i] for i in range(len(headers)))
    out = [line, sep]
    for r in rows:
        out.append(" | ".join(str(r[i]).ljust(widths[i]) for i in range(len(headers))))
    return "\n".join(out)


if __name__ == "__main__":
    info = base_rate_info()
    print(json.dumps(info, indent=2))
    folds = rolling_folds(info["n"])
    print(f"\nrolling-origin folds ({len(folds)}):")
    for k, (tr, va) in enumerate(folds):
        print(f"  fold{k}: train[0:{tr[-1]+1}]  val[{va[0]}:{va[-1]+1}]  "
              f"n_train={len(tr)} n_val={len(va)}")
