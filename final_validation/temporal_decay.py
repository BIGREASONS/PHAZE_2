"""
PHASE C - Temporal decay weighting.

Fixed exponential recency weight on created_date (NO tuning -- a single half-life
chosen a-priori from the data span, then frozen). Per the audit shortlist, only the
three gradient-boosted members are retrained with sample_weight:

    w_i = 0.5 ** (age_days_i / HALF_LIFE_DAYS)
    age_days_i = (max_train_date - date_i) in days   (most-recent row -> weight 1.0)

HALF_LIFE_DAYS is fixed at ~1/3 of the total time span. We deliberately do NOT
search it: with one temporal direction you cannot tune a half-life without
overfitting the fold. The value is locked before any metric is seen.

Decay ensemble = equal-weight average of:
    [decay-CatBoost, decay-LightGBM, decay-XGBoost, plain-RF, plain-ExtraTrees,
     plain-Logistic, plain-TabPFN]
compared against the all-plain frozen incumbent under identical rolling folds.

Output: results/temporal_decay_results.json
"""
import os, json
import numpy as np
import pandas as pd

import common as C
from final_ensemble_validation import get_plain_payload, incumbent_eval

# total span ~192 days (see Phase A dates); lock half-life at ~1/3 of span.
HALF_LIFE_DAYS = 64.0
DECAY_MEMBERS = ["CatBoost", "LightGBM", "XGBoost"]


def decay_weight_fn(t_train: pd.Series):
    """Exponential recency weights for a training block (Series of timestamps)."""
    t = pd.to_datetime(t_train).values.astype("datetime64[ns]")
    tmax = np.nanmax(t)
    age_days = (tmax - t) / np.timedelta64(1, "D")
    age_days = np.nan_to_num(age_days, nan=age_days[~np.isnan(age_days)].mean()
                             if np.isnan(age_days).any() else 0.0)
    return np.power(0.5, age_days / HALF_LIFE_DAYS)


def build_decay_payload():
    """OOF where ONLY the 3 GBMs are decay-weighted; others not retrained here."""
    return C.build_base_oof(weight_fn=decay_weight_fn, tag="decay",
                            include_tabpfn=False, verbose=True)


def main():
    plain = get_plain_payload(verbose=False)
    decay = build_decay_payload()  # contains decay versions of all non-TabPFN members

    # Decay ensemble: decay GBMs + plain RF/ET/Logistic/TabPFN
    def decay_combiner_per_fold(k):
        cols = []
        for m in C.BASE_ORDER:
            if m in DECAY_MEMBERS:
                cols.append(np.array(decay["preds"][m][k]))
            else:
                cols.append(np.array(plain["preds"][m][k]))
        return C.prob_mean(cols)

    # evaluate per fold manually (mixed source), then aggregate
    per_fold = []
    for k in range(len(plain["y_val"])):
        yv = np.array(plain["y_val"][k])
        p = decay_combiner_per_fold(k)
        auc, ap = C.score(yv, p)
        per_fold.append({"fold": k, "n_val": int(len(yv)), "pos_val": int(yv.sum()),
                         "roc_auc": auc, "pr_auc": ap})
    aps = np.array([f["pr_auc"] for f in per_fold])
    aucs = np.array([f["roc_auc"] for f in per_fold])
    decay_res = {"per_fold": per_fold,
                 "roc_auc_mean": float(np.nanmean(aucs)), "roc_auc_std": float(np.nanstd(aucs)),
                 "pr_auc_mean": float(np.nanmean(aps)), "pr_auc_std": float(np.nanstd(aps))}

    inc = incumbent_eval(plain)

    rows, dec_wins, inc_wins = [], 0, 0
    for a, b in zip(inc["per_fold"], decay_res["per_fold"]):
        d = b["pr_auc"] - a["pr_auc"]
        w = "decay" if d > 0 else ("incumbent" if d < 0 else "tie")
        if w == "decay":
            dec_wins += 1
        elif w == "incumbent":
            inc_wins += 1
        rows.append([f"fold{a['fold']}", a["pos_val"],
                     f"{a['pr_auc']:.4f}", f"{b['pr_auc']:.4f}", f"{d:+.4f}", w])

    margin = decay_res["pr_auc_mean"] - inc["pr_auc_mean"]
    noise = max(inc["pr_auc_std"], decay_res["pr_auc_std"])
    majority = ("decay" if dec_wins > inc_wins else
                "incumbent" if inc_wins > dec_wins else "tie")
    robust = (margin > 0) and (majority == "decay") and (margin > noise)

    print("=" * 84)
    print(f"PHASE C  -  TEMPORAL DECAY WEIGHTING  (half-life = {HALF_LIFE_DAYS:.0f} days, FIXED)")
    print("=" * 84)
    print(f"  decay-weighted members: {DECAY_MEMBERS}  |  others = plain frozen members")
    print(C.fmt_table(rows, ["fold", "pos_val", "INCUMBENT PR", "DECAY PR",
                             "delta", "fold winner"]))
    print(f"\n  incumbent PR-AUC = {inc['pr_auc_mean']:.4f} +/- {inc['pr_auc_std']:.4f}")
    print(f"  decay     PR-AUC = {decay_res['pr_auc_mean']:.4f} +/- {decay_res['pr_auc_std']:.4f}")
    print(f"  mean PR-AUC margin (decay - incumbent) = {margin:+.4f}")
    print(f"  noise band (max fold std)              = {noise:.4f}")
    print(f"  majority-fold winner = {majority}")
    print(f"  ROBUST decay>incumbent (margin > noise)? {robust}")

    out = {
        "phase": "C - temporal decay weighting",
        "half_life_days": HALF_LIFE_DAYS,
        "decay_members": DECAY_MEMBERS,
        "tuning": "none -- half-life locked a-priori at ~1/3 of span before any metric seen",
        "incumbent": inc,
        "decay_ensemble": decay_res,
        "decision": {
            "fold_wins": {"incumbent": inc_wins, "decay": dec_wins},
            "majority_fold_winner": majority,
            "mean_pr_auc_margin": float(margin),
            "noise_band_std": float(noise),
            "robust_improvement_over_incumbent": bool(robust),
            "verdict": ("ADOPT decay weighting" if robust else
                        "KEEP incumbent -- decay gain within noise / not majority"),
        },
    }
    path = os.path.join(C.RESULTS_DIR, "temporal_decay_results.json")
    json.dump(out, open(path, "w"), indent=2)
    print(f"\nsaved: {os.path.relpath(path, C.ROOT)}")
    return out


if __name__ == "__main__":
    main()
