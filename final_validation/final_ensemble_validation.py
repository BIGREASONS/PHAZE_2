"""
PHASE A - Rolling-origin evaluation framework + frozen-incumbent baseline.

Run this FIRST. It trains the 7 base members inside each expanding-window fold
(train-only encoders, no future leakage), caches their out-of-fold predictions, and
scores the frozen incumbent = equal-weight 7-model PROBABILITY-averaged ensemble.

Outputs:
  results/base_oof_plain.npz        (via common cache)  -- reused by Phases B/C/D
  results/final_ensemble_validation.json
  stdout: publication-quality per-fold and summary tables.

This module exposes get_plain_payload() so the other phases never retrain the
unweighted members.
"""
import os, json
import numpy as np

import common as C


def get_plain_payload(verbose=True):
    """Train-or-load the unweighted 7-member base OOF under the rolling folds."""
    return C.build_base_oof(weight_fn=None, tag="plain", include_tabpfn=True,
                            tabpfn_recency=False, verbose=verbose)


def incumbent_eval(payload):
    """Frozen incumbent: equal-weight probability average over all 7 members."""
    members = C.BASE_ORDER
    return C.evaluate_combiner(payload, members, C.prob_mean)


def standalone_table(payload):
    """Per-member mean ROC/PR-AUC across folds, for context."""
    rows = []
    for m in C.BASE_ORDER:
        res = C.evaluate_combiner(payload, [m], lambda pl: pl[0])
        rows.append([m, f"{res['roc_auc_mean']:.4f}", f"{res['roc_auc_std']:.4f}",
                     f"{res['pr_auc_mean']:.4f}", f"{res['pr_auc_std']:.4f}"])
    return rows


def main():
    info = C.base_rate_info()
    print("=" * 84)
    print("PHASE A  -  ROLLING-ORIGIN TEMPORAL VALIDATION FRAMEWORK")
    print("=" * 84)
    print(f"N={info['n']}  pos={info['pos']}  pos_rate={info['pos_rate']:.4f}  "
          f"dates {info['date_min'][:10]} -> {info['date_max'][:10]}")

    payload = get_plain_payload()
    folds = payload["folds"]

    # fold geometry table
    geo = []
    for k, (tr, va) in enumerate(folds):
        yv = np.array(payload["y_val"][k])
        geo.append([f"fold{k}", f"[0:{tr[-1]+1}]", f"[{va[0]}:{va[-1]+1}]",
                    len(tr), len(va), int(yv.sum()), f"{yv.mean():.4f}"])
    print("\nTABLE A1 - Expanding-window fold geometry")
    print(C.fmt_table(geo, ["fold", "train_rows", "val_rows", "n_train",
                            "n_val", "pos_val", "pos_rate_val"]))

    # standalone members
    print("\nTABLE A2 - Standalone base members (mean +/- std across folds)")
    print(C.fmt_table(standalone_table(payload),
                      ["member", "ROC-AUC", "ROC sd", "PR-AUC", "PR sd"]))

    # incumbent
    inc = incumbent_eval(payload)
    pf = []
    for f in inc["per_fold"]:
        pf.append([f"fold{f['fold']}", f["n_val"], f["pos_val"],
                   f"{f['roc_auc']:.4f}", f"{f['pr_auc']:.4f}"])
    print("\nTABLE A3 - FROZEN INCUMBENT (equal-weight 7-model probability average)")
    print(C.fmt_table(pf, ["fold", "n_val", "pos_val", "ROC-AUC", "PR-AUC"]))
    print(f"\n  INCUMBENT  ROC-AUC = {inc['roc_auc_mean']:.4f} +/- {inc['roc_auc_std']:.4f}")
    print(f"  INCUMBENT  PR-AUC  = {inc['pr_auc_mean']:.4f} +/- {inc['pr_auc_std']:.4f}")
    print(f"  >> Fold-to-fold PR-AUC noise band (std) = {inc['pr_auc_std']:.4f}  "
          f"(a candidate must beat the incumbent mean by MORE than this to be adopted)")

    out = {
        "dataset": info,
        "framework": {
            "scheme": "rolling-origin expanding window",
            "n_folds": len(folds),
            "first_cut_frac": C.FIRST_CUT_FRAC,
            "members": C.BASE_ORDER,
            "encoders": "fit on train rows only, per fold (no future leakage)",
        },
        "fold_geometry": [
            {"fold": k, "n_train": len(tr), "n_val": len(va),
             "pos_val": int(np.array(payload["y_val"][k]).sum())}
            for k, (tr, va) in enumerate(folds)],
        "standalone": {
            m: C.evaluate_combiner(payload, [m], lambda pl: pl[0]) for m in C.BASE_ORDER},
        "incumbent": inc,
        "noise_band_pr_auc_std": inc["pr_auc_std"],
    }
    path = os.path.join(C.RESULTS_DIR, "final_ensemble_validation.json")
    json.dump(out, open(path, "w"), indent=2)
    print(f"\nsaved: {os.path.relpath(path, C.ROOT)}")
    return out


if __name__ == "__main__":
    main()
