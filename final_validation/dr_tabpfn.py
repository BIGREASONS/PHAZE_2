"""
PHASE D - DR-TabPFN (Drift-Resilient TabPFN).

STEP 1 - Feasibility of the *genuine* DR-TabPFN.
  Drift-Resilient TabPFN (Helli et al., NeurIPS 2024) is a research artifact, not a
  maintained PyPI package. Checked on this machine:
      pip index versions drift-tabpfn   -> no matching distribution
      pip index versions dr-tabpfn       -> no matching distribution
      tabpfn_extensions                  -> NOT installed
  The published implementation ships as a fork of the TabPFN repo plus a separate
  pretrained drift checkpoint that must be downloaded out-of-band. Installing it would
  (a) require a git fork that pins an older tabpfn, risking the working tabpfn 2.0.9
  this project's frozen ensemble depends on, and (b) pull a multi-hundred-MB checkpoint.
  => The real DR-TabPFN is NOT feasible to drop into the frozen stack right now.
  This is recorded as a hard blocker in the JSON (honest stub, per Phase D spec).

STEP 2 - Feasible drift-aware PROXY (clearly labelled as such).
  Instead of leaving Phase D empty, we run the mechanism DR-TabPFN approximates with
  the installed TabPFN: bias the in-context training set toward the most RECENT rows
  (the period closest to the shifted future) instead of a class-balanced random
  sample. This is a legitimate, cheap drift-resilience heuristic -- NOT the published
  model. We add it as a candidate 8th ensemble member and test 7-member vs 8-member.

Output: results/dr_tabpfn_results.json
"""
import os, json, subprocess
import numpy as np

import common as C
from final_ensemble_validation import get_plain_payload, incumbent_eval


def check_real_dr_tabpfn():
    """Probe the environment for a genuine DR-TabPFN install. Returns (feasible, blockers)."""
    blockers = []
    for pkg in ("drift-tabpfn", "dr-tabpfn", "tabpfn-extensions"):
        try:
            r = subprocess.run(["pip", "index", "versions", pkg],
                               capture_output=True, text=True, timeout=60)
            ok = "Available versions" in (r.stdout + r.stderr)
        except Exception:
            ok = False
        if not ok:
            blockers.append(f"PyPI: no installable distribution for '{pkg}'")
    try:
        import tabpfn_extensions  # noqa
        has_ext = True
    except Exception:
        has_ext = False
        blockers.append("python: 'tabpfn_extensions' (hosts drift utilities) not importable")
    feasible = (len(blockers) == 0) and has_ext
    return feasible, blockers


def build_recency_tabpfn_oof():
    """Recency-context TabPFN predictions per rolling fold (the proxy 8th member)."""
    df, y, Xraw, cat_idx = C.load_data()
    folds = C.rolling_folds(len(df))
    key = "recencyTabPFN"
    cpath = os.path.join(C.CACHE_DIR, f"oof_{key}.npz")
    if os.path.exists(cpath):
        print(f"[cache] loading {os.path.basename(cpath)}", flush=True)
        return np.load(cpath, allow_pickle=True)["payload"].item()

    import time
    preds, yval = [], []
    for k, (tr, va) in enumerate(folds):
        t0 = time.time()
        p = C._fit_predict_tabpfn(Xraw.iloc[tr], y[tr], Xraw.iloc[va],
                                  recency_context=True, cap=4000, seed=k)
        preds.append(np.asarray(p, float))
        yval.append(y[va])
        print(f"[recency-TabPFN] fold{k} val={len(va)} ({time.time()-t0:.0f}s)", flush=True)
    payload = {"preds": [p.tolist() for p in preds],
               "y_val": [v.tolist() for v in yval]}
    np.savez_compressed(cpath, payload=np.array(payload, dtype=object))
    return payload


def main():
    feasible, blockers = check_real_dr_tabpfn()
    print("=" * 84)
    print("PHASE D  -  DR-TabPFN  (feasibility + drift-aware proxy)")
    print("=" * 84)
    print(f"  genuine DR-TabPFN feasible here? {feasible}")
    for b in blockers:
        print(f"    BLOCKER: {b}")

    plain = get_plain_payload(verbose=False)
    rec = build_recency_tabpfn_oof()

    # 7-member incumbent vs 8-member (+ recency-TabPFN proxy)
    inc = incumbent_eval(plain)
    per_fold = []
    for k in range(len(plain["y_val"])):
        yv = np.array(plain["y_val"][k])
        cols = [np.array(plain["preds"][m][k]) for m in C.BASE_ORDER]
        cols.append(np.array(rec["preds"][k]))           # 8th member
        p = C.prob_mean(cols)
        auc, ap = C.score(yv, p)
        per_fold.append({"fold": k, "n_val": int(len(yv)), "pos_val": int(yv.sum()),
                         "roc_auc": auc, "pr_auc": ap})
    aps = np.array([f["pr_auc"] for f in per_fold])
    aucs = np.array([f["roc_auc"] for f in per_fold])
    eight = {"per_fold": per_fold,
             "roc_auc_mean": float(np.nanmean(aucs)), "roc_auc_std": float(np.nanstd(aucs)),
             "pr_auc_mean": float(np.nanmean(aps)), "pr_auc_std": float(np.nanstd(aps))}

    rows, w8, w7 = [], 0, 0
    for a, b in zip(inc["per_fold"], eight["per_fold"]):
        d = b["pr_auc"] - a["pr_auc"]
        w = "8-member" if d > 0 else ("7-member" if d < 0 else "tie")
        if w == "8-member":
            w8 += 1
        elif w == "7-member":
            w7 += 1
        rows.append([f"fold{a['fold']}", a["pos_val"],
                     f"{a['pr_auc']:.4f}", f"{b['pr_auc']:.4f}", f"{d:+.4f}", w])

    margin = eight["pr_auc_mean"] - inc["pr_auc_mean"]
    noise = max(inc["pr_auc_std"], eight["pr_auc_std"])
    majority = "8-member" if w8 > w7 else ("7-member" if w7 > w8 else "tie")
    robust = (margin > 0) and (majority == "8-member") and (margin > noise)

    print(C.fmt_table(rows, ["fold", "pos_val", "7-MEMBER PR", "8-MEMBER PR",
                             "delta", "fold winner"]))
    print(f"\n  7-member PR-AUC = {inc['pr_auc_mean']:.4f} +/- {inc['pr_auc_std']:.4f}")
    print(f"  8-member PR-AUC = {eight['pr_auc_mean']:.4f} +/- {eight['pr_auc_std']:.4f}")
    print(f"  mean PR-AUC margin (8 - 7) = {margin:+.4f}   noise band = {noise:.4f}")
    print(f"  majority-fold winner = {majority}   ROBUST? {robust}")

    out = {
        "phase": "D - DR-TabPFN",
        "genuine_dr_tabpfn": {
            "feasible": bool(feasible),
            "blockers": blockers,
            "note": "Real Drift-Resilient TabPFN (Helli et al. 2024) is a research fork "
                    "+ separate checkpoint, not pip-installable; not added to the frozen stack.",
        },
        "proxy": {
            "description": "Recency-context TabPFN: in-context training set biased to the "
                           "most recent rows (all positives kept). A drift heuristic, NOT "
                           "the published DR-TabPFN.",
            "incumbent_7member": inc,
            "candidate_8member": eight,
            "decision": {
                "fold_wins": {"7-member": w7, "8-member": w8},
                "majority_fold_winner": majority,
                "mean_pr_auc_margin": float(margin),
                "noise_band_std": float(noise),
                "robust_improvement_over_incumbent": bool(robust),
                "verdict": ("ADOPT 8-member (add recency-TabPFN)" if robust else
                            "KEEP 7-member incumbent -- proxy gain within noise / not majority"),
            },
        },
    }
    path = os.path.join(C.RESULTS_DIR, "dr_tabpfn_results.json")
    json.dump(out, open(path, "w"), indent=2)
    print(f"\nsaved: {os.path.relpath(path, C.ROOT)}")
    return out


if __name__ == "__main__":
    main()
