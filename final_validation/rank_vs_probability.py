"""
PHASE B - Rank averaging vs probability averaging.

Both ensembles use the SAME 7 frozen members and the SAME rolling-origin folds as
Phase A (loaded from cache -- no retraining). The only thing that changes is the
combiner: mean of probabilities vs mean of normalised within-fold ranks.

Decision (recorded to JSON):
  * majority-fold winner   (which combiner wins PR-AUC on more folds)
  * mean PR-AUC winner
  * statistical margin      (mean PR-AUC delta vs fold-to-fold std = the noise band)

Output: results/rank_vs_probability_results.json
"""
import os, json
import numpy as np

import common as C
from final_ensemble_validation import get_plain_payload


def main():
    payload = get_plain_payload(verbose=False)
    members = C.BASE_ORDER

    prob = C.evaluate_combiner(payload, members, C.prob_mean)
    rank = C.evaluate_combiner(payload, members, C.rank_mean)

    # per-fold comparison
    rows, prob_wins, rank_wins = [], 0, 0
    for pf_p, pf_r in zip(prob["per_fold"], rank["per_fold"]):
        dp = pf_r["pr_auc"] - pf_p["pr_auc"]
        w = "rank" if dp > 0 else ("prob" if dp < 0 else "tie")
        if w == "rank":
            rank_wins += 1
        elif w == "prob":
            prob_wins += 1
        rows.append([f"fold{pf_p['fold']}", pf_p["pos_val"],
                     f"{pf_p['pr_auc']:.4f}", f"{pf_r['pr_auc']:.4f}",
                     f"{dp:+.4f}", w])

    margin = rank["pr_auc_mean"] - prob["pr_auc_mean"]
    noise = max(prob["pr_auc_std"], rank["pr_auc_std"])
    mean_winner = "rank" if margin > 0 else ("prob" if margin < 0 else "tie")
    majority_winner = ("rank" if rank_wins > prob_wins
                       else "prob" if prob_wins > rank_wins else "tie")
    robust = (mean_winner == "rank") and (majority_winner == "rank") and (margin > noise)

    print("=" * 84)
    print("PHASE B  -  RANK AVERAGING vs PROBABILITY AVERAGING  (7 frozen members)")
    print("=" * 84)
    print(C.fmt_table(rows, ["fold", "pos_val", "PROB PR-AUC", "RANK PR-AUC",
                             "delta", "fold winner"]))
    print(f"\n  prob-avg  PR-AUC = {prob['pr_auc_mean']:.4f} +/- {prob['pr_auc_std']:.4f}   "
          f"ROC-AUC = {prob['roc_auc_mean']:.4f}")
    print(f"  rank-avg  PR-AUC = {rank['pr_auc_mean']:.4f} +/- {rank['pr_auc_std']:.4f}   "
          f"ROC-AUC = {rank['roc_auc_mean']:.4f}")
    print(f"  mean PR-AUC margin (rank - prob) = {margin:+.4f}")
    print(f"  noise band (max fold std)        = {noise:.4f}")
    print(f"  majority-fold winner = {majority_winner}   |   mean winner = {mean_winner}")
    print(f"  ROBUST rank>prob (margin > noise AND both criteria)? {robust}")

    out = {
        "phase": "B - rank vs probability averaging",
        "members": members,
        "prob_avg": prob,
        "rank_avg": rank,
        "decision": {
            "fold_wins": {"prob": prob_wins, "rank": rank_wins},
            "majority_fold_winner": majority_winner,
            "mean_pr_auc_winner": mean_winner,
            "mean_pr_auc_margin": float(margin),
            "noise_band_std": float(noise),
            "robust_improvement_over_incumbent": bool(robust),
            "verdict": ("ADOPT rank-avg" if robust else "KEEP prob-avg (incumbent) -- "
                        "rank-avg gain within noise / not majority"),
        },
    }
    path = os.path.join(C.RESULTS_DIR, "rank_vs_probability_results.json")
    json.dump(out, open(path, "w"), indent=2)
    print(f"\nsaved: {os.path.relpath(path, C.ROOT)}")
    return out


if __name__ == "__main__":
    main()
