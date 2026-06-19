"""
PHASE E - Final gate.

Reads the three experiment JSONs + the Phase A baseline, applies the adoption policy,
and writes final_recommendation_report.md (per-fold tables, mean+/-std, margins,
winners, final recommendation) plus a machine-readable final_decision.json that
Phase F consumes.

DECISION POLICY (locked, applied uniformly):
  Adopt a candidate ONLY if ALL hold:
    1. mean PR-AUC margin over the incumbent > 0
    2. candidate wins PR-AUC on a MAJORITY of folds
    3. mean PR-AUC margin > fold-to-fold std (the noise band)
  Otherwise REJECT. Ties go to the incumbent.
"""
import os, json
import numpy as np

import common as C

RES = C.RESULTS_DIR


def load(name):
    p = os.path.join(RES, name)
    if not os.path.exists(p):
        return None
    return json.load(open(p))


def per_fold_md(per_fold):
    rows = []
    for f in per_fold:
        rows.append(f"| {f['fold']} | {f['n_val']} | {f['pos_val']} | "
                    f"{f['roc_auc']:.4f} | {f['pr_auc']:.4f} |")
    return "| fold | n_val | pos_val | ROC-AUC | PR-AUC |\n|---|---|---|---|---|\n" + "\n".join(rows)


def candidate_block(title, incumbent, candidate, decision, extra=""):
    md = [f"### {title}", ""]
    if extra:
        md += [extra, ""]
    md.append("**Incumbent per-fold**")
    md.append("")
    md.append(per_fold_md(incumbent["per_fold"]))
    md.append("")
    md.append("**Candidate per-fold**")
    md.append("")
    md.append(per_fold_md(candidate["per_fold"]))
    md.append("")
    md += [
        f"- Incumbent PR-AUC: **{incumbent['pr_auc_mean']:.4f} ± {incumbent['pr_auc_std']:.4f}**  "
        f"(ROC-AUC {incumbent['roc_auc_mean']:.4f})",
        f"- Candidate PR-AUC: **{candidate['pr_auc_mean']:.4f} ± {candidate['pr_auc_std']:.4f}**  "
        f"(ROC-AUC {candidate['roc_auc_mean']:.4f})",
        f"- Mean PR-AUC margin: **{decision['mean_pr_auc_margin']:+.4f}**  |  "
        f"noise band (std): **{decision['noise_band_std']:.4f}**",
        f"- Fold wins: `{decision['fold_wins']}`  |  majority winner: "
        f"**{decision['majority_fold_winner']}**",
        f"- Robust improvement over incumbent: **{decision['robust_improvement_over_incumbent']}**",
        f"- Verdict: **{decision['verdict']}**",
        "",
    ]
    return "\n".join(md)


def main():
    A = load("final_ensemble_validation.json")
    B = load("rank_vs_probability_results.json")
    Cc = load("temporal_decay_results.json")
    D = load("dr_tabpfn_results.json")
    if A is None:
        raise SystemExit("Phase A results missing -- run final_ensemble_validation.py first.")

    inc = A["incumbent"]
    info = A["dataset"]

    # gather candidate decisions
    candidates = []
    if B:
        candidates.append(("B", "Rank averaging", B["decision"]["robust_improvement_over_incumbent"],
                           B["decision"]))
    if Cc:
        candidates.append(("C", "Temporal decay weighting",
                           Cc["decision"]["robust_improvement_over_incumbent"], Cc["decision"]))
    if D:
        dd = D["proxy"]["decision"]
        candidates.append(("D", "DR-TabPFN proxy (8th member)",
                           dd["robust_improvement_over_incumbent"], dd))

    winners = [c for c in candidates if c[2]]
    # pick the single best winner by margin if multiple
    final_choice = "incumbent"
    final_detail = "Equal-weight 7-model probability-averaged ensemble (frozen)."
    if winners:
        best = max(winners, key=lambda c: c[3]["mean_pr_auc_margin"])
        final_choice = best[1]
        final_detail = best[3]["verdict"]

    # ---- markdown report
    md = []
    md.append("# GridSight AI — Final Recommendation Report")
    md.append("")
    md.append("_Last-chance-before-freeze validation. Rolling-origin expanding-window CV; "
              "all base learners refit per fold on train rows only (no future leakage); "
              "every candidate uses the identical folds and members as the incumbent._")
    md.append("")
    md.append("## 0. Setup")
    md.append("")
    md.append(f"- Dataset: **N={info['n']}**, positives **{info['pos']}** "
              f"(**{info['pos_rate']:.4f}**), dates {info['date_min'][:10]} → {info['date_max'][:10]}")
    md.append(f"- Folds: **{A['framework']['n_folds']}** expanding-window, first cut at "
              f"{int(A['framework']['first_cut_frac']*100)}% of the timeline")
    md.append(f"- Members: {', '.join(A['framework']['members'])}")
    md.append(f"- Incumbent combiner: **equal-weight probability average**")
    md.append("")
    md.append("**Decision policy** — adopt a candidate ONLY if mean PR-AUC margin > 0 "
              "AND it wins a majority of folds AND the margin exceeds the fold-to-fold "
              "std (noise band). Otherwise reject; ties go to the incumbent.")
    md.append("")

    md.append("## 1. Frozen incumbent (baseline)")
    md.append("")
    md.append(per_fold_md(inc["per_fold"]))
    md.append("")
    md.append(f"- **Incumbent PR-AUC = {inc['pr_auc_mean']:.4f} ± {inc['pr_auc_std']:.4f}**, "
              f"ROC-AUC = {inc['roc_auc_mean']:.4f} ± {inc['roc_auc_std']:.4f}")
    md.append(f"- Fold-to-fold PR-AUC noise band (std) = **{inc['pr_auc_std']:.4f}** — "
              "the bar every candidate must clear.")
    md.append("")

    md.append("## 2. Candidate experiments")
    md.append("")
    if B:
        md.append(candidate_block("B — Rank averaging vs probability averaging",
                                  B["prob_avg"], B["rank_avg"], B["decision"]))
    if Cc:
        md.append(candidate_block(
            f"C — Temporal decay weighting (half-life {Cc['half_life_days']:.0f}d, fixed)",
            Cc["incumbent"], Cc["decay_ensemble"], Cc["decision"],
            extra=f"Decay-weighted members: {', '.join(Cc['decay_members'])}; tuning: {Cc['tuning']}"))
    if D:
        extra = ("Genuine DR-TabPFN feasible: **{}**. Blockers: {}. "
                 "Reported result uses a clearly-labelled recency-context TabPFN **proxy**.").format(
            D["genuine_dr_tabpfn"]["feasible"],
            "; ".join(D["genuine_dr_tabpfn"]["blockers"]) or "none")
        md.append(candidate_block("D — DR-TabPFN (proxy, as 8th member)",
                                  D["proxy"]["incumbent_7member"], D["proxy"]["candidate_8member"],
                                  D["proxy"]["decision"], extra=extra))

    md.append("## 3. Summary leaderboard")
    md.append("")
    md.append("| Experiment | Cand. PR-AUC (mean±std) | Margin vs incumbent | "
              "Noise band | Majority winner | Robust? | Run verdict |")
    md.append("|---|---|---|---|---|---|---|")
    if B:
        d = B["decision"]
        md.append(f"| B Rank avg | {B['rank_avg']['pr_auc_mean']:.4f}±{B['rank_avg']['pr_auc_std']:.4f} "
                  f"| {d['mean_pr_auc_margin']:+.4f} | {d['noise_band_std']:.4f} | "
                  f"{d['majority_fold_winner']} | {d['robust_improvement_over_incumbent']} | {d['verdict']} |")
    if Cc:
        d = Cc["decision"]
        md.append(f"| C Temporal decay | {Cc['decay_ensemble']['pr_auc_mean']:.4f}±{Cc['decay_ensemble']['pr_auc_std']:.4f} "
                  f"| {d['mean_pr_auc_margin']:+.4f} | {d['noise_band_std']:.4f} | "
                  f"{d['majority_fold_winner']} | {d['robust_improvement_over_incumbent']} | {d['verdict']} |")
    if D:
        d = D["proxy"]["decision"]
        e = D["proxy"]["candidate_8member"]
        md.append(f"| D DR-TabPFN proxy | {e['pr_auc_mean']:.4f}±{e['pr_auc_std']:.4f} "
                  f"| {d['mean_pr_auc_margin']:+.4f} | {d['noise_band_std']:.4f} | "
                  f"{d['majority_fold_winner']} | {d['robust_improvement_over_incumbent']} | {d['verdict']} |")
    md.append("")

    md.append("## 4. Final recommendation")
    md.append("")
    if winners:
        md.append(f"**ADOPT: {final_choice}.** {final_detail}")
        md.append("")
        md.append("This candidate cleared all three gates (positive margin, fold majority, "
                  "margin above the noise band). Phase F exports it.")
    else:
        md.append("**FREEZE the equal-weight 7-model probability-averaged ensemble.**")
        md.append("")
        md.append("No candidate cleared the noise band on a majority of folds. Under "
                  f"AV-AUC≈0.87 shift and only {info['pos']} positives, every measured "
                  "improvement is within fold-to-fold variance. Per policy, ties go to the "
                  "incumbent. The three checks did their job: they confirm the freeze rather "
                  "than overturn it. Phase F exports the incumbent.")
    md.append("")

    report_path = os.path.join(RES, "final_recommendation_report.md")
    open(report_path, "w", encoding="utf-8").write("\n".join(md))

    decision = {
        "final_choice": final_choice,
        "adopt_candidate": bool(winners),
        "detail": final_detail,
        "incumbent_pr_auc": inc["pr_auc_mean"],
        "incumbent_pr_auc_std": inc["pr_auc_std"],
        "candidates": [{"phase": c[0], "name": c[1], "robust": c[2],
                        "margin": c[3]["mean_pr_auc_margin"]} for c in candidates],
        # explicit config Phase F should export
        "export": {
            "ensemble": "equal-weight probability average",
            "members": C.BASE_ORDER,
            "combiner": "prob_mean",
            "add_recency_tabpfn": final_choice.startswith("DR-TabPFN"),
            "use_rank_avg": final_choice.startswith("Rank"),
            "use_temporal_decay": final_choice.startswith("Temporal"),
        },
    }
    json.dump(decision, open(os.path.join(RES, "final_decision.json"), "w"), indent=2)

    print("=" * 84)
    print("PHASE E  -  FINAL GATE")
    print("=" * 84)
    print("\n".join(md[-12:]))
    print(f"\nsaved: {os.path.relpath(report_path, C.ROOT)}")
    print(f"saved: {os.path.relpath(os.path.join(RES, 'final_decision.json'), C.ROOT)}")
    return decision


if __name__ == "__main__":
    main()
