"""
alloc_robustness.py — Quantifies how the fleet-allocation rule changes the
paper's force-design results (R6 / fig3 / fig4).

Compares two paper-scale farm runs that share the SAME design matrix and the
SAME per-design-point seeds and differ ONLY in how Blue's integer fleet is
built from its budget shares:

  round  : farm.force_from_shares — per-platform round(); mixed fleets can
           overshoot the budget by up to ~+20% (centroid 35.2 -> 42.3 cost).
  capped : farm.best_integer_fleet — exact enumeration under a HARD budget cap.

If the R6 conclusion (value of heterogeneity: best mixed - best pure > 0) and
the fig3 mixture ranking survive the switch, they are real; the difference
between the two runs measures how much of the published effect is a rounding
windfall.

Usage:
    python3 alloc_robustness.py farm_paperscale_results.csv farm_capped_results.csv

Outputs: fig8_alloc_robustness.png, alloc_robustness_summary.md
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROUND_CSV = sys.argv[1] if len(sys.argv) > 1 else "farm_paperscale_results.csv"
CAPPED_CSV = sys.argv[2] if len(sys.argv) > 2 else "farm_capped_results.csv"

PROC_KEYS = ["sigma_b", "sigma_r", "tau", "rho", "p_o", "p_d", "sd", "order"]
ORDERS = ["blue_first", "simultaneous", "red_first"]
MIX_KEYS = ["s_L", "s_M", "s_H"]


def prep(path):
    df = pd.read_csv(path)
    df["pure"] = (df.s_L == 1) | (df.s_M == 1) | (df.s_H == 1)
    return df


def het_values(df):
    """R6 per matched process regime: best mixed - best pure P(victory)."""
    def one(g):
        pure = g[g.pure].p_victory
        mixed = g[~g.pure].p_victory
        if pure.empty or mixed.empty:
            return np.nan
        return mixed.max() - pure.max()
    return (df.groupby(PROC_KEYS, dropna=False)
              .apply(one, include_groups=False).rename("het").reset_index())


def mixture_rank(df, lo=0.95, hi=1.25):
    """fig3: mean P(victory) per mixture at near-parity budgets."""
    par = df[(df.rho > lo) & (df.rho < hi)]
    return par.groupby(MIX_KEYS).p_victory.mean().sort_values(ascending=False)


def mix_label(idx):
    s_L, s_M, s_H = idx
    names = []
    for v, n in ((s_L, "L"), (s_M, "M"), (s_H, "H")):
        if v >= 0.6:
            return n if v == 1 else n + "+"
        if v > 0:
            names.append(n)
    return "".join(names) if len(names) < 3 else "C"


def main():
    rnd, cap = prep(ROUND_CSV), prep(CAPPED_CSV)
    notes = []
    notes.append(f"Runs: round = {ROUND_CSV} ({len(rnd)} dp), "
                 f"capped = {CAPPED_CSV} ({len(cap)} dp); same design+seeds, "
                 f"allocation only difference.")
    if "blue_util" in cap.columns:
        notes.append(f"Capped-run budget utilisation: mean "
                     f"{cap.blue_util.mean():.0%}, min {cap.blue_util.min():.0%}.")

    # --- overall agreement per design point ---
    m = rnd.merge(cap, on=["dp"], suffixes=("_r", "_c"))
    r = np.corrcoef(m.p_victory_r, m.p_victory_c)[0, 1]
    dbar = (m.p_victory_c - m.p_victory_r).mean()
    notes.append(f"Per-design-point P(victory): corr = {r:.3f}; mean shift "
                 f"capped-round = {dbar:+.3f} (capped fleets are never richer "
                 f"than the budget, so a negative shift is expected).")

    # --- R6: value of heterogeneity ---
    hr, hc = het_values(rnd), het_values(cap)
    hm = hr.merge(hc, on=PROC_KEYS, suffixes=("_r", "_c")).dropna()
    lines = []
    for o in ORDERS:
        s = hm[hm.order == o]
        lines.append(f"{o}: round med {s.het_r.median():+.3f} "
                     f"(pos {100*(s.het_r>0).mean():.0f}%) -> capped med "
                     f"{s.het_c.median():+.3f} (pos {100*(s.het_c>0).mean():.0f}%)")
    notes.append("R6 heterogeneity value (best mixed - best pure) by order: "
                 + "; ".join(lines) + ".")
    all_r, all_c = hm.het_r, hm.het_c
    notes.append(f"R6 pooled: round median {all_r.median():+.3f} / mean "
                 f"{all_r.mean():+.3f} -> capped median {all_c.median():+.3f} / "
                 f"mean {all_c.mean():+.3f}; share of regimes with positive "
                 f"heterogeneity value {100*(all_r>0).mean():.0f}% -> "
                 f"{100*(all_c>0).mean():.0f}%. "
                 f"Sign {'PRESERVED' if (all_c.median()>0)==(all_r.median()>0) else 'FLIPPED'} "
                 f"under the hard budget cap.")

    # --- fig3: best design at parity ---
    mr, mc = mixture_rank(rnd), mixture_rank(cap)
    notes.append("fig3 mixture ranking at near-parity (round): "
                 + ", ".join(f"{mix_label(k)}={v:.2f}" for k, v in mr.items()) + ".")
    notes.append("fig3 mixture ranking at near-parity (capped): "
                 + ", ".join(f"{mix_label(k)}={v:.2f}" for k, v in mc.items()) + ".")
    notes.append(f"Best design at parity: round -> {mix_label(mr.index[0])}, "
                 f"capped -> {mix_label(mc.index[0])}.")

    # --- figure: paired R6 boxes + mixture ranking side-by-side ---
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    ax = axes[0]
    data, ticks = [], []
    for i, o in enumerate(ORDERS):
        s = hm[hm.order == o]
        data += [s.het_r, s.het_c]
        ticks.append(o.replace("_", " "))
    pos = [0, 0.45, 1.2, 1.65, 2.4, 2.85]
    bp = ax.boxplot(data, positions=pos, widths=0.38, showfliers=False,
                    patch_artist=True)
    for i, box in enumerate(bp["boxes"]):
        box.set_facecolor("#c7d7ea" if i % 2 == 0 else "#f2c9b0")
        box.set_edgecolor("#444444")
    ax.axhline(0, color="crimson", lw=0.8, ls="--")
    ax.set_xticks([0.22, 1.42, 2.62]); ax.set_xticklabels(ticks, fontsize=8)
    ax.set_ylabel("R6: best mixed − best pure ΔP(vict)")
    ax.set_title("Value of heterogeneity by allocation rule", fontsize=9)
    ax.legend([bp["boxes"][0], bp["boxes"][1]], ["round (may overspend)",
              "capped (hard budget)"], fontsize=7, loc="upper right")
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1]
    labels = [mix_label(k) for k in mr.index]
    x = np.arange(len(labels))
    cap_vals = [mc.get(k, np.nan) for k in mr.index]
    ax.bar(x - 0.19, mr.to_numpy(), width=0.38, color="#c7d7ea",
           edgecolor="#444444", lw=0.5, label="round")
    ax.bar(x + 0.19, cap_vals, width=0.38, color="#f2c9b0",
           edgecolor="#444444", lw=0.5, label="capped")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Mean P(victory), near-parity")
    ax.set_title("fig3 mixture ranking by allocation rule", fontsize=9)
    ax.legend(fontsize=7); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig("fig8_alloc_robustness.png", dpi=150)

    with open("alloc_robustness_summary.md", "w") as f:
        f.write("# Allocation-robustness check (R6 / fig3)\n\n")
        for n in notes:
            f.write("- " + n + "\n\n")
    print("\n".join("- " + n for n in notes))
    print("\nWrote fig8_alloc_robustness.png + alloc_robustness_summary.md")


if __name__ == "__main__":
    main()
