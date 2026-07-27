"""
scale_curve.py — Consolidates the three budget-scale points of the
cross-composition experiment (reference budget = 5x, 10x, 20x the M-hull cost)
into scale curves: how the quantity-quality standings and the budget-vs-design
trade-off move with engagement size.

Reads the three per-scale results CSVs (same design, same seed scheme, same
neutral regime; only the reference budget differs) and produces:

  fig_scale_curve.png   — (A) pure-fleet pairwise log-FER vs scale
                          (B) budget-edge effect vs composition span vs scale
  scale_curve_summary.md

Usage:
    python3 scale_curve.py
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCALES = [(5, "cross_composition_results.csv"),
          (10, "cross10_results.csv"),
          (20, "cross20_results.csv")]
MIX_LABELS = ["L", "M", "H", "LM", "LH", "MH", "C", "L+", "M+", "H+"]


def pure_logfer(par, blue, red):
    row = par[(par.blue_mix == blue) & (par.red_mix == red)]
    return float(row.logfer_mean.iloc[0]) if len(row) else np.nan


def metrics(path):
    df = pd.read_csv(path)
    par = df[df.rho == 1.0]
    rank = par.groupby("blue_mix").p_victory.mean()
    by_rho = df.groupby("rho").p_victory.mean()
    return {
        "L_vs_M": pure_logfer(par, "L", "M"),
        "L_vs_H": pure_logfer(par, "L", "H"),
        "M_vs_H": pure_logfer(par, "M", "H"),
        "design_span": rank.max() - rank.min(),
        "budget_effect": by_rho[1.1] - by_rho[0.9],
        "best_blue": rank.idxmax(),
        "worst_red_toughness": par.groupby("red_mix").p_victory.mean().idxmax(),
        "util_mean": df.blue_util.mean(),
        "util_min": df.blue_util.min(),
    }


def main():
    rows = {s: metrics(p) for s, p in SCALES}
    x = list(rows)
    notes = []
    for s in x:
        m = rows[s]
        notes.append(
            f"scale {s}xM: pure log-FER L-vs-M {m['L_vs_M']:+.2f}, "
            f"L-vs-H {m['L_vs_H']:+.2f}, M-vs-H {m['M_vs_H']:+.2f}; "
            f"design span {m['design_span']:.2f} vs budget(+/-10%) effect "
            f"{m['budget_effect']:+.2f}; best Blue {m['best_blue']}; "
            f"weakest Red {m['worst_red_toughness']}; "
            f"utilisation mean {m['util_mean']:.0%} (min {m['util_min']:.0%}).")

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5))
    ax = axes[0]
    series = [("L_vs_H", "#1a5276", "L vs H"),
              ("M_vs_H", "#1e8449", "M vs H"),
              ("L_vs_M", "#7d6608", "L vs M")]
    for key, c, lab in series:
        ax.plot(x, [rows[s][key] for s in x], marker="o", lw=2, color=c, label=lab)
    ax.axhline(0, color="gray", lw=0.8, ls=":")
    ax.set_xticks(x); ax.set_xticklabels([f"{s}×M" for s in x])
    ax.set_xlabel("Reference budget (engagement scale)")
    ax.set_ylabel("log-FER at parity (>0: first fleet wins)")
    ax.set_title("Pure-fleet standings vs scale:\nthe heavy fleet collapses as salvos grow", fontsize=9)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(x, [rows[s]["design_span"] for s in x], marker="o", lw=2,
            color="#7d3c98", label="composition-choice span (parity)")
    ax.plot(x, [rows[s]["budget_effect"] for s in x], marker="s", lw=2,
            color="#b9770e", label="±10% budget effect")
    ax.set_xticks(x); ax.set_xticklabels([f"{s}×M" for s in x])
    ax.set_xlabel("Reference budget (engagement scale)")
    ax.set_ylabel("Δ mean P(Blue victory)")
    ax.set_title("What matters more, design or budget?\n(the answer depends on scale)", fontsize=9)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig("fig_scale_curve.png", dpi=150)

    with open("scale_curve_summary.md", "w") as f:
        f.write("# Budget-scale curve (5x / 10x / 20x M)\n\n")
        for n in notes:
            f.write("- " + n + "\n\n")
    print("\n".join("- " + n for n in notes))
    print("\nWrote fig_scale_curve.png + scale_curve_summary.md")


if __name__ == "__main__":
    main()
