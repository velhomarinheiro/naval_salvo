"""
parity_scale.py — Consolidated analysis of the strict-parity cross-composition
experiment across the three budget scales (5x, 10x, 20x the M-hull cost).

Strict parity means both sides always hold the SAME budget (rho = 1), so the
outcome is driven by force composition alone: no budget edge, no fusion edge
(neutral regime), same engagement order. Every cell is one Blue composition
facing one Red composition, 20,000 Monte-Carlo battles.

Produces:
  parity_scale_matrix.png    — the three 10x10 log-FER matrices side by side
  parity_scale_curve.png     — how the pure-fleet standings and the spread of
                               outcomes move with engagement scale
  parity_scale_summary.md    — numbers, with Monte-Carlo confidence intervals

Usage:  python3 parity_scale.py
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

SCALES = [(5, "parity5_results.csv"), (10, "parity10_results.csv"),
          (20, "parity20_results.csv")]
MIX_LABELS = ["L", "M", "H", "LM", "LH", "MH", "C", "L+", "M+", "H+"]
PURES = ["L", "M", "H"]
plt.rcParams.update({"figure.dpi": 150, "font.size": 9})


def load(path):
    d = pd.read_csv(path)
    F = d.pivot(index="blue_mix", columns="red_mix", values="logfer_mean").reindex(
        index=MIX_LABELS, columns=MIX_LABELS)
    S = d.pivot(index="blue_mix", columns="red_mix", values="logfer_se").reindex(
        index=MIX_LABELS, columns=MIX_LABELS)
    P = d.pivot(index="blue_mix", columns="red_mix", values="p_victory").reindex(
        index=MIX_LABELS, columns=MIX_LABELS)
    return d, F, S, P


def pair(F, S, b, r):
    """log-FER of b-vs-r with its 95% MC CI, and whether it is decided."""
    est, ci = F.loc[b, r], 1.96 * S.loc[b, r]
    return est, ci, abs(est) > ci


def main():
    data = {s: load(p) for s, p in SCALES}
    notes = []

    # ---- symmetry checks -------------------------------------------------
    for s, (d, F, S, P) in data.items():
        diag = np.abs(np.diag(F.to_numpy())).max()
        anti = np.abs(F.to_numpy() + F.to_numpy().T).max()
        notes.append(f"CHECK {s}xM: mirror-diagonal max |log-FER| = {diag:.3f}; "
                     f"antisymmetry max = {anti:.3f}; median MC 95% CI on a cell "
                     f"= +/-{1.96*np.nanmedian(S.to_numpy()):.3f} "
                     f"({int(d.reps.iloc[0]):,} reps/cell).")

    # ---- pure 3x3 across scales -----------------------------------------
    for s, (d, F, S, P) in data.items():
        bits, wins, undec = [], {p: 0 for p in PURES}, []
        for b in PURES:
            for r in PURES:
                if b == r:
                    continue
                est, ci, dec = pair(F, S, b, r)
                bits.append(f"{b} vs {r}: {est:+.2f}±{ci:.2f}")
                if dec and est > 0:
                    wins[b] += 1
                if not dec:
                    undec.append(f"{b}-{r}")
        best = max(wins, key=wins.get)
        notes.append(f"PURE 3x3 at {s}xM: " + "; ".join(bits) +
                     f". Decided wins {wins} -> strongest pure = '{best}'" +
                     (f"; undecided: {', '.join(undec)}" if undec else "") + ".")

    # ---- full-matrix standings ------------------------------------------
    for s, (d, F, S, P) in data.items():
        row = P.mean(axis=1).sort_values(ascending=False)
        col = P.mean(axis=0).sort_values()
        notes.append(f"BLUE ranking at {s}xM (mean P(vict) vs all Red): " +
                     ", ".join(f"{k}={v:.3f}" for k, v in row.items()) + ".")
        notes.append(f"RED toughness at {s}xM (lower = tougher): " +
                     ", ".join(f"{k}={v:.3f}" for k, v in col.items()) + ".")
        br = P.idxmax(axis=0)
        notes.append(f"BLUE best response at {s}xM: " +
                     ", ".join(f"vs {r}: {br[r]}" for r in MIX_LABELS) + ".")

    # ---- how decisive is composition at each scale? ----------------------
    for s, (d, F, S, P) in data.items():
        off = ~np.eye(len(MIX_LABELS), dtype=bool)
        vals = F.to_numpy()[off]
        ses = S.to_numpy()[off]
        dec = np.abs(vals) > 1.96 * ses
        notes.append(f"DECISIVENESS at {s}xM: of the {off.sum()} non-mirror pairings, "
                     f"{dec.sum()} ({100*dec.mean():.0f}%) are decided at 95% CI; "
                     f"mean |log-FER| = {np.abs(vals).mean():.2f}; "
                     f"max |log-FER| = {np.abs(vals).max():.2f}.")

    # ======================= FIGURES =====================================
    # Fig A: three matrices side by side, shared diverging scale
    lim = max(np.nanmax(np.abs(F.to_numpy())) for _, (_, F, _, _) in data.items())
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim)
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.6))
    for ax, (s, (d, F, S, P)) in zip(axes, data.items()):
        M = F.to_numpy()
        im = ax.imshow(M, cmap="RdBu", norm=norm)
        ax.set_xticks(range(len(MIX_LABELS))); ax.set_xticklabels(MIX_LABELS, fontsize=7)
        ax.set_yticks(range(len(MIX_LABELS))); ax.set_yticklabels(MIX_LABELS, fontsize=7)
        ax.set_title(f"budget {s}×M", fontsize=10)
        ax.set_xlabel("Red composition", fontsize=8)
        if s == 5:
            ax.set_ylabel("Blue composition", fontsize=8)
        # mark the undecided cells (|log-FER| within its own 95% CI) with a dot
        und = np.abs(M) <= 1.96 * S.to_numpy()
        yy, xx = np.where(und)
        ax.scatter(xx, yy, s=5, color="#444444", marker="o", lw=0)
        ax.plot([-0.5, 9.5], [-0.5, 9.5], color="#888888", lw=0.6, ls=":")
    cb = fig.colorbar(im, ax=axes, shrink=0.82, pad=0.015)
    cb.set_label("log-FER  (blue = Blue-favourable; 0 = even)")
    fig.suptitle("Cross-composition at STRICT budget parity, by engagement scale\n"
                 "dots mark pairings not decided at 95% Monte-Carlo CI", fontsize=11)
    fig.savefig("parity_scale_matrix.png", bbox_inches="tight")

    # Fig B: scale curves — pure standings + decisiveness
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.6))
    xs = list(data)
    ax = axes[0]
    for (b, r), c, lab in [(("L", "H"), "#1a5276", "L vs H"),
                           (("M", "H"), "#1e8449", "M vs H"),
                           (("L", "M"), "#b9770e", "L vs M")]:
        est = [data[s][1].loc[b, r] for s in xs]
        ci = [1.96 * data[s][2].loc[b, r] for s in xs]
        ax.errorbar(xs, est, yerr=ci, marker="o", lw=2, capsize=3, color=c, label=lab)
    ax.axhline(0, color="gray", lw=0.8, ls=":")
    ax.set_xticks(xs); ax.set_xticklabels([f"{s}×M" for s in xs])
    ax.set_xlabel("Reference budget (engagement scale)")
    ax.set_ylabel("log-FER at parity")
    ax.set_title("Pure-fleet standings vs scale\n(>0: first fleet wins the exchange)", fontsize=9)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1]
    off = ~np.eye(len(MIX_LABELS), dtype=bool)
    spread = np.array([np.abs(data[s][1].to_numpy()[off]).mean() for s in xs])
    pspan = np.array([data[s][3].mean(axis=1).max() - data[s][3].mean(axis=1).min()
                      for s in xs])
    # The two measures live in different units (log-FER vs probability), so they
    # are indexed to their 5xM value: one dimensionless axis, honest comparison
    # of TRENDS (never two y-scales on one plot).
    ax.plot(xs, spread / spread[0], marker="o", lw=2, color="#7d3c98",
            label="mean |log-FER| over pairings")
    ax.plot(xs, pspan / pspan[0], marker="s", lw=2, color="#b9770e",
            label="span of mean P(victory)")
    for xi, v in zip(xs, spread / spread[0]):
        ax.annotate(f"{v:.2f}", (xi, v), textcoords="offset points", xytext=(0, 7),
                    ha="center", fontsize=7, color="#7d3c98")
    for xi, v in zip(xs, pspan / pspan[0]):
        ax.annotate(f"{v:.2f}", (xi, v), textcoords="offset points", xytext=(0, -12),
                    ha="center", fontsize=7, color="#b9770e")
    ax.axhline(1.0, color="gray", lw=0.8, ls=":")
    ax.set_xticks(xs); ax.set_xticklabels([f"{s}×M" for s in xs])
    ax.set_ylim(0, 1.25)
    ax.set_xlabel("Reference budget (engagement scale)")
    ax.set_ylabel("Indexed to the 5×M value (= 1.0)")
    ax.set_title("How much does composition decide?\n(two measures, indexed — both fall)", fontsize=9)
    ax.legend(fontsize=8, loc="lower left"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig("parity_scale_curve.png")

    with open("parity_scale_summary.md", "w", encoding="utf-8") as f:
        f.write("# Strict-parity cross-composition, by engagement scale\n\n"
                "Both sides always hold the same budget (rho = 1); neutral regime "
                "(simultaneous fire, no fusion edge); 20,000 battles per cell.\n\n")
        for n in notes:
            f.write("- " + n + "\n\n")
    print("\n".join("- " + n for n in notes))
    print("\nWrote parity_scale_matrix.png + parity_scale_curve.png + parity_scale_summary.md")


if __name__ == "__main__":
    main()
