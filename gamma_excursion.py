"""
gamma_excursion.py — Cost-convexity (gamma) robustness excursion for the
LAFusion 2026 experiment (spec sec 5, feeds paper sec 5.5).

Question: gamma > 1 prices concentrated capability superlinearly. As gamma
rises from 1.15 to 1.55, at what value does the budget-optimal Blue force design
flip from QUALITY (H-heavy, few powerful hulls) to QUANTITY (L-heavy, many cheap
hulls)? gamma only reprices Blue's platforms; the budget stays anchored to the
gamma=1.35 Red reference value, so this isolates the high-low mix decision.

Method: fix the process factors at their range centre and a neutral posture
(simultaneous, no fusion edge). Sweep gamma; at each gamma, evaluate every point
of a simplex grid over (s_L, s_M, s_H) and take the design that maximises the
objective. The tipping gamma is where the optimum's quantity-quality index
(s_L - s_H) changes sign.

    python3 gamma_excursion.py --reps 4000

Outputs: gamma_excursion.csv, gamma_excursion.png
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from salvo_mds import monte_carlo
from farm import force_from_shares, RED_FORCE, RED_VALUE, PROCESS_FACTORS, FIXED

# Process centre point + neutral posture (no fusion edge, simultaneous).
CENTER = {k: 0.5 * (lo + hi) for k, (lo, hi) in PROCESS_FACTORS.items()}
CENTER["rho"] = 1.0  # budget parity, so the mix decision is not swamped by budget


def simplex_grid(den=6):
    """All (s_L, s_M, s_H) on a simplex grid with denominator `den` (shares sum to 1)."""
    pts = []
    for i in range(den + 1):
        for j in range(den + 1 - i):
            k = den - i - j
            pts.append((i / den, j / den, k / den))
    return pts


def objective(mix, gamma, reps, seed, response="p_victory"):
    s_L, s_M, s_H = mix
    budget = CENTER["rho"] * RED_VALUE
    blue = force_from_shares(budget, s_L, s_M, s_H, gamma=gamma)
    if not blue:
        return None
    params = dict(order="simultaneous", p_o=CENTER["p_o"], p_d=CENTER["p_d"],
                  sigma_b=CENTER["sigma_b"], sigma_r=CENTER["sigma_r"],
                  tau_b=CENTER["tau"], tau_r=CENTER["tau"], sd=CENTER["sd"], **FIXED)
    out = monte_carlo(blue, RED_FORCE, params, reps=reps, seed=seed)
    return out[response] if response in out else out[response + "_mean"]


def run(gammas, den, reps, response):
    grid = simplex_grid(den)
    rows = []
    for gi, gamma in enumerate(gammas):
        best = None
        for mi, mix in enumerate(grid):
            val = objective(mix, gamma, reps, seed=1000 * gi + mi, response=response)
            if val is None:
                continue
            if best is None or val > best["obj"]:
                best = dict(gamma=gamma, s_L=mix[0], s_M=mix[1], s_H=mix[2], obj=val)
        # quantity-quality index: >0 quantity-leaning (L), <0 quality-leaning (H)
        best["qq_index"] = best["s_L"] - best["s_H"]
        best["regime"] = ("quantity" if best["qq_index"] > 1e-9 else
                          "quality" if best["qq_index"] < -1e-9 else "balanced")
        rows.append(best)
        print(f"  gamma={gamma:.3f}  best mix (L={best['s_L']:.2f}, M={best['s_M']:.2f}, "
              f"H={best['s_H']:.2f})  {response}={best['obj']:.3f}  -> {best['regime']}")
    return pd.DataFrame(rows)


def find_tipping(df):
    """Gamma where the optimum decisively flips from quality (H) to quantity (L).

    Uses the LAST quality->quantity sign change of the quantity-quality index,
    scanning from high gamma. This is robust to the low-gamma region where
    P(victory) saturates near 1 and the arg-max mix is dominated by Monte-Carlo
    noise (many designs win almost always, so ties there are not a real flip)."""
    idx = df["qq_index"].to_numpy()
    g = df["gamma"].to_numpy()
    for a in range(len(idx) - 2, -1, -1):
        if idx[a] < 0 <= idx[a + 1]:
            x0, x1, y0, y1 = g[a], g[a + 1], idx[a], idx[a + 1]
            return x0 + (0 - y0) * (x1 - x0) / (y1 - y0) if y1 != y0 else x1
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=4000)
    ap.add_argument("--den", type=int, default=6, help="simplex-grid denominator")
    ap.add_argument("--g-lo", type=float, default=1.15)
    ap.add_argument("--g-hi", type=float, default=1.55)
    ap.add_argument("--g-step", type=float, default=0.05)
    ap.add_argument("--response", default="p_victory",
                     help="objective to maximise (p_victory or red_loss)")
    ap.add_argument("--rho", type=float, default=None,
                     help="budget ratio at the centre point (default 1.0 = parity). "
                          "A slight edge (e.g. 1.1) keeps P(victory) off the floor at high "
                          "gamma so the optimum-mix flip stays readable.")
    args = ap.parse_args()

    if args.rho is not None:
        CENTER["rho"] = args.rho
    gammas = np.round(np.arange(args.g_lo, args.g_hi + 1e-9, args.g_step), 4)
    print(f"centre rho = {CENTER['rho']}")
    print(f"gamma excursion: {len(gammas)} gammas x {len(simplex_grid(args.den))} mixes "
          f"x {args.reps} reps; objective = {args.response}")
    df = run(gammas, args.den, args.reps, args.response)
    df.to_csv("gamma_excursion.csv", index=False)

    tip = find_tipping(df)
    if tip is not None:
        print(f"\nTIPPING gamma (quality -> quantity): {tip:.3f}")
    else:
        regs = df["regime"].unique()
        print(f"\nNo quality->quantity flip in [{args.g_lo}, {args.g_hi}]; "
              f"optimum stayed: {', '.join(regs)}")

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.plot(df["gamma"], df["s_L"], "-o", ms=3, color="#c0392b", label="s_L (quantity)")
    ax.plot(df["gamma"], df["s_M"], "-o", ms=3, color="#7d6608", label="s_M (balanced)")
    ax.plot(df["gamma"], df["s_H"], "-o", ms=3, color="#1e8449", label="s_H (quality)")
    if tip is not None:
        ax.axvline(tip, ls="--", color="k", lw=0.9)
        ax.annotate(f"tipping γ ≈ {tip:.2f}", xy=(tip, 0.5),
                    xytext=(tip + 0.02, 0.62), fontsize=8,
                    arrowprops=dict(arrowstyle="->", lw=0.7))
    ax.set_xlabel("Cost convexity γ")
    ax.set_ylabel("Budget-optimal share")
    ax.set_title(f"Optimal force mix vs cost convexity\n(objective: {args.response}, process centre, parity)")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig("gamma_excursion.png")
    print("Wrote gamma_excursion.csv + gamma_excursion.png")


if __name__ == "__main__":
    main()
