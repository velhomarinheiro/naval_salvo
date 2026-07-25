"""
cross_composition.py — Cross-composition excursion: force design on BOTH sides.

The primary experiment fixes Red at 5 x M. This excursion releases that
assumption: Blue AND Red are each built from budget shares over the same three
archetypes (L / M / H), and every Blue mixture meets every Red mixture — a full
10 x 10 cross of the simplex-centroid composition set — under near-parity
budgets (Blue budget = rho x Red budget, rho in {0.9, 1.0, 1.1}: parity +/-10%).

Process/posture factors are FIXED at a neutral centre so that composition alone
drives the outcome: simultaneous fire, sigma_B = sigma_R (no fusion edge),
equal tau/p_o/p_d, Tmax=6, theta=0.30 (documented below).

Questions this answers:
  * Is any composition dominant at parity, or is there a rock-paper-scissors
    structure among pure fleets (does L beat H, H beat M, M beat L)?
  * What is Blue's best response to each Red composition?
  * How much does a +/-10% budget edge move the outcome vs. changing the mix?

Internal checks (symmetry of the engine, spec sec 6):
  * mirror cells (same mix, rho=1) must give log-FER ~ 0;
  * log-FER must be antisymmetric: logfer(i,j) ~ -logfer(j,i) at rho=1.

Usage:
    python3 cross_composition.py --reps 5000 --jobs 4

Outputs: cross_composition_results.csv, fig6_cross_matrix.png,
         fig7_budget_sensitivity.png, cross_summary.md
"""
import argparse
import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from salvo_mds import monte_carlo, unit_cost, L_STRIKER, M_BALANCED, H_ESCORT, RED_STD
from farm import build_mixture_design

MASTER_SEED = 20260731
RED_VALUE = 5 * unit_cost(RED_STD)   # 35.2 — the budget-ratio base (spec sec 6)
RHOS = [0.9, 1.0, 1.1]               # parity +/- 10%

# Neutral process centre (documented fixed settings for this excursion):
NEUTRAL = dict(order="simultaneous", Tmax=6, theta=0.30,
               p_o=0.75, p_d=0.75, sigma_b=0.7, sigma_r=0.7,
               tau_b=0.75, tau_r=0.75, sd=0.15)

# Short labels for the 10 simplex-centroid compositions, in build_mixture_design order.
MIX_LABELS = ["L", "M", "H", "LM", "LH", "MH", "C", "L+", "M+", "H+"]


def compositions():
    mix = build_mixture_design()
    assert len(mix) == len(MIX_LABELS)
    return [(MIX_LABELS[i], tuple(mix.iloc[i])) for i in range(len(mix))]


def best_integer_fleet(budget, shares, gamma=1.35):
    """Budget-CAPPED integer fleet closest to the target mixture shares.

    farm.force_from_shares uses per-platform round(), which lets mixed fleets
    overshoot the budget by up to ~20% (e.g. the centroid at 35.2 rounds to a
    42.3-cost fleet) — harmless when Red is fixed and rho varies continuously,
    but fatal in a head-to-head composition comparison at parity: the rounding
    windfall, not the mixture, would decide the outcome.

    Here the integer fleet space is tiny, so we solve the allocation exactly:
    enumerate every (n_L, n_M, n_H) with cost <= budget (HARD cap, never
    overspend) and pick the one minimising the L2 distance between realised
    value shares (spend_k / budget) and the target shares. Unspent budget thus
    counts as share shortfall, so utilisation is maximised subject to mix
    fidelity; ties break toward higher spend. Returns (spec, spend)."""
    plats = (L_STRIKER, M_BALANCED, H_ESCORT)
    costs = [unit_cost(p, gamma=gamma) for p in plats]
    eps = 1e-9 * budget  # float guard: budget built as k*c must admit k hulls
    caps = [int((budget + eps) // c) if s > 0 else 0 for c, s in zip(costs, shares)]
    best, best_key = None, None
    for counts in itertools.product(*(range(c + 1) for c in caps)):
        spend = sum(n * c for n, c in zip(counts, costs))
        if spend > budget + eps or spend == 0:
            continue
        err = sum((n * c / budget - s) ** 2
                  for n, c, s in zip(counts, costs, shares))
        key = (err, -spend)
        if best_key is None or key < best_key:
            best, best_key = counts, key
    if best is None:
        return [], 0.0
    spec = [(p, n) for p, n in zip(plats, best) if n > 0]
    return spec, float(sum(n * c for n, c in zip(best, costs)))


def _run_cell(args):
    (bi, blab, bshares), (ri, rlab, rshares), rho, reps, seed = args
    blue, bspend = best_integer_fleet(rho * RED_VALUE, bshares)
    red, rspend = best_integer_fleet(RED_VALUE, rshares)
    if not blue or not red:
        return None
    out = monte_carlo(blue, red, NEUTRAL, reps=reps, seed=seed)
    return {"blue_mix": blab, "red_mix": rlab, "rho": rho,
            "blue_i": bi, "red_i": ri,
            "s_L_b": bshares[0], "s_M_b": bshares[1], "s_H_b": bshares[2],
            "s_L_r": rshares[0], "s_M_r": rshares[1], "s_H_r": rshares[2],
            "blue_hulls": sum(n for _, n in blue), "red_hulls": sum(n for _, n in red),
            "blue_spend": bspend, "red_spend": rspend,
            "blue_util": bspend / (rho * RED_VALUE), "red_util": rspend / RED_VALUE,
            "seed": seed, **out}


def run(reps, jobs):
    comps = compositions()
    ss = np.random.SeedSequence(MASTER_SEED)
    tasks = []
    seeds = ss.spawn(len(comps) * len(comps) * len(RHOS))
    k = 0
    for rho in RHOS:
        for bi, (blab, bsh) in enumerate(comps):
            for ri, (rlab, rsh) in enumerate(comps):
                tasks.append(((bi, blab, bsh), (ri, rlab, rsh), rho, reps,
                              int(seeds[k].generate_state(1)[0])))
                k += 1
    print(f"cross design: {len(comps)}x{len(comps)} compositions x {len(RHOS)} budget "
          f"ratios = {len(tasks)} cells x {reps} reps = {len(tasks)*reps:,} battles")
    if jobs > 1:
        import multiprocessing as mp
        with mp.Pool(jobs) as pool:
            recs = [r for r in pool.imap_unordered(_run_cell, tasks, chunksize=4)
                    if r is not None]
    else:
        recs = [r for r in map(_run_cell, tasks) if r is not None]
    return pd.DataFrame(recs)


# ----------------------------------------------------------------------
def fleet_table(notes):
    """Realised integer fleets per composition and budget level (transparency:
    shows the quantization the hard budget cap imposes)."""
    for rho in RHOS:
        rows = []
        for lab, sh in compositions():
            spec, spend = best_integer_fleet(rho * RED_VALUE, sh)
            fleet = "+".join(f"{n}{p.name}" for p, n in spec) or "-"
            rows.append(f"{lab}:{fleet}({spend/(rho*RED_VALUE):.0%})")
        notes.append(f"FLEETS rho={rho}: " + "  ".join(rows))


def analyse(df):
    notes = []
    fleet_table(notes)
    notes.append(f"BUDGET UTILISATION under the hard cap: mean "
                 f"{df.blue_util.mean():.0%}, min {df.blue_util.min():.0%} "
                 f"(farm.force_from_shares round() would overshoot up to +20% — "
                 f"see best_integer_fleet docstring).")
    par = df[df.rho == 1.0]
    P = par.pivot(index="blue_mix", columns="red_mix", values="p_victory").reindex(
        index=MIX_LABELS, columns=MIX_LABELS)
    F = par.pivot(index="blue_mix", columns="red_mix", values="logfer_mean").reindex(
        index=MIX_LABELS, columns=MIX_LABELS)

    # --- engine symmetry checks (mirror diagonal + antisymmetry) ---
    diag = np.diag(F.to_numpy())
    anti = np.abs(F.to_numpy() + F.to_numpy().T).max()
    notes.append(f"CHECK: mirror-diagonal |log-FER| max = {np.abs(diag).max():.3f} "
                 f"(~0 expected); antisymmetry max |logfer(i,j)+logfer(j,i)| = "
                 f"{anti:.3f} (Monte-Carlo noise only).")

    # --- pure-vs-pure 3x3: dominance or rock-paper-scissors? ---
    pures = ["L", "M", "H"]
    sub = F.loc[pures, pures]
    beats = {(b, r): sub.loc[b, r] > 0 for b in pures for r in pures if b != r}
    cyc = ""
    for a, b, c in [("L", "M", "H"), ("L", "H", "M")]:
        if beats[(a, b)] and beats[(b, c)] and beats[(c, a)]:
            cyc = f"CYCLE: {a} beats {b}, {b} beats {c}, {c} beats {a} (rock-paper-scissors)."
    if not cyc:
        wins = {p: sum(beats[(p, q)] for q in pures if q != p) for p in pures}
        best = max(wins, key=wins.get)
        cyc = (f"no cycle among pure fleets; pairwise wins {wins} -> "
               f"'{best}' is the strongest pure composition at parity.")
    notes.append(f"PURE 3x3 (log-FER): " + "; ".join(
        f"{b} vs {r}: {sub.loc[b, r]:+.2f}" for b in pures for r in pures if b != r)
        + f". {cyc}")

    # --- dominance over the full 10x10 ---
    row_mean = P.mean(axis=1).sort_values(ascending=False)
    col_mean = P.mean(axis=0).sort_values()          # low = hard to beat as Red
    notes.append("BLUE ranking (mean P(vict) across all Red mixes, rho=1): "
                 + ", ".join(f"{k}={v:.2f}" for k, v in row_mean.items()) + ".")
    notes.append("RED toughness (mean Blue P(vict) against it; lower = tougher): "
                 + ", ".join(f"{k}={v:.2f}" for k, v in col_mean.items()) + ".")

    # --- best response per Red composition ---
    br = P.idxmax(axis=0)
    notes.append("BLUE best response by Red mix: "
                 + ", ".join(f"vs {r}: {br[r]} ({P[r].max():.2f})" for r in MIX_LABELS) + ".")

    # --- budget sensitivity: +/-10% vs composition choice ---
    by_rho = df.groupby("rho").p_victory.mean()
    spread_mix = row_mean.iloc[0] - row_mean.iloc[-1]
    notes.append(f"BUDGET vs DESIGN: mean P(vict) rises {by_rho[0.9]:.2f} -> "
                 f"{by_rho[1.0]:.2f} -> {by_rho[1.1]:.2f} across rho 0.9->1.1 "
                 f"(+/-10% budget ~ {by_rho[1.1]-by_rho[0.9]:+.2f}), while the "
                 f"composition choice spans {spread_mix:.2f} at fixed parity — "
                 f"{'design outweighs' if spread_mix > (by_rho[1.1]-by_rho[0.9]) else 'budget outweighs'} "
                 f"a 10% budget edge in this regime.")
    return P, F, notes


# ----------------------------------------------------------------------
def fig_cross_matrix(P, F, out="fig6_cross_matrix.png"):
    """10x10 cross matrix. Color = log-FER (diverging, symmetric, mirror=0);
    cell text = P(Blue victory). Best Blue response per Red column outlined."""
    M = F.to_numpy()
    lim = np.nanmax(np.abs(M))
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    im = ax.imshow(M, cmap="RdBu", vmin=-lim, vmax=lim)  # blue=+ (Blue wins), red=- ; CVD-safe pair
    ax.set_xticks(range(len(MIX_LABELS))); ax.set_xticklabels(MIX_LABELS, fontsize=8)
    ax.set_yticks(range(len(MIX_LABELS))); ax.set_yticklabels(MIX_LABELS, fontsize=8)
    ax.set_xlabel("Red composition"); ax.set_ylabel("Blue composition")
    ax.set_title("Cross-composition outcome at budget parity\n"
                 "colour: log-FER (blue = Blue-favourable) · text: P(Blue victory)")
    Pm = P.to_numpy()
    best_rows = np.nanargmax(Pm, axis=0)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            dark_bg = abs(M[i, j]) > 0.55 * lim
            ax.text(j, i, f"{Pm[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                    color="white" if dark_bg else "#222222")
    for j, i in enumerate(best_rows):  # outline best response per Red column
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                    edgecolor="#222222", lw=1.6))
    ax.plot([-0.5, len(MIX_LABELS) - 0.5], [-0.5, len(MIX_LABELS) - 0.5],
            color="#888888", lw=0.6, ls=":")  # mirror diagonal
    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    cb.set_label("log-FER (mirror match = 0)")
    fig.tight_layout(); fig.savefig(out, dpi=150)


def fig_budget_sensitivity(df, out="fig7_budget_sensitivity.png"):
    """Mean P(Blue victory) per Blue composition, one line per budget ratio.
    Shows how much a +/-10% budget edge moves each design."""
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    colors = {0.9: "#b2182b", 1.0: "#4d4d4d", 1.1: "#2166ac"}  # red=poorer, gray=parity, blue=richer
    order = (df[df.rho == 1.0].groupby("blue_mix").p_victory.mean()
             .reindex(MIX_LABELS).sort_values(ascending=False).index.tolist())
    for rho in RHOS:
        m = (df[df.rho == rho].groupby("blue_mix").p_victory.mean().reindex(order))
        ax.plot(range(len(order)), m.to_numpy(), marker="o", ms=4, lw=2,
                color=colors[rho], label=f"ρ = {rho:.1f}")
    ax.set_xticks(range(len(order))); ax.set_xticklabels(order, fontsize=8)
    ax.set_xlabel("Blue composition (sorted by parity performance)")
    ax.set_ylabel("Mean P(Blue victory)\nacross all Red compositions")
    ax.set_title("Budget edge (±10%) vs force-design choice")
    ax.legend(fontsize=8, title="Blue/Red budget")
    ax.grid(alpha=0.3); fig.tight_layout(); fig.savefig(out, dpi=150)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5000)
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()

    df = run(args.reps, args.jobs)
    df.to_csv("cross_composition_results.csv", index=False)
    print(f"saved cross_composition_results.csv ({len(df)} cells)")

    P, F, notes = analyse(df)
    fig_cross_matrix(P, F)
    fig_budget_sensitivity(df)
    with open("cross_summary.md", "w") as f:
        f.write(f"# Cross-composition excursion ({args.reps} reps/cell, "
                f"parity ±10%)\n\nFixed neutral regime: {NEUTRAL}\n\n")
        for n in notes:
            f.write("- " + n + "\n\n")
    print()
    for n in notes:
        print("- " + n)
    print("\nWrote fig6_cross_matrix.png + fig7_budget_sensitivity.png + cross_summary.md")


if __name__ == "__main__":
    main()
