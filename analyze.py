"""
analyze.py — Paper figures & tables for the LAFusion 2026 experiment.

Reads farm_results.csv (from farm.py) and produces:
  fig1_fusion_compounding.png  — fusion value vs battle duration (fresh sims, with 95% CIs)
  fig2_partition_tree.png      — partition tree for Red combat-power loss
  fig3_mixture_ternary.png     — P(victory) over the force-design simplex (near-parity)
  fig4_heterogeneity.png       — matched-pair value of heterogeneity, by engagement order
  fig5_fusion_exchange.png     — P(victory) vs fusion advantage (Delta sigma), by budget ratio
  tables_summary.md            — headline numbers for the paper text

Each figure maps to a claim in the spec (sections 1 and 9).
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from salvo_mds import monte_carlo, unit_cost, L_STRIKER, M_BALANCED, H_ESCORT, RED_STD

RESULTS = "farm_results.csv"
RED_FORCE = [(RED_STD, 5)]
RED_VALUE = 5 * unit_cost(RED_STD)
plt.rcParams.update({"figure.dpi": 150, "font.size": 9})
notes = []

df = pd.read_csv(RESULTS)
df["pure"] = (df.s_L == 1) | (df.s_M == 1) | (df.s_H == 1)
df["dsigma"] = df.sigma_b - df.sigma_r

# ----------------------------------------------------------------------
# FIG 1 — Compounding fusion value vs duration (fresh sims, matched seeds, CIs)
def force_from_shares(budget, sL, sM, sH):
    spec = []
    for plat, sh in ((L_STRIKER, sL), (M_BALANCED, sM), (H_ESCORT, sH)):
        n = int(round(budget * sh / unit_cost(plat)))
        if n > 0:
            spec.append((plat, n))
    return spec

blue = force_from_shares(RED_VALUE, 1/3, 1/3, 1/3)
base = dict(order="simultaneous", theta=0.30, p_o=0.8, p_d=0.7,
            tau_b=1.0, tau_r=1.0, sd=0.15, sigma_r=0.6)
Ts, fv, fv_ci = [1, 2, 3, 4, 5, 6], [], []
REPS = 6000
for T in Ts:
    hi = monte_carlo(blue, RED_FORCE, {**base, "Tmax": T, "sigma_b": 0.95}, reps=REPS, seed=100 + T)
    lo = monte_carlo(blue, RED_FORCE, {**base, "Tmax": T, "sigma_b": 0.60}, reps=REPS, seed=200 + T)
    d = hi["red_loss_mean"] - lo["red_loss_mean"]
    se = np.sqrt(hi["red_loss_sd"]**2 / REPS + lo["red_loss_sd"]**2 / REPS)
    fv.append(d); fv_ci.append(1.96 * se)
fig, ax = plt.subplots(figsize=(4.5, 3.2))
ax.errorbar(Ts, fv, yerr=fv_ci, marker="o", capsize=3, color="#1a5276")
ax.set_xlabel("Maximum salvos (battle duration)")
ax.set_ylabel("Fusion value\n(extra Red combat power destroyed)")
ax.set_title("A sustained scouting advantage compounds, then saturates")
ax.axhline(fv[0], ls=":", color="gray", lw=0.8)
ax.annotate(f"single-salvo captures {fv[0]/fv[-1]*100:.0f}% of eventual value",
            xy=(1, fv[0]), xytext=(2.2, fv[0] + 0.04), fontsize=8,
            arrowprops=dict(arrowstyle="->", lw=0.7))
ax.grid(alpha=0.3); fig.tight_layout(); fig.savefig("fig1_fusion_compounding.png")
notes.append(f"FIG1: fusion value grows {fv[0]:.3f} (T=1) -> {fv[-1]:.3f} (T=6); "
             f"single-salvo share = {fv[0]/fv[-1]*100:.0f}%.")

# ----------------------------------------------------------------------
# FIG 2 — Partition tree for Red combat-power loss
from sklearn.tree import DecisionTreeRegressor, plot_tree
FEATS = ["s_L", "s_M", "s_H", "sigma_b", "sigma_r", "tau", "rho", "p_o", "p_d", "sd",
         "ord_blue_first", "ord_red_first"]
X = df.assign(ord_blue_first=(df.order == "blue_first").astype(float),
              ord_red_first=(df.order == "red_first").astype(float))[FEATS]
y = df["red_loss_mean"]
tree = DecisionTreeRegressor(max_depth=3, min_samples_leaf=40, random_state=0).fit(X, y)
fig, ax = plt.subplots(figsize=(9, 4.2))
plot_tree(tree, feature_names=FEATS, filled=True, rounded=True, fontsize=7,
          impurity=False, precision=2, ax=ax)
ax.set_title(f"Partition tree — Red combat-power loss (R² = {tree.score(X, y):.2f})")
fig.tight_layout(); fig.savefig("fig2_partition_tree.png")
imp = pd.Series(tree.feature_importances_, index=FEATS).sort_values(ascending=False)
notes.append("FIG2: top importances " +
             ", ".join(f"{k}={v:.2f}" for k, v in imp[imp > 0.02].items()) + ".")

# ----------------------------------------------------------------------
# FIG 3 — Mixture ternary: P(victory) over the design simplex (near-parity budgets)
par = df[(df.rho > 0.95) & (df.rho < 1.25)]
mix = par.groupby(["s_L", "s_M", "s_H"], as_index=False).p_victory.mean()
# simplex -> 2D: x = s_M + 0.5*s_H, y = (sqrt(3)/2)*s_H  (L at origin, M right, H top)
mx = mix.s_M + 0.5 * mix.s_H
my = (np.sqrt(3) / 2) * mix.s_H
fig, ax = plt.subplots(figsize=(4.6, 4.0))
tri_x, tri_y = [0, 1, 0.5, 0], [0, 0, np.sqrt(3) / 2, 0]
ax.plot(tri_x, tri_y, color="k", lw=0.8)
sc = ax.scatter(mx, my, c=mix.p_victory, s=600, cmap="RdYlGn", vmin=0, vmax=0.7,
                edgecolors="k", linewidths=0.5)
for xi, yi, v in zip(mx, my, mix.p_victory):
    ax.text(xi, yi, f"{v:.2f}", ha="center", va="center", fontsize=7)
ax.text(-0.04, -0.05, "pure L\n(striker)", ha="center", fontsize=8)
ax.text(1.04, -0.05, "pure M\n(balanced)", ha="center", fontsize=8)
ax.text(0.5, np.sqrt(3) / 2 + 0.04, "pure H (escort)", ha="center", fontsize=8)
ax.set_title("Mean P(Blue victory) over the force-design simplex\n(near-parity budgets, all regimes pooled)")
ax.axis("off"); ax.set_aspect("equal")
fig.colorbar(sc, ax=ax, shrink=0.7, label="P(victory)")
fig.tight_layout(); fig.savefig("fig3_mixture_ternary.png")
best = mix.loc[mix.p_victory.idxmax()]
notes.append(f"FIG3: best design at parity = (s_L={best.s_L:.2f}, s_M={best.s_M:.2f}, "
             f"s_H={best.s_H:.2f}), P(vict)={best.p_victory:.2f}; "
             f"pure-H P(vict)={mix[mix.s_H==1].p_victory.iloc[0]:.3f}.")

# ----------------------------------------------------------------------
# FIG 4 — Matched-pair value of heterogeneity, by engagement order
proc_keys = ["sigma_b", "sigma_r", "tau", "rho", "p_o", "p_d", "sd", "order"]
def het_value(g):
    pure_best = g[g.pure].p_victory.max()
    mix_best = g[~g.pure].p_victory.max()
    return mix_best - pure_best
hv = df.groupby(proc_keys, as_index=False).apply(
    lambda g: pd.Series({"het_value": het_value(g)}), include_groups=False)
fig, ax = plt.subplots(figsize=(4.6, 3.2))
for i, o in enumerate(["blue_first", "simultaneous", "red_first"]):
    vals = hv[hv.order == o].het_value.dropna()
    ax.boxplot(vals, positions=[i], widths=0.5, showfliers=False)
    ax.text(i, vals.median() + 0.02, f"med {vals.median():.3f}", ha="center", fontsize=7)
ax.axhline(0, color="crimson", lw=0.8, ls="--")
ax.set_xticks(range(3)); ax.set_xticklabels(["Blue first", "Simultaneous", "Red first"])
ax.set_ylabel("Best mixed − best pure  ΔP(victory)")
ax.set_title("Value of heterogeneity (matched process regimes)")
ax.grid(alpha=0.3, axis="y"); fig.tight_layout(); fig.savefig("fig4_heterogeneity.png")
notes.append("FIG4: median matched-pair heterogeneity value by order: " +
             ", ".join(f"{o}={hv[hv.order==o].het_value.median():+.3f}"
                       for o in ["blue_first", "simultaneous", "red_first"]) + ".")

# ----------------------------------------------------------------------
# FIG 5 — Fusion exchange rate: P(victory) vs Delta sigma, stratified by budget ratio
fig, ax = plt.subplots(figsize=(4.8, 3.4))
bins = [(0.7, 0.95, "#c0392b", "budget ratio < 0.95 (Blue poorer)"),
        (0.95, 1.25, "#7d6608", "0.95–1.25 (near parity)"),
        (1.25, 1.5, "#1e8449", "> 1.25 (Blue richer)")]
for lo_, hi_, c, lab in bins:
    sub = df[(df.rho >= lo_) & (df.rho < hi_)]
    xb = np.linspace(-0.6, 0.6, 13)
    xc = 0.5 * (xb[:-1] + xb[1:])
    ym = [sub[(sub.dsigma >= a) & (sub.dsigma < b)].p_victory.mean() for a, b in zip(xb[:-1], xb[1:])]
    ax.plot(xc, ym, marker="o", ms=3, color=c, label=lab)
ax.axvline(0, color="gray", lw=0.8, ls=":")
ax.set_xlabel("Fusion advantage  Δσ = σ_B − σ_R")
ax.set_ylabel("Mean P(Blue victory)")
ax.set_title("The information–force exchange rate")
ax.legend(fontsize=7); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig("fig5_fusion_exchange.png")
# exchange-rate headline: what Δσ compensates a budget deficit?
poor = df[df.rho < 1.0]; rich = df[df.rho > 1.2]
poor_hi = poor[poor.dsigma > 0.15].p_victory.mean()
rich_lo = rich[rich.dsigma < -0.15].p_victory.mean()
notes.append(f"FIG5: poorer Blue (rho<1.0) with fusion edge (dsig>0.15) P(vict)={poor_hi:.2f} "
             f"[n={len(poor[poor.dsigma > 0.15])}] vs richer Blue (rho>1.2) with fusion "
             f"deficit P(vict)={rich_lo:.2f} [n={len(rich[rich.dsigma < -0.15])}] — information "
             f"{'substitutes for' if poor_hi > rich_lo else 'does not offset'} budget.")

# ----------------------------------------------------------------------
with open("tables_summary.md", "w") as f:
    f.write("# Headline numbers (pilot scale — regenerate at paper scale)\n\n")
    for n in notes:
        f.write("- " + n + "\n")
print("\n".join(notes))
print("\nWrote: fig1..fig5 PNGs + tables_summary.md")
