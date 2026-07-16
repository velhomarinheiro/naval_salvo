"""
farm.py — Design generation, data-farming runner, and metamodels for the
LAFusion 2026 experiment (spec sections 7-9).

Design:
  - Mixture part: simplex-centroid over (s_L, s_M, s_H): 3 vertices (pure forces),
    3 edge midpoints (binary mixes), centroid, + interior fill points.
  - Process part: space-filling Latin Hypercube over the continuous regime factors,
    crossed full-factorially with the 3 engagement orders.
    NOTE for the paper: the published experiment should use the mixed NOB design
    (Vieira et al. 2011; SEED Center spreadsheet). LHS is the reproducible,
    dependency-free stand-in with comparable space-filling properties; swapping
    the design matrix is a one-line change (see build_process_design).

Outputs:
  farm_design.csv   — the design matrix (one row per design point, with seeds)
  farm_results.csv  — design matrix + aggregated responses
  metamodel report  — printed: partition-tree importances + OLS metamodel per response

Usage:
  python3 farm.py --reps 500 --process-points 33   (pilot)
  python3 farm.py --reps 10000 --process-points 128 (paper scale)
"""
import argparse
import itertools
import numpy as np
import pandas as pd
from scipy.stats import qmc

from salvo_mds import (Platform, unit_cost, monte_carlo,
                       L_STRIKER, M_BALANCED, H_ESCORT, RED_STD)

MASTER_SEED = 20260731  # submission-window opening date; fixed for reproducibility

# ----------------------------------------------------------------------
# Factor definitions (spec section 7). Primary run: gamma=1.35 (fixed via
# unit_cost default), Blue=Red damage params; gamma sweep is a secondary excursion.
PROCESS_FACTORS = {
    "sigma_b": (0.4, 1.0),
    "sigma_r": (0.4, 1.0),
    "tau":     (0.5, 1.0),   # applied to both sides (primary-run reduction)
    "rho":     (0.7, 1.5),   # budget ratio
    "p_o":     (0.5, 1.0),
    "p_d":     (0.5, 1.0),
    "sd":      (0.10, 0.20),
}
ORDERS = ["blue_first", "simultaneous", "red_first"]
RED_FORCE = [(RED_STD, 5)]
RED_VALUE = 5 * unit_cost(RED_STD)
FIXED = dict(Tmax=6, theta=0.30)


def build_mixture_design():
    """Simplex-centroid + interior fill over budget shares (s_L, s_M, s_H)."""
    pts = [
        (1, 0, 0), (0, 1, 0), (0, 0, 1),              # vertices: pure forces (required)
        (.5, .5, 0), (.5, 0, .5), (0, .5, .5),        # edge midpoints
        (1/3, 1/3, 1/3),                              # centroid
        (2/3, 1/6, 1/6), (1/6, 2/3, 1/6), (1/6, 1/6, 2/3),  # interior fill
    ]
    return pd.DataFrame(pts, columns=["s_L", "s_M", "s_H"])


def load_nob_design(path, n_points=None, stack_rotations=1, coded_range=(-1.0, 1.0),
                     column_order=None):
    """Load a mixed NOB design (Vieira et al. 2011) from the SEED Center
    spreadsheet and rescale it to PROCESS_FACTORS ranges.

    path            : csv or xlsx file with the NOB design (coded values).
    n_points        : if given, truncate/subsample to this many rows after
                       stacking (None = use all rows produced).
    stack_rotations : number of rotated copies to stack, as in Kesler et al.
                       2019 (they stacked 10x over a 512-point base design to
                       reach 5,120 points). A "rotation" here is a column
                       permutation combined with a coded-space reflection,
                       which preserves the design's balance/orthogonality
                       properties while multiplying the point count.
                       Set to 1 to use the base design unmodified.
    coded_range     : the (low, high) coding used in the source file
                       (SEED Center designs are typically coded to [-1, 1]
                       or [0, 1] — check the spreadsheet's header/README
                       and adjust this argument accordingly).
    column_order    : list of column names in the source file, in the order
                       they should map onto list(PROCESS_FACTORS). If None,
                       assumes the file's columns are already in that order.

    Returns a DataFrame with columns = list(PROCESS_FACTORS), values rescaled
    to the actual factor ranges. Raises clearly on any mismatch rather than
    silently falling back to LHS.
    """
    names = list(PROCESS_FACTORS)
    if path.endswith((".xlsx", ".xls")):
        raw = pd.read_excel(path)
    else:
        raw = pd.read_csv(path)

    if column_order is not None:
        missing = [c for c in column_order if c not in raw.columns]
        if missing:
            raise ValueError(f"load_nob_design: columns {missing} not found in {path}. "
                              f"Available columns: {list(raw.columns)}")
        raw = raw[column_order]
    else:
        if raw.shape[1] < len(names):
            raise ValueError(f"load_nob_design: {path} has {raw.shape[1]} columns, "
                              f"need >= {len(names)} for factors {names}. "
                              f"Pass column_order= to select/reorder explicitly.")
        raw = raw.iloc[:, :len(names)]
    raw.columns = names

    lo_c, hi_c = coded_range
    stacks = [raw.copy()]
    rng = np.random.default_rng(0)
    for r in range(1, stack_rotations):
        rot = raw.copy()
        # Rotation = reflect a pseudo-random subset of coded columns
        # (x -> lo_c+hi_c-x) + a fixed cyclic permutation of columns.
        # This is the standard NOB "rotate and stack" trick: it preserves
        # each column's marginal distribution and the design's pairwise
        # balance while creating new, non-duplicate combinations.
        flip_cols = rng.choice(names, size=len(names) // 2, replace=False)
        for c in flip_cols:
            rot[c] = lo_c + hi_c - rot[c]
        rot = rot[names[r % len(names):] + names[:r % len(names)]]
        rot.columns = names
        stacks.append(rot)
    design = pd.concat(stacks, ignore_index=True)

    if n_points is not None:
        design = design.iloc[:n_points].reset_index(drop=True)

    lo_f = np.array([PROCESS_FACTORS[k][0] for k in names])
    hi_f = np.array([PROCESS_FACTORS[k][1] for k in names])
    frac = (design[names].to_numpy() - lo_c) / (hi_c - lo_c)
    scaled = lo_f + frac * (hi_f - lo_f)
    return pd.DataFrame(scaled, columns=names)


def build_process_design(n_points, seed, nob_path=None, nob_kwargs=None):
    """Space-filling design over the continuous process factors.

    Default: Latin Hypercube (reproducible, dependency-free pilot stand-in).
    Paper-scale: pass nob_path= to use the real mixed NOB design instead
    (Vieira et al. 2011, SEED Center spreadsheet) via load_nob_design above.
    Example:
        build_process_design(n_points=None, seed=0,
                              nob_path="seed_center_nob512.xlsx",
                              nob_kwargs=dict(stack_rotations=10,
                                               coded_range=(-1, 1)))
    """
    names = list(PROCESS_FACTORS)
    if nob_path is not None:
        return load_nob_design(nob_path, n_points=n_points, **(nob_kwargs or {}))
    sampler = qmc.LatinHypercube(d=len(names), seed=seed)
    u = sampler.random(n_points)
    lo = np.array([PROCESS_FACTORS[k][0] for k in names])
    hi = np.array([PROCESS_FACTORS[k][1] for k in names])
    return pd.DataFrame(lo + u * (hi - lo), columns=names)


def force_from_shares(budget, s_L, s_M, s_H):
    spec = []
    for plat, sh in ((L_STRIKER, s_L), (M_BALANCED, s_M), (H_ESCORT, s_H)):
        n = int(round(budget * sh / unit_cost(plat)))
        if n > 0:
            spec.append((plat, n))
    return spec


def build_design(n_process, seed, nob_path=None, nob_kwargs=None):
    mix = build_mixture_design()
    proc = build_process_design(n_process, seed, nob_path=nob_path, nob_kwargs=nob_kwargs)
    rows = []
    ss = np.random.SeedSequence(seed)
    for (_, m), (_, p), order in itertools.product(mix.iterrows(), proc.iterrows(), ORDERS):
        rows.append({**m.to_dict(), **p.to_dict(), "order": order})
    df = pd.DataFrame(rows)
    df.insert(0, "dp", range(len(df)))
    df["seed"] = [int(s.generate_state(1)[0]) for s in ss.spawn(len(df))]
    return df


def run_design_point(row, reps):
    budget = row["rho"] * RED_VALUE
    blue = force_from_shares(budget, row["s_L"], row["s_M"], row["s_H"])
    if not blue:  # degenerate (tiny budget share rounding to zero everywhere)
        return None
    params = dict(order=row["order"], p_o=row["p_o"], p_d=row["p_d"],
                  sigma_b=row["sigma_b"], sigma_r=row["sigma_r"],
                  tau_b=row["tau"], tau_r=row["tau"], sd=row["sd"], **FIXED)
    return monte_carlo(blue, RED_FORCE, params, reps=reps, seed=int(row["seed"]))


def run_farm(design, reps):
    recs = []
    for i, row in design.iterrows():
        out = run_design_point(row, reps)
        if out is None:
            continue
        recs.append({**row.to_dict(), **out})
        if (i + 1) % 100 == 0:
            print(f"  ... {i + 1}/{len(design)} design points")
    return pd.DataFrame(recs)


# ----------------------------------------------------------------------
# Metamodels (spec section 8 toolkit: partition tree + regression)
def fit_metamodels(df, response):
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures

    feats = ["s_L", "s_M", "s_H", "sigma_b", "sigma_r", "tau", "rho",
             "p_o", "p_d", "sd", "ord_blue_first", "ord_red_first"]
    X = df.assign(ord_blue_first=(df["order"] == "blue_first").astype(float),
                  ord_red_first=(df["order"] == "red_first").astype(float))[feats]
    y = df[response]

    tree = DecisionTreeRegressor(max_depth=4, min_samples_leaf=30, random_state=0).fit(X, y)
    imp = pd.Series(tree.feature_importances_, index=feats).sort_values(ascending=False)

    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    Xp = poly.fit_transform(X)
    reg = LinearRegression().fit(Xp, y)
    r2 = reg.score(Xp, y)
    names = poly.get_feature_names_out(feats)
    coefs = pd.Series(reg.coef_, index=names)
    top = coefs.reindex(coefs.abs().sort_values(ascending=False).index)[:10]

    print(f"\n===== Metamodels for {response} =====")
    print(f"Partition tree (depth 4) R2 = {tree.score(X, y):.3f}; top importances:")
    print(imp[imp > 0.01].round(3).to_string())
    print(f"OLS 2nd-order (interactions) R2 = {r2:.3f}; top-10 |coef| terms:")
    print(top.round(3).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=500)
    ap.add_argument("--process-points", type=int, default=33)
    ap.add_argument("--out-prefix", default="farm")
    ap.add_argument("--nob-path", default=None,
                     help="Path to SEED Center NOB design file. If omitted, uses LHS.")
    ap.add_argument("--nob-rotations", type=int, default=1,
                     help="Number of rotate-and-stack copies of the NOB base design.")
    ap.add_argument("--nob-coded-lo", type=float, default=-1.0)
    ap.add_argument("--nob-coded-hi", type=float, default=1.0)
    args = ap.parse_args()

    nob_kwargs = None
    if args.nob_path is not None:
        nob_kwargs = dict(stack_rotations=args.nob_rotations,
                           coded_range=(args.nob_coded_lo, args.nob_coded_hi))

    design = build_design(args.process_points, MASTER_SEED,
                           nob_path=args.nob_path, nob_kwargs=nob_kwargs)
    design.to_csv(f"{args.out_prefix}_design.csv", index=False)
    print(f"Design: {len(design)} points "
          f"({len(build_mixture_design())} mixture x {args.process_points} process x {len(ORDERS)} orders); "
          f"{args.reps} reps each -> {len(design) * args.reps:,} battles")

    results = run_farm(design, args.reps)
    results.to_csv(f"{args.out_prefix}_results.csv", index=False)
    print(f"Saved {args.out_prefix}_results.csv ({len(results)} rows)")

    # R4 uses logfer_mean (logit-transformed FER); the raw fer_mean is a biased
    # mean-of-ratios and is deliberately not metamodeled (see salvo_mds.simulate).
    for resp in ["red_loss_mean", "p_victory", "logfer_mean", "salvos_mean", "overkill_frac_mean"]:
        fit_metamodels(results, resp)


if __name__ == "__main__":
    main()
