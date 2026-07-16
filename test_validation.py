"""
test_validation.py — Automated validation & reproducibility regression tests
for the LAFusion 2026 salvo engine (salvo_mds.py), spec section 10.

These ship WITH the reproducibility package (spec sec 11): every claim the paper
leans on is locked here as a seeded, deterministic assertion. Run with:

    pytest test_validation.py -v
    python3 test_validation.py          # also prints a human-readable summary

Excursions covered:
  * Symmetry (spec sec 6)          — mirror match => symmetric losses, log-FER ~ 0
  * Single-salvo reduction         — Tmax=1 collapses to one exchange
  * Armstrong anchor (sec 10.1)    — homogeneous single-salvo regression baseline
  * Tiah swing (sec 10.2)          — concentration->dispersion flips the exchange
  * Compounding fusion (claim #3)  — fusion value grows with battle duration
  * Allocation robustness (10.3)   — without-replacement lowers overkill, keeps signs
"""
import numpy as np
import pytest

from salvo_mds import (Platform, unit_cost, monte_carlo,
                       L_STRIKER, M_BALANCED, H_ESCORT, RED_STD)


# ----------------------------------------------------------------------
# Shared scenario helpers
def buy(budget, plat):
    """Integer force spec: as many hulls of `plat` as `budget` affords."""
    n = int(round(budget / unit_cost(plat)))
    return [(plat, n)] if n > 0 else []


def base_params(**over):
    p = dict(order="simultaneous", Tmax=6, theta=0.30, p_o=0.8, p_d=0.7,
             sigma_b=0.8, sigma_r=0.8, tau_b=1.0, tau_r=1.0, sd=0.15)
    p.update(over)
    return p


# ======================================================================
# 1. SYMMETRY (spec sec 6): rho=1, s_M=1 => Blue is the exact mirror of Red.
#    Losses must be statistically symmetric and the log-FER must sit at ~0.
def test_symmetry_mirror_match():
    blue = [(M_BALANCED, 5)]
    red = [(RED_STD, 5)]
    o = monte_carlo(blue, red, base_params(sigma_b=0.8, sigma_r=0.8),
                    reps=30000, seed=1)
    assert abs(o["blue_loss_mean"] - o["red_loss_mean"]) < 0.01
    assert abs(o["blue_ships_lost_mean"] - o["red_ships_lost_mean"]) < 0.05
    # log-FER is the robust R4: mirror match => geometric-mean FER ~ 1.
    assert abs(o["logfer_mean"]) < 0.05
    assert abs(o["fer_geom"] - 1.0) < 0.05


def test_raw_fer_is_biased_but_logfer_is_not():
    """Documents WHY R4 uses log-FER: the raw mean-of-ratios is badly biased
    even in a perfectly symmetric fight, while log-FER is not."""
    blue = [(M_BALANCED, 5)]
    o = monte_carlo(blue, [(RED_STD, 5)], base_params(), reps=30000, seed=1)
    assert o["fer_mean"] > 3.0          # raw ratio: wildly asymmetric (misleading)
    assert abs(o["logfer_mean"]) < 0.05  # log-FER: correctly ~symmetric


# ======================================================================
# 2. SINGLE-SALVO REDUCTION: Tmax=1 must terminate every battle in one salvo.
def test_tmax1_single_salvo():
    blue = [(L_STRIKER, 6), (M_BALANCED, 2)]
    red = [(RED_STD, 5)]
    o = monte_carlo(blue, red, base_params(Tmax=1, theta=0.0),
                    reps=5000, seed=3)
    assert o["salvos_mean"] == 1.0
    assert o["salvos_sd"] == 0.0


# ======================================================================
# 3. ARMSTRONG SINGLE-SALVO ANCHOR (spec sec 10.1) — CALIBRATED.
#
#    Reproduces Armstrong (2011), "A verification study of the stochastic salvo
#    combat model", Annals of OR 186(1):23-38, section 5.1 illustrative example
#    (the one scenario the paper reports with exact Monte-Carlo outputs):
#
#      Attacker: 6 ships firing 6 independent offensive missiles, p_alpha = 0.67.
#      Defender: 6 ships attempting 3 interceptions in total, p_z = 0.67.
#      Damage per hit v ~ Normal(mean 0.33 ships, sd 0.11).
#      Armstrong's simulation (50,000 trials): mean loss 0.679 ships, sd 0.467,
#      95th pctile 1.499, P[loss=0] = 0.141.
#
#    NB: the spec's "mean Blue loss ~= 2.64 (sd ~0.99)" does NOT appear anywhere
#    in Armstrong (2011); the 6-on-3 example is the paper's actual, fully
#    specified, published anchor and is what we verify here.
#
#    Armstrong's model is AGGREGATE and continuous: loss = total damage in ships
#    (sum of v over the non-intercepted missiles), truncated to [0, B]. Our
#    engine's `*_damage` response is exactly that quantity; its per-hull INTEGER
#    kills (`*_ships_lost`) add the kill-quantization layer that is this paper's
#    contribution (spec sec 2/R10) and therefore sit far below the aggregate
#    loss (0.33-ship hits rarely stack to a full kill). Both are checked.
ARMSTRONG_TARGET_MEAN = 0.679   # Armstrong (2011) sec 5.1, 50k-trial simulation
ARMSTRONG_TARGET_SD = 0.467
ARMSTRONG_TARGET_MEAN_KILLS_MAX = 0.10  # per-hull kills must stay quantized (<< aggregate)
ARMSTRONG_CALIBRATED = True

# 6-on-3 in engine terms: attacker o=1 x6 ships => 6 missiles; defender d=0.5 x6
# ships => 3 intercepts; u = 1/w = 0.33 ship/hit; defender o=0 isolates one
# direction so `blue_damage` == damage delivered to the defender.
_ARM_ATTACKER = [(Platform("Ra", 1, 0, 1, 1), 6)]
_ARM_DEFENDER = [(Platform("Bd", 0, 0.5, 1.0 / 0.33, 1), 6)]


def _run_armstrong(reps=80000, seed=12345):
    p = dict(order="simultaneous", Tmax=1, theta=0.0, p_o=0.67, p_d=0.67,
             sigma_b=1.0, sigma_r=1.0, tau_b=1.0, tau_r=1.0, sd=0.11)
    return monte_carlo(_ARM_DEFENDER, _ARM_ATTACKER, p, reps=reps, seed=seed)


def test_armstrong_anchor():
    o = _run_armstrong()
    assert o["salvos_sd"] == 0.0  # Tmax=1 => single salvo
    # Aggregate continuous loss reproduces Armstrong's simulation (0.679 / 0.467).
    assert abs(o["blue_damage_mean"] - ARMSTRONG_TARGET_MEAN) < 0.02
    assert abs(o["blue_damage_sd"] - ARMSTRONG_TARGET_SD) < 0.02
    # Per-hull integer kills stay quantized far below the aggregate loss
    # (documents the kill-quantization layer this model adds over Armstrong).
    assert o["blue_ships_lost_mean"] < ARMSTRONG_TARGET_MEAN_KILLS_MAX


# ======================================================================
# 4. TIAH CONCENTRATION-vs-DISPERSION (spec sec 10.2), Blue outnumbered ~3:1.
#    Canonical scenario: Red = 6 x M; budget parity (rho=1).
#      Concentrated  = 2 x H (few big hulls, 3:1 outnumbered), no scouting edge,
#                      simultaneous            -> Blue loses the exchange (FER<1).
#      Dispersed     = many L (offense-tilted), high fusion edge, Blue-first
#                      surprise                -> Blue wins the exchange (FER>1).
#    Reproduces the deterministic swing of Tiah (2007); the exact 0.54/2.0
#    magnitudes need Tiah's precise inputs, so we assert the DIRECTION/sign.
TIAH_RED = [(RED_STD, 6)]
TIAH_BUDGET = 6 * unit_cost(RED_STD)  # rho = 1.0


def _tiah_concentrated(reps=15000, seed=11):
    blue = buy(TIAH_BUDGET, H_ESCORT)          # 2 x H
    p = dict(order="simultaneous", Tmax=1, theta=0.0, p_o=0.8, p_d=0.7,
             sigma_b=0.6, sigma_r=0.6, tau_b=1.0, tau_r=1.0, sd=0.15)
    return monte_carlo(blue, TIAH_RED, p, reps=reps, seed=seed)


def _tiah_dispersed(reps=15000, seed=12):
    blue = buy(TIAH_BUDGET, L_STRIKER)         # 12 x L
    p = dict(order="blue_first", Tmax=1, theta=0.0, p_o=0.8, p_d=0.7,
             sigma_b=0.95, sigma_r=0.6, tau_b=1.0, tau_r=1.0, sd=0.15)
    return monte_carlo(blue, TIAH_RED, p, reps=reps, seed=seed)


def test_tiah_swing():
    conc = _tiah_concentrated()
    disp = _tiah_dispersed()
    # Concentrated, no edge: Blue loses -> unfavourable exchange (log-FER < 0).
    assert conc["logfer_mean"] < 0.0
    # Dispersed + fusion edge + surprise: Blue wins -> favourable (log-FER > 0).
    assert disp["logfer_mean"] > 0.0
    # The swing itself: dispersion strictly improves the exchange ratio.
    assert disp["logfer_mean"] > conc["logfer_mean"] + 1.0


# ======================================================================
# 5. COMPOUNDING FUSION VALUE (contribution #3): the value of a sustained
#    scouting edge (Red loss at high sigma_B minus at low sigma_B) must GROW
#    with battle duration — the result a single-salvo model cannot produce.
def _fusion_value(Tmax, reps=6000):
    blue = (buy(TIAH_BUDGET * 0.34, L_STRIKER)
            + buy(TIAH_BUDGET * 0.33, M_BALANCED)
            + buy(TIAH_BUDGET * 0.33, H_ESCORT))
    base = dict(order="simultaneous", theta=0.30, p_o=0.8, p_d=0.7,
                tau_b=1.0, tau_r=1.0, sd=0.15, sigma_r=0.6, Tmax=Tmax)
    hi = monte_carlo(blue, TIAH_RED, {**base, "sigma_b": 0.95}, reps=reps, seed=10 + Tmax)
    lo = monte_carlo(blue, TIAH_RED, {**base, "sigma_b": 0.60}, reps=reps, seed=50 + Tmax)
    return hi["red_loss_mean"] - lo["red_loss_mean"]


def test_fusion_value_compounds():
    fv1 = _fusion_value(1)
    fv4 = _fusion_value(4)
    assert fv1 > 0.0                 # a scouting edge helps even in one salvo
    assert fv4 > fv1 + 0.1           # ...and compounds substantially over time


# ======================================================================
# 6. ALLOCATION ROBUSTNESS (spec sec 10.3): rerun a slice under
#    uniform-WITHOUT-replacement. Overkill (R10) must drop; the qualitative
#    conclusions and the SIGN of the fusion advantage (R5) must be unchanged.
def _alloc_slice(alloc, reps=15000, seed=7):
    blue = (buy(TIAH_BUDGET * 0.5, L_STRIKER) + buy(TIAH_BUDGET * 0.5, M_BALANCED))
    p = dict(order="blue_first", Tmax=6, theta=0.30, p_o=0.8, p_d=0.7,
             sigma_b=0.9, sigma_r=0.6, tau_b=1.0, tau_r=1.0, sd=0.15, alloc=alloc)
    return monte_carlo(blue, TIAH_RED, p, reps=reps, seed=seed)


def test_allocation_robustness():
    wr = _alloc_slice("with_replacement")
    nr = _alloc_slice("without_replacement")
    # R10: spreading leakers to distinct hulls first strictly reduces overkill.
    assert nr["overkill_frac_mean"] < wr["overkill_frac_mean"] - 0.05
    # R5/R6 sign robustness: the exchange stays favourable to Blue either way.
    assert wr["logfer_mean"] > 0.0
    assert nr["logfer_mean"] > 0.0
    # Conclusions do not flip: Red loss stays high under both allocation rules.
    assert abs(nr["red_loss_mean"] - wr["red_loss_mean"]) < 0.15


# ----------------------------------------------------------------------
def _summary():
    print("=" * 68)
    print("LAFusion 2026 — validation summary (salvo_mds)")
    print("=" * 68)

    o = monte_carlo([(M_BALANCED, 5)], [(RED_STD, 5)], base_params(), reps=30000, seed=1)
    print(f"[sec 6 ] symmetry: blue_loss={o['blue_loss_mean']:.3f} "
          f"red_loss={o['red_loss_mean']:.3f}  logFER={o['logfer_mean']:+.3f} "
          f"(fer_geom={o['fer_geom']:.3f}; raw fer_mean={o['fer_mean']:.1f} <- biased)")

    a = _run_armstrong()
    print(f"[10.1  ] Armstrong 6-on-3 anchor [CALIBRATED]: aggregate loss mean="
          f"{a['blue_damage_mean']:.3f} sd={a['blue_damage_sd']:.3f} "
          f"(Armstrong sim {ARMSTRONG_TARGET_MEAN}/{ARMSTRONG_TARGET_SD}); "
          f"per-hull kills={a['blue_ships_lost_mean']:.3f} (quantized)")

    c, d = _tiah_concentrated(), _tiah_dispersed()
    print(f"[10.2  ] Tiah swing: concentrated logFER={c['logfer_mean']:+.2f} "
          f"(FER_geom={c['fer_geom']:.2f})  ->  dispersed logFER={d['logfer_mean']:+.2f} "
          f"(FER_geom={d['fer_geom']:.2f})")

    fv1, fv4 = _fusion_value(1), _fusion_value(4)
    print(f"[claim3] compounding fusion value: Tmax=1 -> {fv1:.3f}, Tmax=4 -> {fv4:.3f}")

    wr, nr = _alloc_slice("with_replacement"), _alloc_slice("without_replacement")
    print(f"[10.3  ] allocation robustness: overkill with-repl={wr['overkill_frac_mean']:.3f} "
          f"-> without-repl={nr['overkill_frac_mean']:.3f}  "
          f"(logFER {wr['logfer_mean']:+.2f} -> {nr['logfer_mean']:+.2f}, sign unchanged)")
    print("=" * 68)


if __name__ == "__main__":
    _summary()
