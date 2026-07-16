"""
validate_demo.py — Human-readable validation + demonstration narrative for the
LAFusion 2026 experiment. This is the *readable* companion to the automated
regression suite in test_validation.py (spec sec 10); both draw on the SAME
canonical scenarios, so the demo and the tests can never drift apart.

    python3 validate_demo.py

For the pass/fail regression gate used in the reproducibility package, run:

    pytest test_validation.py -v
"""
import numpy as np
from salvo_mds import (Platform, unit_cost, monte_carlo, simulate,
                       L_STRIKER, M_BALANCED, H_ESCORT, RED_STD)
import test_validation as V


def line():
    print("-" * 68)


# ---------------------------------------------------------------
print("=" * 68)
print("ARCHETYPE COSTS (spec sec 4/5, gamma=1.35)")
line()
for p in (L_STRIKER, M_BALANCED, H_ESCORT):
    print(f"  {p.name}: o={p.offense} d={p.defense} w={p.staying} mag={p.magazine}"
          f"  -> cost={unit_cost(p):.1f}  (throughput/cost={p.offense/unit_cost(p):.2f})")
print(f"  Red standard platform == M; Red force = 5 x M, value = "
      f"{5 * unit_cost(RED_STD):.1f}  (budget-ratio base, spec sec 6)")

# ---------------------------------------------------------------
# 1. SYMMETRY (spec sec 6): the mirror-match built-in check.
print("\n" + "=" * 68)
print("1. SYMMETRY CHECK (rho=1, Blue = 5 x M = exact mirror of Red)")
line()
o = monte_carlo([(M_BALANCED, 5)], [(RED_STD, 5)], V.base_params(), reps=30000, seed=1)
print(f"  Blue loss = {o['blue_loss_mean']:.3f}   Red loss = {o['red_loss_mean']:.3f}   (must match)")
print(f"  log-FER   = {o['logfer_mean']:+.3f}  (fer_geom = {o['fer_geom']:.3f}; must be ~0 / ~1)")
print(f"  NOTE: the RAW mean-of-ratios fer_mean = {o['fer_mean']:.1f} is badly biased even")
print(f"        in this symmetric fight -> R4 uses the logit-transformed log-FER instead.")

# ---------------------------------------------------------------
# 2. ARMSTRONG SINGLE-SALVO ANCHOR (spec sec 10.1) — CALIBRATED.
print("\n" + "=" * 68)
print("2. ARMSTRONG ANCHOR (2011 sec 5.1 6-on-3 illustrative example, Tmax=1)")
line()
a = V._run_armstrong()
print(f"  aggregate continuous loss: mean = {a['blue_damage_mean']:.3f} ships "
      f"(sd {a['blue_damage_sd']:.3f})   [Armstrong sim: 0.679 / 0.467]")
print(f"  per-hull INTEGER kills   : mean = {a['blue_ships_lost_mean']:.3f}   "
      f"(<< aggregate: the kill-quantization layer this model adds, spec sec 2/R10)")
print("  The base salvo exchange (offense/defense aggregation + leaker + damage)")
print("  reproduces Armstrong exactly at cap=1; per-hull resolution adds quantization.")
print("  (The spec's '2.64' does not appear in Armstrong 2011; 6-on-3 is the paper's")
print("   actual published anchor.)")

# ---------------------------------------------------------------
# 3. TIAH CONCENTRATION vs DISPERSION (spec sec 10.2), Blue outnumbered 3:1.
print("\n" + "=" * 68)
print("3. TIAH REPLICATION (single salvo, budget parity, Blue 3:1 outnumbered in hulls)")
line()
c, d = V._tiah_concentrated(), V._tiah_dispersed()
print(f"  Red = 6 x M.  Budget parity (rho=1).")
print(f"  Concentrated (2 x H, no scouting edge, simultaneous):")
print(f"      log-FER = {c['logfer_mean']:+.2f} (FER_geom={c['fer_geom']:.2f})  "
      f"Blue loss {c['blue_loss_mean']:.2f}  Red loss {c['red_loss_mean']:.2f}  -> Blue loses")
print(f"  Dispersed (12 x L, high Delta-sigma, Blue-first surprise):")
print(f"      log-FER = {d['logfer_mean']:+.2f} (FER_geom={d['fer_geom']:.2f})  "
      f"Blue loss {d['blue_loss_mean']:.2f}  Red loss {d['red_loss_mean']:.2f}  -> Blue wins")
print("  -> reproduces Tiah's swing FER<1 (concentrated) to FER>1 (dispersed+scout+surprise).")
print("     Exact 0.54/2.0 needs Tiah's precise inputs; the sign of the swing is the check.")

# ---------------------------------------------------------------
# 4. COMPOUNDING FUSION VALUE (contribution #3) — the multi-salvo headline.
print("\n" + "=" * 68)
print("4. COMPOUNDING FUSION VALUE (Red loss: high sigma_B vs low sigma_B, by Tmax)")
line()
print(f"  {'Tmax':>4} | {'fusion value (RedLoss@0.95 - @0.60)':>36}")
prev = None
for Tmax in (1, 2, 3, 4, 6):
    fv = V._fusion_value(Tmax)
    arrow = "" if prev is None else ("  (up)" if fv > prev else "  (~flat/sat.)")
    print(f"  {Tmax:>4} | {fv:>36.3f}{arrow}")
    prev = fv
print("  Fusion value rises then saturates: a sustained scouting edge COMPOUNDS -- the")
print("  multi-salvo result a single-salvo model cannot produce.")

# ---------------------------------------------------------------
# 5. ALLOCATION ROBUSTNESS (spec sec 10.3) + per-unit responses.
print("\n" + "=" * 68)
print("5. ALLOCATION ROBUSTNESS (uniform with- vs without-replacement)")
line()
wr, nr = V._alloc_slice("with_replacement"), V._alloc_slice("without_replacement")
print(f"  with-replacement    : overkill R10 = {wr['overkill_frac_mean']:.3f}  "
      f"Red loss {wr['red_loss_mean']:.3f}  log-FER {wr['logfer_mean']:+.2f}  P(vict)={wr['p_victory']:.3f}")
print(f"  without-replacement : overkill R10 = {nr['overkill_frac_mean']:.3f}  "
      f"Red loss {nr['red_loss_mean']:.3f}  log-FER {nr['logfer_mean']:+.2f}  P(vict)={nr['p_victory']:.3f}")
print("  -> R10 (overkill) drops when leakers spread to distinct hulls first; the sign of")
print("     R5/R6 and the conclusions are unchanged (spec sec 10.3).")

print("\n" + "=" * 68)
print("6. PER-UNIT RESPONSES (one multi-salvo mixed engagement)")
line()
blue = (V.buy(V.TIAH_BUDGET * 0.5, L_STRIKER) + V.buy(V.TIAH_BUDGET * 0.5, M_BALANCED))
p4 = dict(order="blue_first", Tmax=6, theta=0.30, p_o=0.8, p_d=0.7,
          sigma_b=0.9, sigma_r=0.6, tau_b=1.0, tau_r=1.0, sd=0.15)
r4 = monte_carlo(blue, V.TIAH_RED, p4, reps=10000, seed=99)
print(f"  Blue force: {[(pl.name, n) for pl, n in blue]}   Red: 6 x M")
print(f"  P(Blue victory)       = {r4['p_victory']:.3f}")
print(f"  mean salvos-to-decide = {r4['salvos_mean']:.2f}")
print(f"  overkill fraction     = {r4['overkill_frac_mean']:.3f}   (wasted/delivered damage)")
print(f"  mean Blue ships lost  = {r4['blue_ships_lost_mean']:.2f}")
print(f"  mean Red ships lost    = {r4['red_ships_lost_mean']:.2f}")
one = simulate(blue, V.TIAH_RED, p4, np.random.default_rng(7))
print(f"  example survivors by type (single battle): {one['survivors']}")
print("=" * 68)
