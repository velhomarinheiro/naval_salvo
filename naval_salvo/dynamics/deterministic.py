"""
naval_salvo.dynamics.deterministic
==================================

Deterministic, *pulsed* (jump) regime of the heterogeneous multi-domain
salvo equation.

This module implements the canonical jump equation of Phase 1 (working
document 1.4 §3, "Equação canônica em regime pulsado"):

    A^{(d)}_i (t_k^+) = A^{(d)}_i (t_k^-)
                      - (1 / s^{(d)}_i)
                        sum_{d'} sum_j  1^{(d', d)}
                                        [ T^atq_{ji}(t_k^-) - T^def_{ij}(t_k^-) ]_+

where

    [x]_+               = max(0, x)             (no negative attrition)
    T^atq_{ji}          = sigma_o * eta_o * p_o * B_j(t_k^-)
    T^def_{ij}          = sigma_d * eta_d * p_d * A_i(t_k^-)
    1^{(d', d)}         = admissibility coefficient
                          (attacker domain d' to defender domain d)
    s^{(d)}_i            = staying power of unit type i in domain d

The exchange is *simultaneous*: both sides compute their incoming
attrition based on the pre-salvo state, then both update.  This
matches Hughes (1995), Johns-Pilnick-Hughes (2001), and Armstrong (2005)
deterministic.  The *sequential* variant (Armstrong 2014) lives in a
separate module that this one will be a wrapper-target for in step 6.

Continuous regime
-----------------
The continuous (EDO) version of the canonical equation will land in the
solver / dynamics-dispatch layer of step 3+, paired with an event
integrator.  This module is intentionally *just the jumps*; that
restriction makes the JPH / Hughes degenerate validation a one-line
arithmetic check and decouples the integrator's mechanics from the
attrition kernel.

References
----------
- Hughes (1995) NRL 42:267-289, eqs. for ΔA, ΔB.
- Johns, Pilnick & Hughes (2001) NPS-IJWA-01-010, eq. (2.16)-(2.18).
- Armstrong (2005) Operations Research 53:830-841, simultaneous-fire
  deterministic baseline.
- Working document 1.4 §3.3 (canonical pulsed equation).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..admissibility import Admissibility
from ..domains import Domain
from ..parameters import EngagementParameters
from ..state import BattleState, Force


# ---------------------------------------------------------------------------
# Public-facing salvo step
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SalvoResult:
    """
    Outcome of a single simultaneous salvo exchange.

    Attributes
    ----------
    blue_strength_pre, red_strength_pre : ndarray
        Strength vectors *before* the salvo.  Same ordering as
        ``Force.unit_types``.
    blue_strength_post, red_strength_post : ndarray
        Strength vectors *after* the salvo (post-clip in [0, initial]).
    blue_losses, red_losses : ndarray
        Element-wise losses, ``pre - post``.  Always non-negative.
    blue_raw_kernel, red_raw_kernel : ndarray
        Raw per-target kernel ``sum_j max(0, T^atq - T^def)`` *before*
        dividing by staying power and clipping.  Same shape as the
        respective force.  Useful for diagnostics and for validating
        analytical predictions.
    """

    blue_strength_pre: np.ndarray
    red_strength_pre: np.ndarray
    blue_strength_post: np.ndarray
    red_strength_post: np.ndarray
    blue_losses: np.ndarray
    red_losses: np.ndarray
    blue_raw_kernel: np.ndarray
    red_raw_kernel: np.ndarray


def salvo_step(
    state: BattleState,
    params: EngagementParameters,
    admissibility: Admissibility,
    *,
    apply: bool = True,
    record_time: Optional[float] = None,
) -> SalvoResult:
    """
    Apply one simultaneous salvo exchange to ``state``.

    Computes, for each side, the incoming attrition implied by the
    canonical heterogeneous multi-domain salvo equation evaluated at the
    pre-salvo strengths, then -- if ``apply`` is True -- writes the new
    strengths back into the Force objects in ``state``.

    Parameters
    ----------
    state : BattleState
        The live battle state.  Read for pre-salvo strengths; mutated
        in place if ``apply`` is True.
    params : EngagementParameters
        Per-pair offensive / defensive throughputs and effectiveness
        coefficients for both directions.
    admissibility : Admissibility
        5x5 cross-domain admissibility matrix.  Cells equal to zero
        completely zero out the corresponding attacker-defender domain
        pair.  Cells equal to a marginal value chi multiply both the
        offensive and the defensive kernels of the matching pairs by
        chi (i.e. the indicator is interpreted as a *coefficient*, not
        as a hard 0/1 mask -- this is the Phase 1 design choice that
        permits calibrating chi).
    apply : bool, default True
        If True, write the post-salvo strengths back into
        ``state.blue`` and ``state.red``, and (if ``record_time`` is not
        None) append ``record_time`` to ``state.salvo_times``.  If
        False, the SalvoResult is computed but the state is not
        modified, which is useful for what-if analysis and tests.
    record_time : float, optional
        If provided and ``apply`` is True, the time at which the salvo
        is logged in ``state.salvo_times``.  Defaults to leaving the
        salvo log unchanged.

    Returns
    -------
    SalvoResult
        Pre / post / loss vectors and the raw kernels.

    Notes
    -----
    The implementation follows the JPH (2001) matrix form (NPS-IJWA-01-010
    p. 15-16), in which the per-target attrition is

        ΔA_i = -(1/s_i) * max(0,  Σ_j O_{ji} B_j  -  Σ_j D_{ij} A_i)

    i.e. attacks are summed over all attackers first, defences are
    summed over all attackers first, and the *aggregate* difference is
    clipped at zero.  This is the same convention as Hughes (1995) and
    Armstrong (2005), and matches the worked numerical example in JPH
    p. 27 (Coral Sea).  Cross-domain admissibility χ^{(d', d)} enters
    *inside* the sum, multiplying each (j, i) contribution to both
    O_{ji} B_j and D_{ij} A_i: this preserves the JPH structure
    exactly when χ ≡ 1, which is what the degenerate-case validation
    relies on.

    Per-pair clipping is *not* used.  Two one-on-many engagements with
    overlapping defenders would otherwise create non-physical "ghost"
    interceptions where the defender intercepts more shots than were
    actually fired.  The aggregate clip is the canonical resolution.
    """
    # Pre-salvo strengths.
    A = state.blue.strength_vector()                 # shape (n_blue,)
    B = state.red.strength_vector()                  # shape (n_red,)

    # Incoming attrition on Blue (Red attacks Blue).
    blue_raw = _aggregate_incoming_kernel(
        attacker_force=state.red,
        defender_force=state.blue,
        attacker_strengths=B,
        defender_strengths=A,
        offensive_kernel=params.red_attacks_blue.offensive_kernel_matrix(),
        defensive_kernel=params.red_attacks_blue.defensive_kernel_matrix(),
        admissibility=admissibility,
    )

    # Incoming attrition on Red (Blue attacks Red).
    red_raw = _aggregate_incoming_kernel(
        attacker_force=state.blue,
        defender_force=state.red,
        attacker_strengths=A,
        defender_strengths=B,
        offensive_kernel=params.blue_attacks_red.offensive_kernel_matrix(),
        defensive_kernel=params.blue_attacks_red.defensive_kernel_matrix(),
        admissibility=admissibility,
    )

    # Convert hits -> unit losses by dividing by staying power, then
    # apply the canonical "no amplification" rule: losses are clipped at
    # the pre-salvo strength (you can't lose more units than you had)
    # and post-salvo strength is clipped at zero from below.  Initial
    # strength is *not* an upper bound on post-salvo: the model has no
    # regeneration in this regime, so post <= pre <= initial holds
    # automatically.
    s_blue = state.blue.staying_power_vector()
    s_red = state.red.staying_power_vector()

    blue_losses = np.minimum(blue_raw / s_blue, A)
    red_losses = np.minimum(red_raw / s_red, B)

    A_post = A - blue_losses
    B_post = B - red_losses
    # Defensive belt: clip tiny negative round-off to zero.
    A_post = np.maximum(A_post, 0.0)
    B_post = np.maximum(B_post, 0.0)

    result = SalvoResult(
        blue_strength_pre=A,
        red_strength_pre=B,
        blue_strength_post=A_post,
        red_strength_post=B_post,
        blue_losses=blue_losses,
        red_losses=red_losses,
        blue_raw_kernel=blue_raw,
        red_raw_kernel=red_raw,
    )

    if apply:
        state.blue.set_strength_vector(A_post)
        state.red.set_strength_vector(B_post)
        if record_time is not None:
            state.record_salvo(record_time)

    return result


# ---------------------------------------------------------------------------
# Internal: per-direction kernel aggregator
# ---------------------------------------------------------------------------


def _aggregate_incoming_kernel(
    *,
    attacker_force: Force,
    defender_force: Force,
    attacker_strengths: np.ndarray,
    defender_strengths: np.ndarray,
    offensive_kernel: np.ndarray,
    defensive_kernel: np.ndarray,
    admissibility: Admissibility,
) -> np.ndarray:
    """
    Compute the per-defender raw attrition kernel for one direction.

    Returns
    -------
    ndarray of shape (n_defender,)
        Element ``i`` is

            max(0,  Σ_j χ^{(d_j, d_i)} O_{ji} B_j
                  - Σ_j χ^{(d_j, d_i)} D_{ij} A_i)

        where d_j and d_i are the domains of attacker unit type j and
        defender unit type i respectively, and χ is the admissibility
        coefficient.  Attacks and defences are aggregated over all
        attackers j *before* the clip, matching the canonical JPH
        (2001) reading and reducing exactly to Hughes (1995) when both
        forces collapse to one unit type per side.

    Notes
    -----
    The admissibility coefficient is applied *uniformly* to both T^atq
    and T^def of the same (j, i) pair.  This is the right semantics
    because χ reflects the doctrinal feasibility of the engagement
    geometry: if a torpedo can't reach an aircraft (χ = 0), then
    neither can the torpedo's offensive throughput nor the aircraft's
    point-defense vs. that torpedo enter the equation.  This is also
    what makes the canonical recovery of JPH (2001) clean: setting
    χ = 1 for (S, S) and 0 elsewhere reproduces JPH exactly.
    """
    n_atk = attacker_force.n_unit_types
    n_def = defender_force.n_unit_types

    # Build the full χ(j, i) matrix once.
    chi = np.empty((n_atk, n_def), dtype=np.float64)
    for j, ut_j in enumerate(attacker_force.unit_types):
        for i, ut_i in enumerate(defender_force.unit_types):
            chi[j, i] = admissibility[ut_j.domain, ut_i.domain]

    # Per-(attacker, defender) attrition tensor, χ-weighted.
    # T_atq[j, i] = χ * O_{ji} * B_j               (j-th attacker fires)
    # T_def[j, i] = χ * D_{ij} * A_i               (i-th defender intercepts)
    weighted_off = chi * offensive_kernel
    weighted_def = chi * defensive_kernel

    # Aggregate over j (attackers), giving per-defender totals.
    total_offense_per_target = (weighted_off * attacker_strengths[:, None]).sum(axis=0)
    total_defense_per_target = (weighted_def * defender_strengths[None, :]).sum(axis=0)

    # *Aggregate* clip-at-zero (canonical JPH reading; matches Hughes 1995).
    return np.maximum(0.0, total_offense_per_target - total_defense_per_target)
