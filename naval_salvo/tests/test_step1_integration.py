"""
Integration smoke test for Step 1.

Exercises the four foundational modules together: build a JPH-style
2v2 surface engagement, populate the canonical degenerate admissibility,
and verify that the resulting numerical objects (kernel matrices,
strength vectors, admissibility lookups) line up dimensionally with
each other.  This locks in the contract that step 2 (deterministic
dynamics) will rely on.
"""

from __future__ import annotations

import numpy as np
import pytest

from naval_salvo import (
    Admissibility,
    BattleState,
    DirectionalParameters,
    Domain,
    EngagementParameters,
    Force,
    PairParameters,
    UnitType,
)


def _build_jph_like_engagement():
    """
    A JPH-style 2-vs-2 surface engagement.

    Both sides have two surface unit types so that we exercise the
    heterogeneous-within-domain path while staying in the degenerate
    (S, S only) admissibility regime.

    Numerical values are illustrative -- they are not calibrated against
    any historical battle.  The point of this test is structural, not
    numerical.
    """
    blue = Force(
        label="Blue",
        unit_types=[
            UnitType("BF1", Domain.SURFACE, staying_power=3.0,
                     initial_strength=4),
            UnitType("BF2", Domain.SURFACE, staying_power=2.0,
                     initial_strength=2),
        ],
    )
    red = Force(
        label="Red",
        unit_types=[
            UnitType("RF1", Domain.SURFACE, staying_power=4.0,
                     initial_strength=3),
            UnitType("RF2", Domain.SURFACE, staying_power=2.0,
                     initial_strength=2),
        ],
    )

    bar = DirectionalParameters.zeros(blue, red)   # Blue->Red
    rab = DirectionalParameters.zeros(red, blue)   # Red->Blue

    # Symmetric, fully-trained, fully-aimed exchange with throughputs.
    for bj in ("BF1", "BF2"):
        for ri in ("RF1", "RF2"):
            bar.set(bj, ri, PairParameters(
                sigma_offense=0.5, eta_offense=0.9, p_offense=2.0,
                sigma_defense=0.5, eta_defense=0.8, p_defense=1.0,
            ))
    for rj in ("RF1", "RF2"):
        for bi in ("BF1", "BF2"):
            rab.set(rj, bi, PairParameters(
                sigma_offense=0.5, eta_offense=0.9, p_offense=2.0,
                sigma_defense=0.5, eta_defense=0.8, p_defense=1.0,
            ))

    ep = EngagementParameters(
        blue=blue, red=red,
        blue_attacks_red=bar, red_attacks_blue=rab,
    )
    bs = BattleState(blue=blue, red=red)
    return ep, bs


class TestStep1Integration:

    def test_kernel_dimensions_match_force_sizes(self):
        ep, bs = _build_jph_like_engagement()

        Kbar_off = ep.blue_attacks_red.offensive_kernel_matrix()
        Kbar_def = ep.blue_attacks_red.defensive_kernel_matrix()
        Krab_off = ep.red_attacks_blue.offensive_kernel_matrix()
        Krab_def = ep.red_attacks_blue.defensive_kernel_matrix()

        # (n_blue x n_red) and (n_red x n_blue).
        assert Kbar_off.shape == (ep.blue.n_unit_types, ep.red.n_unit_types)
        assert Kbar_def.shape == (ep.blue.n_unit_types, ep.red.n_unit_types)
        assert Krab_off.shape == (ep.red.n_unit_types, ep.blue.n_unit_types)
        assert Krab_def.shape == (ep.red.n_unit_types, ep.blue.n_unit_types)

    def test_degenerate_admissibility_filters_out_non_surface(self):
        # Even though all four unit types are SURFACE, this test pins the
        # contract that the admissibility lookup is keyed by Domain pair.
        # In the JPH-degenerate matrix only (S, S) is active.
        adm = Admissibility.degenerate()
        # All cross pairs in this all-surface engagement go through (S, S)
        # which is admissible (= 1).
        assert adm.is_admissible(Domain.SURFACE, Domain.SURFACE)
        # Anything involving a non-surface domain is structurally null.
        for d in (Domain.UNDERWATER, Domain.AIR, Domain.COASTAL, Domain.CYBER):
            assert not adm.is_admissible(d, d)
            assert not adm.is_admissible(Domain.SURFACE, d)
            assert not adm.is_admissible(d, Domain.SURFACE)

    def test_jph_first_salvo_kernel_value(self):
        """
        Compute the first-salvo attrition that the JPH equation predicts
        for unit BF1 of Blue (target i=0) under attack by Red, *by hand*
        from the kernels and admissibility.  This is the equation that
        step 2 will implement; here we only check that the data layer
        gives us all the pieces with the right shapes and values.

        JPH (2001) eq. (2.18), specialised to the per-salvo (jump) regime
        with chi = 1 for (S,S):

            Delta A_i = -(1 / s_i) * sum_j [ T^atq_{ji} - T^def_{ij} ]_+

        with
            T^atq_{ji} = sigma^atq * eta^atq * p^atq * B_j(t-)
            T^def_{ij} = sigma^def * eta^def * p^def * A_i(t-)

        We compute it manually here and just confirm dimensions / values.
        """
        ep, bs = _build_jph_like_engagement()
        adm = Admissibility.degenerate()

        # Strengths at t = 0 (= initial strengths because we just built it).
        A = ep.blue.strength_vector()         # shape (2,)
        B = ep.red.strength_vector()          # shape (2,)
        s_blue = ep.blue.staying_power_vector()  # shape (2,)

        # Red attacks Blue.
        Krab_off = ep.red_attacks_blue.offensive_kernel_matrix()  # (n_red, n_blue)
        Krab_def = ep.red_attacks_blue.defensive_kernel_matrix()  # (n_red, n_blue)

        # In the degenerate admissibility every pair is in (S, S) = 1.
        # T^atq[j -> i] for the salvo Red -> Blue:
        T_atq = Krab_off * B[:, None]        # (n_red, n_blue)
        T_def = Krab_def * A[None, :]        # (n_red, n_blue)

        # Aggregate over attacker index j (axis 0):
        kernel_per_target = np.maximum(0.0, (T_atq - T_def).sum(axis=0))
        # The clip-at-zero implements the "no negative attrition" rule
        # which JPH assume implicitly.

        # Hand value for BF1 (i = 0):
        #   p_off = 0.5 * 0.9 * 2.0 = 0.9
        #   p_def = 0.5 * 0.8 * 1.0 = 0.4
        #   T_atq sum_j (B = 3 + 2 = 5)  -> 0.9 * 5 = 4.5
        #   T_def sum_j (each j sees A_i = 4) -> 0.4 * 4 * 2 = 3.2
        #   raw kernel = 4.5 - 3.2 = 1.3
        #   delta_A_0  = -1.3 / 3.0 = -0.4333...
        assert kernel_per_target.shape == (ep.blue.n_unit_types,)
        assert kernel_per_target[0] == pytest.approx(4.5 - 3.2)
        # And the implied per-salvo delta (just for reference; the actual
        # dynamics module will compute this).
        delta_A = -kernel_per_target / s_blue
        assert delta_A[0] == pytest.approx(-(4.5 - 3.2) / 3.0)

    def test_battle_state_is_alive_at_t0(self):
        ep, bs = _build_jph_like_engagement()
        assert bs.is_terminated() is False
        assert bs.time == 0.0

    def test_admissibility_is_canonical_or_degenerate_disjoint(self):
        # Sanity: the canonical and degenerate matrices disagree on at
        # least one cell (otherwise something is wrong with the spec).
        canon = Admissibility.canonical().matrix
        degen = Admissibility.degenerate().matrix
        assert not np.allclose(canon, degen)
