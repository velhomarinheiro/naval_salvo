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
        step 2 implements; here we only check that the data layer
        gives us all the pieces with the right shapes and values.

        Canonical JPH (2001) reading, eq. (2.18) matrix form, in
        per-salvo (jump) regime with χ = 1 for (S, S):

            ΔA_i = -(1/s_i) * max(0,  Σ_j O_{ji} B_j(t-) - Σ_j D_{ij} A_i(t-))

        i.e. attacks summed first, defences summed first, then the
        aggregate difference clipped at zero.  We compute it manually
        here and just confirm dimensions / values.
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
        # Aggregate offence/defence over attackers j first, *then* clip.
        total_offense = (Krab_off * B[:, None]).sum(axis=0)   # (n_blue,)
        total_defense = (Krab_def * A[None, :]).sum(axis=0)   # (n_blue,)
        kernel_per_target = np.maximum(0.0, total_offense - total_defense)

        # Hand value for BF1 (i = 0):
        #   O_{ji} = 0.5 * 0.9 * 2.0 = 0.9   (same for all j, i in this symmetric setup)
        #   D_{ij} = 0.5 * 0.8 * 1.0 = 0.4
        #   Σ_j O B_j = 0.9 * (3 + 2)           = 4.5
        #   Σ_j D A_i = 0.4 * 4 * 2             = 3.2
        #   raw kernel = max(0, 4.5 - 3.2)      = 1.3
        #   delta_A_0  = -1.3 / 3.0             ≈ -0.4333
        assert kernel_per_target.shape == (ep.blue.n_unit_types,)
        assert kernel_per_target[0] == pytest.approx(4.5 - 3.2)
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
