"""Tests for naval_salvo.dynamics.deterministic."""

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
    SalvoResult,
    UnitType,
    salvo_step,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _simple_homogeneous(A0=4.0, B0=3.0, alpha=2.0, beta=2.0,
                        z=1.0, y=1.0, w=2.0, x=2.0):
    """1-vs-1 homogeneous surface engagement, Hughes parameter mapping."""
    blue = Force("Blue", [UnitType("B", Domain.SURFACE,
                                   staying_power=w, initial_strength=B0)])
    red = Force("Red", [UnitType("A", Domain.SURFACE,
                                 staying_power=x, initial_strength=A0)])
    bar = DirectionalParameters.zeros(blue, red)
    bar.set("B", "A", PairParameters(p_offense=beta, p_defense=y))
    rab = DirectionalParameters.zeros(red, blue)
    rab.set("A", "B", PairParameters(p_offense=alpha, p_defense=z))
    ep = EngagementParameters(blue=blue, red=red,
                              blue_attacks_red=bar,
                              red_attacks_blue=rab)
    bs = BattleState(blue=blue, red=red)
    adm = Admissibility.degenerate()
    return bs, ep, adm


# ---------------------------------------------------------------------------
# Smoke / shape tests
# ---------------------------------------------------------------------------


class TestSalvoStepBasic:

    def test_returns_salvoresult(self):
        bs, ep, adm = _simple_homogeneous()
        out = salvo_step(bs, ep, adm)
        assert isinstance(out, SalvoResult)

    def test_pre_strengths_match_initial(self):
        bs, ep, adm = _simple_homogeneous(A0=5, B0=7)
        out = salvo_step(bs, ep, adm, apply=False)
        np.testing.assert_array_equal(out.red_strength_pre, [5.0])
        np.testing.assert_array_equal(out.blue_strength_pre, [7.0])

    def test_apply_false_does_not_mutate(self):
        bs, ep, adm = _simple_homogeneous(A0=5, B0=7)
        salvo_step(bs, ep, adm, apply=False)
        np.testing.assert_array_equal(bs.blue.strength_vector(), [7.0])
        np.testing.assert_array_equal(bs.red.strength_vector(), [5.0])

    def test_apply_true_mutates(self):
        bs, ep, adm = _simple_homogeneous(A0=4, B0=3,
                                          alpha=4.0, beta=4.0,
                                          z=1.0, y=1.0,
                                          w=2.0, x=2.0)
        salvo_step(bs, ep, adm, apply=True)
        # Hughes:  ΔA = -(βB - yA)/x = -(4*3 - 1*4)/2 = -4
        #          A_post = max(0, 4 - 4) = 0
        np.testing.assert_array_almost_equal(
            bs.red.strength_vector(), [0.0]
        )

    def test_record_time(self):
        bs, ep, adm = _simple_homogeneous()
        assert bs.salvo_times == []
        salvo_step(bs, ep, adm, apply=True, record_time=1.5)
        assert bs.salvo_times == [1.5]


# ---------------------------------------------------------------------------
# Edge cases of the kernel
# ---------------------------------------------------------------------------


class TestSalvoStepEdgeCases:

    def test_no_offense_no_change(self):
        bs, ep, adm = _simple_homogeneous(alpha=0.0, beta=0.0)
        out = salvo_step(bs, ep, adm)
        np.testing.assert_array_equal(out.blue_losses, [0.0])
        np.testing.assert_array_equal(out.red_losses, [0.0])

    def test_defense_dominates_no_loss(self):
        # β·B = 1·3 = 3   vs.   y·A = 5·4 = 20   →   max(0, -17) = 0
        bs, ep, adm = _simple_homogeneous(beta=1.0, y=5.0)
        out = salvo_step(bs, ep, adm, apply=False)
        np.testing.assert_array_equal(out.red_losses, [0.0])

    def test_loss_capped_at_initial_strength(self):
        # Make β so large that one salvo would notionally kill 1000 ships;
        # we have only 4.
        bs, ep, adm = _simple_homogeneous(A0=4.0, B0=3.0,
                                          alpha=0.0, beta=1000.0,
                                          z=0.0, y=0.0,
                                          w=1.0, x=1.0)
        out = salvo_step(bs, ep, adm, apply=False)
        # Cap: -ΔA ≤ A.
        assert out.red_losses[0] == pytest.approx(4.0)
        assert out.red_strength_post[0] == 0.0

    def test_admissibility_zero_blocks_attrition(self):
        # Build an engagement whose only attacker-defender pair has
        # admissibility 0; output should be zero-loss.
        blue = Force("Blue", [UnitType("BSub", Domain.UNDERWATER, 2.0, 1)])
        red = Force("Red",  [UnitType("RAir", Domain.AIR,        2.0, 1)])
        bar = DirectionalParameters.zeros(blue, red)
        bar.set("BSub", "RAir", PairParameters(p_offense=10.0, p_defense=0.0))
        rab = DirectionalParameters.zeros(red, blue)
        rab.set("RAir", "BSub", PairParameters(p_offense=10.0, p_defense=0.0))
        ep = EngagementParameters(blue=blue, red=red,
                                  blue_attacks_red=bar,
                                  red_attacks_blue=rab)
        bs = BattleState(blue=blue, red=red)
        # In the canonical matrix, (U, A) = 0 (torpedo vs aircraft).
        adm = Admissibility.canonical()
        out = salvo_step(bs, ep, adm, apply=False)
        # The Blue→Red direction is U→A = 0; so Red suffers nothing.
        np.testing.assert_array_equal(out.red_losses, [0.0])

    def test_admissibility_chi_scales_kernel_linearly(self):
        # Two scenarios, identical except chi for the cross-domain cell.
        # The raw kernel should scale linearly with chi.
        def scenario(chi_value):
            blue = Force("Blue", [UnitType("BS", Domain.SURFACE, 2.0, 4)])
            red = Force("Red",  [UnitType("RU", Domain.UNDERWATER, 2.0, 4)])
            bar = DirectionalParameters.zeros(blue, red)
            bar.set("BS", "RU", PairParameters(p_offense=5.0, p_defense=0.0))
            rab = DirectionalParameters.zeros(red, blue)
            rab.set("RU", "BS", PairParameters(p_offense=0.0, p_defense=0.0))
            ep = EngagementParameters(blue=blue, red=red,
                                      blue_attacks_red=bar,
                                      red_attacks_blue=rab)
            bs = BattleState(blue=blue, red=red)
            M = np.zeros((5, 5), dtype=np.float64)
            M[Domain.SURFACE.index, Domain.UNDERWATER.index] = chi_value
            adm = Admissibility.from_array(M)
            return bs, ep, adm

        bs1, ep1, adm1 = scenario(1.0)
        bs2, ep2, adm2 = scenario(0.3)
        out1 = salvo_step(bs1, ep1, adm1, apply=False)
        out2 = salvo_step(bs2, ep2, adm2, apply=False)
        # raw_kernel(chi=0.3) = 0.3 * raw_kernel(chi=1.0)
        np.testing.assert_allclose(
            out2.red_raw_kernel, 0.3 * out1.red_raw_kernel
        )


# ---------------------------------------------------------------------------
# Simultaneity: both sides see the *pre-salvo* state
# ---------------------------------------------------------------------------


class TestSimultaneity:

    def test_first_strike_does_not_get_full_credit(self):
        """
        If Blue has β=10 z=0 w=1 and Red has α=0 y=0 x=1, the simultaneous
        salvo step kills B*10 of A's units (capped at A).  In a *sequential*
        Blue-first model Red would have zero return fire and Blue would be
        intact; in a sequential Red-first model Blue would be wiped out.
        Simultaneous fire computes ΔB based on Red's *pre-salvo* α=0 (so
        Blue takes no losses), while Red is destroyed.  The point of this
        test is to lock in that *simultaneity* semantic.
        """
        bs, ep, adm = _simple_homogeneous(A0=5.0, B0=2.0,
                                          alpha=0.0, beta=10.0,
                                          z=0.0, y=0.0,
                                          w=1.0, x=1.0)
        out = salvo_step(bs, ep, adm, apply=True)
        # Blue's losses: ΔB = -(αA - zB)/w = -(0 - 0)/1 = 0.
        np.testing.assert_array_equal(bs.blue.strength_vector(), [2.0])
        # Red's losses: ΔA = -(βB - yA)/x = -(10·2 - 0)/1 = -20, capped at A.
        np.testing.assert_array_equal(bs.red.strength_vector(), [0.0])

    def test_mutual_kill_possible(self):
        """
        With high lethality on both sides and equal forces, both can be
        wiped out in a single simultaneous exchange.  This is *not*
        possible in a sequential model.
        """
        bs, ep, adm = _simple_homogeneous(A0=2.0, B0=2.0,
                                          alpha=10.0, beta=10.0,
                                          z=0.0, y=0.0,
                                          w=1.0, x=1.0)
        out = salvo_step(bs, ep, adm, apply=True)
        np.testing.assert_array_equal(bs.blue.strength_vector(), [0.0])
        np.testing.assert_array_equal(bs.red.strength_vector(), [0.0])


# ---------------------------------------------------------------------------
# Sanity properties (derived from canonical equation)
# ---------------------------------------------------------------------------


class TestSanityProperties:

    def test_monotonic_in_offensive_throughput(self):
        """Holding all else equal, larger β → larger Red losses."""
        bs1, ep1, adm = _simple_homogeneous(beta=2.0)
        bs2, ep2, _   = _simple_homogeneous(beta=4.0)
        out1 = salvo_step(bs1, ep1, adm, apply=False)
        out2 = salvo_step(bs2, ep2, adm, apply=False)
        assert out2.red_losses[0] >= out1.red_losses[0]

    def test_monotonic_in_defensive_throughput(self):
        """Holding all else equal, larger y → smaller Red losses."""
        bs1, ep1, adm = _simple_homogeneous(y=0.5)
        bs2, ep2, _   = _simple_homogeneous(y=2.0)
        out1 = salvo_step(bs1, ep1, adm, apply=False)
        out2 = salvo_step(bs2, ep2, adm, apply=False)
        assert out2.red_losses[0] <= out1.red_losses[0]

    def test_strengths_nonnegative_after_salvo(self):
        # Random-ish parameters; just confirm we never go negative.
        rng = np.random.default_rng(42)
        for _ in range(20):
            params = dict(
                A0=rng.uniform(1, 10), B0=rng.uniform(1, 10),
                alpha=rng.uniform(0, 5), beta=rng.uniform(0, 5),
                z=rng.uniform(0, 3),     y=rng.uniform(0, 3),
                w=rng.uniform(0.5, 3),   x=rng.uniform(0.5, 3),
            )
            bs, ep, adm = _simple_homogeneous(**params)
            salvo_step(bs, ep, adm, apply=True)
            assert bs.blue.strength_vector()[0] >= 0.0
            assert bs.red.strength_vector()[0] >= 0.0

    def test_initial_strength_is_upper_bound(self):
        """In this regime (no regeneration), strengths never increase."""
        rng = np.random.default_rng(123)
        for _ in range(20):
            params = dict(
                A0=rng.uniform(1, 10), B0=rng.uniform(1, 10),
                alpha=rng.uniform(0, 5), beta=rng.uniform(0, 5),
                z=rng.uniform(0, 3),     y=rng.uniform(0, 3),
                w=rng.uniform(0.5, 3),   x=rng.uniform(0.5, 3),
            )
            bs, ep, adm = _simple_homogeneous(**params)
            A0 = bs.red.strength_vector()[0]
            B0 = bs.blue.strength_vector()[0]
            for _ in range(5):
                salvo_step(bs, ep, adm, apply=True)
                assert bs.red.strength_vector()[0] <= A0 + 1e-12
                assert bs.blue.strength_vector()[0] <= B0 + 1e-12


# ---------------------------------------------------------------------------
# Heterogeneous-within-domain (JPH 2v2 surface)
# ---------------------------------------------------------------------------


class TestHeterogeneousWithinDomain:
    """
    Kicks the "multiple unit types in the same domain" path of the
    targeting / kernel aggregation, using a 2-vs-2 surface engagement
    similar to the integration test of step 1 (which only exercised
    the data layer).
    """

    @staticmethod
    def _build_2v2():
        blue = Force("Blue", [
            UnitType("BF1", Domain.SURFACE, staying_power=3.0,
                     initial_strength=4),
            UnitType("BF2", Domain.SURFACE, staying_power=2.0,
                     initial_strength=2),
        ])
        red = Force("Red", [
            UnitType("RF1", Domain.SURFACE, staying_power=4.0,
                     initial_strength=3),
            UnitType("RF2", Domain.SURFACE, staying_power=2.0,
                     initial_strength=2),
        ])
        bar = DirectionalParameters.zeros(blue, red)
        rab = DirectionalParameters.zeros(red, blue)
        params = PairParameters(
            sigma_offense=0.5, eta_offense=0.9, p_offense=2.0,
            sigma_defense=0.5, eta_defense=0.8, p_defense=1.0,
        )
        for bj in ("BF1", "BF2"):
            for ri in ("RF1", "RF2"):
                bar.set(bj, ri, params)
        for rj in ("RF1", "RF2"):
            for bi in ("BF1", "BF2"):
                rab.set(rj, bi, params)
        ep = EngagementParameters(blue=blue, red=red,
                                  blue_attacks_red=bar,
                                  red_attacks_blue=rab)
        bs = BattleState(blue=blue, red=red)
        adm = Admissibility.degenerate()
        return bs, ep, adm

    def test_locks_in_step1_integration_value(self):
        """
        The Step-1 integration test computed by hand:
        kernel for BF1 = 4.5 - 3.2 = 1.3, ΔA_0 = -1.3 / 3.0.

        The deterministic dynamics module must reproduce this exactly.
        """
        bs, ep, adm = self._build_2v2()
        out = salvo_step(bs, ep, adm, apply=True)
        assert out.blue_raw_kernel[0] == pytest.approx(1.3)
        assert out.blue_losses[0] == pytest.approx(1.3 / 3.0)
        # Post-salvo strength of BF1.
        assert bs.blue.strength_of("BF1") == pytest.approx(4.0 - 1.3 / 3.0)

    def test_symmetry_in_symmetric_setup(self):
        """
        With identical parameters and identical force composition the
        two sides should suffer identical kernels (target by target).
        Our 2v2 setup is *not* fully symmetric (different staying powers
        on RF1 vs RF2 etc.), so we set up a *fully symmetric* one here.
        """
        blue = Force("Blue", [
            UnitType("U1", Domain.SURFACE, 2.0, 3),
            UnitType("U2", Domain.SURFACE, 2.0, 2),
        ])
        red = Force("Red", [
            UnitType("V1", Domain.SURFACE, 2.0, 3),
            UnitType("V2", Domain.SURFACE, 2.0, 2),
        ])
        bar = DirectionalParameters.zeros(blue, red)
        rab = DirectionalParameters.zeros(red, blue)
        params = PairParameters(
            sigma_offense=1.0, eta_offense=1.0, p_offense=1.5,
            sigma_defense=1.0, eta_defense=1.0, p_defense=0.5,
        )
        for bj in ("U1", "U2"):
            for ri in ("V1", "V2"):
                bar.set(bj, ri, params)
        for rj in ("V1", "V2"):
            for bi in ("U1", "U2"):
                rab.set(rj, bi, params)
        ep = EngagementParameters(blue=blue, red=red,
                                  blue_attacks_red=bar,
                                  red_attacks_blue=rab)
        bs = BattleState(blue=blue, red=red)
        out = salvo_step(bs, ep, Admissibility.degenerate(), apply=False)
        # By construction, target i in Blue mirrors target i in Red.
        np.testing.assert_allclose(
            out.blue_raw_kernel, out.red_raw_kernel
        )
