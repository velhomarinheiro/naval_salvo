"""Tests for naval_salvo.solver."""

from __future__ import annotations

import numpy as np
import pytest

from naval_salvo import (
    Admissibility,
    BattleState,
    CampaignTrajectory,
    DirectionalParameters,
    Domain,
    EngagementParameters,
    Force,
    PairParameters,
    StrengthProportional,
    UnitType,
    run_campaign,
)


def _simple_engagement(A0=4.0, B0=4.0, alpha=2.0, beta=2.0,
                       z=1.0, y=1.0, w=2.0, x=2.0):
    """A 1-vs-1 surface engagement; same construction as the Hughes
    recovery harness but inlined here so the test is self-contained."""
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
    return bs, ep, Admissibility.degenerate()


# ---------------------------------------------------------------------------
# Basic shape & contract
# ---------------------------------------------------------------------------


class TestRunCampaignContract:
    def test_returns_trajectory(self):
        bs, ep, adm = _simple_engagement()
        traj = run_campaign(bs, ep, adm, n_salvos=3,
                            stop_on_combat_ineffective=False)
        assert isinstance(traj, CampaignTrajectory)

    def test_history_shapes(self):
        bs, ep, adm = _simple_engagement()
        traj = run_campaign(bs, ep, adm, n_salvos=5,
                            stop_on_combat_ineffective=False)
        # 5 salvos -> 6 strength snapshots (initial + 5 post-salvo).
        assert traj.blue_strength_history.shape == (6, 1)
        assert traj.red_strength_history.shape == (6, 1)
        assert traj.times.shape == (6,)
        assert len(traj.per_step_results) == 5

    def test_times_increment_by_step(self):
        bs, ep, adm = _simple_engagement()
        traj = run_campaign(bs, ep, adm, n_salvos=4, time_step=0.5,
                            stop_on_combat_ineffective=False)
        np.testing.assert_array_equal(traj.times, [0., 0.5, 1.0, 1.5, 2.0])

    def test_initial_history_row_matches_initial_strength(self):
        bs, ep, adm = _simple_engagement(A0=7, B0=3)
        traj = run_campaign(bs, ep, adm, n_salvos=2,
                            stop_on_combat_ineffective=False)
        np.testing.assert_array_equal(traj.blue_strength_history[0], [3.0])
        np.testing.assert_array_equal(traj.red_strength_history[0], [7.0])

    def test_n_salvos_zero_rejected(self):
        bs, ep, adm = _simple_engagement()
        with pytest.raises(ValueError):
            run_campaign(bs, ep, adm, n_salvos=0)


# ---------------------------------------------------------------------------
# Termination logic
# ---------------------------------------------------------------------------


class TestEarlyTermination:
    def test_terminates_when_one_side_dies(self):
        # Blue lethal enough to wipe Red in one salvo.
        bs, ep, adm = _simple_engagement(A0=2, B0=2,
                                         alpha=0, beta=10,
                                         z=0, y=0, w=1, x=1)
        traj = run_campaign(bs, ep, adm, n_salvos=10,
                            stop_on_combat_ineffective=True)
        assert traj.terminated_early is True
        # Should stop after exactly 1 salvo.
        assert traj.n_completed_salvos == 1

    def test_runs_to_n_salvos_when_termination_disabled(self):
        bs, ep, adm = _simple_engagement(A0=2, B0=2,
                                         alpha=0, beta=10,
                                         z=0, y=0, w=1, x=1)
        traj = run_campaign(bs, ep, adm, n_salvos=5,
                            stop_on_combat_ineffective=False)
        assert traj.terminated_early is False
        assert traj.n_completed_salvos == 5

    def test_no_termination_when_balanced(self):
        # Defensive parity: both sides survive indefinitely.
        bs, ep, adm = _simple_engagement(A0=4, B0=4,
                                         alpha=1, beta=1,
                                         z=10, y=10,
                                         w=2, x=2)
        traj = run_campaign(bs, ep, adm, n_salvos=5,
                            stop_on_combat_ineffective=True)
        assert traj.terminated_early is False
        assert traj.n_completed_salvos == 5


# ---------------------------------------------------------------------------
# Targeting refresh between salvos
# ---------------------------------------------------------------------------


class TestTargetingRefresh:
    def test_no_policy_keeps_initial_sigma(self):
        # Without a policy, σ values from the constructed EP should
        # remain unchanged across salvos.
        bs, ep, adm = _simple_engagement()
        sigma_pre = ep.blue_attacks_red.get("B", "A").sigma_offense
        run_campaign(bs, ep, adm, n_salvos=2,
                     stop_on_combat_ineffective=False)
        sigma_post = ep.blue_attacks_red.get("B", "A").sigma_offense
        assert sigma_pre == sigma_post

    def test_policy_recomputes_sigma_after_strength_change(self):
        # 1 attacker vs 2 defenders, Strength-proportional.  After the
        # first salvo strengths shift; we just check the campaign runs
        # cleanly.  (Full σ-update test is in test_coefficients.)
        blue = Force("Blue", [UnitType("B", Domain.SURFACE, 2.0, 1)])
        red = Force("Red", [
            UnitType("Big",   Domain.SURFACE, 1.0, 5),
            UnitType("Small", Domain.SURFACE, 1.0, 2),
        ])
        bar = DirectionalParameters.zeros(blue, red)
        bar.set("B", "Big",   PairParameters(p_offense=2.0))
        bar.set("B", "Small", PairParameters(p_offense=2.0))
        rab = DirectionalParameters.zeros(red, blue)
        rab.set("Big",   "B", PairParameters(p_offense=0.5))
        rab.set("Small", "B", PairParameters(p_offense=0.5))
        ep = EngagementParameters(blue=blue, red=red,
                                  blue_attacks_red=bar,
                                  red_attacks_blue=rab)
        bs = BattleState(blue=blue, red=red)
        traj = run_campaign(bs, ep, admissibility=Admissibility.degenerate(),
                            n_salvos=4,
                            targeting_policy=StrengthProportional(),
                            stop_on_combat_ineffective=True)
        # No assertion crashes; trajectory recorded.
        assert traj.n_completed_salvos >= 1


# ---------------------------------------------------------------------------
# Per-domain aggregation utility
# ---------------------------------------------------------------------------


class TestDomainAggregation:
    def test_aggregation_keys_and_shapes(self):
        # Mixed-domain Blue: 1 surface, 1 underwater.
        blue = Force("Blue", [
            UnitType("S1", Domain.SURFACE, 2.0, 3),
            UnitType("U1", Domain.UNDERWATER, 2.0, 1),
        ])
        red = Force("Red", [UnitType("R", Domain.SURFACE, 2.0, 2)])
        bar = DirectionalParameters.zeros(blue, red)
        bar.set("S1", "R", PairParameters(p_offense=1.0))
        bar.set("U1", "R", PairParameters(p_offense=1.0))
        rab = DirectionalParameters.zeros(red, blue)
        rab.set("R", "S1", PairParameters(p_offense=1.0))
        rab.set("R", "U1", PairParameters(p_offense=0.0))   # χ=0 in deg.
        ep = EngagementParameters(blue=blue, red=red,
                                  blue_attacks_red=bar,
                                  red_attacks_blue=rab)
        bs = BattleState(blue=blue, red=red)
        traj = run_campaign(bs, ep, Admissibility.degenerate(), n_salvos=2,
                            stop_on_combat_ineffective=False)
        per_dom = traj.total_strength_history_by_domain("blue")
        # All five domains must appear as keys.
        assert set(d.value for d in per_dom.keys()) == {"S", "U", "A", "C", "X"}
        # Shapes match (n_salvos + 1,)
        for d, arr in per_dom.items():
            assert arr.shape == (3,)
        # Initial: SURFACE = 3 (S1), UNDERWATER = 1 (U1), others = 0.
        assert per_dom[Domain.SURFACE][0] == 3.0
        assert per_dom[Domain.UNDERWATER][0] == 1.0
        assert per_dom[Domain.AIR][0] == 0.0

    def test_aggregation_unknown_side_raises(self):
        bs, ep, adm = _simple_engagement()
        traj = run_campaign(bs, ep, adm, n_salvos=1,
                            stop_on_combat_ineffective=False)
        with pytest.raises(ValueError):
            traj.total_strength_history_by_domain("green")


# ---------------------------------------------------------------------------
# Multi-salvo Hughes recovery (regression against the analytical solver)
# ---------------------------------------------------------------------------


class TestHughesViaCampaign:
    """run_campaign must agree with hughes_analytical for the homogeneous
    1v1 surface engagement."""

    def test_matches_hughes_analytical(self):
        from naval_salvo import (
            HughesScenario, build_hughes_homogeneous_engagement,
            hughes_analytical,
        )
        scn = HughesScenario(A0=6, B0=4, alpha=1.0, beta=4.0,
                             z=1.0, y=1.0, w=2.0, x=2.0, n_salvos=4)
        bs, ep, adm = build_hughes_homogeneous_engagement(scn)
        A_an, B_an = hughes_analytical(scn)

        traj = run_campaign(bs, ep, adm, n_salvos=scn.n_salvos,
                            stop_on_combat_ineffective=False)
        np.testing.assert_allclose(
            traj.red_strength_history[:, 0], A_an, atol=1e-12
        )
        np.testing.assert_allclose(
            traj.blue_strength_history[:, 0], B_an, atol=1e-12
        )
