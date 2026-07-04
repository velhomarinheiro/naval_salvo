"""Tests for the naval_salvo.stochastic module.

Coverage:
    - Exact reproduction of Armstrong's deterministic example (2014, p.1594)
    - Adherence to the published stochastic scenario (A=6, p=0.67, p.1597)
    - Ordering property: 1st >= simultaneous >= 2nd (on average)
    - Submarine cyber immunity (χ)
    - Loss truncation within [0, stock]
    - Seed reproducibility
    - Verdict over naval forces only
"""

import numpy as np
import pytest

from naval_salvo.stochastic import (
    DOMAINS,
    EngagementOrder,
    HomogeneousForce,
    MultiDomainForce,
    UnitGroup,
    phi,
    run_homogeneous_battle,
    run_multidomain_battle,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def deterministic_pair():
    """B=4, A=5, alpha=beta=4, y=z=2, u=v=1/3 (Armstrong 2014, p.1594)."""
    blue = HomogeneousForce(4, 4, 1.0, 2, 1.0, 1 / 3, 0.0)
    red = HomogeneousForce(5, 4, 1.0, 2, 1.0, 1 / 3, 0.0)
    return blue, red


def stochastic_pair():
    """B=4, A=6, n=4/2, p=0.67, mu=0.33, sigma=0.11 (Armstrong 2014, p.1597)."""
    blue = HomogeneousForce(4, 4, 0.67, 2, 0.67, 0.33, 0.11)
    red = HomogeneousForce(6, 4, 0.67, 2, 0.67, 0.33, 0.11)
    return blue, red


def md_forces(cyber_blue=0.0, cyber_red=0.0):
    blue = MultiDomainForce("Blue", [
        UnitGroup("frigate", "s", 4, 4, 0.67, 2, 0.67, 0.33, 0.11),
        UnitGroup("aircraft", "a", 2, 2, 0.75, 0, 0.0, 0.5, 0.15),
    ], cyber_offense=cyber_blue)
    red = MultiDomainForce("Red", [
        UnitGroup("escort", "s", 5, 4, 0.60, 2, 0.60, 0.33, 0.11),
        UnitGroup("submarine", "u", 1, 4, 0.80, 0, 0.0, 0.5, 0.15),
    ], cyber_offense=cyber_red)
    return blue, red


# ---------------------------------------------------------------------------
# Armstrong (2014) validation -- deterministic case
# ---------------------------------------------------------------------------
class TestDeterministicArmstrong2014:
    def test_blue_first_matches_published(self):
        blue, red = deterministic_pair()
        bl, rl = run_homogeneous_battle(blue, red, EngagementOrder.BLUE_FIRST,
                                        n_sim=50)
        assert np.allclose(rl, 2.0, atol=1e-9)
        assert np.allclose(bl, 4.0 / 3.0, atol=1e-9)

    def test_simultaneous_matches_published(self):
        blue, red = deterministic_pair()
        bl, rl = run_homogeneous_battle(blue, red,
                                        EngagementOrder.SIMULTANEOUS, n_sim=50)
        assert np.allclose(bl, 4.0, atol=1e-9)   # Blue annihilated
        assert np.allclose(rl, 2.0, atol=1e-9)

    def test_red_first_blue_annihilated_no_return_fire_effect(self):
        blue, red = deterministic_pair()
        bl, rl = run_homogeneous_battle(blue, red, EngagementOrder.RED_FIRST,
                                        n_sim=50)
        assert np.allclose(bl, 4.0, atol=1e-9)
        assert np.allclose(rl, 0.0, atol=1e-9)


# ---------------------------------------------------------------------------
# Armstrong (2014) validation -- published stochastic scenario
# ---------------------------------------------------------------------------
class TestStochasticArmstrong2014:
    """Published simulation: mean 2.637, sd 0.989, P[0]=0.8%, P[all]=12.5%."""

    @pytest.fixture(scope="class")
    def losses(self):
        blue, red = stochastic_pair()
        bl, _ = run_homogeneous_battle(blue, red, EngagementOrder.BLUE_FIRST,
                                       n_sim=20_000, seed=123)
        return bl

    def test_mean_close_to_published(self, losses):
        assert losses.mean() == pytest.approx(2.637, abs=0.05)

    def test_std_close_to_published(self, losses):
        assert losses.std(ddof=1) == pytest.approx(0.989, abs=0.05)

    def test_p_zero_loss_close_to_published(self, losses):
        p0 = float(np.mean(losses < 0.165))
        assert p0 == pytest.approx(0.008, abs=0.01)

    def test_p_total_loss_close_to_published(self, losses):
        pall = float(np.mean(losses >= 4 - 0.165))
        assert pall == pytest.approx(0.125, abs=0.05)


# ---------------------------------------------------------------------------
# Theoretical properties
# ---------------------------------------------------------------------------
class TestOrderingProperty:
    def test_first_leq_simultaneous_leq_second_mean_losses(self):
        blue, red = stochastic_pair()
        means = {}
        for order in EngagementOrder:
            bl, _ = run_homogeneous_battle(blue, red, order,
                                           n_sim=20_000, seed=99)
            means[order] = bl.mean()
        assert means[EngagementOrder.BLUE_FIRST] <= \
            means[EngagementOrder.SIMULTANEOUS] + 0.02
        assert means[EngagementOrder.SIMULTANEOUS] <= \
            means[EngagementOrder.RED_FIRST] + 0.02

    def test_multidomain_initiative_increases_win_probability(self):
        blue, red = md_forces()
        p = {}
        for order in EngagementOrder:
            r = run_multidomain_battle(blue, red, order, n_salvos=3,
                                       n_sim=4_000, seed=7)
            p[order] = r.p_blue_win
        assert p[EngagementOrder.BLUE_FIRST] > p[EngagementOrder.SIMULTANEOUS]
        assert p[EngagementOrder.SIMULTANEOUS] > p[EngagementOrder.RED_FIRST]


# ---------------------------------------------------------------------------
# Cyber modulation and submarine immunity
# ---------------------------------------------------------------------------
class TestCyberModulation:
    def test_phi_is_one_for_nonpositive_intensity(self):
        assert phi(0.0) == 1.0
        assert phi(-2.5) == 1.0

    def test_phi_decreasing_in_intensity(self):
        assert phi(0.5) > phi(1.0) > phi(2.0)

    def test_submarine_immune_to_cyber(self):
        """Under strong Red cyber, only the submarine keeps its effectiveness."""
        from naval_salvo.stochastic import _p_off_effective, _p_def_effective
        sub = UnitGroup("sub", "u", 1, 4, 0.8, 2, 0.5, 0.5, 0.1)
        surf = UnitGroup("nav", "s", 1, 4, 0.8, 2, 0.5, 0.5, 0.1)
        phi_val = 0.3
        assert _p_off_effective(sub, phi_val) == pytest.approx(0.8)
        assert _p_def_effective(sub, phi_val) == pytest.approx(0.5)
        assert _p_off_effective(surf, phi_val) == pytest.approx(0.8 * 0.09)
        assert _p_def_effective(surf, phi_val) == pytest.approx(0.5 * 0.3)

    def test_cyber_dominance_can_invert_outcome(self):
        """Consistent with the deterministic model's sensitivity finding."""
        blue, red = md_forces(cyber_blue=3.0, cyber_red=0.0)
        r = run_multidomain_battle(blue, red, EngagementOrder.SIMULTANEOUS,
                                   n_salvos=3, n_sim=4_000, seed=7,
                                   k_cyber=0.5)
        assert r.p_blue_win > 0.9


# ---------------------------------------------------------------------------
# Structural invariants
# ---------------------------------------------------------------------------
class TestInvariants:
    def test_losses_bounded_by_initial_stock(self):
        blue, red = stochastic_pair()
        for order in EngagementOrder:
            bl, rl = run_homogeneous_battle(blue, red, order,
                                            n_sim=2_000, seed=1)
            assert (bl >= -1e-9).all() and (bl <= blue.units + 1e-9).all()
            assert (rl >= -1e-9).all() and (rl <= red.units + 1e-9).all()

    def test_seed_reproducibility(self):
        blue, red = md_forces()
        r1 = run_multidomain_battle(blue, red, EngagementOrder.SIMULTANEOUS,
                                    n_sim=500, seed=42)
        r2 = run_multidomain_battle(blue, red, EngagementOrder.SIMULTANEOUS,
                                    n_sim=500, seed=42)
        assert np.array_equal(r1.blue_naval_final, r2.blue_naval_final)
        assert np.array_equal(r1.red_naval_final, r2.red_naval_final)

    def test_naval_verdict_ignores_air_units(self):
        """Aircraft do not count toward the verdict (naval force = s + u)."""
        blue, _ = md_forces()
        stocks = blue.initial_stocks()
        assert blue.naval_strength(stocks) == pytest.approx(4.0)  # frigates only

    def test_invalid_domain_raises(self):
        with pytest.raises(ValueError):
            UnitGroup("x", "z", 1, 1, 0.5, 0, 0.0, 0.5, 0.1)

    def test_probabilities_sum_to_one(self):
        blue, red = md_forces()
        r = run_multidomain_battle(blue, red, EngagementOrder.SIMULTANEOUS,
                                   n_sim=1_000, seed=3)
        assert r.p_blue_win + r.p_red_win + r.p_draw == pytest.approx(1.0)
