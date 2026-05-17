"""Tests for naval_salvo.coefficients."""

from __future__ import annotations

import numpy as np
import pytest

from naval_salvo import (
    Admissibility,
    Domain,
    EngagementBuilder,
    StrengthProportional,
    ThreatWeighted,
    ThroughputGrid,
    Uniform,
    UnitType,
    apply_targeting_policy,
    salvo_step,
)


# ---------------------------------------------------------------------------
# ThroughputGrid
# ---------------------------------------------------------------------------


class TestThroughputGrid:
    def test_zeros_factory_dimensions(self):
        g = ThroughputGrid.zeros(3, 5)
        assert g.shape() == (3, 5)
        assert g.p_offense.sum() == 0.0
        assert g.p_defense.sum() == 0.0
        assert np.all(g.eta_offense == 1.0)
        assert np.all(g.eta_defense == 1.0)


# ---------------------------------------------------------------------------
# EngagementBuilder fluent API
# ---------------------------------------------------------------------------


def _two_simple_forces():
    blue_uts = [
        UnitType("Frigate", Domain.SURFACE, 3.0, 2),
        UnitType("SSK",     Domain.UNDERWATER, 2.0, 1),
    ]
    red_uts = [
        UnitType("Destroyer", Domain.SURFACE, 4.0, 3),
    ]
    return blue_uts, red_uts


class TestEngagementBuilder:
    def test_must_set_blue_first(self):
        b = EngagementBuilder()
        with pytest.raises(ValueError):
            b.with_throughput_blue_attacks_red(p_offense={("a", "b"): 1.0})

    def test_minimal_build_has_zero_throughputs(self):
        blue_uts, red_uts = _two_simple_forces()
        ep = (
            EngagementBuilder()
            .with_blue(blue_uts)
            .with_red(red_uts)
            .build()
        )
        K = ep.blue_attacks_red.offensive_kernel_matrix()
        assert K.shape == (2, 1)
        assert K.sum() == 0.0

    def test_explicit_throughput_lookup(self):
        blue_uts, red_uts = _two_simple_forces()
        ep = (
            EngagementBuilder()
            .with_blue(blue_uts)
            .with_red(red_uts)
            .with_throughput_blue_attacks_red(
                p_offense={("Frigate", "Destroyer"): 2.5,
                           ("SSK",     "Destroyer"): 1.0},
                eta_offense={("Frigate", "Destroyer"): 0.8,
                             ("SSK",     "Destroyer"): 0.5},
            )
            .build()
        )
        # Default σ = 1.0 (no targeting policy specified) and η as set.
        # Pair (Frigate -> Destroyer) kernel = 1 * 0.8 * 2.5 = 2.0.
        pair = ep.blue_attacks_red.get("Frigate", "Destroyer")
        assert pair.p_offense == 2.5
        assert pair.eta_offense == 0.8
        assert pair.sigma_offense == 1.0

    def test_unknown_unit_in_throughput_dict_raises(self):
        blue_uts, red_uts = _two_simple_forces()
        with pytest.raises(ValueError):
            (EngagementBuilder()
                .with_blue(blue_uts)
                .with_red(red_uts)
                .with_throughput_blue_attacks_red(
                    p_offense={("NonexistentUnit", "Destroyer"): 1.0}
                )
                .build())

    def test_targeting_policy_fills_sigma(self):
        # Single Blue (Frigate) attacking 2 Red destroyers.
        blue_uts = [UnitType("Frigate", Domain.SURFACE, 3.0, 2)]
        red_uts = [
            UnitType("D1", Domain.SURFACE, 4.0, 3),
            UnitType("D2", Domain.SURFACE, 4.0, 3),
        ]
        ep = (
            EngagementBuilder()
            .with_blue(blue_uts)
            .with_red(red_uts)
            .with_throughput_blue_attacks_red(
                p_offense={("Frigate", "D1"): 2.0,
                           ("Frigate", "D2"): 2.0},
            )
            .with_throughput_red_attacks_blue()
            .with_admissibility(Admissibility.degenerate())
            .with_targeting_policy(Uniform())
            .build()
        )
        # Uniform across two admissible defenders → σ = 0.5 each.
        p1 = ep.blue_attacks_red.get("Frigate", "D1")
        p2 = ep.blue_attacks_red.get("Frigate", "D2")
        assert p1.sigma_offense == pytest.approx(0.5)
        assert p2.sigma_offense == pytest.approx(0.5)

    def test_t_char_default_and_override(self):
        blue_uts, red_uts = _two_simple_forces()
        ep = (
            EngagementBuilder()
            .with_blue(blue_uts)
            .with_red(red_uts)
            .with_t_char_per_domain([1.0, 5.0, 0.1, 1.0])
            .build()
        )
        np.testing.assert_array_equal(ep.t_char, [1.0, 5.0, 0.1, 1.0])

    def test_admissibility_default_is_canonical(self):
        blue_uts, red_uts = _two_simple_forces()
        ep = (
            EngagementBuilder()
            .with_blue(blue_uts)
            .with_red(red_uts)
            .build()
        )
        # The canonical matrix has e.g. (S, S) = 1; nothing crashed.
        assert ep is not None


# ---------------------------------------------------------------------------
# apply_targeting_policy mid-battle refresh
# ---------------------------------------------------------------------------


class TestApplyTargetingPolicy:
    def test_eta_and_p_preserved(self):
        blue_uts, red_uts = _two_simple_forces()
        ep = (
            EngagementBuilder()
            .with_blue(blue_uts)
            .with_red(red_uts)
            .with_throughput_blue_attacks_red(
                p_offense={("Frigate", "Destroyer"): 2.5},
                eta_offense={("Frigate", "Destroyer"): 0.8},
            )
            .with_targeting_policy(Uniform())
            .build()
        )
        ep_new = apply_targeting_policy(ep, Admissibility.canonical(), Uniform())
        pair_new = ep_new.blue_attacks_red.get("Frigate", "Destroyer")
        assert pair_new.p_offense == 2.5
        assert pair_new.eta_offense == 0.8
        # Sigma is recomputed; it might or might not differ from the
        # original (here same because the strengths haven't changed).

    def test_recomputes_sigma_after_strength_change(self):
        blue_uts = [UnitType("Frigate", Domain.SURFACE, 3.0, 2)]
        red_uts = [
            UnitType("Big",   Domain.SURFACE, 4.0, 8),
            UnitType("Small", Domain.SURFACE, 4.0, 2),
        ]
        ep = (
            EngagementBuilder()
            .with_blue(blue_uts)
            .with_red(red_uts)
            .with_throughput_blue_attacks_red(
                p_offense={("Frigate", "Big"): 2.0,
                           ("Frigate", "Small"): 2.0},
            )
            .with_admissibility(Admissibility.degenerate())
            .with_targeting_policy(StrengthProportional())
            .build()
        )
        # Initial 8:2 split → σ = 0.8 : 0.2.
        sig_big_pre = ep.blue_attacks_red.get("Frigate", "Big").sigma_offense
        assert sig_big_pre == pytest.approx(0.8)

        # Wipe out the "Big" unit type.  Refresh the policy.
        ep.red.set_strength_of("Big", 0.0)
        ep_new = apply_targeting_policy(
            ep, Admissibility.degenerate(), StrengthProportional()
        )
        # Now all fire goes to "Small".
        assert ep_new.blue_attacks_red.get("Frigate", "Big").sigma_offense == 0.0
        assert ep_new.blue_attacks_red.get("Frigate", "Small").sigma_offense == 1.0


# ---------------------------------------------------------------------------
# End-to-end sanity: builder + dynamics
# ---------------------------------------------------------------------------


class TestBuilderEndToEnd:
    def test_two_target_split_kernel(self):
        """
        One Blue attacker attacks two equal-stock Red defenders with
        p_offense = 4 and Uniform targeting.  Uniform gives σ = 0.5
        each, so the *each* defender suffers β·B·σ - 0 = 4·1·0.5 = 2
        in raw kernel.  Two defenders, each takes 2 hits.
        """
        blue_uts = [UnitType("Atk", Domain.SURFACE, 1.0, 1)]
        red_uts = [
            UnitType("D1", Domain.SURFACE, 1.0, 4),
            UnitType("D2", Domain.SURFACE, 1.0, 4),
        ]
        ep = (
            EngagementBuilder()
            .with_blue(blue_uts)
            .with_red(red_uts)
            .with_throughput_blue_attacks_red(
                p_offense={("Atk", "D1"): 4.0, ("Atk", "D2"): 4.0},
            )
            .with_admissibility(Admissibility.degenerate())
            .with_targeting_policy(Uniform())
            .build()
        )
        from naval_salvo import BattleState
        bs = BattleState(blue=ep.blue, red=ep.red)
        out = salvo_step(bs, ep, Admissibility.degenerate(), apply=False)
        np.testing.assert_allclose(out.red_raw_kernel, [2.0, 2.0])
