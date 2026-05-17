"""
Tests for naval_salvo.validation.jph_coronel.

The minute-1 worked example of Johns, Pilnick & Hughes (2001) NPS-IJWA-01-010
p.22 is reproduced bit-for-bit (modulo machine epsilon).  This locks in
the *intra-domain heterogeneous* path of the engine: multiple distinct
unit types within the same domain (surface), each with its own staying
power, combat power, and effectiveness, with explicit time-varying
targeting fractions.
"""

from __future__ import annotations

import numpy as np
import pytest

from naval_salvo import salvo_step
from naval_salvo.validation import (
    BRITISH_GROUPS,
    GERMAN_GROUPS,
    build_coronel_engagement,
    coronel_forces,
    coronel_minute_one_targeting,
    jph_minute_one_delta_good_hope,
)


# ---------------------------------------------------------------------------
# Static data sanity
# ---------------------------------------------------------------------------


class TestCoronelData:
    def test_british_groups_count(self):
        # Two unit-type groups: GH+Mon, Glasgow.
        assert len(BRITISH_GROUPS) == 2

    def test_german_groups_count(self):
        # Two: Sch+Gn, Lp+Dr.
        assert len(GERMAN_GROUPS) == 2

    def test_per_ship_combat_power_mapping(self):
        # JPH p.22 reports β_{Sch+Gn} = 2.16 (group total 4.32, n=2).
        sch_gn = GERMAN_GROUPS[0]
        assert sch_gn.per_ship_power == pytest.approx(4.32 / 2)
        # Glasgow: 0.42 / 1 = 0.42.
        glasgow = BRITISH_GROUPS[1]
        assert glasgow.per_ship_power == pytest.approx(0.42)

    def test_staying_power_per_ship(self):
        # Beall Table 1: GH+Mon group ς = 3.21 → per ship = 1.605.
        gh_mon = BRITISH_GROUPS[0]
        assert gh_mon.staying_power_per_ship == pytest.approx(3.21 / 2)
        # Sch+Gn: 3.330 / 2 = 1.665.
        sch_gn = GERMAN_GROUPS[0]
        assert sch_gn.staying_power_per_ship == pytest.approx(3.330 / 2)


class TestCoronelForces:
    def test_force_structure(self):
        british, german = coronel_forces()
        assert british.label == "British"
        assert german.label == "German"
        # 2 distinct unit types per side -- this is the *heterogeneous
        # intra-domain* path that Step 3 exercises.
        assert british.n_unit_types == 2
        assert german.n_unit_types == 2

    def test_initial_strengths(self):
        british, german = coronel_forces()
        # GH+Mon = 2 ships, Glasgow = 1 ship.
        np.testing.assert_array_equal(
            british.strength_vector(), [2.0, 1.0]
        )
        # Sch+Gn = 2 ships, Lp+Dr = 2 ships.
        np.testing.assert_array_equal(
            german.strength_vector(), [2.0, 2.0]
        )


# ---------------------------------------------------------------------------
# Targeting matrix
# ---------------------------------------------------------------------------


class TestMinuteOneTargeting:
    def test_german_targeting_matrix(self):
        targ = coronel_minute_one_targeting()
        # Sch+Gn engages GH+Mon (group 1) only, full share.
        assert targ.german_attacks_british[0, 0] == 1.0
        assert targ.german_attacks_british[0, 1] == 0.0
        # Lp+Dr engages Glasgow (group 2) only.
        assert targ.german_attacks_british[1, 0] == 0.0
        assert targ.german_attacks_british[1, 1] == 1.0

    def test_british_silent_in_minute_one(self):
        targ = coronel_minute_one_targeting()
        np.testing.assert_array_equal(
            targ.british_attacks_german, np.zeros((2, 2))
        )


# ---------------------------------------------------------------------------
# THE critical regression: ΔA_1 (Good Hope) at minute 1
# ---------------------------------------------------------------------------


class TestJPHMinute1Reproduction:
    """
    Locks the engine output against JPH p.22 worked example to machine
    precision.  This is the *contract* of Step 3: any change to the
    targeting/coefficients/dynamics layer that breaks this test means
    the model has drifted from JPH 2001.
    """

    def test_analytical_helper_value(self):
        # JPH report 0.037 (3 d.p.); the precise analytical value is
        # 2 · (0.028 · 2.16 · 0.5) / 1.605 = 0.037682...
        v = jph_minute_one_delta_good_hope()
        assert v == pytest.approx(0.0376822429906542, abs=1e-15)
        # Sanity: rounded to 3 d.p. matches the paper.
        assert round(v, 3) == 0.038

    def test_engine_per_ship_attrition_on_good_hope_matches_jph(self):
        bs, ep, adm = build_coronel_engagement()
        out = salvo_step(bs, ep, adm, apply=False)

        # The engine's blue_losses[0] is the *group-total* loss for
        # GH+Mon (2 ships).  JPH's worked example reports per-ship
        # attrition ΔA_1 ≈ 0.037.
        per_ship_loss = out.blue_losses[0] / BRITISH_GROUPS[0].n_ships
        assert per_ship_loss == pytest.approx(
            jph_minute_one_delta_good_hope(), abs=1e-12
        )

    def test_german_takes_zero_damage_in_minute_one(self):
        # British silent → no damage to either German group.
        bs, ep, adm = build_coronel_engagement()
        out = salvo_step(bs, ep, adm, apply=False)
        np.testing.assert_array_equal(out.red_raw_kernel, [0.0, 0.0])
        np.testing.assert_array_equal(out.red_losses, [0.0, 0.0])

    def test_glasgow_takes_damage_from_lp_dr(self):
        """
        Lp+Dr engage Glasgow with ε = 0.012, β = 2.165, ψ = 1.0.
        Glasgow's per-ship attrition in minute 1 should be:
            ΔA_2 = 2 · (0.012 · 2.165 · 1.0 · 1.0) / 1.23
                 = 2 · 0.02598 / 1.23
                 ≈ 0.04224

        Engine returns *group-total* loss; Glasgow has only 1 ship so
        the two coincide.
        """
        bs, ep, adm = build_coronel_engagement()
        out = salvo_step(bs, ep, adm, apply=False)
        expected = 2 * (0.012 * 2.165 * 1.0) / 1.23
        # Glasgow is index 1 in the British force (after GH+Mon).
        assert out.blue_losses[1] == pytest.approx(expected, abs=1e-12)


# ---------------------------------------------------------------------------
# Hand-computed kernel arithmetic (broader internal regression)
# ---------------------------------------------------------------------------


class TestKernelArithmetic:
    """
    For minute 1 of Coronel:
        Group 1 (GH+Mon, A=2 ships, ς=1.605, η=0.028 [outgoing], silent)
        Group 2 (Glasgow, A=1, ς=1.23, η=0.028, silent in min 1)
        Group 3 (Sch+Gn,  B=2, ς=1.665, η=0.028, β=2.16, ψ→GH=1.0, ψ→Glas=0)
        Group 4 (Lp+Dr,   B=2, ς=1.115, η=0.012, β=2.165, ψ→GH=0, ψ→Glas=1.0)

    Kernel on GH+Mon = ε · β · ψ · B = (Sch+Gn term) + (Lp+Dr term)
                     = 0.028 · 2.16 · 1.0 · 2 + 0.012 · 2.165 · 0.0 · 2
                     = 0.12096

    Group-total loss = 0.12096 / 1.605 = 0.0753645   ✓
    Per-ship loss    = 0.0376822                     (matches JPH)
    """

    def test_full_minute1_blue_kernel(self):
        bs, ep, adm = build_coronel_engagement()
        out = salvo_step(bs, ep, adm, apply=False)
        # GH+Mon kernel = 0.12096
        assert out.blue_raw_kernel[0] == pytest.approx(0.12096, abs=1e-12)
        # Glasgow kernel = 0.012 · 2.165 · 1.0 · 2 = 0.05196
        assert out.blue_raw_kernel[1] == pytest.approx(0.05196, abs=1e-12)
