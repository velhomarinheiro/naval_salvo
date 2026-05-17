"""Tests for naval_salvo.state."""

from __future__ import annotations

import numpy as np
import pytest

from naval_salvo.domains import DOMAIN_ORDER, Domain
from naval_salvo.state import (
    BattleState,
    Force,
    UnitType,
    UnitTypeState,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_blue() -> Force:
    return Force(
        label="Blue",
        unit_types=[
            UnitType("Frigate", Domain.SURFACE, staying_power=3.0,
                     initial_strength=4),
            UnitType("Corvette", Domain.SURFACE, staying_power=2.0,
                     initial_strength=2,
                     subtype="USV-mothership"),
            UnitType("SSK", Domain.UNDERWATER, staying_power=2.0,
                     initial_strength=1),
        ],
    )


def _make_red() -> Force:
    return Force(
        label="Red",
        unit_types=[
            UnitType("Destroyer", Domain.SURFACE, staying_power=4.0,
                     initial_strength=3),
            UnitType("Coastal-Bty", Domain.COASTAL, staying_power=1.0,
                     initial_strength=2),
            UnitType("Cyber-C2", Domain.CYBER, staying_power=1.0,
                     initial_strength=2,
                     subtype="X_C2"),
        ],
    )


# ---------------------------------------------------------------------------
# UnitType
# ---------------------------------------------------------------------------


class TestUnitType:
    def test_construction_minimal(self):
        ut = UnitType("Frigate", Domain.SURFACE, 3.0, 4)
        assert ut.name == "Frigate"
        assert ut.domain is Domain.SURFACE
        assert ut.staying_power == 3.0
        assert ut.initial_strength == 4
        assert ut.subtype is None

    def test_subtype_optional(self):
        ut = UnitType("Heli-MotherShip", Domain.SURFACE, 3.0, 1.0,
                      subtype="helicopter-carrier")
        assert ut.subtype == "helicopter-carrier"

    def test_string_domain_coerced(self):
        ut = UnitType("X1", "AIR", 1.0, 1.0)
        assert ut.domain is Domain.AIR

    def test_invalid_staying_power(self):
        with pytest.raises(ValueError):
            UnitType("X", Domain.SURFACE, 0.0, 1.0)
        with pytest.raises(ValueError):
            UnitType("X", Domain.SURFACE, -1.0, 1.0)
        with pytest.raises(ValueError):
            UnitType("X", Domain.SURFACE, np.inf, 1.0)

    def test_invalid_initial_strength(self):
        with pytest.raises(ValueError):
            UnitType("X", Domain.SURFACE, 1.0, -0.1)
        with pytest.raises(ValueError):
            UnitType("X", Domain.SURFACE, 1.0, np.nan)

    def test_blank_name(self):
        with pytest.raises(ValueError):
            UnitType("", Domain.SURFACE, 1.0, 1.0)


# ---------------------------------------------------------------------------
# UnitTypeState
# ---------------------------------------------------------------------------


class TestUnitTypeState:
    def test_fractional_strength(self):
        ut = UnitType("X", Domain.SURFACE, 1.0, 4.0)
        s = UnitTypeState(ut, 2.0)
        assert s.fractional_strength == 0.5

    def test_fractional_strength_zero_initial(self):
        ut = UnitType("X", Domain.SURFACE, 1.0, 0.0)
        s = UnitTypeState(ut, 0.0)
        assert s.fractional_strength == 0.0

    def test_invalid_current_strength(self):
        ut = UnitType("X", Domain.SURFACE, 1.0, 1.0)
        with pytest.raises(ValueError):
            UnitTypeState(ut, -0.5)
        with pytest.raises(ValueError):
            UnitTypeState(ut, np.inf)


# ---------------------------------------------------------------------------
# Force
# ---------------------------------------------------------------------------


class TestForce:
    def test_construction(self):
        blue = _make_blue()
        assert blue.label == "Blue"
        assert blue.n_unit_types == 3
        assert len(blue.states) == 3

    def test_initial_state_matches_initial_strength(self):
        blue = _make_blue()
        for ut, st in zip(blue.unit_types, blue.states):
            assert st.current_strength == ut.initial_strength

    def test_blank_label(self):
        with pytest.raises(ValueError):
            Force(label="", unit_types=[])

    def test_duplicate_names_rejected(self):
        with pytest.raises(ValueError):
            Force(
                label="X",
                unit_types=[
                    UnitType("Same", Domain.SURFACE, 1.0, 1.0),
                    UnitType("Same", Domain.AIR, 1.0, 1.0),
                ],
            )

    def test_strength_vector(self):
        blue = _make_blue()
        v = blue.strength_vector()
        assert v.shape == (3,)
        assert v.dtype == np.float64
        np.testing.assert_array_equal(v, [4.0, 2.0, 1.0])

    def test_set_strength_vector(self):
        blue = _make_blue()
        blue.set_strength_vector(np.array([3.0, 1.0, 0.5]))
        np.testing.assert_array_equal(
            blue.strength_vector(), [3.0, 1.0, 0.5]
        )

    def test_set_strength_vector_clips_negatives(self):
        # Decision 1.4 §2.3.f: no amplification, strengths cannot go negative.
        blue = _make_blue()
        blue.set_strength_vector(np.array([-1.0, 2.0, 1.0]))
        np.testing.assert_array_equal(
            blue.strength_vector(), [0.0, 2.0, 1.0]
        )

    def test_set_strength_vector_wrong_shape(self):
        blue = _make_blue()
        with pytest.raises(ValueError):
            blue.set_strength_vector(np.array([1.0, 2.0]))

    def test_set_strength_vector_non_finite(self):
        blue = _make_blue()
        with pytest.raises(ValueError):
            blue.set_strength_vector(np.array([1.0, np.nan, 1.0]))

    def test_strength_of_and_set_strength_of(self):
        blue = _make_blue()
        assert blue.strength_of("Frigate") == 4.0
        blue.set_strength_of("Frigate", 2.5)
        assert blue.strength_of("Frigate") == 2.5

    def test_set_strength_of_clips_negative(self):
        blue = _make_blue()
        blue.set_strength_of("Frigate", -2.0)
        assert blue.strength_of("Frigate") == 0.0

    def test_indices_in(self):
        blue = _make_blue()
        # Two surface unit types (Frigate, Corvette), one underwater.
        assert blue.indices_in(Domain.SURFACE) == [0, 1]
        assert blue.indices_in(Domain.UNDERWATER) == [2]
        assert blue.indices_in(Domain.AIR) == []

    def test_unit_types_in(self):
        blue = _make_blue()
        names = [ut.name for ut in blue.unit_types_in(Domain.SURFACE)]
        assert names == ["Frigate", "Corvette"]

    def test_total_strength_by_domain(self):
        blue = _make_blue()
        d = blue.total_strength_by_domain()
        # All five domains must be keys.
        assert set(d.keys()) == set(DOMAIN_ORDER)
        assert d[Domain.SURFACE] == 4.0 + 2.0
        assert d[Domain.UNDERWATER] == 1.0
        assert d[Domain.AIR] == 0.0
        assert d[Domain.CYBER] == 0.0

    def test_add_unit_type(self):
        blue = _make_blue()
        blue.add_unit_type(
            UnitType("MPA", Domain.AIR, staying_power=1.0, initial_strength=2)
        )
        assert blue.n_unit_types == 4
        assert blue.strength_of("MPA") == 2.0
        np.testing.assert_array_equal(
            blue.strength_vector(), [4.0, 2.0, 1.0, 2.0]
        )

    def test_add_duplicate_name(self):
        blue = _make_blue()
        with pytest.raises(ValueError):
            blue.add_unit_type(
                UnitType("Frigate", Domain.SURFACE, 1.0, 1.0)
            )

    def test_iter(self):
        blue = _make_blue()
        items = list(iter(blue))
        assert len(items) == 3
        for ut, st in items:
            assert isinstance(ut, UnitType)
            assert isinstance(st, UnitTypeState)

    def test_is_combat_ineffective(self):
        blue = _make_blue()
        assert blue.is_combat_ineffective() is False
        blue.set_strength_vector(np.zeros(3))
        assert blue.is_combat_ineffective() is True

    def test_initial_strength_vector(self):
        blue = _make_blue()
        np.testing.assert_array_equal(
            blue.initial_strength_vector(), [4.0, 2.0, 1.0]
        )
        # Stays the same even after strengths change.
        blue.set_strength_vector(np.zeros(3))
        np.testing.assert_array_equal(
            blue.initial_strength_vector(), [4.0, 2.0, 1.0]
        )

    def test_staying_power_vector(self):
        blue = _make_blue()
        np.testing.assert_array_equal(
            blue.staying_power_vector(), [3.0, 2.0, 2.0]
        )


# ---------------------------------------------------------------------------
# BattleState
# ---------------------------------------------------------------------------


class TestBattleState:
    def test_construction(self):
        blue, red = _make_blue(), _make_red()
        bs = BattleState(blue=blue, red=red)
        assert bs.time == 0.0
        assert bs.salvo_times == []

    def test_force_lookup(self):
        blue, red = _make_blue(), _make_red()
        bs = BattleState(blue=blue, red=red)
        assert bs.force("blue") is blue
        assert bs.force("RED") is red
        assert bs.force("Blue") is blue

    def test_opposing_force(self):
        blue, red = _make_blue(), _make_red()
        bs = BattleState(blue=blue, red=red)
        assert bs.opposing_force("blue") is red
        assert bs.opposing_force("red") is blue

    def test_unknown_side_raises(self):
        bs = BattleState(blue=_make_blue(), red=_make_red())
        with pytest.raises(KeyError):
            bs.force("green")

    def test_record_salvo(self):
        bs = BattleState(blue=_make_blue(), red=_make_red())
        bs.record_salvo(0.5)
        bs.record_salvo(1.0)
        assert bs.salvo_times == [0.5, 1.0]

    def test_is_terminated_kinetic_wipeout(self):
        blue, red = _make_blue(), _make_red()
        bs = BattleState(blue=blue, red=red)
        assert bs.is_terminated() is False
        # Zero out all kinetic strengths in Blue (Frigate, Corvette, SSK).
        blue.set_strength_vector(np.zeros(3))
        assert bs.is_terminated() is True

    def test_is_terminated_cyber_only_does_not_count(self):
        # If only the cyber stock of a side is depleted but kinetic stocks
        # remain, the engagement is *not* over (kinetic combat continues).
        blue, red = _make_blue(), _make_red()
        bs = BattleState(blue=blue, red=red)
        # Drop Red cyber unit to zero.  Red still has kinetic units.
        red.set_strength_of("Cyber-C2", 0.0)
        assert bs.is_terminated() is False

    def test_same_label_for_both_sides_raises(self):
        b1 = Force(label="Same", unit_types=[
            UnitType("X", Domain.SURFACE, 1.0, 1.0)])
        b2 = Force(label="Same", unit_types=[
            UnitType("Y", Domain.SURFACE, 1.0, 1.0)])
        with pytest.raises(ValueError):
            BattleState(blue=b1, red=b2)

    def test_non_finite_time_raises(self):
        with pytest.raises(ValueError):
            BattleState(blue=_make_blue(), red=_make_red(), time=np.nan)
