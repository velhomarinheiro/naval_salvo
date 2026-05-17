"""Tests for naval_salvo.parameters."""

from __future__ import annotations

import numpy as np
import pytest

from naval_salvo.domains import Domain
from naval_salvo.state import Force, UnitType
from naval_salvo.parameters import (
    DirectionalParameters,
    EngagementParameters,
    PairParameters,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _two_forces() -> tuple[Force, Force]:
    blue = Force(
        label="Blue",
        unit_types=[
            UnitType("BF1", Domain.SURFACE, 3.0, 2),
            UnitType("BF2", Domain.SURFACE, 3.0, 1),
        ],
    )
    red = Force(
        label="Red",
        unit_types=[
            UnitType("RF1", Domain.SURFACE, 4.0, 3),
        ],
    )
    return blue, red


# ---------------------------------------------------------------------------
# PairParameters
# ---------------------------------------------------------------------------


class TestPairParameters:
    def test_default_is_inert(self):
        p = PairParameters()
        # All sigma/eta default to 1, all p default to 0 -- no effect.
        assert p.offensive_kernel() == 0.0
        assert p.defensive_kernel() == 0.0

    def test_offensive_kernel(self):
        p = PairParameters(
            sigma_offense=0.5, eta_offense=0.8, p_offense=4.0,
            sigma_defense=1.0, eta_defense=1.0, p_defense=0.0,
        )
        assert p.offensive_kernel() == pytest.approx(0.5 * 0.8 * 4.0)

    def test_defensive_kernel(self):
        p = PairParameters(
            sigma_defense=0.6, eta_defense=0.5, p_defense=2.0,
        )
        assert p.defensive_kernel() == pytest.approx(0.6 * 0.5 * 2.0)

    @pytest.mark.parametrize("field_name",
                             ["sigma_offense", "sigma_defense",
                              "eta_offense", "eta_defense"])
    @pytest.mark.parametrize("bad", [-0.1, 1.5, np.nan, np.inf])
    def test_sigma_eta_must_be_in_unit_interval(self, field_name, bad):
        kwargs = {field_name: bad}
        with pytest.raises(ValueError):
            PairParameters(**kwargs)

    @pytest.mark.parametrize("field_name", ["p_offense", "p_defense"])
    @pytest.mark.parametrize("bad", [-1.0, np.nan, np.inf])
    def test_p_must_be_nonneg_finite(self, field_name, bad):
        kwargs = {field_name: bad}
        with pytest.raises(ValueError):
            PairParameters(**kwargs)

    def test_p_can_exceed_one(self):
        # Throughputs are not bounded above.
        p = PairParameters(p_offense=10.0, p_defense=20.0)
        assert p.p_offense == 10.0
        assert p.p_defense == 20.0

    def test_immutable(self):
        p = PairParameters()
        with pytest.raises(Exception):
            p.sigma_offense = 0.5  # type: ignore[misc]


# ---------------------------------------------------------------------------
# DirectionalParameters
# ---------------------------------------------------------------------------


class TestDirectionalParameters:
    def test_zeros_factory(self):
        blue, red = _two_forces()
        d = DirectionalParameters.zeros(blue, red)
        assert d.pairs.shape == (2, 1)
        assert isinstance(d.pairs[0, 0], PairParameters)
        # All zero throughput.
        assert d.offensive_kernel_matrix().sum() == 0.0
        assert d.defensive_kernel_matrix().sum() == 0.0

    def test_get_set(self):
        blue, red = _two_forces()
        d = DirectionalParameters.zeros(blue, red)
        d.set("BF1", "RF1",
              PairParameters(sigma_offense=1.0, eta_offense=1.0, p_offense=2.5))
        assert d.get("BF1", "RF1").p_offense == 2.5
        # Other cells untouched.
        assert d.get("BF2", "RF1").p_offense == 0.0

    def test_offensive_kernel_matrix_shape_and_values(self):
        blue, red = _two_forces()
        d = DirectionalParameters.zeros(blue, red)
        d.set("BF1", "RF1",
              PairParameters(sigma_offense=0.5, eta_offense=1.0, p_offense=4.0))
        d.set("BF2", "RF1",
              PairParameters(sigma_offense=1.0, eta_offense=0.5, p_offense=2.0))
        K = d.offensive_kernel_matrix()
        assert K.shape == (2, 1)
        assert K.dtype == np.float64
        assert K[0, 0] == pytest.approx(2.0)
        assert K[1, 0] == pytest.approx(1.0)

    def test_defensive_kernel_matrix(self):
        blue, red = _two_forces()
        d = DirectionalParameters.zeros(blue, red)
        d.set("BF1", "RF1",
              PairParameters(sigma_defense=0.5, eta_defense=0.5, p_defense=4.0))
        K = d.defensive_kernel_matrix()
        assert K[0, 0] == pytest.approx(1.0)
        assert K[1, 0] == 0.0

    def test_shape_mismatch_rejected(self):
        blue, red = _two_forces()
        with pytest.raises(ValueError):
            DirectionalParameters(
                attacker=blue, defender=red,
                pairs=np.empty((3, 3), dtype=object),
            )

    def test_dtype_must_be_object(self):
        blue, red = _two_forces()
        with pytest.raises(ValueError):
            DirectionalParameters(
                attacker=blue, defender=red,
                pairs=np.zeros((2, 1), dtype=np.float64),
            )

    def test_non_pairparameters_entries_rejected(self):
        blue, red = _two_forces()
        bad = np.empty((2, 1), dtype=object)
        bad[0, 0] = "not a PairParameters"
        bad[1, 0] = PairParameters()
        with pytest.raises(ValueError):
            DirectionalParameters(attacker=blue, defender=red, pairs=bad)


# ---------------------------------------------------------------------------
# EngagementParameters
# ---------------------------------------------------------------------------


class TestEngagementParameters:
    def test_with_zero_couplings(self):
        blue, red = _two_forces()
        ep = EngagementParameters.with_zero_couplings(blue, red)
        assert ep.blue is blue
        assert ep.red is red
        assert ep.t_char.shape == (4,)
        assert ep.rho.shape == (4,)
        np.testing.assert_array_equal(ep.t_char, np.ones(4))
        np.testing.assert_array_equal(ep.rho, np.zeros(4))

    def test_directional_blocks_must_match_forces(self):
        blue, red = _two_forces()
        # Wrong: red_attacks_blue passed in the slot for blue_attacks_red.
        bar = DirectionalParameters.zeros(blue, red)
        rab = DirectionalParameters.zeros(red, blue)
        with pytest.raises(ValueError):
            EngagementParameters(
                blue=blue, red=red,
                blue_attacks_red=rab,         # wrong direction
                red_attacks_blue=bar,         # wrong direction
            )

    def test_t_char_validation(self):
        blue, red = _two_forces()
        bar = DirectionalParameters.zeros(blue, red)
        rab = DirectionalParameters.zeros(red, blue)
        with pytest.raises(ValueError):
            EngagementParameters(
                blue=blue, red=red,
                blue_attacks_red=bar, red_attacks_blue=rab,
                t_char=np.array([1.0, 1.0, 1.0]),  # wrong shape
            )
        with pytest.raises(ValueError):
            EngagementParameters(
                blue=blue, red=red,
                blue_attacks_red=bar, red_attacks_blue=rab,
                t_char=np.array([1.0, 1.0, 1.0, 0.0]),  # zero
            )
        with pytest.raises(ValueError):
            EngagementParameters(
                blue=blue, red=red,
                blue_attacks_red=bar, red_attacks_blue=rab,
                t_char=np.array([1.0, 1.0, 1.0, -1.0]),  # negative
            )

    def test_rho_default_is_zero(self):
        # Decision 1.4 §2.3.f: baseline has no regeneration.
        blue, red = _two_forces()
        ep = EngagementParameters.with_zero_couplings(blue, red)
        assert np.all(ep.rho == 0.0)

    def test_rho_nonneg_required(self):
        blue, red = _two_forces()
        bar = DirectionalParameters.zeros(blue, red)
        rab = DirectionalParameters.zeros(red, blue)
        with pytest.raises(ValueError):
            EngagementParameters(
                blue=blue, red=red,
                blue_attacks_red=bar, red_attacks_blue=rab,
                rho=np.array([0.0, 0.0, 0.0, -0.1]),
            )

    def test_t_char_per_domain_can_differ(self):
        # Hausken-Moxnes (2026) motivate differing tempos by domain;
        # check that the data structure permits it.
        blue, red = _two_forces()
        ep = EngagementParameters.with_zero_couplings(
            blue, red, t_char=np.array([1.0, 5.0, 0.1, 1.0])
        )
        np.testing.assert_array_equal(
            ep.t_char, [1.0, 5.0, 0.1, 1.0]
        )
