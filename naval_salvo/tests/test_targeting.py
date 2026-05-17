"""Tests for naval_salvo.targeting."""

from __future__ import annotations

import numpy as np
import pytest

from naval_salvo import (
    Admissibility,
    Domain,
    Force,
    Manual,
    StrengthProportional,
    ThreatWeighted,
    Uniform,
    UnitType,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _force_2_surface_1_underwater(label):
    return Force(
        label=label,
        unit_types=[
            UnitType("S1", Domain.SURFACE, 2.0, 4),
            UnitType("S2", Domain.SURFACE, 2.0, 2),
            UnitType("U",  Domain.UNDERWATER, 2.0, 1),
        ],
    )


# ---------------------------------------------------------------------------
# Uniform
# ---------------------------------------------------------------------------


class TestUniform:
    def test_equal_split_under_full_admissibility(self):
        # 1 attacker vs 3 defenders all admissible -- σ = 1/3 each.
        atk = Force("A", [UnitType("a", Domain.SURFACE, 1.0, 1)])
        defn = Force("D", [
            UnitType("d1", Domain.SURFACE, 1.0, 1),
            UnitType("d2", Domain.SURFACE, 1.0, 1),
            UnitType("d3", Domain.SURFACE, 1.0, 1),
        ])
        adm = Admissibility.degenerate()
        sig_o, sig_d = Uniform().compute(atk, defn, adm)
        np.testing.assert_allclose(sig_o[0], [1/3, 1/3, 1/3])

    def test_admissibility_zero_excluded(self):
        # Mixed force: U attacker with degenerate adm matrix
        # (S→S only) ⇒ U has no admissible target ⇒ row = 0.
        atk = _force_2_surface_1_underwater("A")
        defn = _force_2_surface_1_underwater("D")
        adm = Admissibility.degenerate()
        sig_o, _ = Uniform().compute(atk, defn, adm)
        # S1 (row 0): admissible to S1, S2 (cols 0, 1) → σ = 0.5 each.
        np.testing.assert_allclose(sig_o[0], [0.5, 0.5, 0.0])
        # S2 same as S1.
        np.testing.assert_allclose(sig_o[1], [0.5, 0.5, 0.0])
        # U (row 2): no admissible target → all zero.
        np.testing.assert_allclose(sig_o[2], [0.0, 0.0, 0.0])

    def test_chi_intermediate_propagates(self):
        atk = Force("A", [UnitType("Sair", Domain.AIR, 1.0, 1)])
        defn = Force("D", [
            UnitType("dS", Domain.SURFACE,    1.0, 1),
            UnitType("dU", Domain.UNDERWATER, 1.0, 1),
        ])
        # Custom adm: A->S = 1.0, A->U = 0.5
        M = np.zeros((5, 5))
        M[Domain.AIR.index, Domain.SURFACE.index]    = 1.0
        M[Domain.AIR.index, Domain.UNDERWATER.index] = 0.5
        adm = Admissibility.from_array(M)
        sig_o, _ = Uniform().compute(atk, defn, adm)
        # Row sum = 1, with split 1.0 : 0.5 → 2/3 vs 1/3.
        np.testing.assert_allclose(sig_o[0], [2/3, 1/3])

    def test_defensive_column_normalised(self):
        # Two attackers vs one defender, all admissible.  Defender's
        # σ_def column over (j) must sum to 1.
        atk = Force("A", [
            UnitType("a1", Domain.SURFACE, 1.0, 1),
            UnitType("a2", Domain.SURFACE, 1.0, 1),
        ])
        defn = Force("D", [UnitType("d1", Domain.SURFACE, 1.0, 1)])
        _, sig_d = Uniform().compute(atk, defn, Admissibility.degenerate())
        # Column 0: [0.5, 0.5]
        np.testing.assert_allclose(sig_d[:, 0], [0.5, 0.5])


# ---------------------------------------------------------------------------
# StrengthProportional
# ---------------------------------------------------------------------------


class TestStrengthProportional:
    def test_proportional_to_defender_stock(self):
        atk = Force("A", [UnitType("a", Domain.SURFACE, 1.0, 1)])
        defn = Force("D", [
            UnitType("big",   Domain.SURFACE, 1.0, 8),    # 4× the small one
            UnitType("small", Domain.SURFACE, 1.0, 2),
        ])
        sig_o, _ = StrengthProportional().compute(
            atk, defn, Admissibility.degenerate()
        )
        # 8 : 2 split → 0.8 : 0.2
        np.testing.assert_allclose(sig_o[0], [0.8, 0.2])

    def test_recovers_uniform_for_equal_stocks(self):
        atk = Force("A", [UnitType("a", Domain.SURFACE, 1.0, 1)])
        defn = Force("D", [
            UnitType("d1", Domain.SURFACE, 1.0, 4),
            UnitType("d2", Domain.SURFACE, 1.0, 4),
            UnitType("d3", Domain.SURFACE, 1.0, 4),
        ])
        sig_o, _ = StrengthProportional().compute(
            atk, defn, Admissibility.degenerate()
        )
        np.testing.assert_allclose(sig_o[0], [1/3, 1/3, 1/3])

    def test_zero_stock_target_gets_zero_share(self):
        atk = Force("A", [UnitType("a", Domain.SURFACE, 1.0, 1)])
        defn = Force("D", [
            UnitType("alive", Domain.SURFACE, 1.0, 5),
            UnitType("dead",  Domain.SURFACE, 1.0, 0),
        ])
        sig_o, _ = StrengthProportional().compute(
            atk, defn, Admissibility.degenerate()
        )
        np.testing.assert_allclose(sig_o[0], [1.0, 0.0])

    def test_all_zero_admissible_targets_yields_zero_row(self):
        atk = Force("A", [UnitType("a", Domain.SURFACE, 1.0, 1)])
        defn = Force("D", [UnitType("d", Domain.AIR, 1.0, 5)])
        # Default admissibility: SURFACE→AIR = chi (0.5 default), but
        # we can also test the limit χ = 0 by handing a custom matrix.
        adm = Admissibility.from_array(np.zeros((5, 5)))
        sig_o, _ = StrengthProportional().compute(atk, defn, adm)
        np.testing.assert_array_equal(sig_o[0], [0.0])

    def test_defensive_proportional_to_attacker_stock(self):
        atk = Force("A", [
            UnitType("big",   Domain.SURFACE, 1.0, 6),
            UnitType("small", Domain.SURFACE, 1.0, 2),
        ])
        defn = Force("D", [UnitType("d", Domain.SURFACE, 1.0, 1)])
        _, sig_d = StrengthProportional().compute(
            atk, defn, Admissibility.degenerate()
        )
        # Defender splits 6 : 2 = 0.75 : 0.25
        np.testing.assert_allclose(sig_d[:, 0], [0.75, 0.25])


# ---------------------------------------------------------------------------
# ThreatWeighted
# ---------------------------------------------------------------------------


class TestThreatWeighted:
    def test_weights_drive_split(self):
        atk = Force("A", [UnitType("a", Domain.SURFACE, 1.0, 1)])
        defn = Force("D", [
            UnitType("d1", Domain.SURFACE, 1.0, 1),
            UnitType("d2", Domain.SURFACE, 1.0, 1),
        ])
        # Weights 3:1 → σ = 0.75 : 0.25.
        sig_o, _ = ThreatWeighted(
            offensive_weights=np.array([3.0, 1.0])
        ).compute(atk, defn, Admissibility.degenerate())
        np.testing.assert_allclose(sig_o[0], [0.75, 0.25])

    def test_default_weights_recover_uniform(self):
        atk = Force("A", [UnitType("a", Domain.SURFACE, 1.0, 1)])
        defn = Force("D", [
            UnitType("d1", Domain.SURFACE, 1.0, 1),
            UnitType("d2", Domain.SURFACE, 1.0, 1),
        ])
        sig_o, _ = ThreatWeighted().compute(
            atk, defn, Admissibility.degenerate()
        )
        np.testing.assert_allclose(sig_o[0], [0.5, 0.5])

    def test_negative_weight_rejected(self):
        with pytest.raises(ValueError):
            ThreatWeighted(offensive_weights=np.array([1.0, -0.1]))

    def test_wrong_shape_rejected(self):
        atk = Force("A", [UnitType("a", Domain.SURFACE, 1.0, 1)])
        defn = Force("D", [
            UnitType("d1", Domain.SURFACE, 1.0, 1),
            UnitType("d2", Domain.SURFACE, 1.0, 1),
        ])
        # Weights of length 3 against 2 defenders.
        policy = ThreatWeighted(offensive_weights=np.array([1.0, 2.0, 3.0]))
        with pytest.raises(ValueError):
            policy.compute(atk, defn, Admissibility.degenerate())


# ---------------------------------------------------------------------------
# Manual
# ---------------------------------------------------------------------------


class TestManual:
    def test_round_trip(self):
        atk = Force("A", [UnitType("a", Domain.SURFACE, 1.0, 1)])
        defn = Force("D", [UnitType("d", Domain.SURFACE, 1.0, 1)])
        sig_o = np.array([[0.7]])
        sig_d = np.array([[0.3]])
        out_o, out_d = Manual(sig_o, sig_d).compute(
            atk, defn, Admissibility.degenerate()
        )
        np.testing.assert_array_equal(out_o, sig_o)
        np.testing.assert_array_equal(out_d, sig_d)

    def test_required_arrays(self):
        with pytest.raises(ValueError):
            Manual()  # both arrays missing

    def test_out_of_range_rejected(self):
        with pytest.raises(ValueError):
            Manual(sigma_offense=np.array([[1.5]]),
                   sigma_defense=np.array([[0.0]]))
        with pytest.raises(ValueError):
            Manual(sigma_offense=np.array([[-0.1]]),
                   sigma_defense=np.array([[0.0]]))

    def test_shape_mismatch_at_compute(self):
        atk = Force("A", [UnitType("a", Domain.SURFACE, 1.0, 1)])
        defn = Force("D", [
            UnitType("d1", Domain.SURFACE, 1.0, 1),
            UnitType("d2", Domain.SURFACE, 1.0, 1),
        ])
        # σ has shape (1, 1) but force pair is (1, 2).
        with pytest.raises(ValueError):
            Manual(sigma_offense=np.array([[1.0]]),
                   sigma_defense=np.array([[1.0]])).compute(
                atk, defn, Admissibility.degenerate())

    def test_returns_copies(self):
        # Mutating the policy's internal array must not affect the
        # output (and vice versa).
        sig_o_in = np.array([[0.5]])
        sig_d_in = np.array([[0.5]])
        policy = Manual(sigma_offense=sig_o_in, sigma_defense=sig_d_in)
        atk = Force("A", [UnitType("a", Domain.SURFACE, 1.0, 1)])
        defn = Force("D", [UnitType("d", Domain.SURFACE, 1.0, 1)])
        out_o, _ = policy.compute(atk, defn, Admissibility.degenerate())
        out_o[0, 0] = 99.0
        # Re-compute -- internal state should be untouched.
        out_o2, _ = policy.compute(atk, defn, Admissibility.degenerate())
        assert out_o2[0, 0] == 0.5
