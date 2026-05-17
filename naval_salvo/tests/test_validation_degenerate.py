"""
Validation tests for naval_salvo: the multi-domain heterogeneous engine
must reduce *exactly* to Hughes (1995) / JPH (2001) when restricted to
their canonical scope.

The contract is bit-for-bit (modulo machine epsilon), as required by
working document 1.4 §3.4.
"""

from __future__ import annotations

import numpy as np
import pytest

from naval_salvo import (
    Admissibility,
    Domain,
    HughesScenario,
    build_hughes_homogeneous_engagement,
    hughes_analytical,
    salvo_step,
)


# ---------------------------------------------------------------------------
# Hughes (1995) homogeneous, single salvo
# ---------------------------------------------------------------------------


class TestHughesAnalyticalSelfConsistency:
    """Sanity checks on the closed-form iteration itself."""

    def test_no_offense_no_change(self):
        scn = HughesScenario(A0=4, B0=3, alpha=0, beta=0,
                             z=0, y=0, w=1, x=1, n_salvos=3)
        A, B = hughes_analytical(scn)
        np.testing.assert_array_equal(A, [4.0, 4.0, 4.0, 4.0])
        np.testing.assert_array_equal(B, [3.0, 3.0, 3.0, 3.0])

    def test_no_negative_strength(self):
        scn = HughesScenario(A0=2, B0=2, alpha=10, beta=10,
                             z=0, y=0, w=1, x=1, n_salvos=5)
        A, B = hughes_analytical(scn)
        # Both sides wiped out in one salvo, then frozen at zero.
        assert np.all(A >= 0.0)
        assert np.all(B >= 0.0)
        assert A[1] == 0.0 and B[1] == 0.0

    def test_christiansen_textbook_form(self):
        # ΔA = (βB - a3·A)/a1, ΔB = (αA - b3·B)/b1   (Christiansen 2008
        # quasi-code, Hughes 1995 notation).  With α=4, β=3, a1=2, a3=1,
        # b1=2, b3=1, A0=5, B0=4:
        #   ΔA = (3·4 - 1·5)/2 = 7/2 = 3.5
        #   ΔB = (4·5 - 1·4)/2 = 16/2 = 8 → capped at B0 = 4
        scn = HughesScenario(A0=5, B0=4, alpha=4, beta=3,
                             z=1, y=1, w=2, x=2, n_salvos=1)
        A, B = hughes_analytical(scn)
        assert A[1] == pytest.approx(5 - 3.5)
        assert B[1] == 0.0


class TestHughesEngineRecovery:
    """The deterministic engine must match hughes_analytical()."""

    @pytest.mark.parametrize("scn_kwargs", [
        # Sub-saturation regime: defenses absorb everything.
        dict(A0=4, B0=3, alpha=1, beta=1, z=10, y=10, w=2, x=2),
        # Symmetric, balanced exchange.
        dict(A0=4, B0=4, alpha=2, beta=2, z=1, y=1, w=2, x=2),
        # Asymmetric: Blue much more lethal.
        dict(A0=6, B0=4, alpha=1, beta=4, z=1, y=1, w=2, x=2),
        # Mutual annihilation regime.
        dict(A0=2, B0=2, alpha=10, beta=10, z=0, y=0, w=1, x=1),
        # Heavy staying power.
        dict(A0=10, B0=10, alpha=3, beta=3, z=2, y=2, w=5, x=5),
    ])
    def test_single_salvo_matches_analytical(self, scn_kwargs):
        scn = HughesScenario(n_salvos=1, **scn_kwargs)
        A_an, B_an = hughes_analytical(scn)

        bs, ep, adm = build_hughes_homogeneous_engagement(scn)
        salvo_step(bs, ep, adm, apply=True)
        np.testing.assert_allclose(bs.red.strength_of("A"), A_an[1],
                                   rtol=0, atol=1e-12)
        np.testing.assert_allclose(bs.blue.strength_of("B"), B_an[1],
                                   rtol=0, atol=1e-12)

    @pytest.mark.parametrize("scn_kwargs", [
        dict(A0=4, B0=3, alpha=1, beta=1, z=10, y=10, w=2, x=2),
        dict(A0=4, B0=4, alpha=2, beta=2, z=1, y=1, w=2, x=2),
        dict(A0=6, B0=4, alpha=1, beta=4, z=1, y=1, w=2, x=2),
        dict(A0=10, B0=10, alpha=3, beta=3, z=2, y=2, w=5, x=5),
    ])
    def test_multi_salvo_matches_analytical(self, scn_kwargs):
        scn = HughesScenario(n_salvos=10, **scn_kwargs)
        A_an, B_an = hughes_analytical(scn)

        bs, ep, adm = build_hughes_homogeneous_engagement(scn)
        for _ in range(scn.n_salvos):
            salvo_step(bs, ep, adm, apply=True)
        np.testing.assert_allclose(bs.red.strength_of("A"), A_an[-1],
                                   rtol=0, atol=1e-12)
        np.testing.assert_allclose(bs.blue.strength_of("B"), B_an[-1],
                                   rtol=0, atol=1e-12)


# ---------------------------------------------------------------------------
# JPH (2001) heterogeneous-within-domain recovery
# ---------------------------------------------------------------------------


class TestJPHHeterogeneousRecovery:
    """
    The JPH (2001) heterogeneous salvo equation, in matrix form, is

        ΔA_i = -(1/s_i) * max(0, Σ_j O_{ji} B_j - Σ_j D_{ij} A_i)

    Our engine *is* this equation (with χ-weighting that reduces to 1
    in the JPH-degenerate admissibility).  This test checks the
    multi-attacker-per-target arithmetic on a worked example with
    known target indices.
    """

    def test_multi_attacker_aggregation(self):
        """
        A single Blue defender attacked by two Red unit types.  The total
        offence on Blue should be the sum of the Red contributions; the
        total defence should accumulate over the same attackers.
        """
        from naval_salvo import (
            BattleState, DirectionalParameters, EngagementParameters,
            Force, PairParameters, UnitType,
        )

        blue = Force("Blue", [UnitType("B1", Domain.SURFACE, 2.0, 5)])
        red = Force("Red", [
            UnitType("R1", Domain.SURFACE, 2.0, 3),  # contributes 3·O_{R1,B1}
            UnitType("R2", Domain.SURFACE, 2.0, 2),  # contributes 2·O_{R2,B1}
        ])

        bar = DirectionalParameters.zeros(blue, red)
        # Blue can't really hit Red here -- zero throughputs.
        rab = DirectionalParameters.zeros(red, blue)
        rab.set("R1", "B1",
                PairParameters(p_offense=2.0, p_defense=0.5))
        rab.set("R2", "B1",
                PairParameters(p_offense=1.0, p_defense=0.2))
        ep = EngagementParameters(blue=blue, red=red,
                                  blue_attacks_red=bar,
                                  red_attacks_blue=rab)
        bs = BattleState(blue=blue, red=red)

        # Expected:
        #   Σ_j O B_j  = 2*3 + 1*2 = 8
        #   Σ_j D A_i  = 0.5*5 + 0.2*5 = 3.5
        #   raw kernel = max(0, 8 - 3.5) = 4.5
        #   ΔB1        = -4.5 / 2.0 = -2.25
        out = salvo_step(bs, ep, Admissibility.degenerate(), apply=False)
        assert out.blue_raw_kernel[0] == pytest.approx(4.5)
        assert out.blue_losses[0] == pytest.approx(4.5 / 2.0)


# ---------------------------------------------------------------------------
# Cross-domain admissibility on degenerate set
# ---------------------------------------------------------------------------


class TestCrossDomainInDegenerateMode:
    """
    Cross-domain unit types must be invisible to each other when the
    admissibility matrix is the JPH-degenerate (only (S, S) = 1).
    Putting an air unit on either side should change *nothing*.
    """

    def test_phantom_air_unit_does_nothing(self):
        from naval_salvo import (
            BattleState, DirectionalParameters, EngagementParameters,
            Force, PairParameters, UnitType,
        )

        # Reference: pure surface 1-vs-1.
        scn = HughesScenario(A0=4, B0=3, alpha=2, beta=2,
                             z=1, y=1, w=2, x=2, n_salvos=3)
        A_ref, B_ref = hughes_analytical(scn)

        # Same engagement, but each side gets an extra Air unit type
        # with non-zero throughputs.  Under degenerate admissibility,
        # nothing involving the air domain should be active.
        blue = Force("Blue", [
            UnitType("B", Domain.SURFACE, scn.w, scn.B0),
            UnitType("B_air_phantom", Domain.AIR, 1.0, 5),
        ])
        red = Force("Red", [
            UnitType("A", Domain.SURFACE, scn.x, scn.A0),
            UnitType("A_air_phantom", Domain.AIR, 1.0, 5),
        ])

        bar = DirectionalParameters.zeros(blue, red)
        rab = DirectionalParameters.zeros(red, blue)
        # Surface↔Surface:  Hughes mapping.
        bar.set("B", "A", PairParameters(p_offense=scn.beta,  p_defense=scn.y))
        rab.set("A", "B", PairParameters(p_offense=scn.alpha, p_defense=scn.z))
        # Air↔* : large but should be neutralised by χ = 0.
        for atk, defn, dirn in [
            ("B_air_phantom", "A",            bar),
            ("B_air_phantom", "A_air_phantom", bar),
            ("B",            "A_air_phantom", bar),
            ("A_air_phantom", "B",            rab),
            ("A_air_phantom", "B_air_phantom", rab),
            ("A",            "B_air_phantom", rab),
        ]:
            dirn.set(atk, defn, PairParameters(p_offense=10.0, p_defense=10.0))

        ep = EngagementParameters(blue=blue, red=red,
                                  blue_attacks_red=bar,
                                  red_attacks_blue=rab)
        bs = BattleState(blue=blue, red=red)
        adm = Admissibility.degenerate()

        for _ in range(scn.n_salvos):
            salvo_step(bs, ep, adm, apply=True)

        # Surface units evolve exactly as in the pure-surface reference.
        np.testing.assert_allclose(bs.red.strength_of("A"), A_ref[-1],
                                   rtol=0, atol=1e-12)
        np.testing.assert_allclose(bs.blue.strength_of("B"), B_ref[-1],
                                   rtol=0, atol=1e-12)
        # Air phantoms untouched.
        assert bs.blue.strength_of("B_air_phantom") == 5.0
        assert bs.red.strength_of("A_air_phantom") == 5.0
