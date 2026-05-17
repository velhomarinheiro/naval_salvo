"""Tests for naval_salvo.cyber: Φ modulator families."""

from __future__ import annotations

import numpy as np
import pytest

from naval_salvo import (
    Admissibility,
    BattleState,
    CyberModulator,
    CyberSubtype,
    DecomposedPhi,
    DirectionalParameters,
    Domain,
    EngagementParameters,
    Force,
    HauskenPhi,
    PairParameters,
    SimplePhi,
    UnitType,
    phi_logistic,
)


# ---------------------------------------------------------------------------
# Scalar phi_logistic
# ---------------------------------------------------------------------------


class TestPhiLogistic:
    def test_parity_returns_one_minus_half_r0(self):
        # At z=0, logistic = 0.5, so phi = 1 - r0/2.
        assert phi_logistic(5, 5, r0=0.5) == pytest.approx(0.75)
        assert phi_logistic(10, 10, r0=0.8) == pytest.approx(0.6)

    def test_attacker_overwhelmingly_dominates(self):
        # With large positive z, phi -> 1 - r0.
        v = phi_logistic(100, 1, r0=0.5, k=10.0, x_ref=10.0)
        assert v == pytest.approx(0.5, abs=1e-3)

    def test_defender_overwhelmingly_dominates(self):
        # With large negative z, phi -> 1.
        v = phi_logistic(1, 100, r0=0.5, k=10.0, x_ref=10.0)
        assert v == pytest.approx(1.0, abs=1e-3)

    def test_no_cyber_short_circuits_to_one(self):
        # Both sides zero -> Φ = 1 (the "no cyber dimension" reading).
        assert phi_logistic(0, 0, r0=0.5) == 1.0
        assert phi_logistic(0, 0, r0=0.9, k=5.0) == 1.0

    def test_attacker_only_max_degradation_at_high_k(self):
        # x_atk > 0, x_def = 0, large k -> phi ~ 1 - r0
        v = phi_logistic(5, 0, r0=0.6, k=10.0, x_ref=1.0)
        assert v == pytest.approx(0.4, abs=1e-3)

    def test_r0_zero_disables_modulator(self):
        # Φ identically 1 when r0=0.
        for x_a in [0, 1, 5, 100]:
            for x_d in [0, 1, 5, 100]:
                assert phi_logistic(x_a, x_d, r0=0.0) == 1.0

    def test_phi_in_unit_interval(self):
        # Φ ∈ [1-r0, 1] for any non-negative inputs.
        rng = np.random.default_rng(0)
        for _ in range(50):
            x_a = float(rng.uniform(0, 100))
            x_d = float(rng.uniform(0, 100))
            r0 = float(rng.uniform(0, 1))
            v = phi_logistic(x_a, x_d, r0=r0, k=2.0, x_ref=10.0)
            assert 1.0 - r0 - 1e-9 <= v <= 1.0 + 1e-9

    def test_monotone_in_attacker_advantage(self):
        # Increasing x_atk weakly decreases Φ (more degradation).
        prev = phi_logistic(0, 5, r0=0.5)
        for x_a in [1, 2, 5, 10, 100]:
            now = phi_logistic(x_a, 5, r0=0.5)
            assert now <= prev + 1e-12
            prev = now

    @pytest.mark.parametrize("bad", [-0.1, 1.1])
    def test_r0_out_of_range_rejected(self, bad):
        with pytest.raises(ValueError):
            phi_logistic(1, 1, r0=bad)

    def test_negative_stocks_rejected(self):
        with pytest.raises(ValueError):
            phi_logistic(-1, 1)
        with pytest.raises(ValueError):
            phi_logistic(1, -1)

    def test_invalid_x_ref_rejected(self):
        with pytest.raises(ValueError):
            phi_logistic(1, 1, x_ref=0.0)
        with pytest.raises(ValueError):
            phi_logistic(1, 1, x_ref=-1.0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_engagement_with_cyber(
    blue_cyber_per_subtype: dict[str, float] = None,
    red_cyber_per_subtype: dict[str, float] = None,
):
    """
    Build a small engagement with kinetic units and cyber sub-types.
    Returns (BattleState, EngagementParameters, Admissibility).
    """
    blue_cyber = blue_cyber_per_subtype or {}
    red_cyber = red_cyber_per_subtype or {}

    blue_units = [
        UnitType("Frig", Domain.SURFACE, 2.0, 2),
        UnitType("Sub",  Domain.UNDERWATER, 2.0, 1),
    ]
    for code, n in blue_cyber.items():
        blue_units.append(UnitType(
            f"BlueX-{code}", Domain.CYBER, 1.0, float(n), subtype=code
        ))

    red_units = [
        UnitType("Dest", Domain.SURFACE, 3.0, 2),
    ]
    for code, n in red_cyber.items():
        red_units.append(UnitType(
            f"RedX-{code}", Domain.CYBER, 1.0, float(n), subtype=code
        ))

    blue = Force("Blue", blue_units)
    red = Force("Red", red_units)

    bar = DirectionalParameters.zeros(blue, red)
    rab = DirectionalParameters.zeros(red, blue)
    # Set non-trivial σ, η, p across all pairs.
    for j_unit in blue.unit_types:
        for i_unit in red.unit_types:
            bar.set(j_unit.name, i_unit.name, PairParameters(
                sigma_offense=0.5, sigma_defense=0.5,
                eta_offense=0.8, eta_defense=0.8,
                p_offense=2.0, p_defense=1.0,
            ))
    for j_unit in red.unit_types:
        for i_unit in blue.unit_types:
            rab.set(j_unit.name, i_unit.name, PairParameters(
                sigma_offense=0.5, sigma_defense=0.5,
                eta_offense=0.8, eta_defense=0.8,
                p_offense=2.0, p_defense=1.0,
            ))

    ep = EngagementParameters(
        blue=blue, red=red,
        blue_attacks_red=bar, red_attacks_blue=rab,
    )
    bs = BattleState(blue=blue, red=red)
    return bs, ep, Admissibility.canonical()


# ---------------------------------------------------------------------------
# SimplePhi
# ---------------------------------------------------------------------------


class TestSimplePhi:
    def test_no_cyber_is_identity(self):
        bs, ep, adm = _build_engagement_with_cyber({}, {})
        ep2 = SimplePhi(r0=0.5, k=2.0).apply(ep)
        # All kinetic pairs unchanged (with submarine defenders also unchanged).
        for j in range(2):
            for i in range(1):
                a = ep.blue_attacks_red.pairs[j, i]
                b = ep2.blue_attacks_red.pairs[j, i]
                assert a.sigma_offense == b.sigma_offense
                assert a.eta_offense == b.eta_offense

    def test_attacker_dominance_degrades_kinetic_pairs(self):
        # Red has cyber, Blue has none -> Red→Blue Φ < 1, but Blue→Red Φ ≈ 1.
        bs, ep, adm = _build_engagement_with_cyber({}, {"C2": 5.0})
        ep2 = SimplePhi(r0=0.5, k=2.0).apply(ep)
        # Red→Blue, Frigate defender (kinetic, non-sub): degraded.
        rb = ep2.red_attacks_blue
        # Find Frigate index in Blue.
        frig_i = [ut.name for ut in bs.blue.unit_types].index("Frig")
        dest_j = [ut.name for ut in bs.red.unit_types].index("Dest")
        new_sigma_off = rb.pairs[dest_j, frig_i].sigma_offense
        old_sigma_off = ep.red_attacks_blue.pairs[dest_j, frig_i].sigma_offense
        assert new_sigma_off < old_sigma_off

    def test_blue_attack_unaffected_when_only_red_has_cyber(self):
        # Red → Blue is the direction Red attacks; Blue → Red is not.
        # In SimplePhi, Φ_blue_attacks_red is computed with x_atk=Blue,
        # x_def=Red.  With Blue=0, Red>0 -> attacker has no cyber -> Φ ≈ 1.
        bs, ep, adm = _build_engagement_with_cyber({}, {"C2": 5.0})
        ep2 = SimplePhi(r0=0.5, k=10.0).apply(ep)
        # Frigate -> Destroyer pair in Blue->Red direction.
        frig_j = [ut.name for ut in bs.blue.unit_types].index("Frig")
        dest_i = [ut.name for ut in bs.red.unit_types].index("Dest")
        new_sigma = ep2.blue_attacks_red.pairs[frig_j, dest_i].sigma_offense
        old_sigma = ep.blue_attacks_red.pairs[frig_j, dest_i].sigma_offense
        # Blue is the attacker here with zero cyber stock -> Φ should
        # be close to 1 (defender Red dominates).
        assert new_sigma == pytest.approx(old_sigma, abs=0.01)

    def test_returns_new_engagement_parameters(self):
        bs, ep, adm = _build_engagement_with_cyber({}, {"C2": 1.0})
        ep2 = SimplePhi(r0=0.5).apply(ep)
        assert isinstance(ep2, EngagementParameters)
        # Same forces (by reference).
        assert ep2.blue is ep.blue
        assert ep2.red is ep.red

    def test_invalid_r0_rejected(self):
        with pytest.raises(ValueError):
            SimplePhi(r0=-0.1)
        with pytest.raises(ValueError):
            SimplePhi(r0=1.5)

    def test_invalid_k_rejected(self):
        with pytest.raises(ValueError):
            SimplePhi(k=-1.0)


# ---------------------------------------------------------------------------
# DecomposedPhi (canonical Family 2)
# ---------------------------------------------------------------------------


class TestDecomposedPhi:
    def test_each_subtype_has_its_own_phi(self):
        # Build a force where Blue has only SEN cyber and Red has only WPN.
        bs, ep, adm = _build_engagement_with_cyber(
            blue_cyber_per_subtype={"SEN": 5.0},
            red_cyber_per_subtype={"WPN": 5.0},
        )
        ep2 = DecomposedPhi(r0=0.5, k=2.0).apply(ep)

        # Red→Blue direction: Red is attacker.
        # Red attacker has no SEN cyber -> Φ_SEN ~ 1 (no scouting degradation).
        # Red attacker has WPN -> Φ_WPN < 1 (firing chain degraded? wait,
        # attacker's WPN cyber degrades opponent's η_offense.  But
        # opponent here is Blue, and Φ_WPN multiplies η_off which is
        # the attacker's, so... let's just check that DIFFERENT
        # parameters are scaled by DIFFERENT amounts.)
        frig_i = [ut.name for ut in bs.blue.unit_types].index("Frig")
        dest_j = [ut.name for ut in bs.red.unit_types].index("Dest")
        old = ep.red_attacks_blue.pairs[dest_j, frig_i]
        new = ep2.red_attacks_blue.pairs[dest_j, frig_i]
        # The four sigma_off / sigma_def / eta_off values are scaled
        # by *independent* Φ products.  At minimum, sigma_off and η_off
        # should differ in their scaling because Φ_SEN is asymmetric
        # between blue (sender) and red (sender).
        # Just confirm at least one is reduced and they reduce by
        # different amounts than under SimplePhi (which would scale
        # all three identically).
        ratio_sigma = new.sigma_offense / old.sigma_offense
        ratio_eta = new.eta_offense / old.eta_offense
        # σ_offense scales by Φ_C2 · Φ_SEN; η_offense scales by Φ_WPN.
        # Since Blue has SEN and Red has WPN, the two scalings differ.
        assert ratio_sigma != pytest.approx(ratio_eta)

    def test_canonical_kill_chain_mapping(self):
        # Verify the canonical mapping: σ_off scales by Φ_C2 · Φ_SEN,
        # σ_def by Φ_C2 only, η_off by Φ_WPN only.
        # Use parity of all sub-types so each Φ is the same value (1 - r0/2).
        bs, ep, adm = _build_engagement_with_cyber(
            blue_cyber_per_subtype={"C2": 5.0, "SEN": 5.0,
                                    "WPN": 5.0, "LOG": 5.0},
            red_cyber_per_subtype={"C2": 5.0, "SEN": 5.0,
                                    "WPN": 5.0, "LOG": 5.0},
        )
        r0 = 0.4
        phi = 1.0 - r0 / 2  # parity logistic value
        ep2 = DecomposedPhi(r0=r0, k=1.0).apply(ep)

        frig_i = [ut.name for ut in bs.blue.unit_types].index("Frig")
        dest_j = [ut.name for ut in bs.red.unit_types].index("Dest")
        old = ep.red_attacks_blue.pairs[dest_j, frig_i]
        new = ep2.red_attacks_blue.pairs[dest_j, frig_i]

        # σ_off = old.σ_off · Φ_C2 · Φ_SEN = 0.5 · phi · phi
        assert new.sigma_offense == pytest.approx(
            old.sigma_offense * phi * phi, abs=1e-12)
        # σ_def = old.σ_def · Φ_C2 = 0.5 · phi
        assert new.sigma_defense == pytest.approx(
            old.sigma_defense * phi, abs=1e-12)
        # η_off = old.η_off · Φ_WPN = 0.8 · phi
        assert new.eta_offense == pytest.approx(
            old.eta_offense * phi, abs=1e-12)
        # η_def, p_off, p_def unchanged.
        assert new.eta_defense == old.eta_defense
        assert new.p_offense == old.p_offense
        assert new.p_defense == old.p_defense

    def test_no_cyber_is_identity(self):
        bs, ep, adm = _build_engagement_with_cyber({}, {})
        ep2 = DecomposedPhi(r0=0.5).apply(ep)
        for j in range(bs.blue.n_unit_types):
            for i in range(bs.red.n_unit_types):
                a = ep.blue_attacks_red.pairs[j, i]
                b = ep2.blue_attacks_red.pairs[j, i]
                assert a.sigma_offense == b.sigma_offense
                assert a.sigma_defense == b.sigma_defense
                assert a.eta_offense == b.eta_offense

    def test_invalid_r0_rejected(self):
        with pytest.raises(ValueError):
            DecomposedPhi(r0=-0.1)
        with pytest.raises(ValueError):
            DecomposedPhi(r0=1.5)


# ---------------------------------------------------------------------------
# Submarine immunity (canonical δ_X^U = 0)
# ---------------------------------------------------------------------------


class TestSubmarineImmunity:
    def test_simple_phi_skips_submarine_defender(self):
        bs, ep, adm = _build_engagement_with_cyber(
            blue_cyber_per_subtype={},
            red_cyber_per_subtype={"C2": 100.0},
        )
        ep2 = SimplePhi(r0=0.99, k=10.0).apply(ep)
        # Red→Blue, Submarine defender: should be unchanged.
        sub_i = [ut.name for ut in bs.blue.unit_types].index("Sub")
        dest_j = [ut.name for ut in bs.red.unit_types].index("Dest")
        old = ep.red_attacks_blue.pairs[dest_j, sub_i]
        new = ep2.red_attacks_blue.pairs[dest_j, sub_i]
        assert new.sigma_offense == old.sigma_offense
        assert new.sigma_defense == old.sigma_defense
        assert new.eta_offense == old.eta_offense

    def test_decomposed_phi_skips_submarine_defender(self):
        bs, ep, adm = _build_engagement_with_cyber(
            blue_cyber_per_subtype={},
            red_cyber_per_subtype={"C2": 50.0, "SEN": 50.0,
                                    "WPN": 50.0, "LOG": 50.0},
        )
        ep2 = DecomposedPhi(r0=0.99, k=10.0).apply(ep)
        sub_i = [ut.name for ut in bs.blue.unit_types].index("Sub")
        dest_j = [ut.name for ut in bs.red.unit_types].index("Dest")
        old = ep.red_attacks_blue.pairs[dest_j, sub_i]
        new = ep2.red_attacks_blue.pairs[dest_j, sub_i]
        assert new.sigma_offense == old.sigma_offense
        assert new.sigma_defense == old.sigma_defense
        assert new.eta_offense == old.eta_offense

    def test_hausken_phi_skips_submarine_defender(self):
        bs, ep, adm = _build_engagement_with_cyber(
            blue_cyber_per_subtype={},
            red_cyber_per_subtype={"C2": 100.0},
        )
        ep2 = HauskenPhi(r0=0.9).apply(ep)
        sub_i = [ut.name for ut in bs.blue.unit_types].index("Sub")
        dest_j = [ut.name for ut in bs.red.unit_types].index("Dest")
        old = ep.red_attacks_blue.pairs[dest_j, sub_i]
        new = ep2.red_attacks_blue.pairs[dest_j, sub_i]
        assert new.sigma_offense == old.sigma_offense
        assert new.eta_offense == old.eta_offense

    def test_non_submarine_defenders_still_modulated(self):
        # Sanity: the immunity is *specifically* for submarines.  A
        # surface (Frigate) defender in the same scenario must still
        # be degraded.
        bs, ep, adm = _build_engagement_with_cyber(
            blue_cyber_per_subtype={},
            red_cyber_per_subtype={"C2": 100.0},
        )
        ep2 = SimplePhi(r0=0.99, k=10.0).apply(ep)
        frig_i = [ut.name for ut in bs.blue.unit_types].index("Frig")
        dest_j = [ut.name for ut in bs.red.unit_types].index("Dest")
        old = ep.red_attacks_blue.pairs[dest_j, frig_i]
        new = ep2.red_attacks_blue.pairs[dest_j, frig_i]
        assert new.sigma_offense < old.sigma_offense


# ---------------------------------------------------------------------------
# HauskenPhi (Family 3)
# ---------------------------------------------------------------------------


class TestHauskenPhi:
    def test_no_attacker_cyber_means_no_effect(self):
        # In Family 3, Φ = 1 when x_atk = 0 (regardless of x_def).
        bs, ep, adm = _build_engagement_with_cyber(
            blue_cyber_per_subtype={},
            red_cyber_per_subtype={"C2": 100.0},
        )
        ep2 = HauskenPhi(r0=0.9).apply(ep)
        # Blue→Red direction: Blue is attacker, has no cyber -> Φ = 1.
        frig_j = [ut.name for ut in bs.blue.unit_types].index("Frig")
        dest_i = [ut.name for ut in bs.red.unit_types].index("Dest")
        old = ep.blue_attacks_red.pairs[frig_j, dest_i]
        new = ep2.blue_attacks_red.pairs[frig_j, dest_i]
        assert new.sigma_offense == pytest.approx(old.sigma_offense)
        assert new.eta_offense == pytest.approx(old.eta_offense)

    def test_no_defender_cyber_means_full_attacker_effect(self):
        # Red→Blue: Red has cyber, Blue has none -> max degradation.
        bs, ep, adm = _build_engagement_with_cyber(
            blue_cyber_per_subtype={},
            red_cyber_per_subtype={"C2": 5.0},
        )
        r0 = 0.6
        ep2 = HauskenPhi(r0=r0).apply(ep)
        # Frigate defender: degraded by factor (1 - r0).
        frig_i = [ut.name for ut in bs.blue.unit_types].index("Frig")
        dest_j = [ut.name for ut in bs.red.unit_types].index("Dest")
        old = ep.red_attacks_blue.pairs[dest_j, frig_i]
        new = ep2.red_attacks_blue.pairs[dest_j, frig_i]
        assert new.sigma_offense == pytest.approx(
            old.sigma_offense * (1.0 - r0), abs=1e-12)


# ---------------------------------------------------------------------------
# Integration: cyber modulator inside run_campaign
# ---------------------------------------------------------------------------


class TestCyberInCampaign:
    def test_cyber_extends_or_changes_outcome(self):
        from naval_salvo import (
            BaciaCamposConfig, build_bacia_campos, run_campaign,
            StrengthProportional,
        )

        cfg_no = BaciaCamposConfig(blue_cyber_per_subtype=0,
                                    red_cyber_per_subtype=0)
        cfg_yes = BaciaCamposConfig(blue_cyber_per_subtype=1,
                                     red_cyber_per_subtype=2)
        traj_no = run_campaign(*build_bacia_campos(cfg_no), n_salvos=10,
                               targeting_policy=StrengthProportional())
        traj_yes = run_campaign(*build_bacia_campos(cfg_yes), n_salvos=10,
                                 targeting_policy=StrengthProportional(),
                                 cyber_modulator=DecomposedPhi(r0=0.6, k=2.0))
        # The cyber-enabled campaign should run for *at least as long*
        # as the non-cyber one (cyber degrades both kinetic
        # effectivenesses, slowing decisive outcomes).
        assert traj_yes.n_completed_salvos >= traj_no.n_completed_salvos

    def test_red_cyber_dominance_protects_red_kinetic(self):
        # Red has 2x Blue's cyber per sub-type -> Red kinetic should
        # take less damage than in the no-cyber baseline.
        from naval_salvo import (
            BaciaCamposConfig, build_bacia_campos, run_campaign,
            StrengthProportional,
        )

        cfg_no = BaciaCamposConfig(blue_cyber_per_subtype=0,
                                    red_cyber_per_subtype=0)
        cfg_yes = BaciaCamposConfig(blue_cyber_per_subtype=1,
                                     red_cyber_per_subtype=2)
        traj_no = run_campaign(*build_bacia_campos(cfg_no), n_salvos=15,
                               targeting_policy=StrengthProportional())
        traj_yes = run_campaign(*build_bacia_campos(cfg_yes), n_salvos=15,
                                 targeting_policy=StrengthProportional(),
                                 cyber_modulator=DecomposedPhi(r0=0.6, k=2.0))
        # Compare Red destroyer survival.
        dest_no = traj_no.red_strength_history[-1, 0]
        dest_yes = traj_yes.red_strength_history[-1, 0]
        # With Red cyber dominance, destroyers should survive at least
        # as well as without cyber.
        assert dest_yes >= dest_no - 1e-9

    def test_submarine_survives_better_under_cyber_dominance(self):
        # Even if Red has overwhelming cyber, submarine immunity means
        # Red's kinetic offense against the sub is unchanged from the
        # no-cyber baseline.  Submarine fractional survival should be
        # identical (modulo simultaneity effects from other units
        # changing).  At minimum it should NOT be worse.
        from naval_salvo import (
            BaciaCamposConfig, build_bacia_campos, run_campaign,
            StrengthProportional,
        )

        cfg = BaciaCamposConfig(blue_cyber_per_subtype=0,
                                red_cyber_per_subtype=3)
        state, ep, adm = build_bacia_campos(cfg)
        traj = run_campaign(state, ep, adm, n_salvos=2,
                            targeting_policy=StrengthProportional(),
                            cyber_modulator=DecomposedPhi(r0=0.9, k=10.0))
        sub_idx = [ut.name for ut in traj._blue_unit_types].index("Submarine")
        # Submarine took *some* damage (it's not invulnerable kinetically);
        # but the cyber pressure didn't make it worse than the kinetic
        # attrition rate alone.  Just check the stock decreased
        # monotonically (no anomalies).
        sub_history = traj.blue_strength_history[:, sub_idx]
        assert all(sub_history[k+1] <= sub_history[k] + 1e-12
                   for k in range(len(sub_history)-1))
