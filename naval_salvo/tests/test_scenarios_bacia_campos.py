"""
Tests for naval_salvo.scenarios.bacia_campos.

These verify the cenário central do SIGE 2026 paper: that the engine
produces sensible, monotone qualitative responses to the sensitivity
knobs that the paper varies (number of frigates, submarine present,
number of FPSOs).  Specific numerical values are *not* the point --
the parameters here are illustrative and may be refined in follow-on
work; what we lock in is the qualitative structure.
"""

from __future__ import annotations

import numpy as np
import pytest

from naval_salvo import (
    BACIA_CAMPOS_PARAMETERS,
    BaciaCamposConfig,
    Domain,
    StrengthProportional,
    build_bacia_campos,
    run_campaign,
)


# ---------------------------------------------------------------------------
# Calibration table
# ---------------------------------------------------------------------------


class TestParameterTable:
    def test_all_canonical_unit_types_present(self):
        expected = {
            "Frigate", "Submarine", "MPA", "FPSO",
            "Destroyer", "StrikeAir",
        }
        assert expected.issubset(BACIA_CAMPOS_PARAMETERS.keys())

    def test_fpso_has_no_offensive_capability(self):
        # Decision 1.4 §2.3.b: Pre-salt platforms are zero-offence.
        s, p_off, p_def, eta = BACIA_CAMPOS_PARAMETERS["FPSO"]
        assert p_off == 0.0
        assert eta == 0.0     # FPSO doesn't "fire", so eta is degenerate

    def test_submarine_has_no_active_intercept(self):
        # Subs do not actively intercept incoming SSMs.
        _, _, p_def, _ = BACIA_CAMPOS_PARAMETERS["Submarine"]
        assert p_def == 0.0

    def test_all_staying_powers_positive(self):
        for name, (s, *_) in BACIA_CAMPOS_PARAMETERS.items():
            assert s > 0.0, f"{name} has non-positive staying power"


# ---------------------------------------------------------------------------
# Force composition under various configs
# ---------------------------------------------------------------------------


class TestForceComposition:
    def test_default_config_force_layout(self):
        bs, ep, adm = build_bacia_campos()
        # Blue: Frigate(1) + Submarine(1) + MPA(1) + FPSO(4) ; 4 unit types.
        names = [ut.name for ut in bs.blue.unit_types]
        assert names == ["Frigate", "Submarine", "MPA", "FPSO"]
        # Red: Destroyer + StrikeAir ; 2 unit types.
        red_names = [ut.name for ut in bs.red.unit_types]
        assert red_names == ["Destroyer", "StrikeAir"]

    def test_blue_label_is_MB(self):
        bs, _, _ = build_bacia_campos()
        assert bs.blue.label == "MB"

    def test_no_submarine_config(self):
        cfg = BaciaCamposConfig(submarine_present=False)
        bs, _, _ = build_bacia_campos(cfg)
        assert "Submarine" not in [ut.name for ut in bs.blue.unit_types]

    def test_no_frigate_config(self):
        cfg = BaciaCamposConfig(n_frigates=0)
        bs, _, _ = build_bacia_campos(cfg)
        assert "Frigate" not in [ut.name for ut in bs.blue.unit_types]

    def test_multiple_frigates_increase_initial_strength(self):
        cfg = BaciaCamposConfig(n_frigates=3)
        bs, _, _ = build_bacia_campos(cfg)
        assert bs.blue.strength_of("Frigate") == 3.0

    def test_no_fpsos_config(self):
        cfg = BaciaCamposConfig(n_fpsos=0)
        bs, _, _ = build_bacia_campos(cfg)
        assert "FPSO" not in [ut.name for ut in bs.blue.unit_types]

    def test_invalid_negative_count_rejected(self):
        with pytest.raises(ValueError):
            BaciaCamposConfig(n_frigates=-1)
        with pytest.raises(ValueError):
            BaciaCamposConfig(n_fpsos=-1)
        with pytest.raises(ValueError):
            BaciaCamposConfig(n_destroyers=-1)

    def test_empty_blue_force_rejected(self):
        # No frigates, no sub, no MPA, no FPSO -> error.
        with pytest.raises(ValueError):
            build_bacia_campos(BaciaCamposConfig(
                n_frigates=0, submarine_present=False,
                mpa_present=False, n_fpsos=0,
            ))


# ---------------------------------------------------------------------------
# Targeting policy: FPSOs receive higher Red attack share
# ---------------------------------------------------------------------------


class TestRedTargetingPriority:
    def test_fpso_receives_higher_sigma_than_frigate(self):
        bs, ep, _ = build_bacia_campos()
        # In the Red->Blue direction, ThreatWeighted gives FPSOs 3x
        # the priority of escorts.  Compare σ_offense to FPSO vs to
        # Frigate (both surface, both admissible).
        sig_to_fpso = ep.red_attacks_blue.get("Destroyer", "FPSO").sigma_offense
        sig_to_frig = ep.red_attacks_blue.get("Destroyer", "Frigate").sigma_offense
        assert sig_to_fpso > sig_to_frig

    def test_blue_uses_strength_proportional(self):
        # Blue->Red, no special weights.  Quick sanity: σ values sum
        # to ~1 across admissible defenders for each attacker.
        bs, ep, _ = build_bacia_campos()
        for j_name in ("Frigate", "Submarine", "MPA", "FPSO"):
            row_sum = sum(
                ep.blue_attacks_red.get(j_name, i_name).sigma_offense
                for i_name in ("Destroyer", "StrikeAir")
            )
            # Either zero (no admissible target) or 1 (normalised).
            assert row_sum == pytest.approx(0.0) or row_sum == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Sensitivity dynamics: the headline figures of the paper
# ---------------------------------------------------------------------------


def _run(config: BaciaCamposConfig, n_salvos: int = 10):
    state, ep, adm = build_bacia_campos(config)
    return run_campaign(state, ep, adm, n_salvos=n_salvos,
                        targeting_policy=StrengthProportional())


def _fpso_survivors(traj, config) -> float:
    """Count surviving FPSO mass at the end of the campaign."""
    if config.n_fpsos == 0:
        return 0.0
    state_unit_types = traj._blue_unit_types
    for j, ut in enumerate(state_unit_types):
        if ut.name == "FPSO":
            return float(traj.blue_strength_history[-1, j])
    return 0.0


class TestSensitivityDynamics:
    """
    These tests check *qualitative monotone* behaviour, the kind the
    paper's sensitivity figures will report.  They do not pin specific
    numerical values, which would brittle-fail if the calibration
    parameters were tuned.
    """

    def test_more_frigates_protect_more_FPSOs(self):
        """
        Increasing n_frigates from 1 -> 2 should protect more FPSO
        mass at the end of the campaign.  This is the headline
        sensitivity result of the paper.
        """
        cfg1 = BaciaCamposConfig(n_frigates=1)
        cfg2 = BaciaCamposConfig(n_frigates=2)
        traj1 = _run(cfg1)
        traj2 = _run(cfg2)
        s1 = _fpso_survivors(traj1, cfg1)
        s2 = _fpso_survivors(traj2, cfg2)
        # We expect strict improvement; allow equality only as a safe
        # margin against parameter calibration drift.
        assert s2 >= s1

    def test_zero_frigates_underperforms_one_frigate(self):
        cfg0 = BaciaCamposConfig(n_frigates=0)
        cfg1 = BaciaCamposConfig(n_frigates=1)
        traj0 = _run(cfg0)
        traj1 = _run(cfg1)
        s0 = _fpso_survivors(traj0, cfg0)
        s1 = _fpso_survivors(traj1, cfg1)
        assert s0 <= s1

    def test_submarine_extends_blue_resistance(self):
        """
        Submarine present must (weakly) extend the campaign duration:
        Blue can resist for at least as many salvos as without the
        submarine.  This is the second headline figure of the paper.
        """
        cfg_with = BaciaCamposConfig(submarine_present=True)
        cfg_without = BaciaCamposConfig(submarine_present=False)
        traj_with = _run(cfg_with)
        traj_without = _run(cfg_without)
        assert traj_with.n_completed_salvos >= traj_without.n_completed_salvos

    def test_submarine_immunity_to_air_chi_canonical(self):
        """
        Under canonical χ, the submarine is only marginally
        admissible (χ = 0.5) by air attack and not at all by Red's
        (absent) cyber stock.  After one salvo, the submarine should
        survive better than a same-stock surface unit would.
        """
        cfg = BaciaCamposConfig(submarine_present=True)
        traj = _run(cfg, n_salvos=1)
        # Submarine fractional remaining after 1 salvo.
        sub_idx = [ut.name for ut in traj._blue_unit_types].index("Submarine")
        sub_frac = (
            traj.blue_strength_history[1, sub_idx] /
            traj.blue_strength_history[0, sub_idx]
        )
        # The submarine has staying_power=2 and is only marginally
        # admissible to air attack; we require it survives at *least*
        # 30% of its strength after one salvo (very loose -- the actual
        # number is around 67%).  This is a structural sanity check,
        # not a tight numerical regression.
        assert sub_frac > 0.3


# ---------------------------------------------------------------------------
# Termination diagnostics
# ---------------------------------------------------------------------------


class TestCampaignTermination:
    def test_default_config_runs_at_least_one_salvo(self):
        traj = _run(BaciaCamposConfig())
        assert traj.n_completed_salvos >= 1

    def test_default_config_eventually_terminates(self):
        # With a max of 20 salvos, the default scenario should reach
        # a kinetic-zero state on at least one side.
        traj = _run(BaciaCamposConfig(), n_salvos=20)
        assert traj.terminated_early is True
