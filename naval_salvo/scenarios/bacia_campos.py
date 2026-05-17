"""
naval_salvo.scenarios.bacia_campos
==================================

Defesa da Bacia de Campos -- the central case study of the SIGE 2026
paper.

Scope (Phase 1 doc 1.4 §5, Step 4 of the implementation plan):
    Blue (Brazilian Navy / defender):
        - 1-2 frigates (Tamandare-class equivalent)         domain S
        - 1 conventional submarine (Riachuelo-class)        domain U
        - 1 maritime patrol aircraft (P-3AM equivalent)     domain A
        - 4 FPSO platforms                                   domain S, sub-type
                                                              "FPSO" with
                                                              p_offense = 0.
    Red (generic adversary):
        - 2-3 surface combatants (destroyer-class)          domain S
        - 1 strike aircraft squadron (with SSMs)            domain A
        - (cyber capability lands in Step 5)

The scenario module exports:

- ``BaciaCamposConfig``         dataclass collecting the *force
                                 sizing* knobs that the sensitivity
                                 analysis varies (number of frigates,
                                 submarine present yes/no, number of
                                 FPSOs, etc.).
- ``build_bacia_campos()``      assembles a fully-specified
                                 ``BattleState`` + ``EngagementParameters``
                                 + ``Admissibility`` from a config.
- ``BACIA_CAMPOS_PARAMETERS``   the calibrated per-pair parameters as
                                 a dictionary.  Documented per-cell
                                 with provenance so the paper's
                                 parameter table can be generated
                                 directly from this constant.

Parameter calibration sources
-----------------------------
The published salvo-equation literature gives consistent ranges that
we adopt here:

- **Frigate (Tamandare/Niteroi-modernised) ~3500-6000 ton**:
    staying_power x = 2.0   (Hughes 1995, Christiansen 2008 LCS/NSC)
    p_offense β   = 4.0     (4 SSMs per salvo with p_hit = 1 baseline)
    p_defense z   = 2.0     (point defense + CIWS; from JPH §IV)
    eta            = 0.85    (well-trained crew, modern systems)

- **Conventional submarine (Riachuelo-class)**:
    staying_power = 2.0     (similar to non-frigate combatants)
    p_offense     = 2.0     (heavy torpedo salvo)
    p_defense     = 0.0     (subs do not actively intercept SSMs)
    eta           = 0.90    (high stealth, training premium)

- **Maritime patrol aircraft (P-3AM / equivalent)**:
    staying_power = 1.0     (single hit puts MPA out)
    p_offense     = 1.5     (modest ASW/anti-surface payload)
    p_defense     = 0.0
    eta           = 0.70    (long mission cycles, lower availability)

- **FPSO platform (Pre-salt)**:
    staying_power = 3.0     (large stationary platform; realistic
                              that 3 missile hits = mission-kill,
                              consistent with FPSO survivability
                              studies for Bacia de Campos)
    p_offense     = 0.0     (NO offensive capability by construction;
                              decision 1.4 §2.3.b)
    p_defense     = 0.0     (no native point defense; protection
                              depends on χ-coupling with escort
                              frigates -- a known limitation of
                              the canonical salvo formulation that
                              the paper discusses explicitly)
    eta           = 0.0     (degenerate: FPSO is a target, not a unit)

- **Generic destroyer (adversary)**:
    staying_power = 3.0     (somewhat tougher than frigate)
    p_offense     = 4.0     (4 SSMs; lower than peer-NATO assumption
                              because the scenario assumes regional
                              adversary rather than great-power peer)
    p_defense     = 2.0     (modern but not Aegis-class)
    eta           = 0.80    (peer-competitor training)

- **Adversary strike aircraft squadron**:
    staying_power = 1.0     (one good SAM hit = kill)
    p_offense     = 2.5     (SSM-armed strike package, modest payload)
    p_defense     = 0.0
    eta           = 0.75

- **Cyber units (X domain, sub-typed C2/SEN/WPN/LOG)**:
    staying_power = 1.0     (cyber teams have no kinetic vulnerability;
                             attrited only by intra-X cyber attack)
    p_offense     = 0.5     (modest direct attrition on opponent's
                             cyber stock per salvo)
    p_defense     = 0.5     (counter-cyber capacity)
    eta           = 1.0
    Note: cyber's main effect is *not* kinetic attrition (which the
    χ matrix mostly nulls or marginalises) but the Φ modulator that
    re-scales σ and η of opponent's kinetic units.

These are *illustrative* values for proof-of-concept; the SIGE 2026
paper notes they will be refined through MB doctrinal sources in
follow-on work.  What the scenario demonstrates is *qualitative*:
the multi-domain coupling effects that single-domain (homogeneous or
JPH 2001 surface-only) models cannot capture.

References
----------
- Phase 1 working document 1.4 §5 (cenario specification).
- Christiansen (2008) "Cost-Effective Procurement of Distributed
  Lethality Capabilities" -- LCS/NSC parameter table (NPS thesis).
- Casola (2017) NPS thesis -- MDUSV scenario task force composition,
  p.43 Table 2 (canonical staying/offensive/defensive ranges).
- 369154011020210128.pdf (PKR105 / Iver Huitfeldt comparison) --
  frigate-class parameter ranges (ATP-31B Above Water Warfare manual).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..admissibility import Admissibility, canonical_matrix
from ..coefficients import EngagementBuilder
from ..domains import Domain
from ..parameters import EngagementParameters
from ..state import BattleState, Force, UnitType
from ..targeting import StrengthProportional, ThreatWeighted, Uniform


# ---------------------------------------------------------------------------
# Calibrated parameters (single source of truth for the paper table)
# ---------------------------------------------------------------------------

#: Calibrated unit-type parameter dictionary.  Each entry is a tuple
#: (staying_power, p_offense, p_defense, eta).  Values rationale is in
#: the module docstring above.
BACIA_CAMPOS_PARAMETERS: dict[str, tuple[float, float, float, float]] = {
    # name                  s,    p_off,  p_def,  eta
    "Frigate":             (2.0,  4.0,    2.0,    0.85),
    "Submarine":           (2.0,  2.0,    0.0,    0.90),
    "MPA":                 (1.0,  1.5,    0.0,    0.70),
    "FPSO":                (3.0,  0.0,    0.0,    0.00),
    "Destroyer":           (3.0,  4.0,    2.0,    0.80),
    "StrikeAir":           (1.0,  2.5,    0.0,    0.75),
    # Cyber unit types (X domain, sub-typed by suffix).  Same params
    # for all four sub-types under the canonical baseline; calibration
    # may differentiate them in future work.
    "Cyber-C2":            (1.0,  0.5,    0.5,    1.00),
    "Cyber-SEN":           (1.0,  0.5,    0.5,    1.00),
    "Cyber-WPN":           (1.0,  0.5,    0.5,    1.00),
    "Cyber-LOG":           (1.0,  0.5,    0.5,    1.00),
}


# ---------------------------------------------------------------------------
# Scenario configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BaciaCamposConfig:
    """
    Force-sizing knobs for the Bacia de Campos scenario.

    Defaults match the canonical Phase 1 specification (1 frigate,
    1 submarine, 1 MPA, 4 FPSOs vs 2 destroyers + 1 strike-air group).
    The sensitivity analysis varies each knob.

    Attributes
    ----------
    n_frigates : int
        Number of Tamandare-equivalent frigates.  Sensitivity: 0..3.
    submarine_present : bool
        Whether the Riachuelo-class submarine is on station.
        Sensitivity: True / False (the headline figure).
    mpa_present : bool
        Whether the patrol aircraft is on station.  Default True.
    n_fpsos : int
        Number of FPSO platforms (the protected assets).  Default 4.
    n_destroyers : int
        Adversary surface combatants.  Default 2.
    strike_air_present : bool
        Whether adversary strike-air sortie is launched.  Default True.
    blue_cyber_per_subtype : int
        Number of cyber teams of each sub-type (C2/SEN/WPN/LOG) on
        the Blue side.  0 (default) means Blue has no cyber stock;
        the sensitivity analysis varies this.
    red_cyber_per_subtype : int
        Same, for the Red side.  Defaults to a modest baseline (1)
        reflecting the Phase-1 specification of "modest Blue cyber,
        significant Red cyber" -- but the *significant* asymmetry is
        better captured by setting Blue=0 and Red=2 in the comparison
        runs.
    use_canonical_admissibility : bool
        If True, use ``Admissibility.canonical()``.  If False, use a
        scenario-tuned matrix that, in particular, sets χ(A→U) = 0.5
        (MPA can attack submarines but only marginally) and χ(S→U) =
        0.3 (frigates have limited ASW reach).
    chi : float
        χ value used for marginal cells of the canonical admissibility
        matrix when ``use_canonical_admissibility`` is True.
    """

    n_frigates: int = 1
    submarine_present: bool = True
    mpa_present: bool = True
    n_fpsos: int = 4
    n_destroyers: int = 2
    strike_air_present: bool = True
    blue_cyber_per_subtype: int = 0
    red_cyber_per_subtype: int = 0
    use_canonical_admissibility: bool = True
    chi: float = 0.5

    def __post_init__(self) -> None:
        if self.n_frigates < 0:
            raise ValueError(f"n_frigates must be >= 0; got {self.n_frigates}")
        if self.n_fpsos < 0:
            raise ValueError(f"n_fpsos must be >= 0; got {self.n_fpsos}")
        if self.n_destroyers < 0:
            raise ValueError(
                f"n_destroyers must be >= 0; got {self.n_destroyers}")
        if self.blue_cyber_per_subtype < 0:
            raise ValueError(
                f"blue_cyber_per_subtype must be >= 0; "
                f"got {self.blue_cyber_per_subtype}")
        if self.red_cyber_per_subtype < 0:
            raise ValueError(
                f"red_cyber_per_subtype must be >= 0; "
                f"got {self.red_cyber_per_subtype}")
        if not 0.0 <= self.chi <= 1.0:
            raise ValueError(f"chi must be in [0, 1]; got {self.chi}")


# ---------------------------------------------------------------------------
# Scenario builder
# ---------------------------------------------------------------------------


def build_bacia_campos(
    config: Optional[BaciaCamposConfig] = None,
) -> tuple[BattleState, EngagementParameters, Admissibility]:
    """
    Assemble the full Bacia de Campos engagement.

    Returns
    -------
    state : BattleState
    params : EngagementParameters
    adm : Admissibility

    Notes
    -----
    Targeting policy: Red attacks Blue using ``ThreatWeighted`` with
    FPSOs receiving 3x the priority of frigates / submarine / MPA.
    Blue attacks Red using ``StrengthProportional`` (defenders simply
    react to the most numerous threat).
    """
    if config is None:
        config = BaciaCamposConfig()

    blue, red = _build_forces(config)

    # Per-pair throughput dictionaries.
    bar_p_off, bar_eta_off = _build_directional_throughputs(blue, red)
    rab_p_off, rab_eta_off = _build_directional_throughputs(red, blue)
    bar_p_def, bar_eta_def = _build_directional_defenses(blue, red)
    rab_p_def, rab_eta_def = _build_directional_defenses(red, blue)

    # Admissibility.
    if config.use_canonical_admissibility:
        adm = Admissibility(canonical_matrix(chi=config.chi))
    else:
        adm = Admissibility.canonical()

    # Build red threat weights: FPSOs get the highest priority because
    # they are the operationally meaningful target; military escorts
    # are secondary.
    red_attack_weights = _build_red_threat_weights(blue)

    # Build with no global targeting policy, then apply per-direction
    # policies separately (each direction has different semantics:
    # Red prioritises FPSOs via ThreatWeighted, Blue reacts to numerous
    # threats via StrengthProportional).
    builder = (
        EngagementBuilder()
        .with_blue(blue.unit_types, label=blue.label)
        .with_red(red.unit_types, label=red.label)
        .with_throughput_blue_attacks_red(
            p_offense=bar_p_off,
            eta_offense=bar_eta_off,
            p_defense=bar_p_def,
            eta_defense=bar_eta_def,
        )
        .with_throughput_red_attacks_blue(
            p_offense=rab_p_off,
            eta_offense=rab_eta_off,
            p_defense=rab_p_def,
            eta_defense=rab_eta_def,
        )
        .with_admissibility(adm)
    )
    ep = builder.build()

    # Now overlay per-direction targeting:
    #   Blue -> Red: StrengthProportional (defenders react to the
    #                most numerous attacker)
    #   Red -> Blue: ThreatWeighted (attacker prioritises FPSOs 3x)
    from ..coefficients import _refresh_sigma
    sig_o_bar, sig_d_bar = StrengthProportional().compute(blue, red, adm)
    sig_o_rab, sig_d_rab = ThreatWeighted(
        offensive_weights=red_attack_weights
    ).compute(red, blue, adm)
    bar_new = _refresh_sigma(ep.blue_attacks_red, sig_o_bar, sig_d_bar)
    rab_new = _refresh_sigma(ep.red_attacks_blue, sig_o_rab, sig_d_rab)
    ep = EngagementParameters(
        blue=ep.blue,
        red=ep.red,
        blue_attacks_red=bar_new,
        red_attacks_blue=rab_new,
        t_char=ep.t_char,
        rho=ep.rho,
    )

    state = BattleState(blue=ep.blue, red=ep.red)
    return state, ep, adm


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_forces(config: BaciaCamposConfig) -> tuple[Force, Force]:
    """Assemble the two Force objects from a BaciaCamposConfig."""
    blue_unit_types: list[UnitType] = []

    # Frigates (zero or more).
    s, p_off, p_def, eta = BACIA_CAMPOS_PARAMETERS["Frigate"]
    if config.n_frigates > 0:
        blue_unit_types.append(UnitType(
            name="Frigate",
            domain=Domain.SURFACE,
            staying_power=s,
            initial_strength=float(config.n_frigates),
        ))

    # Submarine.
    if config.submarine_present:
        s, p_off, p_def, eta = BACIA_CAMPOS_PARAMETERS["Submarine"]
        blue_unit_types.append(UnitType(
            name="Submarine",
            domain=Domain.UNDERWATER,
            staying_power=s,
            initial_strength=1.0,
        ))

    # MPA.
    if config.mpa_present:
        s, p_off, p_def, eta = BACIA_CAMPOS_PARAMETERS["MPA"]
        blue_unit_types.append(UnitType(
            name="MPA",
            domain=Domain.AIR,
            staying_power=s,
            initial_strength=1.0,
        ))

    # FPSO platforms (zero-offence surface sub-type).
    if config.n_fpsos > 0:
        s, p_off, p_def, eta = BACIA_CAMPOS_PARAMETERS["FPSO"]
        blue_unit_types.append(UnitType(
            name="FPSO",
            domain=Domain.SURFACE,
            staying_power=s,
            initial_strength=float(config.n_fpsos),
            subtype="pre-salt-platform",
        ))

    # Cyber units (4 sub-types, one Force-level UnitType each).
    if config.blue_cyber_per_subtype > 0:
        for sub_code, name in [("C2",  "Cyber-C2"),
                               ("SEN", "Cyber-SEN"),
                               ("WPN", "Cyber-WPN"),
                               ("LOG", "Cyber-LOG")]:
            s, p_off, p_def, eta = BACIA_CAMPOS_PARAMETERS[name]
            blue_unit_types.append(UnitType(
                name=name,
                domain=Domain.CYBER,
                staying_power=s,
                initial_strength=float(config.blue_cyber_per_subtype),
                subtype=sub_code,
            ))

    if not blue_unit_types:
        raise ValueError(
            "BaciaCamposConfig produced an empty Blue force; need "
            "at least one defender or one FPSO."
        )

    blue = Force(label="MB", unit_types=blue_unit_types)

    # Red side.
    red_unit_types: list[UnitType] = []
    if config.n_destroyers > 0:
        s, p_off, p_def, eta = BACIA_CAMPOS_PARAMETERS["Destroyer"]
        red_unit_types.append(UnitType(
            name="Destroyer",
            domain=Domain.SURFACE,
            staying_power=s,
            initial_strength=float(config.n_destroyers),
        ))
    if config.strike_air_present:
        s, p_off, p_def, eta = BACIA_CAMPOS_PARAMETERS["StrikeAir"]
        red_unit_types.append(UnitType(
            name="StrikeAir",
            domain=Domain.AIR,
            staying_power=s,
            initial_strength=4.0,                 # squadron of 4 aircraft
        ))
    if config.red_cyber_per_subtype > 0:
        for sub_code, name in [("C2",  "Cyber-C2"),
                               ("SEN", "Cyber-SEN"),
                               ("WPN", "Cyber-WPN"),
                               ("LOG", "Cyber-LOG")]:
            s, p_off, p_def, eta = BACIA_CAMPOS_PARAMETERS[name]
            red_unit_types.append(UnitType(
                name=name,
                domain=Domain.CYBER,
                staying_power=s,
                initial_strength=float(config.red_cyber_per_subtype),
                subtype=sub_code,
            ))

    if not red_unit_types:
        raise ValueError(
            "BaciaCamposConfig produced an empty Red force; need "
            "at least one attacker."
        )

    red = Force(label="Adversario", unit_types=red_unit_types)
    return blue, red


def _build_directional_throughputs(
    attacker: Force, defender: Force
) -> tuple[dict, dict]:
    """Build (p_offense, eta_offense) dicts for one direction."""
    p_off: dict[tuple[str, str], float] = {}
    eta_off: dict[tuple[str, str], float] = {}
    for atk in attacker.unit_types:
        s_a, p_off_a, p_def_a, eta_a = BACIA_CAMPOS_PARAMETERS[atk.name]
        for defn in defender.unit_types:
            p_off[(atk.name, defn.name)] = p_off_a
            eta_off[(atk.name, defn.name)] = eta_a
    return p_off, eta_off


def _build_directional_defenses(
    attacker: Force, defender: Force
) -> tuple[dict, dict]:
    """Build (p_defense, eta_defense) dicts for one direction."""
    p_def_d: dict[tuple[str, str], float] = {}
    eta_def_d: dict[tuple[str, str], float] = {}
    for atk in attacker.unit_types:
        for defn in defender.unit_types:
            s_d, p_off_d, p_def_d_val, eta_d = BACIA_CAMPOS_PARAMETERS[
                defn.name
            ]
            p_def_d[(atk.name, defn.name)] = p_def_d_val
            eta_def_d[(atk.name, defn.name)] = eta_d
    return p_def_d, eta_def_d


def _build_red_threat_weights(blue: Force) -> np.ndarray:
    """
    Per-defender threat weights expressing the Red attacker's priorities.

    FPSOs are the protected operational target -- 3x weight.  Military
    escorts get baseline 1.0.
    """
    weights = np.empty(blue.n_unit_types, dtype=np.float64)
    for j, ut in enumerate(blue.unit_types):
        if ut.name == "FPSO":
            weights[j] = 3.0
        else:
            weights[j] = 1.0
    return weights
