"""
naval_salvo.validation.jph_coronel
==================================

Minute-by-minute reproduction of the Battle of Coronel (1 November 1914)
worked example from Johns, Pilnick & Hughes (2001), NPS-IJWA-01-010 §III.B.

This is the canonical *intra-domain heterogeneous* validation: both
forces are entirely on the surface domain, but each side has multiple
distinct unit types with different staying powers, combat powers, and
effectiveness parameters.  The model has to keep them straight cell by
cell -- which is exactly the path Step 3 of the implementation
exercises.

Beall data, JPH p.21 Table 1 (combat power per group):

    Group                            E_ji   ς (staying power)   duration (min)
    ---------------------------------------------------------------------
    (1) Good Hope + Monmouth         0.00   3.21                0  (silent)
    (2) Glasgow                      0.028  1.23                15
    (3) Scharnhorst + Gneisenau      0.028  3.330               28
    (4) Leipzig + Dresden            0.012  2.23                2

The "combat power per group" of JPH is the *aggregate* group total;
individual ship combat powers are then ρ_{ji} = group_total /
n_ships_in_attacker_group, which is what their worked example on p.22
uses (β_{ji} = 2.16 for Good Hope or Monmouth attacked by Sch+Gn:
4.32 / 2 = 2.16).

The paper reports, after 28 minutes:
    - Good Hope, Monmouth destroyed (ΔA_i ≈ 1)
    - Glasgow: ΔA_3 = 0.084  (16-percent loss reduces force to 0.916)
    - Scharnhorst + Gneisenau: ΔB_1 = 0.053  (force → 3.894)
    - Total British firepower loss:  94.76%
    - Total German firepower loss:    2.67%

Both totals are *firepower* losses (combat power weighted), not unit
counts.  We replicate the same arithmetic.

Worked first-minute example (JPH p.22):
    A_1 = Good Hope, attacked by Scharnhorst+Gneisenau (each engages
    half of A_1, half of A_2 → ψ = 0.5).  Leipzig+Dresden engage
    Glasgow, not GH/Mon → their ψ to GH/Mon is 0.

        ΔA_1 = 2 · (0.028 · 2.16 · 0.5 · 1.0) / 1.605
             = 2 · 0.0302 / 1.605
             ≈ 0.0377

    JPH report 0.037 (rounded to 3 d.p.).  Our test locks 0.0377...
    to machine precision.

References
----------
- Johns, Pilnick & Hughes (2001) NPS-IJWA-01-010 §III.B and p.22
  worked example.
- Beall (1990) "Naval gunnery and naval salvo combat data" -- original
  source of the per-group combat power and effectiveness numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..admissibility import Admissibility
from ..coefficients import EngagementBuilder
from ..domains import Domain
from ..parameters import EngagementParameters
from ..state import BattleState, Force, UnitType
from ..targeting import Manual


# ---------------------------------------------------------------------------
# Per-side parameter records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CoronelGroup:
    """
    Beall-style group record.

    Stores the group's *total* combat power and the staying power *per
    ship*; we then derive the per-ship offensive throughput by dividing
    by the number of ships in the group.
    """

    name: str
    n_ships: int
    group_combat_power: float    # β · n_ships  (Beall total)
    eta: float                   # offensive effectiveness E_ji (= σ τ ρ_p)
    staying_power_per_ship: float

    @property
    def per_ship_power(self) -> float:
        return self.group_combat_power / self.n_ships


# Canonical Beall numbers (JPH p.21 Table 1, p.22 worked text).
# Note that *individual* values reported by JPH are:
#   β_{Sch/Gn -> GH/Mon} = 4.32 / 2 = 2.16   ✓
#   β_{LP/DR -> Glasgow} = 4.33 / 2 = 2.165  ✓
#   β_{Glasgow -> any}   = 0.42  / 1 = 0.42   ✓
#   ε (eta) per group as listed below.
BRITISH_GROUPS: tuple[CoronelGroup, ...] = (
    # (1) Good Hope + Monmouth -- they fire zero (E = 0 in Beall data).
    CoronelGroup(
        name="Good Hope + Monmouth",
        n_ships=2,
        group_combat_power=7.27,
        eta=0.000,
        staying_power_per_ship=1.605,    # = 3.21 / 2
    ),
    # (2) Glasgow -- single ship, lower combat power, modest E.
    CoronelGroup(
        name="Glasgow",
        n_ships=1,
        group_combat_power=0.42,
        eta=0.028,
        staying_power_per_ship=1.23,
    ),
)

GERMAN_GROUPS: tuple[CoronelGroup, ...] = (
    # (3) Scharnhorst + Gneisenau -- the principal hitters.
    CoronelGroup(
        name="Scharnhorst + Gneisenau",
        n_ships=2,
        group_combat_power=4.32,
        eta=0.028,
        staying_power_per_ship=1.665,    # = 3.33 / 2
    ),
    # (4) Leipzig + Dresden -- engage Glasgow only.
    CoronelGroup(
        name="Leipzig + Dresden",
        n_ships=2,
        group_combat_power=4.33,
        eta=0.012,
        staying_power_per_ship=1.115,    # = 2.23 / 2
    ),
)


# ---------------------------------------------------------------------------
# Targeting schedule -- minute-by-minute ψ matrices (JPH §III.B)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CoronelTargetingMinute:
    """
    Targeting fractions for one specific minute of the battle.

    Convention: rows index attacker groups in the *same* canonical
    order used by ``coronel_forces()`` (Brit: GH+Mon, Glasgow; Ger:
    Sch+Gn, Lp+Dr).  The matrices are (n_attacker_groups,
    n_defender_groups), indexed [j, i] = "attacker j against defender
    i".  Entries are ψ ∈ [0, 1] as in JPH.
    """

    minute: int
    british_attacks_german: np.ndarray   # shape (2, 2)
    german_attacks_british: np.ndarray   # shape (2, 2)


def coronel_minute_one_targeting() -> CoronelTargetingMinute:
    """
    JPH p.22 first-minute targeting matrix (worked example).

    German side:
      - Scharnhorst+Gneisenau split their fire 50/50 between Good Hope
        and Monmouth.  Both ships are inside group 1 (GH+Mon) so the
        *group-level* targeting share is 1.0 against group 1, 0.0
        against Glasgow.
      - Leipzig+Dresden engage Glasgow only -> 0 against group 1, 1
        against Glasgow.

    British side:
      - In minute 1, JPH's worked example shows the British have not
        yet returned fire; they are silent.  Their targeting matrix is
        therefore zero.  (This matches the "duration of fire = 0" entry
        in Beall Table 1 for GH+Mon and a not-yet-15-min Glasgow.)
    """
    # Index 0 = Sch+Gn, 1 = Lp+Dr.  Index 0 = GH+Mon, 1 = Glasgow.
    german = np.array(
        [
            [1.0, 0.0],     # Sch+Gn -> GH+Mon  (full group share)
            [0.0, 1.0],     # Lp+Dr  -> Glasgow
        ],
        dtype=np.float64,
    )
    british = np.zeros((2, 2), dtype=np.float64)   # silent in minute 1
    return CoronelTargetingMinute(
        minute=1,
        british_attacks_german=british,
        german_attacks_british=german,
    )


# ---------------------------------------------------------------------------
# Force builder
# ---------------------------------------------------------------------------


def coronel_forces() -> tuple[Force, Force]:
    """
    Build the British (Blue) and German (Red) forces as a single unit
    type per *group* (not per ship).

    JPH 2001 use one combat unit per Beall group, with the per-ship
    parameters scaled up by group size.  We mirror that exactly: the
    "British" force has 2 unit types (group 1 and group 2), each with
    initial strength = n_ships_in_group, staying power per ship, and
    combat power per ship.

    Each British or German "unit" therefore literally is one ship, but
    the targeting σ matrix indexes *groups* of identically-typed ships.
    Initial strengths:
        GH+Mon : 2.0        Sch+Gn : 2.0
        Glasgow: 1.0        Lp+Dr  : 2.0
    """
    british = Force(
        label="British",
        unit_types=[
            UnitType(
                name=g.name,
                domain=Domain.SURFACE,
                staying_power=g.staying_power_per_ship,
                initial_strength=float(g.n_ships),
            )
            for g in BRITISH_GROUPS
        ],
    )
    german = Force(
        label="German",
        unit_types=[
            UnitType(
                name=g.name,
                domain=Domain.SURFACE,
                staying_power=g.staying_power_per_ship,
                initial_strength=float(g.n_ships),
            )
            for g in GERMAN_GROUPS
        ],
    )
    return british, german


# ---------------------------------------------------------------------------
# Engagement builder using the Manual targeting policy
# ---------------------------------------------------------------------------


def build_coronel_engagement(
    targeting: Optional[CoronelTargetingMinute] = None,
) -> tuple[BattleState, EngagementParameters, Admissibility]:
    """
    Assemble a Coronel engagement, given the targeting fractions for
    the current minute.

    If ``targeting`` is None, defaults to the minute-1 schedule
    (silent British, Sch+Gn -> GH+Mon, Lp+Dr -> Glasgow).

    Returns
    -------
    BattleState, EngagementParameters, Admissibility
    """
    if targeting is None:
        targeting = coronel_minute_one_targeting()

    british, german = coronel_forces()

    # Per-pair offensive throughput p_offense = β_{j,i}.  In Beall, β
    # is the *per-ship* combat power of attacker group j.  In our model
    # the kernel is σ · η · p · B_j; here B_j = 2 ships for groups with
    # 2 ships and 1 ship for Glasgow.  So we set p_offense = per-ship
    # combat power of the attacker, and the total fire is correctly
    # scaled by B_j.
    bar_p_off: dict[tuple[str, str], float] = {}     # British -> German
    bar_eta_off: dict[tuple[str, str], float] = {}
    for jg in BRITISH_GROUPS:
        for ig in GERMAN_GROUPS:
            bar_p_off[(jg.name, ig.name)] = jg.per_ship_power
            bar_eta_off[(jg.name, ig.name)] = jg.eta

    rab_p_off: dict[tuple[str, str], float] = {}     # German -> British
    rab_eta_off: dict[tuple[str, str], float] = {}
    for jg in GERMAN_GROUPS:
        for ig in BRITISH_GROUPS:
            rab_p_off[(jg.name, ig.name)] = jg.per_ship_power
            rab_eta_off[(jg.name, ig.name)] = jg.eta

    # Defensive parameters: JPH's Coronel reproduction uses pure gunfire
    # exchange (no missile defense).  Set p_defense = 0, η_def = 1.

    builder = (
        EngagementBuilder()
        .with_blue(british.unit_types, label="British")
        .with_red(german.unit_types, label="German")
        .with_throughput_blue_attacks_red(
            p_offense=bar_p_off, eta_offense=bar_eta_off,
        )
        .with_throughput_red_attacks_blue(
            p_offense=rab_p_off, eta_offense=rab_eta_off,
        )
        .with_admissibility(Admissibility.degenerate())   # only (S, S)
        .with_targeting_policy(Manual(
            sigma_offense=targeting.british_attacks_german.copy(),
            sigma_defense=np.zeros_like(targeting.british_attacks_german),
        ))
    )
    ep = builder.build()
    # Builder applied the British->German σ.  We need to override the
    # German->British direction with its own Manual matrix.
    from ..coefficients import _refresh_sigma
    rab_new = _refresh_sigma(
        ep.red_attacks_blue,
        sigma_off=targeting.german_attacks_british.copy(),
        sigma_def=np.zeros_like(targeting.german_attacks_british),
    )
    ep = EngagementParameters(
        blue=ep.blue,
        red=ep.red,
        blue_attacks_red=ep.blue_attacks_red,
        red_attacks_blue=rab_new,
        t_char=ep.t_char,
        rho=ep.rho,
    )

    bs = BattleState(blue=ep.blue, red=ep.red)
    adm = Admissibility.degenerate()
    return bs, ep, adm


# ---------------------------------------------------------------------------
# Analytical reference values
# ---------------------------------------------------------------------------


def jph_minute_one_delta_good_hope() -> float:
    """
    JPH p.22 worked first-minute attrition on Good Hope (group 1, ship 1).

    From the paper, expanded:

        ΔA_1 = (Sch term) + (Gn term) + (Lp term) + (Dr term)

    Sch contributes:  ε · β · ψ · 1.0 / ς
                   = 0.028 · 2.16 · 0.5 · 1.0 / 1.605
    Gn contributes:  same
    Lp, Dr contribute zero because their ψ to GH+Mon is 0.

    Returns the analytical scalar (≈ 0.0377...).

    NOTE: JPH Equations are written *per ship* of A_1, so the result
    is a per-ship attrition (ΔA_1 ≈ 0.037).  Our model reads the
    matching kernel as *total hits* on the GH+Mon group (2 ships); to
    compare with JPH we therefore divide by the group size.
    """
    eps = 0.028
    beta = 2.16
    psi = 0.5
    sigma = 1.605
    return 2 * (eps * beta * psi * 1.0) / sigma
