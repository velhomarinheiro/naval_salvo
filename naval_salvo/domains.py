"""
domains.py
==========

Defines the canonical five domains of the heterogeneous multi-domain salvo
model and provides helper utilities for indexing and validation.

Phase 1 specification (document 1.4 §2):
    D = {S, U, A, C, X}
    where:
        S  -- Surface       (surface combatants, USVs, helicopters carried,
                              surface mines, Pre-Salt platforms)
        U  -- Underwater    (submarines, UUVs, submarine mines)
        A  -- Air           (aircraft, UAVs)
        C  -- Coastal       (coastal artillery, anti-ship missile batteries,
                              coastal mines)
        X  -- Cyber-EM      (cyber/electromagnetic effects; sub-types
                              X_C2, X_SEN, X_WPN, X_LOG)

The set is fixed and ordered (S, U, A, C, X). The ordering matters because
many objects in the model are indexed by domain (eg. admissibility matrix,
state vectors). This module centralises that ordering so that the rest of
the code base does not need to hard-code it.

References
----------
- Johns, Pilnick & Hughes (2001), NPS-IJWA-01-010, "Heterogeneous Salvo
  Model for the Navy After Next" -- baseline 2-force, 1-domain (kinetic
  surface) heterogeneous model that this module generalises.
- Hausken & Moxnes (2026), Annals of OR 357:1003-1019 -- multi-equipment
  Lanchester model with proportional reallocation, motivating the
  cross-domain coupling through the admissibility matrix.
- Working document 1.4, §2.1 (canonical domain set) and §2.2
  (admissibility matrix structure).
"""

from __future__ import annotations

from enum import Enum
from typing import Final


class Domain(str, Enum):
    """
    Canonical domains of the heterogeneous multi-domain salvo model.

    Inherits from ``str`` so that instances behave like their string codes
    in serialisation, dictionary keys, logging and error messages, while
    still benefiting from the type safety of an Enum.

    Members
    -------
    SURFACE   ('S') : Surface domain.  Surface combatants (frigates,
                      corvettes, destroyers, USVs).  Helicopters embarked
                      on a mother-ship are *folded into* the parent unit's
                      offensive/defensive capability and do not appear as
                      a separate domain (decision 1.4 §2.3.b).  Pre-salt
                      platforms are sub-types of S with zero offensive
                      capability and reduced staying power.
    UNDERWATER('U') : Submarines and UUVs.  Immune to cyber by construction:
                      delta_X = 0 for all U-typed units (decision 1.4
                      §2.3.d).  Submarine mines are a U sub-type.
    AIR       ('A') : Aircraft, UAVs.  Highest combat tempo (Hausken-Moxnes
                      2026 motivates t_char^A << t_char^S).
    COASTAL   ('C') : Coastal anti-ship missile batteries, coastal artillery,
                      coastal mines (sub-types of C in this model).
    CYBER     ('X') : Cyber-electromagnetic effects.  Decomposed internally
                      into sub-types X_C2, X_SEN, X_WPN, X_LOG with
                      *intra-X* lethal attrition only (no regeneration:
                      rho_X = 0).  Effects on the four kinetic domains
                      flow through the modulators delta_offense,
                      delta_defense rather than through direct attrition
                      (decision 1.4 §2.3.c).
    """

    SURFACE = "S"
    UNDERWATER = "U"
    AIR = "A"
    COASTAL = "C"
    CYBER = "X"

    @property
    def is_kinetic(self) -> bool:
        """
        True for the four kinetic domains (S, U, A, C); False for X.

        The kinetic / non-kinetic distinction is central to the model
        because cyber acts on the kinetic domains *multiplicatively*
        (through delta_offense, delta_defense modulators) rather than
        through direct hit-attrition.
        """
        return self is not Domain.CYBER

    @property
    def index(self) -> int:
        """
        Position of this domain in the canonical ordering (0 .. 4).

        Useful for indexing into the admissibility matrix and other
        per-domain structures stored as numpy arrays.
        """
        return DOMAIN_ORDER.index(self)


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

#: Canonical ordering of the five domains.  Frozen tuple so that downstream
#: code can rely on a stable index.  The ordering follows document 1.4 §2.1.
DOMAIN_ORDER: Final[tuple[Domain, ...]] = (
    Domain.SURFACE,
    Domain.UNDERWATER,
    Domain.AIR,
    Domain.COASTAL,
    Domain.CYBER,
)

#: Number of domains.  Defined as a constant so that array allocations
#: throughout the package can refer to it by name rather than the magic
#: number 5.
N_DOMAINS: Final[int] = len(DOMAIN_ORDER)

#: The four kinetic (non-cyber) domains, in canonical order.
KINETIC_DOMAINS: Final[tuple[Domain, ...]] = tuple(
    d for d in DOMAIN_ORDER if d.is_kinetic
)


# ---------------------------------------------------------------------------
# Cyber sub-types
# ---------------------------------------------------------------------------


class CyberSubtype(str, Enum):
    """
    Sub-types of the cyber-electromagnetic domain X.

    Decision 1.4 §2.3.c: cyber capabilities are decomposed into four
    functional sub-types whose targeting is biased toward distinct
    components of the opponent's kill chain.  Within X, attrition is
    purely intra-domain and lethal (no regeneration; rho_X = 0).

    Members
    -------
    C2  : Command-and-control disruption.  Primary target of
          opportunity is the opponent's tactical decision loop
          (degrades sigma_offense and sigma_defense aiming
          parameters).
    SEN : Sensor/ISR degradation (jamming, spoofing, blinding).
          Primarily reduces offensive scouting/aiming
          (sigma_offense).
    WPN : Weapon-system attack (firing-chain interference,
          guidance disruption).  Primarily reduces effective
          offensive coefficient (eta_offense).
    LOG : Logistics / sustainment cyber attack (degrades
          regeneration rho on the kinetic side).
    """

    C2 = "C2"
    SENSOR = "SEN"
    WEAPON = "WPN"
    LOGISTICS = "LOG"


CYBER_SUBTYPES: Final[tuple[CyberSubtype, ...]] = tuple(CyberSubtype)


# ---------------------------------------------------------------------------
# Convenience parsers
# ---------------------------------------------------------------------------


def parse_domain(value: str | Domain) -> Domain:
    """
    Coerce a string code or Domain instance to a Domain.

    Accepts either the single-letter canonical code ('S', 'U', 'A', 'C',
    'X') or the long member name ('SURFACE', 'UNDERWATER', ...).  Case is
    ignored.

    Parameters
    ----------
    value : str or Domain
        The value to coerce.

    Returns
    -------
    Domain
        The corresponding Domain enum member.

    Raises
    ------
    ValueError
        If ``value`` does not correspond to any Domain.

    Examples
    --------
    >>> parse_domain('S') is Domain.SURFACE
    True
    >>> parse_domain('surface') is Domain.SURFACE
    True
    >>> parse_domain(Domain.AIR) is Domain.AIR
    True
    """
    if isinstance(value, Domain):
        return value
    if not isinstance(value, str):
        raise ValueError(
            f"Cannot parse Domain from object of type {type(value).__name__}"
        )

    key = value.strip().upper()
    # Try the short code first (Domain.SURFACE.value == 'S').
    for d in DOMAIN_ORDER:
        if d.value == key:
            return d
    # Then the long name.
    try:
        return Domain[key]
    except KeyError as exc:
        valid = [d.value for d in DOMAIN_ORDER] + [d.name for d in DOMAIN_ORDER]
        raise ValueError(
            f"Unknown domain '{value}'. Valid values: {valid}."
        ) from exc
