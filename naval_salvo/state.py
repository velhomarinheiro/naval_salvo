"""
state.py
========

Data structures representing the *state* of a heterogeneous multi-domain
salvo battle at a given instant in time.

The model carries, for each of the two sides (Blue and Red), a collection
of *unit types*.  Each unit type belongs to one of the five canonical
domains (S, U, A, C, X) and has

    A_i^{(d)}(t)            -- current strength (count or fractional count)
    A_i^{(d)}(0)            -- initial strength (used for normalisation /
                                staying-power calculations)
    s_i^{(d)}               -- staying power (varsigma in the canonical
                                notation; one parameter per unit type)
    label / metadata         -- a human-readable name and free-form tags
                                (eg. 'helicopter-as-S-subtype', 'plataforma-presal').

The numerical engine works on flat numpy vectors for speed; the data
classes here are the *human-friendly* layer that builds and maintains
those vectors.

Design choices
--------------
- Mutable strengths.  We use a separate ``UnitTypeState`` for the live
  current strength so that integrators (deterministic, stochastic,
  sequential) can update it in place during a salvo step.  The static
  parameters of each unit type live in ``UnitType``.
- One-side-at-a-time.  Force objects know which side they belong to
  (Blue or Red).  Cross-side coupling is handled by ``BattleState``.
- Numpy interop.  ``Force.strength_vector()`` and
  ``Force.set_strength_vector()`` give access to the raw numerical
  state for the dynamics modules.

References
----------
- Working document 1.4 §3 (state representation; intra-domain
  heterogeneity follows Hughes 1995 + Johns-Pilnick-Hughes 2001).
- The split between ``UnitType`` (static) and ``UnitTypeState`` (dynamic)
  mirrors the standard practice in agent-based / DES code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Iterator, Optional

import numpy as np

from .domains import (
    DOMAIN_ORDER,
    KINETIC_DOMAINS,
    Domain,
    parse_domain,
)


# ---------------------------------------------------------------------------
# Unit-type level
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UnitType:
    """
    Static description of one heterogeneous unit type.

    Attributes
    ----------
    name : str
        Human-readable identifier (eg. 'Tamandare-class frigate',
        'SBR-Riachuelo', 'P-3-equivalent MPA', 'X_C2-team-1').  Must be
        unique within a Force.
    domain : Domain
        Canonical domain to which this unit type belongs.
    staying_power : float
        Number of standard hits the unit type can absorb before being
        rendered combat-ineffective (varsigma in the canonical notation).
        Must be strictly positive.
    initial_strength : float
        Number of units of this type at t = 0.  Allowed to be a non-integer
        because some applications model fractional or aggregated stock.
        Must be non-negative.
    subtype : str, optional
        Free-form tag for sub-categorisation within a domain
        (eg. 'helicopter', 'USV', 'mine', 'pre-salt-platform', 'X_C2',
        'X_SEN').  Used by validation routines and by the cyber module
        which dispatches on cyber sub-type.

    Notes
    -----
    The class is frozen to emphasise that the *parameters* of a unit
    type do not change during a battle.  The *strength* of units of
    that type does change, and is held in :class:`UnitTypeState`.
    """

    name: str
    domain: Domain
    staying_power: float
    initial_strength: float
    subtype: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("UnitType.name must be a non-empty string.")
        if not isinstance(self.domain, Domain):
            # Allow string codes in user-facing constructors via parse.
            object.__setattr__(self, "domain", parse_domain(self.domain))
        if self.staying_power <= 0.0 or not np.isfinite(self.staying_power):
            raise ValueError(
                f"UnitType {self.name!r}: staying_power must be > 0 "
                f"and finite; got {self.staying_power}."
            )
        if self.initial_strength < 0.0 or not np.isfinite(self.initial_strength):
            raise ValueError(
                f"UnitType {self.name!r}: initial_strength must be >= 0 "
                f"and finite; got {self.initial_strength}."
            )


@dataclass
class UnitTypeState:
    """
    Mutable live state of a unit type during a battle.

    A thin wrapper around ``current_strength`` so that integrators can
    update it in place without rebinding attributes on the (frozen)
    :class:`UnitType`.  Keeps a reference to its parent :class:`UnitType`
    so that staying power, initial strength etc. are always one indirection
    away.
    """

    unit_type: UnitType
    current_strength: float

    def __post_init__(self) -> None:
        if self.current_strength < 0.0 or not np.isfinite(self.current_strength):
            raise ValueError(
                f"UnitTypeState for {self.unit_type.name!r}: "
                f"current_strength must be >= 0 and finite; "
                f"got {self.current_strength}."
            )

    @property
    def fractional_strength(self) -> float:
        """
        Current strength relative to initial strength (in [0, 1] for
        non-amplified models).  Returns 0.0 if the unit type started with
        zero strength.
        """
        s0 = self.unit_type.initial_strength
        if s0 <= 0.0:
            return 0.0
        return self.current_strength / s0


# ---------------------------------------------------------------------------
# Force level (one side)
# ---------------------------------------------------------------------------


@dataclass
class Force:
    """
    A complete description of one side (Blue or Red) of the engagement.

    A Force owns:

    - A list of :class:`UnitType` objects (static parameters).
    - A parallel list of :class:`UnitTypeState` objects (live strengths).
    - A ``label`` ('Blue' or 'Red' typically) for diagnostic output.

    The list ordering of unit types is fixed at construction time and
    never changes; this lets the dynamics modules cache index lookups
    safely.

    Parameters
    ----------
    label : str
        Side identifier, typically 'Blue' or 'Red'.
    unit_types : iterable of UnitType
        Unit types making up this force.  Names must be unique.

    Examples
    --------
    >>> from naval_salvo.domains import Domain
    >>> blue = Force(
    ...     label='Blue',
    ...     unit_types=[
    ...         UnitType('Frigate', Domain.SURFACE, staying_power=3,
    ...                  initial_strength=4),
    ...         UnitType('SSK',     Domain.UNDERWATER, staying_power=2,
    ...                  initial_strength=2),
    ...     ],
    ... )
    >>> blue.n_unit_types
    2
    >>> blue.strength_of('Frigate')
    4.0
    """

    label: str
    unit_types: list[UnitType] = field(default_factory=list)
    states: list[UnitTypeState] = field(init=False)

    # Cached lookup: name -> position in unit_types/states.  Built once
    # in __post_init__ and updated only by add_unit_type().
    _index_by_name: dict[str, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.label:
            raise ValueError("Force.label must be a non-empty string.")

        # Defensive copy so that callers can't mutate the list under us.
        self.unit_types = list(self.unit_types)

        names = [ut.name for ut in self.unit_types]
        if len(set(names)) != len(names):
            dup = [n for n in names if names.count(n) > 1]
            raise ValueError(
                f"Force {self.label!r}: duplicate unit type names: "
                f"{sorted(set(dup))}."
            )

        self.states = [
            UnitTypeState(unit_type=ut, current_strength=ut.initial_strength)
            for ut in self.unit_types
        ]
        self._index_by_name = {ut.name: i for i, ut in enumerate(self.unit_types)}

    # ---- collection methods ----------------------------------------------

    @property
    def n_unit_types(self) -> int:
        """Number of unit types in this force (across all domains)."""
        return len(self.unit_types)

    def add_unit_type(self, ut: UnitType) -> None:
        """Append a new unit type to the force, with the canonical initial state."""
        if ut.name in self._index_by_name:
            raise ValueError(
                f"Force {self.label!r}: unit type {ut.name!r} already present."
            )
        self.unit_types.append(ut)
        self.states.append(
            UnitTypeState(unit_type=ut, current_strength=ut.initial_strength)
        )
        self._index_by_name[ut.name] = len(self.unit_types) - 1

    def __iter__(self) -> Iterator[tuple[UnitType, UnitTypeState]]:
        """Iterate over (unit_type, state) pairs in insertion order."""
        return zip(self.unit_types, self.states)

    def __len__(self) -> int:
        return self.n_unit_types

    # ---- domain views -----------------------------------------------------

    def unit_types_in(self, domain: Domain | str) -> list[UnitType]:
        """Return all UnitTypes whose domain matches ``domain`` (insertion order)."""
        d = parse_domain(domain)
        return [ut for ut in self.unit_types if ut.domain is d]

    def indices_in(self, domain: Domain | str) -> list[int]:
        """Return the indices (in self.unit_types) of unit types in ``domain``."""
        d = parse_domain(domain)
        return [i for i, ut in enumerate(self.unit_types) if ut.domain is d]

    # ---- strength accessors ----------------------------------------------

    def strength_of(self, name: str) -> float:
        """Return the current strength of the unit type with the given name."""
        return self.states[self._index_by_name[name]].current_strength

    def set_strength_of(self, name: str, value: float) -> None:
        """Set the current strength of the named unit type (clipping at zero)."""
        if not np.isfinite(value):
            raise ValueError(f"strength must be finite; got {value}.")
        idx = self._index_by_name[name]
        self.states[idx].current_strength = max(0.0, float(value))

    def strength_vector(self) -> np.ndarray:
        """
        Return current strengths as a 1-D float64 array.

        Order follows ``self.unit_types``.  This is the canonical numerical
        representation used by the dynamics modules.
        """
        return np.array(
            [s.current_strength for s in self.states], dtype=np.float64
        )

    def set_strength_vector(self, vec: np.ndarray) -> None:
        """
        Update all current strengths from a 1-D array.

        Negative entries are clipped to 0 (canonical no-amplification
        rule, decision 1.4 §2.3.f).  Length must match ``n_unit_types``.
        """
        v = np.asarray(vec, dtype=np.float64)
        if v.shape != (self.n_unit_types,):
            raise ValueError(
                f"strength vector has shape {v.shape}; expected "
                f"({self.n_unit_types},)."
            )
        if not np.all(np.isfinite(v)):
            raise ValueError("strength vector contains non-finite values.")
        v = np.maximum(v, 0.0)
        for i, val in enumerate(v):
            self.states[i].current_strength = float(val)

    def initial_strength_vector(self) -> np.ndarray:
        """Return initial strengths as a 1-D float64 array."""
        return np.array(
            [ut.initial_strength for ut in self.unit_types], dtype=np.float64
        )

    def staying_power_vector(self) -> np.ndarray:
        """Return staying powers as a 1-D float64 array."""
        return np.array(
            [ut.staying_power for ut in self.unit_types], dtype=np.float64
        )

    # ---- summary ----------------------------------------------------------

    def total_strength_by_domain(self) -> dict[Domain, float]:
        """Return current total strength summed within each domain."""
        out: dict[Domain, float] = {d: 0.0 for d in DOMAIN_ORDER}
        for ut, st in zip(self.unit_types, self.states):
            out[ut.domain] += st.current_strength
        return out

    def is_combat_ineffective(self, threshold: float = 0.0) -> bool:
        """
        True if every unit type has current strength <= ``threshold``.

        Used by the solver as a stopping condition for the kinetic side.
        Note that a force can be cyber-degraded but kinetically intact
        and vice versa, so applications that distinguish the two should
        use ``total_strength_by_domain`` directly.
        """
        return all(st.current_strength <= threshold for st in self.states)


# ---------------------------------------------------------------------------
# Two-side battle state
# ---------------------------------------------------------------------------


@dataclass
class BattleState:
    """
    Live state of a Blue-vs-Red engagement.

    Holds the two :class:`Force` objects together with the current
    simulation time and a list of past salvo timestamps for diagnostic
    plotting and history.

    Attributes
    ----------
    blue : Force
        The Blue (own) force.
    red  : Force
        The Red (opponent) force.
    time : float
        Current simulation time, in the same units as t_char (typically
        a normalised dimensionless time).
    salvo_times : list of float
        Times at which a salvo (jump) has been applied so far.
    """

    blue: Force
    red: Force
    time: float = 0.0
    salvo_times: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.blue.label == self.red.label:
            raise ValueError(
                f"Blue and Red forces must have distinct labels; "
                f"both are {self.blue.label!r}."
            )
        if not np.isfinite(self.time):
            raise ValueError(f"BattleState.time must be finite; got {self.time}.")

    def force(self, side: str) -> Force:
        """Return the Force for ``'blue'`` or ``'red'`` (case-insensitive)."""
        s = side.strip().lower()
        if s == self.blue.label.lower():
            return self.blue
        if s == self.red.label.lower():
            return self.red
        if s == "blue":
            return self.blue
        if s == "red":
            return self.red
        raise KeyError(
            f"Unknown side {side!r}; expected one of "
            f"{self.blue.label!r}, {self.red.label!r}, 'blue', 'red'."
        )

    def opposing_force(self, side: str) -> Force:
        """Return the *opponent* of ``side``."""
        own = self.force(side)
        return self.red if own is self.blue else self.blue

    def record_salvo(self, t: float) -> None:
        """Append a salvo timestamp to the history."""
        self.salvo_times.append(float(t))

    def is_terminated(self) -> bool:
        """Either side combat-ineffective in *all* kinetic domains."""
        for force in (self.blue, self.red):
            kinetic_total = sum(
                st.current_strength
                for ut, st in zip(force.unit_types, force.states)
                if ut.domain.is_kinetic
            )
            if kinetic_total <= 0.0:
                return True
        return False
