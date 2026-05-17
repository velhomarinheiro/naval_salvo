"""
naval_salvo.coefficients
========================

High-level builder for ``EngagementParameters``.

The Step-1 and Step-2 layers are deliberately low-level: every
``PairParameters`` is set cell by cell.  That works fine for the
2-vs-2 worked examples and the 1-vs-1 Hughes recovery, but it scales
badly: a Bacia de Campos scenario with five domains and ~10 unit
types per side has 100 cells per direction.

This module provides:

- ``EngagementBuilder``    fluent-API constructor for the full set of
                            EngagementParameters of a two-side
                            engagement.
- ``apply_targeting_policy`` write the σ values produced by a
                            ``TargetingPolicy`` (see ``targeting.py``)
                            into one direction of an existing
                            ``EngagementParameters``.

References
----------
- Working document 1.4 §4 (architecture), section "coefficients.py"
  — wraps σ, η and p into the per-pair throughputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Union

import numpy as np

from .admissibility import Admissibility
from .domains import Domain
from .parameters import (
    DirectionalParameters,
    EngagementParameters,
    PairParameters,
)
from .state import Force, UnitType
from .targeting import TargetingPolicy


# ---------------------------------------------------------------------------
# Per-direction "throughput grid" -- the η, p inputs the user actually
# specifies.  σ is filled in separately via a TargetingPolicy.
# ---------------------------------------------------------------------------


@dataclass
class ThroughputGrid:
    """
    Per-(attacker, defender) effectiveness and throughput, *without* σ.

    Attributes
    ----------
    eta_offense, eta_defense : ndarray, (n_attacker, n_defender)
        Composite offensive / defensive effectiveness in [0, 1].
    p_offense, p_defense : ndarray, (n_attacker, n_defender)
        Offensive / defensive throughput (hits or intercepts per unit
        per salvo).  Non-negative, otherwise unbounded.
    """

    eta_offense: np.ndarray
    eta_defense: np.ndarray
    p_offense: np.ndarray
    p_defense: np.ndarray

    @classmethod
    def zeros(cls, n_attacker: int, n_defender: int) -> "ThroughputGrid":
        shape = (n_attacker, n_defender)
        return cls(
            eta_offense=np.ones(shape, dtype=np.float64),
            eta_defense=np.ones(shape, dtype=np.float64),
            p_offense=np.zeros(shape, dtype=np.float64),
            p_defense=np.zeros(shape, dtype=np.float64),
        )

    def shape(self) -> tuple[int, int]:
        return self.p_offense.shape


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


@dataclass
class EngagementBuilder:
    """
    Fluent-API constructor for ``EngagementParameters``.

    Typical usage::

        b = (EngagementBuilder()
                .with_blue([
                    UnitType('Frigate', Domain.SURFACE, 3.0, 2),
                    UnitType('SSK',     Domain.UNDERWATER, 2.0, 1),
                ])
                .with_red([
                    UnitType('Destroyer', Domain.SURFACE, 4.0, 3),
                ])
                .with_throughput_blue_attacks_red(
                    p_offense={('Frigate', 'Destroyer'): 2.0,
                               ('SSK',     'Destroyer'): 1.5},
                    p_defense={('Frigate', 'Destroyer'): 1.0},
                )
                .with_throughput_red_attacks_blue(...)
                .with_targeting_policy(StrengthProportional())
                .with_admissibility(Admissibility.canonical())
                .with_t_char_per_domain([1.0, 5.0, 0.1, 1.0])
            )
        ep = b.build()

    The builder validates as it goes (eg. unit names referenced in
    ``p_offense`` must exist in the relevant force) and finalises into
    a stateful :class:`naval_salvo.parameters.EngagementParameters`.
    """

    blue: Optional[Force] = None
    red: Optional[Force] = None
    bar_grid: Optional[ThroughputGrid] = None     # Blue->Red
    rab_grid: Optional[ThroughputGrid] = None     # Red->Blue
    targeting_policy: Optional[TargetingPolicy] = None
    admissibility: Optional[Admissibility] = None
    t_char: Optional[np.ndarray] = None
    rho: Optional[np.ndarray] = None

    # ---- force composition ------------------------------------------------

    def with_blue(self, unit_types: Iterable[UnitType], label: str = "Blue") -> "EngagementBuilder":
        self.blue = Force(label=label, unit_types=list(unit_types))
        return self

    def with_red(self, unit_types: Iterable[UnitType], label: str = "Red") -> "EngagementBuilder":
        self.red = Force(label=label, unit_types=list(unit_types))
        return self

    # ---- throughputs ------------------------------------------------------

    def with_throughput_blue_attacks_red(
        self,
        *,
        eta_offense: Optional[dict[tuple[str, str], float]] = None,
        eta_defense: Optional[dict[tuple[str, str], float]] = None,
        p_offense:   Optional[dict[tuple[str, str], float]] = None,
        p_defense:   Optional[dict[tuple[str, str], float]] = None,
    ) -> "EngagementBuilder":
        if self.blue is None or self.red is None:
            raise ValueError(
                "with_blue() and with_red() must be called before "
                "with_throughput_blue_attacks_red()."
            )
        self.bar_grid = self._build_grid(
            self.blue, self.red,
            eta_offense, eta_defense, p_offense, p_defense,
        )
        return self

    def with_throughput_red_attacks_blue(
        self,
        *,
        eta_offense: Optional[dict[tuple[str, str], float]] = None,
        eta_defense: Optional[dict[tuple[str, str], float]] = None,
        p_offense:   Optional[dict[tuple[str, str], float]] = None,
        p_defense:   Optional[dict[tuple[str, str], float]] = None,
    ) -> "EngagementBuilder":
        if self.blue is None or self.red is None:
            raise ValueError(
                "with_blue() and with_red() must be called before "
                "with_throughput_red_attacks_blue()."
            )
        self.rab_grid = self._build_grid(
            self.red, self.blue,
            eta_offense, eta_defense, p_offense, p_defense,
        )
        return self

    # ---- targeting / admissibility / global ------------------------------

    def with_targeting_policy(self, policy: TargetingPolicy) -> "EngagementBuilder":
        self.targeting_policy = policy
        return self

    def with_admissibility(self, adm: Admissibility) -> "EngagementBuilder":
        self.admissibility = adm
        return self

    def with_t_char_per_domain(
        self, t_char: Union[np.ndarray, list[float]]
    ) -> "EngagementBuilder":
        self.t_char = np.asarray(t_char, dtype=np.float64)
        return self

    def with_rho_per_domain(
        self, rho: Union[np.ndarray, list[float]]
    ) -> "EngagementBuilder":
        self.rho = np.asarray(rho, dtype=np.float64)
        return self

    # ---- finalise --------------------------------------------------------

    def build(self) -> EngagementParameters:
        """
        Combine all configured pieces into a fully-validated
        :class:`EngagementParameters`.

        Raises
        ------
        ValueError
            If essential fields are missing.
        """
        if self.blue is None:
            raise ValueError("Builder.build(): with_blue(...) was never called.")
        if self.red is None:
            raise ValueError("Builder.build(): with_red(...) was never called.")
        if self.bar_grid is None:
            self.bar_grid = ThroughputGrid.zeros(
                self.blue.n_unit_types, self.red.n_unit_types
            )
        if self.rab_grid is None:
            self.rab_grid = ThroughputGrid.zeros(
                self.red.n_unit_types, self.blue.n_unit_types
            )
        if self.admissibility is None:
            self.admissibility = Admissibility.canonical()

        # If a targeting policy was supplied, compute the σ matrices.
        # Otherwise default to all-ones (the historical JPH default; the
        # scaling is then absorbed into η).
        if self.targeting_policy is not None:
            sig_o_bar, sig_d_bar = self.targeting_policy.compute(
                self.blue, self.red, self.admissibility
            )
            sig_o_rab, sig_d_rab = self.targeting_policy.compute(
                self.red, self.blue, self.admissibility
            )
        else:
            sig_o_bar = np.ones((self.blue.n_unit_types, self.red.n_unit_types))
            sig_d_bar = np.ones_like(sig_o_bar)
            sig_o_rab = np.ones((self.red.n_unit_types, self.blue.n_unit_types))
            sig_d_rab = np.ones_like(sig_o_rab)

        bar = self._assemble_directional(
            self.blue, self.red, self.bar_grid, sig_o_bar, sig_d_bar,
        )
        rab = self._assemble_directional(
            self.red, self.blue, self.rab_grid, sig_o_rab, sig_d_rab,
        )

        kwargs = dict(
            blue=self.blue,
            red=self.red,
            blue_attacks_red=bar,
            red_attacks_blue=rab,
        )
        if self.t_char is not None:
            kwargs["t_char"] = self.t_char
        if self.rho is not None:
            kwargs["rho"] = self.rho
        return EngagementParameters(**kwargs)

    # ---- internal helpers ------------------------------------------------

    @staticmethod
    def _build_grid(
        attacker: Force,
        defender: Force,
        eta_offense: Optional[dict[tuple[str, str], float]],
        eta_defense: Optional[dict[tuple[str, str], float]],
        p_offense: Optional[dict[tuple[str, str], float]],
        p_defense: Optional[dict[tuple[str, str], float]],
    ) -> ThroughputGrid:
        nA, nD = attacker.n_unit_types, defender.n_unit_types
        grid = ThroughputGrid.zeros(nA, nD)

        def _fill(target: np.ndarray,
                  src: Optional[dict[tuple[str, str], float]],
                  field_name: str) -> None:
            if src is None:
                return
            for (atk_name, def_name), value in src.items():
                if atk_name not in attacker._index_by_name:
                    raise ValueError(
                        f"{field_name}: attacker unit {atk_name!r} not in "
                        f"force {attacker.label!r}."
                    )
                if def_name not in defender._index_by_name:
                    raise ValueError(
                        f"{field_name}: defender unit {def_name!r} not in "
                        f"force {defender.label!r}."
                    )
                j = attacker._index_by_name[atk_name]
                i = defender._index_by_name[def_name]
                target[j, i] = float(value)

        _fill(grid.eta_offense, eta_offense, "eta_offense")
        _fill(grid.eta_defense, eta_defense, "eta_defense")
        _fill(grid.p_offense,   p_offense,   "p_offense")
        _fill(grid.p_defense,   p_defense,   "p_defense")
        return grid

    @staticmethod
    def _assemble_directional(
        attacker: Force,
        defender: Force,
        grid: ThroughputGrid,
        sigma_off: np.ndarray,
        sigma_def: np.ndarray,
    ) -> DirectionalParameters:
        nA = attacker.n_unit_types
        nD = defender.n_unit_types
        if grid.shape() != (nA, nD):
            raise ValueError(
                f"ThroughputGrid shape {grid.shape()} does not match "
                f"({nA}, {nD})."
            )
        if sigma_off.shape != (nA, nD):
            raise ValueError(
                f"sigma_off shape {sigma_off.shape} != ({nA}, {nD})."
            )
        if sigma_def.shape != (nA, nD):
            raise ValueError(
                f"sigma_def shape {sigma_def.shape} != ({nA}, {nD})."
            )
        pairs = np.empty((nA, nD), dtype=object)
        for j in range(nA):
            for i in range(nD):
                pairs[j, i] = PairParameters(
                    sigma_offense=float(sigma_off[j, i]),
                    sigma_defense=float(sigma_def[j, i]),
                    eta_offense=float(grid.eta_offense[j, i]),
                    eta_defense=float(grid.eta_defense[j, i]),
                    p_offense=float(grid.p_offense[j, i]),
                    p_defense=float(grid.p_defense[j, i]),
                )
        return DirectionalParameters(
            attacker=attacker, defender=defender, pairs=pairs,
        )


# ---------------------------------------------------------------------------
# Mid-battle σ refresh
# ---------------------------------------------------------------------------


def apply_targeting_policy(
    params: EngagementParameters,
    admissibility: Admissibility,
    policy: TargetingPolicy,
) -> EngagementParameters:
    """
    Recompute the σ values of an existing :class:`EngagementParameters`
    using the given targeting policy and the *current* state.

    Returns a *new* EngagementParameters with the updated σ; the η and
    p values of every pair are preserved exactly.

    Use this between salvos when the policy is dynamic
    (StrengthProportional, ThreatWeighted with time-varying weights):
    after each :func:`salvo_step`, the strengths change, so the σ
    allocation generally changes too, and the user should refresh.

    Parameters
    ----------
    params : EngagementParameters
    admissibility : Admissibility
    policy : TargetingPolicy

    Returns
    -------
    EngagementParameters
        A new instance, with the same η, p and global parameters but
        updated σ matrices.

    Notes
    -----
    The returned object reuses the same :class:`Force` instances as
    ``params`` (they are referenced, not copied).  The forces' live
    strengths therefore continue to be the source of truth, exactly as
    callers expect.
    """
    sig_o_bar, sig_d_bar = policy.compute(params.blue, params.red, admissibility)
    sig_o_rab, sig_d_rab = policy.compute(params.red, params.blue, admissibility)

    bar = _refresh_sigma(params.blue_attacks_red, sig_o_bar, sig_d_bar)
    rab = _refresh_sigma(params.red_attacks_blue, sig_o_rab, sig_d_rab)

    return EngagementParameters(
        blue=params.blue,
        red=params.red,
        blue_attacks_red=bar,
        red_attacks_blue=rab,
        t_char=params.t_char,
        rho=params.rho,
    )


def _refresh_sigma(
    direction: DirectionalParameters,
    sigma_off: np.ndarray,
    sigma_def: np.ndarray,
) -> DirectionalParameters:
    nA, nD = direction.pairs.shape
    new_pairs = np.empty((nA, nD), dtype=object)
    for j in range(nA):
        for i in range(nD):
            old = direction.pairs[j, i]
            new_pairs[j, i] = PairParameters(
                sigma_offense=float(sigma_off[j, i]),
                sigma_defense=float(sigma_def[j, i]),
                eta_offense=old.eta_offense,
                eta_defense=old.eta_defense,
                p_offense=old.p_offense,
                p_defense=old.p_defense,
            )
    return DirectionalParameters(
        attacker=direction.attacker,
        defender=direction.defender,
        pairs=new_pairs,
    )
