"""
parameters.py
=============

Containers for the calibrable parameters of one unit-type-to-unit-type
interaction (offensive coefficient, defensive coefficient, targeting
shares, effectiveness composites) and a top-level ``EngagementParameters``
object that bundles everything the dynamics modules need to step the
state forward.

The parameter naming follows the canonical notation fixed in working
document 1.4 §2:

    sigma_offense_{j -> i}   sigma^atq_{ji}
        Offensive aiming / targeting share of attacker j against
        defender i.  In the JPH (2001) formulation: psi_{ji}, the
        "fraction of B_j units that engage A_i units".  Here treated
        as a free targeting share in [0, 1].

    sigma_defense_{i, j}     sigma^def_{ij}
        Defensive aiming / targeting share of defender i against
        attacker j.  In JPH (2001): theta_{ij}.

    eta_offense_{j -> i}     eta^atq_{ji}
        *Composite* offensive effectiveness of j vs i.  Bundles in
        scouting (sigma), training (tau), and distraction (rho_p) of
        JPH eq. (2.18); the canonical separation is

            eta^atq_{ji} = sigma^scout_{ji} * tau^train_{ji} * rho^dist_{ij}

        but for code we keep one composite scalar per ordered pair so
        the equations stay close to the JPH matrix form.

    eta_defense_{i, j}       eta^def_{ij}
        Composite defensive effectiveness of i against j.

    p_offense_{j -> i}       p^atq_{ji}  (was 'beta' in Hughes 1995)
        Number of offensive shots / hits that one j-unit can deliver
        per salvo against one i-unit (before sigma, eta degrade it).

    p_defense_{i, j}         p^def_{ij}  (was 'a3' in Hughes 1995)
        Number of incoming j-shots that one i-unit can intercept per
        salvo (before sigma, eta degrade it).

The full per-salvo "kernel" T = T^atq - T^def of the JPH equation thus
reads, for a single attacker j attacking defender i:

    T^atq_{ji} = sigma^atq_{ji} * eta^atq_{ji} * p^atq_{ji} * B_j
    T^def_{ij} = sigma^def_{ij} * eta^def_{ij} * p^def_{ij} * A_i

Cross-domain coupling enters the dynamics through the admissibility
indicator ``1^{(d',d)}`` (see ``admissibility.py``); these per-pair
parameters are agnostic about which domain each side belongs to.

References
----------
- Johns, Pilnick & Hughes (2001) NPS-IJWA-01-010, eqs. (2.16)-(2.18)
  and matrix form on p. 15-16.
- Hughes (1995) NRL 42:267-289, original homogeneous salvo equation.
- Working document 1.4 §3.2 (canonical notation table).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .domains import Domain
from .state import Force


# ---------------------------------------------------------------------------
# Per-pair parameter record
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PairParameters:
    """
    Calibrable parameters for one ordered pair (attacker j, defender i).

    Stored as a frozen dataclass because per-pair parameters are usually
    fixed for the duration of an engagement; if the modeller needs
    time-varying parameters they should construct a new
    :class:`EngagementParameters` at each salvo.

    Attributes
    ----------
    sigma_offense : float in [0, 1]
        Offensive targeting share, sigma^atq_{ji}.
    sigma_defense : float in [0, 1]
        Defensive targeting share, sigma^def_{ij}.
    eta_offense : float in [0, 1]
        Composite offensive effectiveness, eta^atq_{ji}.
    eta_defense : float in [0, 1]
        Composite defensive effectiveness, eta^def_{ij}.
    p_offense : float >= 0
        Offensive throughput, p^atq_{ji} (hits per attacker per salvo).
    p_defense : float >= 0
        Defensive throughput, p^def_{ij} (intercepts per defender per salvo).

    Notes
    -----
    All four sigma/eta parameters are constrained to [0, 1] by the
    canonical specification (decision 1.4 §2.3.f, "no amplification").
    Throughput parameters p are unbounded above (they are pure counts).
    """

    sigma_offense: float = 1.0
    sigma_defense: float = 1.0
    eta_offense: float = 1.0
    eta_defense: float = 1.0
    p_offense: float = 0.0
    p_defense: float = 0.0

    def __post_init__(self) -> None:
        for name in ("sigma_offense", "sigma_defense", "eta_offense", "eta_defense"):
            v = getattr(self, name)
            if not (0.0 <= v <= 1.0) or not np.isfinite(v):
                raise ValueError(
                    f"PairParameters.{name} must be in [0, 1] and finite; "
                    f"got {v}."
                )
        for name in ("p_offense", "p_defense"):
            v = getattr(self, name)
            if v < 0.0 or not np.isfinite(v):
                raise ValueError(
                    f"PairParameters.{name} must be >= 0 and finite; "
                    f"got {v}."
                )

    def offensive_kernel(self) -> float:
        """Return the multiplicative product sigma^atq * eta^atq * p^atq."""
        return self.sigma_offense * self.eta_offense * self.p_offense

    def defensive_kernel(self) -> float:
        """Return the multiplicative product sigma^def * eta^def * p^def."""
        return self.sigma_defense * self.eta_defense * self.p_defense


# ---------------------------------------------------------------------------
# Direction (attacker side -> defender side) parameter block
# ---------------------------------------------------------------------------


@dataclass
class DirectionalParameters:
    """
    All per-pair parameters for one attack direction.

    "One attack direction" means one ordered pair of sides:
    Blue-attacks-Red, or Red-attacks-Blue.  For each direction we store
    a (n_attacker x n_defender) grid of :class:`PairParameters`.

    The arrays are indexed in the same order as ``Force.unit_types`` of
    the corresponding side.

    Attributes
    ----------
    attacker : Force
        The force that attacks in this direction (eg. Blue if this is
        the Blue->Red block).
    defender : Force
        The force that is attacked.
    pairs : ndarray of dtype object, shape (n_attacker, n_defender)
        Per-pair parameters.  ``pairs[j, i]`` describes attacker unit
        type ``j`` attacking defender unit type ``i`` (note the row /
        column orientation matches the JPH "O" matrix: row = attacker,
        col = defender).
    """

    attacker: Force
    defender: Force
    pairs: np.ndarray = field()

    def __post_init__(self) -> None:
        nA = self.attacker.n_unit_types
        nD = self.defender.n_unit_types
        if self.pairs.shape != (nA, nD):
            raise ValueError(
                f"DirectionalParameters.pairs has shape "
                f"{self.pairs.shape}; expected ({nA}, {nD})."
            )
        if self.pairs.dtype != object:
            raise ValueError(
                "DirectionalParameters.pairs must be an object array of "
                "PairParameters instances."
            )
        for j in range(nA):
            for i in range(nD):
                if not isinstance(self.pairs[j, i], PairParameters):
                    raise ValueError(
                        f"pairs[{j}, {i}] is not a PairParameters instance "
                        f"(got {type(self.pairs[j, i]).__name__})."
                    )

    @classmethod
    def zeros(cls, attacker: Force, defender: Force) -> "DirectionalParameters":
        """
        Build a directional block with all-zero throughput and unit
        sigma/eta (the natural "everything inert" starting point).
        """
        nA = attacker.n_unit_types
        nD = defender.n_unit_types
        pairs = np.empty((nA, nD), dtype=object)
        default = PairParameters()
        for j in range(nA):
            for i in range(nD):
                pairs[j, i] = default
        return cls(attacker=attacker, defender=defender, pairs=pairs)

    def get(self, attacker_name: str, defender_name: str) -> PairParameters:
        """Return the PairParameters for the named (attacker, defender)."""
        j = self.attacker._index_by_name[attacker_name]
        i = self.defender._index_by_name[defender_name]
        return self.pairs[j, i]

    def set(
        self,
        attacker_name: str,
        defender_name: str,
        params: PairParameters,
    ) -> None:
        """Set the PairParameters for the named (attacker, defender)."""
        if not isinstance(params, PairParameters):
            raise TypeError(
                f"params must be a PairParameters; got {type(params).__name__}."
            )
        j = self.attacker._index_by_name[attacker_name]
        i = self.defender._index_by_name[defender_name]
        self.pairs[j, i] = params

    # ---- numpy accessors used by dynamics --------------------------------

    def offensive_kernel_matrix(self) -> np.ndarray:
        """
        Return the (n_attacker, n_defender) matrix of products
        sigma^atq * eta^atq * p^atq.

        This is the per-attacker-per-defender contribution before
        multiplying by the current attacker strength B_j and the
        admissibility indicator.
        """
        nA, nD = self.pairs.shape
        K = np.zeros((nA, nD), dtype=np.float64)
        for j in range(nA):
            for i in range(nD):
                K[j, i] = self.pairs[j, i].offensive_kernel()
        return K

    def defensive_kernel_matrix(self) -> np.ndarray:
        """
        Return the (n_attacker, n_defender) matrix of products
        sigma^def * eta^def * p^def.

        Note: ``[j, i]`` here means "defender i defending against
        attacker j", consistent with the offensive matrix' indexing.
        """
        nA, nD = self.pairs.shape
        K = np.zeros((nA, nD), dtype=np.float64)
        for j in range(nA):
            for i in range(nD):
                K[j, i] = self.pairs[j, i].defensive_kernel()
        return K


# ---------------------------------------------------------------------------
# Top-level engagement parameter bundle
# ---------------------------------------------------------------------------


@dataclass
class EngagementParameters:
    """
    All numerical inputs to the dynamics modules, except the live state.

    The model is two-sided so we always carry two directional blocks:

    - ``blue_attacks_red`` : Blue is the attacker, Red the defender.
    - ``red_attacks_blue`` : Red is the attacker, Blue the defender.

    Plus a few global / cross-cutting parameters:

    - ``t_char`` : characteristic time per kinetic domain (vector of
      length 4 ordered as (S, U, A, C)).  Used by the continuous-EDO
      regime to set the domain-specific tempo.  Hausken & Moxnes (2026)
      motivate making air faster than surface, surface faster than
      submarine warfare.
    - ``rho`` : regeneration rate per kinetic domain (also length 4).
      Set to 0 in the canonical baseline (decision 1.4 §2.3.f) but kept
      as a parameter so calibrations that include reload / replenishment
      can be plugged in without touching the equations.

    The cyber modulator (delta_offense, delta_defense) and the precise
    family selector (simple / decomposed / Hausken) live in a separate
    ``CyberParameters`` object that we will define in step 5; for the
    foundational data structures of step 1 we keep
    ``EngagementParameters`` agnostic.

    Attributes
    ----------
    blue : Force
    red  : Force
    blue_attacks_red : DirectionalParameters
    red_attacks_blue : DirectionalParameters
    t_char : ndarray of shape (4,)
        Characteristic times for the four kinetic domains, ordered
        (S, U, A, C).
    rho : ndarray of shape (4,)
        Regeneration rates for the four kinetic domains.
    """

    blue: Force
    red: Force
    blue_attacks_red: DirectionalParameters
    red_attacks_blue: DirectionalParameters
    t_char: np.ndarray = field(
        default_factory=lambda: np.ones(4, dtype=np.float64)
    )
    rho: np.ndarray = field(
        default_factory=lambda: np.zeros(4, dtype=np.float64)
    )

    def __post_init__(self) -> None:
        # Cross-check that the directional blocks reference the right
        # forces.  Compare by identity; the user is expected to pass
        # the exact same Force objects.
        if self.blue_attacks_red.attacker is not self.blue:
            raise ValueError(
                "blue_attacks_red.attacker must be the same Force object "
                "as 'blue'."
            )
        if self.blue_attacks_red.defender is not self.red:
            raise ValueError(
                "blue_attacks_red.defender must be the same Force object "
                "as 'red'."
            )
        if self.red_attacks_blue.attacker is not self.red:
            raise ValueError(
                "red_attacks_blue.attacker must be the same Force object "
                "as 'red'."
            )
        if self.red_attacks_blue.defender is not self.blue:
            raise ValueError(
                "red_attacks_blue.defender must be the same Force object "
                "as 'blue'."
            )

        self.t_char = np.asarray(self.t_char, dtype=np.float64)
        self.rho = np.asarray(self.rho, dtype=np.float64)
        if self.t_char.shape != (4,):
            raise ValueError(
                f"t_char must have shape (4,); got {self.t_char.shape}."
            )
        if self.rho.shape != (4,):
            raise ValueError(
                f"rho must have shape (4,); got {self.rho.shape}."
            )
        if np.any(self.t_char <= 0.0) or not np.all(np.isfinite(self.t_char)):
            raise ValueError(
                f"t_char entries must be > 0 and finite; got {self.t_char}."
            )
        if np.any(self.rho < 0.0) or not np.all(np.isfinite(self.rho)):
            raise ValueError(
                f"rho entries must be >= 0 and finite; got {self.rho}."
            )

    @classmethod
    def with_zero_couplings(
        cls,
        blue: Force,
        red: Force,
        t_char: Optional[np.ndarray] = None,
        rho: Optional[np.ndarray] = None,
    ) -> "EngagementParameters":
        """
        Build an EngagementParameters with both directions filled with
        zero throughputs (everything inert).  Useful as a starting point
        in tests and tutorials.
        """
        return cls(
            blue=blue,
            red=red,
            blue_attacks_red=DirectionalParameters.zeros(blue, red),
            red_attacks_blue=DirectionalParameters.zeros(red, blue),
            t_char=np.ones(4, dtype=np.float64) if t_char is None else t_char,
            rho=np.zeros(4, dtype=np.float64) if rho is None else rho,
        )
