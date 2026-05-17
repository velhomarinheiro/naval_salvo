"""
naval_salvo.targeting
=====================

Targeting policies that fill in the σ^atq (offensive aiming) and σ^def
(defensive aiming) shares of every (attacker, defender) pair.

Phase 1 (working document 1.4 §3.2) fixes four canonical policies:

1. **Uniform**.  σ_{j→i} = 1 / n_admissible_targets_of_j.  This is the
   default reading of Hughes (1995) and the JPH (2001) "fraction of B_j
   units that engage A_i" when no other information is available.  It
   is also Beall's reading in the Coronel/Coral Sea/Savo Island
   reproductions: when both Scharnhorst and Gneisenau engage both Good
   Hope and Monmouth, ψ = 0.5 each (cf. JPH p. 22 worked example).

2. **StrengthProportional** (MacKay 2009; canonical in Hausken-Moxnes
   2026).  σ_{j→i} ∝ A_i(t).  Each attacker spreads its fire across
   admissible defenders in proportion to the *current* defender stock,
   so larger / more numerous targets attract more fire.  This is the
   semantics the SBPO 2025 proceedings paper recommends as the
   "realistic targeting heuristic" baseline.

3. **ThreatWeighted**.  σ_{j→i} ∝ w_i, where w is a user-supplied
   per-target weight vector.  The weights typically encode "value" or
   "threat" priorities (FPSOs > escort vessels for an opponent
   targeting Bacia de Campos; carriers > destroyers in a classic
   surface engagement).  Weights are static within one salvo step but
   may be updated by the caller between steps.

4. **Manual**.  σ_{j→i} given explicitly as (n_attacker, n_defender)
   matrices, one per direction.  This is the mode used to reproduce
   historical battles where the targeting fractions are observed (Beall
   data; JPH worked examples).  It is also the mode that
   ``DirectionalParameters.set(...)`` already supports under the hood:
   the user fills σ cell by cell.  ``Manual`` exists here mainly so
   that the *signature* "compute σ(state, params, admissibility) →
   matrices" works uniformly, regardless of policy.

The output of every policy is a pair of numpy arrays of shape
``(n_attacker, n_defender)`` giving σ^atq and σ^def for one direction.

Notes on conservation
---------------------
Hughes (1995) and JPH (2001) treat the σ values as *fractions* in
[0, 1], and JPH explicitly remarks that they are "fractions of B_j
units" (i.e. they should sum to 1 across i for any fixed attacker j).
We honour that by default in Uniform, StrengthProportional, and
ThreatWeighted.  The Manual policy does *not* enforce any such
constraint -- the user is free to supply σ values that exceed 1 in
exotic settings (eg. modelling concentrated-fire amplification by
a doctrine).  However, decision 1.4 §2.3.f is "no amplification", so
we still validate σ ∈ [0, 1] in :class:`PairParameters`.

Defensive σ
-----------
The defensive σ is interpreted as "fraction of A_i's defensive
firepower that is dedicated to incoming j-shots".  In a setting where
the defender allocates dynamically (eg. an Aegis system prioritising
the closest threat), Uniform / StrengthProportional / ThreatWeighted
read the *attacker* stock as the relevant magnitude.  In static-
geometry historical reproductions, one usually sets defensive σ to a
fixed, hand-calibrated value (use Manual).

References
----------
- Johns, Pilnick & Hughes (2001) NPS-IJWA-01-010 §3.B (Coronel),
  worked example p.22 -- ψ = 0.5 for Sch+Gn vs GH+Mon.
- MacKay (2009) JORS 60:1421-1427 -- semi-dynamic proportional
  allocation.
- Hausken & Moxnes (2026) Annals of OR 357:1003-1019 §2.1 -- ρ_dji
  proportional reallocation, exactly the StrengthProportional reading.
- Working document 1.4 §3.2 (canonical policies inventory).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .admissibility import Admissibility
from .state import Force


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _admissibility_mask(
    attacker: Force, defender: Force, admissibility: Admissibility
) -> np.ndarray:
    """
    (n_attacker, n_defender) array of χ values gated by domain pair.

    Returned values lie in [0, 1].  A row that is all-zero means the
    attacker can engage no admissible defender at all and downstream
    code should leave its σ row untouched (zero).
    """
    n_atk = attacker.n_unit_types
    n_def = defender.n_unit_types
    chi = np.empty((n_atk, n_def), dtype=np.float64)
    for j, ut_j in enumerate(attacker.unit_types):
        for i, ut_i in enumerate(defender.unit_types):
            chi[j, i] = admissibility[ut_j.domain, ut_i.domain]
    return chi


def _row_normalise(W: np.ndarray) -> np.ndarray:
    """
    Normalise each row of ``W`` to sum to 1, leaving all-zero rows zero.

    A safety helper for any allocation policy that produces unnormalised
    weights.  Floating-point safe: any row whose sum is below the
    machine precision threshold is left as zeros (which is the right
    answer when the attacker has no admissible target).
    """
    sums = W.sum(axis=1, keepdims=True)
    out = np.zeros_like(W)
    nonzero = sums.ravel() > 0.0
    out[nonzero] = W[nonzero] / sums[nonzero]
    return out


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class TargetingPolicy(ABC):
    """
    Abstract targeting policy.

    A policy maps the current battle state (forces + admissibility) to
    a pair of σ matrices for *one* direction (attacker -> defender).

    Subclasses implement :meth:`compute`.
    """

    @abstractmethod
    def compute(
        self,
        attacker: Force,
        defender: Force,
        admissibility: Admissibility,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return ``(sigma_offense, sigma_defense)`` for the (attacker, defender)
        direction.

        Both arrays have shape ``(n_attacker, n_defender)`` and entries
        in [0, 1].  The convention is:

        - ``sigma_offense[j, i]``  = fraction of attacker j's offensive
                                     firepower aimed at defender i.
        - ``sigma_defense[j, i]``  = fraction of defender i's defensive
                                     firepower allocated to incoming
                                     attacker-j shots.
        """


# ---------------------------------------------------------------------------
# Uniform
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Uniform(TargetingPolicy):
    """
    Equal split across admissible targets (Hughes 1995 default reading).

    For each attacker j, σ^atq_{j, i} = χ_{ji} / Σ_k χ_{jk}.  When all
    χ are 0 or 1 this reduces to "1 / n_admissible".  When χ takes
    intermediate values, the share is proportional to χ (so e.g. an
    air unit against a marginal-admissible submarine target gets a
    smaller share than against a primary-admissible surface target,
    even before considering the kernel itself).

    The defensive σ is the symmetric counterpart: σ^def_{j, i} =
    χ_{ji} / Σ_l χ_{li} (column-normalised).
    """

    def compute(
        self,
        attacker: Force,
        defender: Force,
        admissibility: Admissibility,
    ) -> tuple[np.ndarray, np.ndarray]:
        chi = _admissibility_mask(attacker, defender, admissibility)
        sigma_off = _row_normalise(chi)
        # Defensive: fraction of i's defenses against shots from j.
        # Symmetry-by-transpose: row-normalise the *transpose* of χ
        # (treating the defender as the "attacker of defenses").
        sigma_def = _row_normalise(chi.T).T
        return sigma_off, sigma_def


# ---------------------------------------------------------------------------
# Strength-proportional (MacKay 2009 / Hausken-Moxnes 2026)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StrengthProportional(TargetingPolicy):
    """
    σ allocated proportionally to the *current* opposing stock.

    For each attacker j:

        σ^atq_{j, i}  ∝  χ_{ji} · A_i(t)

    so that more numerous defenders attract proportionally more fire,
    matching the "semi-dynamic targeting" heuristic of MacKay (2009)
    and the variable-kill-rate Lanchester model of Hausken-Moxnes
    (2026).

    For the defensive direction:

        σ^def_{j, i}  ∝  χ_{ji} · B_j(t)

    i.e. defender i prioritises incoming threats from the *more
    numerous* attacker types.  This is the "ratio of strengths"
    heuristic the SBPO 2025 paper recommends.

    Edge cases
    ----------
    - If all admissible defenders for some attacker j currently have
      zero strength, the policy returns σ^atq_{j, ·} = 0 (the attacker
      effectively does not fire).  This is consistent with "no targets
      → no shots".
    - If the attacker has zero strength itself, the σ values are
      undefined but harmless because they multiply 0 in the kernel.
    """

    def compute(
        self,
        attacker: Force,
        defender: Force,
        admissibility: Admissibility,
    ) -> tuple[np.ndarray, np.ndarray]:
        chi = _admissibility_mask(attacker, defender, admissibility)
        A = defender.strength_vector()         # (n_def,)
        B = attacker.strength_vector()         # (n_atk,)
        # Offensive: weight by current defender stock.
        W_off = chi * A[None, :]
        sigma_off = _row_normalise(W_off)
        # Defensive: weight by current attacker stock.
        W_def = chi * B[:, None]
        # Column-normalise (each defender splits its defense across the
        # admissible attackers in proportion to attacker stock).
        col_sums = W_def.sum(axis=0, keepdims=True)
        sigma_def = np.zeros_like(W_def)
        nonzero = col_sums.ravel() > 0.0
        sigma_def[:, nonzero] = W_def[:, nonzero] / col_sums[:, nonzero]
        return sigma_off, sigma_def


# ---------------------------------------------------------------------------
# Threat-weighted (calibrated priorities)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ThreatWeighted(TargetingPolicy):
    """
    σ proportional to a calibrated per-target weight vector.

    Useful when domain knowledge prescribes priorities that do not flow
    automatically from the current strengths -- for example, in the
    Bacia de Campos scenario the FPSO platforms are *both* zero-offence
    targets *and* the operational priority for the attacker, so the
    StrengthProportional policy would correctly increase fire on the
    FPSOs (they are numerous) but ThreatWeighted lets us also encode
    "FPSOs are *more* important than escort frigates per unit" by hand.

    Parameters
    ----------
    offensive_weights : ndarray, shape (n_defender,) or None
        Per-defender priorities for the offensive direction.  If None,
        falls back to all-ones (uniform).
    defensive_weights : ndarray, shape (n_attacker,) or None
        Per-attacker priorities for the defensive direction.  If None,
        falls back to all-ones.

    Notes
    -----
    Weights are *not* required to sum to one; they are normalised
    automatically per row / column.  Negative weights are not allowed
    and raise ValueError on construction.
    """

    offensive_weights: Optional[np.ndarray] = None
    defensive_weights: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        for nm in ("offensive_weights", "defensive_weights"):
            w = getattr(self, nm)
            if w is None:
                continue
            w = np.asarray(w, dtype=np.float64)
            if w.ndim != 1:
                raise ValueError(
                    f"ThreatWeighted.{nm} must be 1-D; got shape {w.shape}."
                )
            if np.any(w < 0.0) or not np.all(np.isfinite(w)):
                raise ValueError(
                    f"ThreatWeighted.{nm} must be non-negative and finite."
                )
            object.__setattr__(self, nm, w)

    def compute(
        self,
        attacker: Force,
        defender: Force,
        admissibility: Admissibility,
    ) -> tuple[np.ndarray, np.ndarray]:
        chi = _admissibility_mask(attacker, defender, admissibility)
        n_atk, n_def = chi.shape

        w_off = (
            np.ones(n_def, dtype=np.float64)
            if self.offensive_weights is None
            else self.offensive_weights
        )
        w_def = (
            np.ones(n_atk, dtype=np.float64)
            if self.defensive_weights is None
            else self.defensive_weights
        )
        if w_off.shape != (n_def,):
            raise ValueError(
                f"offensive_weights has shape {w_off.shape}; expected "
                f"({n_def},) to match the defender."
            )
        if w_def.shape != (n_atk,):
            raise ValueError(
                f"defensive_weights has shape {w_def.shape}; expected "
                f"({n_atk},) to match the attacker."
            )

        sigma_off = _row_normalise(chi * w_off[None, :])

        W_def = chi * w_def[:, None]
        col_sums = W_def.sum(axis=0, keepdims=True)
        sigma_def = np.zeros_like(W_def)
        nonzero = col_sums.ravel() > 0.0
        sigma_def[:, nonzero] = W_def[:, nonzero] / col_sums[:, nonzero]
        return sigma_off, sigma_def


# ---------------------------------------------------------------------------
# Manual (user-supplied)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Manual(TargetingPolicy):
    """
    Bring-your-own σ matrices.

    Parameters
    ----------
    sigma_offense : ndarray, shape (n_attacker, n_defender)
    sigma_defense : ndarray, shape (n_attacker, n_defender)

    The arrays are stored as-is and must have entries in [0, 1].
    Useful for reproducing historical worked examples (eg. JPH p.22
    Coronel where ψ values are observed and time-varying).
    """

    sigma_offense: np.ndarray = None
    sigma_defense: np.ndarray = None

    def __post_init__(self) -> None:
        for nm in ("sigma_offense", "sigma_defense"):
            v = getattr(self, nm)
            if v is None:
                raise ValueError(f"Manual.{nm} is required.")
            v = np.asarray(v, dtype=np.float64)
            if v.ndim != 2:
                raise ValueError(
                    f"Manual.{nm} must be 2-D; got shape {v.shape}."
                )
            if np.any(v < 0.0) or np.any(v > 1.0) or not np.all(np.isfinite(v)):
                raise ValueError(
                    f"Manual.{nm} must lie in [0, 1] and be finite."
                )
            object.__setattr__(self, nm, v)

    def compute(
        self,
        attacker: Force,
        defender: Force,
        admissibility: Admissibility,
    ) -> tuple[np.ndarray, np.ndarray]:
        expected = (attacker.n_unit_types, defender.n_unit_types)
        if self.sigma_offense.shape != expected:
            raise ValueError(
                f"Manual.sigma_offense has shape {self.sigma_offense.shape}; "
                f"expected {expected} for this attacker/defender pair."
            )
        if self.sigma_defense.shape != expected:
            raise ValueError(
                f"Manual.sigma_defense has shape {self.sigma_defense.shape}; "
                f"expected {expected} for this attacker/defender pair."
            )
        return self.sigma_offense.copy(), self.sigma_defense.copy()
