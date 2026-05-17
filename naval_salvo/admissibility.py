"""
admissibility.py
================

Implements the 5x5 cross-domain admissibility matrix that gates which
attacker-domain / defender-domain pairs can interact.

The matrix follows the three-level encoding fixed in working document 1.4
§2.2:

    1   -- primary admissible (full doctrinal capability)
    chi -- marginal, calibrable in [0, 1]   (e.g. helicopter dipping
           sonar from S into U; ASW aircraft into U; coastal artillery
           against deep submarines)
    0   -- structurally null (e.g. submarine torpedo against high-altitude
           aircraft; coastal battery against own coastal battery in
           friendly geometry)

The matrix appears in the dynamic equations as the indicator
``1^{(d', d)}`` which gates the attacker-domain d' acting on
defender-domain d.  In code we represent it as a numpy array of floats
in [0, 1] so that the marginal level chi can take any calibrated value.

Convention
----------
``M[i, j] = M[d_attacker.index, d_defender.index]``

i.e. rows are *attackers* and columns are *defenders*.  This matches
the kernel ``1^{(d', d)} [T^atq_{ji} - T^def_{ij}]`` of the canonical
salvo equation, where d' is the attacker domain and d is the defender
(target) domain.

References
----------
- Working document 1.4 §2.2 (encoding rules and default chi values).
- Hausken & Moxnes (2026) Table 1 -- nine-equipment cross-impact table
  motivating sparse off-diagonal couplings.
- Johns, Pilnick & Hughes (2001) -- baseline degenerate case has only
  the (S, S) cell active, which is the canonical Hughes 1995 / homogeneous
  recovery (decision 1.4 §3.4).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np

from .domains import (
    DOMAIN_ORDER,
    KINETIC_DOMAINS,
    N_DOMAINS,
    Domain,
    parse_domain,
)


# ---------------------------------------------------------------------------
# Default values for the marginal level
# ---------------------------------------------------------------------------

#: Default value for the "marginal admissible" level chi.  Document 1.4 §2.2
#: leaves chi as a calibrable parameter in [0, 1]; we set it to 0.5 as a
#: neutral mid-point.  The user can override this via the ``chi`` argument
#: of :class:`Admissibility` or :func:`canonical_matrix`.
DEFAULT_CHI: float = 0.5


# ---------------------------------------------------------------------------
# Canonical baseline matrices
# ---------------------------------------------------------------------------


def degenerate_johns_pilnick_hughes() -> np.ndarray:
    """
    Build the admissibility matrix for the JPH (2001) degenerate case.

    Only the (Surface, Surface) cell is active; all other cross-domain
    interactions are turned off.  This is the configuration in which the
    multi-domain model must reduce *exactly* to the original heterogeneous
    salvo model of Johns, Pilnick & Hughes (2001), so it serves as the
    primary regression test in the validation suite (decision 1.4 §3.4 and
    document 1.4 §5 Step 2).

    Returns
    -------
    numpy.ndarray
        A (N_DOMAINS, N_DOMAINS) float64 array with M[S, S] = 1 and all
        other entries equal to 0.
    """
    M = np.zeros((N_DOMAINS, N_DOMAINS), dtype=np.float64)
    s = Domain.SURFACE.index
    M[s, s] = 1.0
    return M


def canonical_matrix(chi: float = DEFAULT_CHI) -> np.ndarray:
    """
    Build the canonical default 5x5 admissibility matrix.

    The default values follow document 1.4 §2.2 Table 1 (illustrative
    defaults; calibration may override any cell).  Concretely:

    Attacker \\ Defender |  S    U    A    C    X
    ---------------------+------------------------
    S (surface)          |  1    chi  chi  1    chi
    U (underwater)       |  1    1    0    chi  0
    A (air)              |  1    chi  1    1    chi
    C (coastal)          |  1    chi  chi  1    chi
    X (cyber)            |  chi  0    chi  chi  1

    Reasoning:

    - S vs S, A vs A, U vs U, C vs C, X vs X are doctrinal "1" except
      that the U-on-U case includes ASW peer combat (also primary).
    - U vs A (defender) is structurally 0: a torpedo cannot reach an
      aircraft at altitude.
    - X vs U is 0 by decision 1.4 §2.3.d (submarines immune to cyber
      via delta_X = 0).  Note this is enforced doubly: both the
      admissibility matrix and the cyber modulator delta_X^U set the
      contribution to zero.  Belt-and-braces is intentional.
    - Marginal cells (chi) include S->U (helicopters dipping; surface
      ASW), A->U (MPA), C->U (only useful in shallow littorals).
    - X cells are largely chi because cyber rarely *destroys* kinetic
      stock directly; the heavy lifting of the cyber-kinetic coupling
      runs through the delta modulators in ``cyber.py``.

    Parameters
    ----------
    chi : float, default 0.5
        Value to use for marginal-admissible cells.  Must be in [0, 1].

    Returns
    -------
    numpy.ndarray
        A (5, 5) float64 admissibility matrix indexed in canonical
        domain order.

    Raises
    ------
    ValueError
        If ``chi`` is outside [0, 1].
    """
    if not 0.0 <= chi <= 1.0:
        raise ValueError(f"chi must be in [0, 1]; got {chi}.")

    M = np.zeros((N_DOMAINS, N_DOMAINS), dtype=np.float64)
    S, U, A, C, X = (d.index for d in DOMAIN_ORDER)

    # Surface attacker
    M[S, S] = 1.0
    M[S, U] = chi
    M[S, A] = chi
    M[S, C] = 1.0
    M[S, X] = chi

    # Underwater attacker
    M[U, S] = 1.0
    M[U, U] = 1.0
    M[U, A] = 0.0     # torpedo vs aircraft -- structurally null
    M[U, C] = chi
    M[U, X] = 0.0     # submarines do not project cyber in this model

    # Air attacker
    M[A, S] = 1.0
    M[A, U] = chi     # MPA / ASW air
    M[A, A] = 1.0
    M[A, C] = 1.0
    M[A, X] = chi

    # Coastal attacker
    M[C, S] = 1.0
    M[C, U] = chi
    M[C, A] = chi     # coastal SAM
    M[C, C] = 1.0
    M[C, X] = chi

    # Cyber attacker -- mostly marginal; submarines are immune (0).
    M[X, S] = chi
    M[X, U] = 0.0
    M[X, A] = chi
    M[X, C] = chi
    M[X, X] = 1.0

    return M


# ---------------------------------------------------------------------------
# Wrapper class
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Admissibility:
    """
    Immutable wrapper around a 5x5 admissibility matrix.

    Stores a numpy array internally but exposes domain-aware accessors so
    that downstream code (targeting, dynamics) does not need to keep
    track of integer indices.

    Construction can be done in three ways:

    1. ``Admissibility.canonical(chi=...)``  -- the default cross-domain
       matrix described in document 1.4 §2.2.
    2. ``Admissibility.degenerate()``        -- only (S, S) active; used
       by the JPH validation harness.
    3. ``Admissibility.from_array(M)``       -- bring your own 5x5 matrix
       (with sanity checks).

    Examples
    --------
    >>> adm = Admissibility.canonical()
    >>> adm[Domain.SURFACE, Domain.SURFACE]
    1.0
    >>> adm[Domain.UNDERWATER, Domain.AIR]      # structurally null
    0.0
    >>> Admissibility.degenerate()[Domain.AIR, Domain.AIR]
    0.0
    """

    matrix: np.ndarray = field()

    # ---- construction -----------------------------------------------------

    def __post_init__(self) -> None:
        # Validate shape and value range immediately on construction so
        # that downstream code can rely on the invariant.
        m = np.asarray(self.matrix, dtype=np.float64)
        if m.shape != (N_DOMAINS, N_DOMAINS):
            raise ValueError(
                f"Admissibility matrix must have shape ({N_DOMAINS}, "
                f"{N_DOMAINS}); got {m.shape}."
            )
        if np.any(m < 0.0) or np.any(m > 1.0):
            bad = m[(m < 0.0) | (m > 1.0)]
            raise ValueError(
                f"All admissibility entries must lie in [0, 1]; "
                f"out-of-range values: {bad.tolist()}."
            )
        if not np.all(np.isfinite(m)):
            raise ValueError("Admissibility matrix contains non-finite values.")

        # Because the dataclass is frozen we have to use object.__setattr__
        # to install the cleaned numpy array (and freeze it for safety).
        m_frozen = m.copy()
        m_frozen.setflags(write=False)
        object.__setattr__(self, "matrix", m_frozen)

    @classmethod
    def canonical(cls, chi: float = DEFAULT_CHI) -> "Admissibility":
        """Return the canonical default admissibility (see :func:`canonical_matrix`)."""
        return cls(canonical_matrix(chi=chi))

    @classmethod
    def degenerate(cls) -> "Admissibility":
        """Return the (S, S)-only admissibility matrix used for JPH validation."""
        return cls(degenerate_johns_pilnick_hughes())

    @classmethod
    def from_array(cls, M: np.ndarray) -> "Admissibility":
        """Wrap an externally constructed 5x5 array (with validation)."""
        return cls(np.asarray(M, dtype=np.float64))

    # ---- accessors --------------------------------------------------------

    def __getitem__(self, key: tuple[Domain | str, Domain | str]) -> float:
        """
        Look up an entry by (attacker_domain, defender_domain).

        Parameters
        ----------
        key : tuple of (Domain or str, Domain or str)
            Pair (d_attacker, d_defender).  Strings are passed through
            :func:`parse_domain`.

        Returns
        -------
        float
            The admissibility coefficient in [0, 1].
        """
        if not isinstance(key, tuple) or len(key) != 2:
            raise KeyError(
                "Admissibility lookup requires a (attacker, defender) tuple."
            )
        d_atk, d_def = parse_domain(key[0]), parse_domain(key[1])
        return float(self.matrix[d_atk.index, d_def.index])

    def is_admissible(
        self, attacker: Domain | str, defender: Domain | str
    ) -> bool:
        """
        True iff the (attacker, defender) cell is non-zero.

        This is the boolean view that dynamics code uses to skip whole
        sub-blocks of the targeting computation.
        """
        return self[attacker, defender] > 0.0

    def admissible_pairs(self) -> Iterable[tuple[Domain, Domain]]:
        """Iterate over (attacker, defender) pairs with non-zero admissibility."""
        for d_atk in DOMAIN_ORDER:
            for d_def in DOMAIN_ORDER:
                if self.matrix[d_atk.index, d_def.index] > 0.0:
                    yield (d_atk, d_def)

    # ---- diagnostics ------------------------------------------------------

    def as_dataframe_dict(self) -> dict[str, dict[str, float]]:
        """
        Return a nested dict suitable for pretty-printing or pandas export.

        Outer key is the attacker code, inner key is the defender code.
        """
        out: dict[str, dict[str, float]] = {}
        for d_atk in DOMAIN_ORDER:
            row: dict[str, float] = {}
            for d_def in DOMAIN_ORDER:
                row[d_def.value] = float(
                    self.matrix[d_atk.index, d_def.index]
                )
            out[d_atk.value] = row
        return out
