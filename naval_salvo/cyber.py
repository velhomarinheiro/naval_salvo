"""
naval_salvo.cyber
=================

Cyber-electromagnetic modulation: the Φ family of multiplicative
modifiers that scale the kinetic kill-chain coefficients (σ, ρ, δ) of
non-cyber units as a function of the *relative cyber strength*
between the two sides.

This module implements the formulation of working document 1.4 §3.5
and Sections 4.1-4.5 of the SBPO paper (Equação de Salva
Multidominio, draft sections 2-4).

CANONICAL FORM (paper eqs. 12-13)
---------------------------------

For each cyber channel ``p ∈ {σ, ρ, δ}`` and each kinetic
attacker-defender pair (j, i):

    Φ^p_{ji}(t) = 1 / [1 + (R^p(t) / r_0^p)^k^p]            ... (12)

    R^p(t) = [Σ_k w^p_k Q_k^(y)(t)] / [Σ_k w^p_k P_k^(y)(t)] ... (13)

where:

- ``P_k^(y)(t)`` is the cyber stock of the **side whose kinetic
  coefficient is being modulated**, broken down by sub-type
  ``k ∈ {y_σ, y_ρ, y_δ, y_def}``;
- ``Q_k^(y)(t)`` is the cyber stock of **the opponent**;
- ``w^p_k`` are channel-specific weights emphasising the cyber
  sub-types relevant for channel p (high weight on the same-channel
  sub-type and on y_def, low weight on the others);
- ``r_0^p > 0`` is the half-degradation reference ratio;
- ``k^p ≥ 1`` controls the sigmoid steepness.

Behaviour:

- Q^(y) → 0 (opponent has no cyber)  ⇒ R^p → 0, Φ^p → 1  (no modulation)
- balanced (R^p = r_0^p)              ⇒ Φ^p = 0.5         (half degradation)
- Q^(y) → ∞ (opponent dominates)      ⇒ R^p → ∞, Φ^p → 0  (full collapse)

Φ multiplies the affected channel of the kinetic offensive/defensive
coefficient (paper eqs. 10-11):

    O^(d',d)_{ji}(t) = O^0_{ji} · Φ^σ_{ji}(t) · Φ^ρ_{ij}(t)
    D^(d,d')_{ij}(t) = D^0_{ij} · Φ^δ_{ij}(t)

Following paper §4.1 the four cyber sub-types are:

    y_σ  : jamming, SIGINT, sensor blinding -- attacks scouting/σ
    y_ρ  : spoofing, decoys, datalink falsification -- attacks ρ (distraction)
    y_δ  : CMS attack, defensive network attack -- attacks δ (alertness)
    y_def: COMSEC, hardening, monitoring -- defensive cyber

Mapping to the kinetic coefficients in our code base
----------------------------------------------------

The paper's σ, ρ, δ are factors INSIDE the composite η_offense (paper:
σ τ ρ β) and η_defense (paper: δ τ ζ).  Because our PairParameters
already aggregate σ τ ρ into ``eta_offense`` and δ τ into
``eta_defense``, we apply Φ multiplicatively to those aggregates:

    eta_offense_modulated = eta_offense × Φ^σ × Φ^ρ
    eta_defense_modulated = eta_defense × Φ^δ
    sigma_offense, sigma_defense, p_offense, p_defense unchanged

The operational content is identical to the paper's eq. (10)-(11);
the bookkeeping is one level less granular.

NOTE ON eq. (13) ORIENTATION
----------------------------

The paper's eq. (13) writes ``R^p = Σ w_k B_k / Σ w_k A_k`` without
fully fixing whether A and B in eq. (13) are the same A, B as in
eq. (4) (the two sides of the engagement) or whether they refer to
"side whose coefficient is being modulated" vs. "opponent" generically.
The text of §4.2 ("When B^(y) → 0, ... Φ^p → 1; when the cyber
attacker dominates, Φ^p → 0") is consistent with the reading
"B = cyber attacker against this channel, A = the side whose kinetic
coefficient is being modulated (i.e., the cyber defender for this
channel)".  This implementation uses that operational reading.  If
the researcher wishes the literal eq.-(13) reading where A and B
denote the two engagement sides regardless of channel, they should
verify and (if needed) flip the orientation in :func:`_compute_R`.

Submarine immunity (paper §4.3, decision 1.4 §2.3.d)
----------------------------------------------------

Φ^p_{ji} ≡ 1 for any pair where the defender i is in
Domain.UNDERWATER.  Belt-and-braces with the χ-matrix gating.

Three switchable families
-------------------------

- ``ChannelPhi``    (canonical, paper eqs. 12-13)
- ``SimplePhi``     (Family 1: aggregate scalar Φ from totals)
- ``HauskenPhi``    (Family 3: ratio-based Φ ≡ 1 - r₀ x_atk/(x_atk+x_def))
- ``DecomposedPhi`` (Step-5 legacy, kept for backward compatibility)

References
----------
- Working document 1.4 §3.5
- Paper draft (Equação de Salva Multidominio) §4.1-4.5, eqs. (12)-(13)
- Hausken & Moxnes (2026) Annals of OR 357:1003-1019
- Phase 1 doc 1.4 §2.3.d (submarine immunity)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .domains import CyberSubtype, Domain
from .parameters import (
    DirectionalParameters,
    EngagementParameters,
    PairParameters,
)
from .state import Force


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

#: Mapping from paper's cyber sub-type codes to our CyberSubtype enum.
PAPER_SUBTYPE_CHANNELS: dict[str, CyberSubtype] = {
    "y_sigma": CyberSubtype.SENSOR,
    "y_rho":   CyberSubtype.WEAPON,
    "y_delta": CyberSubtype.C2,
    "y_def":   CyberSubtype.LOGISTICS,
}


def _cyber_stock_total(force: Force) -> float:
    total = 0.0
    for ut, st in zip(force.unit_types, force.states):
        if ut.domain is Domain.CYBER:
            total += st.current_strength
    return total


def _cyber_stock_by_subtype(force: Force) -> dict[CyberSubtype, float]:
    out = {s: 0.0 for s in CyberSubtype}
    valid_codes = {s.value: s for s in CyberSubtype}
    for ut, st in zip(force.unit_types, force.states):
        if ut.domain is not Domain.CYBER:
            continue
        if ut.subtype is None:
            continue
        s = valid_codes.get(ut.subtype)
        if s is not None:
            out[s] += st.current_strength
    return out


# ---------------------------------------------------------------------------
# Core sigmoids
# ---------------------------------------------------------------------------


def phi_sigmoid(R: float, r0: float = 1.0, k: float = 2.0) -> float:
    """
    Paper eq. (12): inverse sigmoid in the cyber-strength ratio.

        Φ(R) = 1 / [1 + (R / r₀)^k]

    Parameters
    ----------
    R : float >= 0
        Cyber-strength ratio (see eq. 13).  Inf is allowed.
    r0 : float > 0, default 1.0
        Half-degradation reference: Φ = 0.5 when R = r₀.
    k : float >= 1, default 2.0
        Sigmoid steepness.

    Returns
    -------
    Φ in (0, 1].
        - R = 0 → Φ = 1   (opponent has no cyber; no modulation)
        - R = r₀ → Φ = 0.5
        - R → ∞ → Φ → 0   (opponent has overwhelming cyber dominance)
    """
    if r0 <= 0.0 or not np.isfinite(r0):
        raise ValueError(f"r0 must be > 0 and finite; got {r0}")
    if k < 1.0 or not np.isfinite(k):
        raise ValueError(f"k must be >= 1 and finite; got {k}")
    if R < 0.0:
        raise ValueError(f"R must be >= 0; got {R}")
    if R == 0.0:
        return 1.0
    if not np.isfinite(R):
        return 0.0
    return 1.0 / (1.0 + (R / r0) ** k)


def phi_logistic(
    x_atk: float,
    x_def: float,
    *,
    r0: float = 0.5,
    k: float = 1.0,
    x_ref: float = 1.0,
) -> float:
    """
    DEPRECATED in favour of :func:`phi_sigmoid`.  Retained for backward
    compatibility with Step 5 tests.  Implements the logistic-of-
    difference form

        Φ = 1 - r₀ / (1 + exp(-k(x_atk - x_def)/x_ref))

    Returns 1.0 when both stocks are zero (no-cyber short-circuit).
    """
    if not 0.0 <= r0 <= 1.0:
        raise ValueError(f"r0 must be in [0, 1]; got {r0}")
    if k < 0.0 or not np.isfinite(k):
        raise ValueError(f"k must be >= 0 and finite; got {k}")
    if x_ref <= 0.0 or not np.isfinite(x_ref):
        raise ValueError(f"x_ref must be > 0 and finite; got {x_ref}")
    if x_atk < 0.0 or x_def < 0.0:
        raise ValueError(
            f"Cyber stocks must be non-negative; got x_atk={x_atk}, "
            f"x_def={x_def}"
        )
    if x_atk == 0.0 and x_def == 0.0:
        return 1.0
    z = k * (x_atk - x_def) / x_ref
    return 1.0 - r0 / (1.0 + np.exp(-z))


# ---------------------------------------------------------------------------
# Modulator base class
# ---------------------------------------------------------------------------


class CyberModulator(ABC):
    """
    Abstract Φ modulator.  Subclasses implement :meth:`apply`.
    """

    @abstractmethod
    def apply(self, params: EngagementParameters) -> EngagementParameters:
        ...


# ---------------------------------------------------------------------------
# CANONICAL: ChannelPhi  (paper eqs. 12-13)
# ---------------------------------------------------------------------------


# Default channel weights w^p_k (paper §4.2).  Each channel emphasises
# its corresponding y_p sub-type and y_def.
DEFAULT_CHANNEL_WEIGHTS: dict[str, dict[CyberSubtype, float]] = {
    "sigma": {
        CyberSubtype.SENSOR:    1.0,   # y_σ (direct match)
        CyberSubtype.WEAPON:    0.1,
        CyberSubtype.C2:        0.2,
        CyberSubtype.LOGISTICS: 0.3,   # y_def
    },
    "rho": {
        CyberSubtype.SENSOR:    0.1,
        CyberSubtype.WEAPON:    1.0,   # y_ρ
        CyberSubtype.C2:        0.1,
        CyberSubtype.LOGISTICS: 0.3,
    },
    "delta": {
        CyberSubtype.SENSOR:    0.1,
        CyberSubtype.WEAPON:    0.1,
        CyberSubtype.C2:        1.0,   # y_δ
        CyberSubtype.LOGISTICS: 0.3,
    },
}


@dataclass
class ChannelPhi(CyberModulator):
    """
    CANONICAL Φ modulator following paper eqs. (12)-(13).

    Three channels (σ, ρ, δ), each with its own sigmoid Φ^p computed
    from a weighted ratio of cyber stocks, applied as:

        eta_offense ← eta_offense × Φ^σ × Φ^ρ
        eta_defense ← eta_defense × Φ^δ

    Submarine defenders are exempt (paper §4.3).

    Parameters
    ----------
    r0_sigma, r0_rho, r0_delta : float > 0
        Half-degradation reference ratios.  Default 1.0.
    k_sigma, k_rho, k_delta : float >= 1
        Sigmoid steepness.  Default 2.0.
    weights : dict[str, dict[CyberSubtype, float]], optional
        Channel weights w^p_k.  If None, uses DEFAULT_CHANNEL_WEIGHTS.
    """

    r0_sigma: float = 1.0
    r0_rho:   float = 1.0
    r0_delta: float = 1.0
    k_sigma:  float = 2.0
    k_rho:    float = 2.0
    k_delta:  float = 2.0
    weights:  Optional[dict[str, dict[CyberSubtype, float]]] = None

    def __post_init__(self) -> None:
        for nm in ("r0_sigma", "r0_rho", "r0_delta"):
            v = getattr(self, nm)
            if v <= 0.0 or not np.isfinite(v):
                raise ValueError(f"{nm} must be > 0 and finite; got {v}")
        for nm in ("k_sigma", "k_rho", "k_delta"):
            v = getattr(self, nm)
            if v < 1.0 or not np.isfinite(v):
                raise ValueError(f"{nm} must be >= 1 and finite; got {v}")

    def _channel_weights(self, channel: str) -> dict[CyberSubtype, float]:
        if self.weights is not None:
            return self.weights[channel]
        return DEFAULT_CHANNEL_WEIGHTS[channel]

    def _compute_R(
        self,
        own_stock: dict[CyberSubtype, float],
        opp_stock: dict[CyberSubtype, float],
        channel: str,
    ) -> float:
        """
        Eq. (13) under the operational reading: R is the ratio of
        opponent's weighted cyber stock to own weighted cyber stock.

        Return 0 when opp has no cyber, +inf when opp has cyber but
        own has none, finite ratio otherwise.
        """
        w = self._channel_weights(channel)
        num = sum(w[s] * opp_stock[s] for s in CyberSubtype)
        den = sum(w[s] * own_stock[s] for s in CyberSubtype)
        if num == 0.0:
            return 0.0
        if den == 0.0:
            return float("inf")
        return num / den

    def _phi_for_side(
        self,
        own_stock: dict[CyberSubtype, float],
        opp_stock: dict[CyberSubtype, float],
    ) -> tuple[float, float, float]:
        R_sigma = self._compute_R(own_stock, opp_stock, "sigma")
        R_rho   = self._compute_R(own_stock, opp_stock, "rho")
        R_delta = self._compute_R(own_stock, opp_stock, "delta")
        phi_s = phi_sigmoid(R_sigma, self.r0_sigma, self.k_sigma)
        phi_r = phi_sigmoid(R_rho,   self.r0_rho,   self.k_rho)
        phi_d = phi_sigmoid(R_delta, self.r0_delta, self.k_delta)
        return phi_s, phi_r, phi_d

    def apply(self, params: EngagementParameters) -> EngagementParameters:
        blue_stock = _cyber_stock_by_subtype(params.blue)
        red_stock  = _cyber_stock_by_subtype(params.red)

        phi_blue_s, phi_blue_r, phi_blue_d = self._phi_for_side(
            own_stock=blue_stock, opp_stock=red_stock
        )
        phi_red_s, phi_red_r, phi_red_d = self._phi_for_side(
            own_stock=red_stock, opp_stock=blue_stock
        )

        # Blue→Red: Blue's offensive coefficients use Φ_blue;
        # Red's defensive coefficients (against Blue's incoming fire)
        # use Φ_red.
        bar_new = _modulate_directional_channel(
            params.blue_attacks_red,
            phi_off_sigma=phi_blue_s, phi_off_rho=phi_blue_r,
            phi_def_delta=phi_red_d,
        )
        rab_new = _modulate_directional_channel(
            params.red_attacks_blue,
            phi_off_sigma=phi_red_s, phi_off_rho=phi_red_r,
            phi_def_delta=phi_blue_d,
        )
        return EngagementParameters(
            blue=params.blue, red=params.red,
            blue_attacks_red=bar_new,
            red_attacks_blue=rab_new,
            t_char=params.t_char,
            rho=params.rho,
        )


# ---------------------------------------------------------------------------
# Family 1: SimplePhi (legacy, retained for sensitivity comparison)
# ---------------------------------------------------------------------------


@dataclass
class SimplePhi(CyberModulator):
    """
    Legacy Family 1: aggregate scalar Φ via :func:`phi_logistic`.
    Applied uniformly to σ_off, σ_def, η_off of every kinetic pair.
    Skips submarine defenders.
    """

    r0: float = 0.5
    k: float = 1.0
    x_ref: Optional[float] = None
    _x_ref_locked: dict = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if not 0.0 <= self.r0 <= 1.0:
            raise ValueError(f"r0 must be in [0, 1]; got {self.r0}")
        if self.k < 0.0 or not np.isfinite(self.k):
            raise ValueError(f"k must be >= 0 and finite; got {self.k}")
        if self.x_ref is not None and self.x_ref <= 0.0:
            raise ValueError(f"x_ref must be > 0; got {self.x_ref}")

    def apply(self, params: EngagementParameters) -> EngagementParameters:
        x_blue = _cyber_stock_total(params.blue)
        x_red  = _cyber_stock_total(params.red)

        if "scale" not in self._x_ref_locked:
            scale = self.x_ref if self.x_ref is not None else max(
                1e-9, x_blue + x_red
            )
            self._x_ref_locked["scale"] = scale
        scale = self._x_ref_locked["scale"]

        phi_red_attacks_blue = phi_logistic(
            x_atk=x_red, x_def=x_blue,
            r0=self.r0, k=self.k, x_ref=scale,
        )
        phi_blue_attacks_red = phi_logistic(
            x_atk=x_blue, x_def=x_red,
            r0=self.r0, k=self.k, x_ref=scale,
        )

        bar_new = _modulate_directional_uniform(
            params.blue_attacks_red, phi=phi_blue_attacks_red,
        )
        rab_new = _modulate_directional_uniform(
            params.red_attacks_blue, phi=phi_red_attacks_blue,
        )
        return EngagementParameters(
            blue=params.blue, red=params.red,
            blue_attacks_red=bar_new,
            red_attacks_blue=rab_new,
            t_char=params.t_char,
            rho=params.rho,
        )


# ---------------------------------------------------------------------------
# Legacy DecomposedPhi (Step-5 implementation, kept for backward compat)
# ---------------------------------------------------------------------------


@dataclass
class DecomposedPhi(CyberModulator):
    """
    LEGACY: Step-5 implementation that maps each CyberSubtype to a
    specific kinetic kill-chain component (C2 → σ, SEN → σ extra,
    WPN → η, LOG → ρ-regen).  Retained for sensitivity comparison;
    the canonical paper formulation is :class:`ChannelPhi`.
    """

    r0: float = 0.5
    k: float = 1.0
    x_ref: Optional[float] = None
    _x_ref_locked: dict = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if not 0.0 <= self.r0 <= 1.0:
            raise ValueError(f"r0 must be in [0, 1]; got {self.r0}")
        if self.k < 0.0 or not np.isfinite(self.k):
            raise ValueError(f"k must be >= 0 and finite; got {self.k}")
        if self.x_ref is not None and self.x_ref <= 0.0:
            raise ValueError(f"x_ref must be > 0; got {self.x_ref}")

    def apply(self, params: EngagementParameters) -> EngagementParameters:
        x_blue = _cyber_stock_by_subtype(params.blue)
        x_red  = _cyber_stock_by_subtype(params.red)

        if "scale" not in self._x_ref_locked:
            total_initial = sum(x_blue.values()) + sum(x_red.values())
            scale = (
                self.x_ref if self.x_ref is not None
                else max(1e-9, total_initial)
            )
            self._x_ref_locked["scale"] = scale
        scale = self._x_ref_locked["scale"]

        phi_blue_atk: dict[CyberSubtype, float] = {}
        phi_red_atk:  dict[CyberSubtype, float] = {}
        for s in CyberSubtype:
            phi_blue_atk[s] = phi_logistic(
                x_atk=x_blue[s], x_def=x_red[s],
                r0=self.r0, k=self.k, x_ref=scale,
            )
            phi_red_atk[s]  = phi_logistic(
                x_atk=x_red[s], x_def=x_blue[s],
                r0=self.r0, k=self.k, x_ref=scale,
            )

        bar_new = _modulate_directional_decomposed(
            params.blue_attacks_red, phi=phi_blue_atk,
        )
        rab_new = _modulate_directional_decomposed(
            params.red_attacks_blue, phi=phi_red_atk,
        )
        return EngagementParameters(
            blue=params.blue, red=params.red,
            blue_attacks_red=bar_new,
            red_attacks_blue=rab_new,
            t_char=params.t_char,
            rho=params.rho,
        )


# ---------------------------------------------------------------------------
# Family 3: HauskenPhi
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HauskenPhi(CyberModulator):
    """
    Family 3 (Hausken-Moxnes 2026): Φ = 1 - r₀ x_atk/(x_atk + x_def).
    Applied uniformly.  Skips submarine defenders.
    """

    r0: float = 0.5

    def __post_init__(self) -> None:
        if not 0.0 <= self.r0 <= 1.0:
            raise ValueError(f"r0 must be in [0, 1]; got {self.r0}")

    def apply(self, params: EngagementParameters) -> EngagementParameters:
        x_blue = _cyber_stock_total(params.blue)
        x_red  = _cyber_stock_total(params.red)

        def _ratio_phi(x_atk: float, x_def: float) -> float:
            denom = x_atk + x_def
            if denom <= 0.0:
                return 1.0
            return 1.0 - self.r0 * x_atk / denom

        phi_blue_attacks_red = _ratio_phi(x_atk=x_blue, x_def=x_red)
        phi_red_attacks_blue = _ratio_phi(x_atk=x_red,  x_def=x_blue)
        bar_new = _modulate_directional_uniform(
            params.blue_attacks_red, phi=phi_blue_attacks_red,
        )
        rab_new = _modulate_directional_uniform(
            params.red_attacks_blue, phi=phi_red_attacks_blue,
        )
        return EngagementParameters(
            blue=params.blue, red=params.red,
            blue_attacks_red=bar_new,
            red_attacks_blue=rab_new,
            t_char=params.t_char,
            rho=params.rho,
        )


# ---------------------------------------------------------------------------
# Direction-level helpers
# ---------------------------------------------------------------------------


def _is_submarine_defender(direction: DirectionalParameters, i: int) -> bool:
    return direction.defender.unit_types[i].domain is Domain.UNDERWATER


def _is_kinetic_pair(direction: DirectionalParameters, j: int, i: int) -> bool:
    return (
        direction.attacker.unit_types[j].domain.is_kinetic
        and direction.defender.unit_types[i].domain.is_kinetic
    )


def _modulate_directional_channel(
    direction: DirectionalParameters,
    *,
    phi_off_sigma: float,
    phi_off_rho:   float,
    phi_def_delta: float,
) -> DirectionalParameters:
    """Canonical channel modulation (paper eqs. 10-11):
        eta_offense ← eta_offense × Φ^σ × Φ^ρ
        eta_defense ← eta_defense × Φ^δ
    """
    nA, nD = direction.pairs.shape
    new_pairs = np.empty((nA, nD), dtype=object)
    for j in range(nA):
        for i in range(nD):
            old = direction.pairs[j, i]
            if (
                _is_kinetic_pair(direction, j, i)
                and not _is_submarine_defender(direction, i)
            ):
                new_pairs[j, i] = PairParameters(
                    sigma_offense=old.sigma_offense,
                    sigma_defense=old.sigma_defense,
                    eta_offense=_clip01(
                        old.eta_offense * phi_off_sigma * phi_off_rho
                    ),
                    eta_defense=_clip01(old.eta_defense * phi_def_delta),
                    p_offense=old.p_offense,
                    p_defense=old.p_defense,
                )
            else:
                new_pairs[j, i] = old
    return DirectionalParameters(
        attacker=direction.attacker,
        defender=direction.defender,
        pairs=new_pairs,
    )


def _modulate_directional_uniform(
    direction: DirectionalParameters,
    *,
    phi: float,
) -> DirectionalParameters:
    nA, nD = direction.pairs.shape
    new_pairs = np.empty((nA, nD), dtype=object)
    for j in range(nA):
        for i in range(nD):
            old = direction.pairs[j, i]
            if (
                _is_kinetic_pair(direction, j, i)
                and not _is_submarine_defender(direction, i)
            ):
                new_pairs[j, i] = PairParameters(
                    sigma_offense=_clip01(old.sigma_offense * phi),
                    sigma_defense=_clip01(old.sigma_defense * phi),
                    eta_offense=_clip01(old.eta_offense * phi),
                    eta_defense=old.eta_defense,
                    p_offense=old.p_offense,
                    p_defense=old.p_defense,
                )
            else:
                new_pairs[j, i] = old
    return DirectionalParameters(
        attacker=direction.attacker,
        defender=direction.defender,
        pairs=new_pairs,
    )


def _modulate_directional_decomposed(
    direction: DirectionalParameters,
    *,
    phi: dict[CyberSubtype, float],
) -> DirectionalParameters:
    nA, nD = direction.pairs.shape
    new_pairs = np.empty((nA, nD), dtype=object)
    phi_c2  = phi[CyberSubtype.C2]
    phi_sen = phi[CyberSubtype.SENSOR]
    phi_wpn = phi[CyberSubtype.WEAPON]
    for j in range(nA):
        for i in range(nD):
            old = direction.pairs[j, i]
            if (
                _is_kinetic_pair(direction, j, i)
                and not _is_submarine_defender(direction, i)
            ):
                new_pairs[j, i] = PairParameters(
                    sigma_offense=_clip01(
                        old.sigma_offense * phi_c2 * phi_sen
                    ),
                    sigma_defense=_clip01(old.sigma_defense * phi_c2),
                    eta_offense=_clip01(old.eta_offense * phi_wpn),
                    eta_defense=old.eta_defense,
                    p_offense=old.p_offense,
                    p_defense=old.p_defense,
                )
            else:
                new_pairs[j, i] = old
    return DirectionalParameters(
        attacker=direction.attacker,
        defender=direction.defender,
        pairs=new_pairs,
    )


def _clip01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return float(x)
