"""naval_salvo.stochastic -- Stochastic Multi-Domain Salvo Equation.

Stochastic extension of the multi-domain model, with simultaneous or
sequential fire exchange (Blue first / Red first).

Foundations:
    - Armstrong (2005): stochastic salvo model (binomial fires, normal
      damage per non-intercepted missile).
    - Armstrong (2014): sequential fire exchange -- return fire is
      executed by the survivors of the first salvo.
    - Canonical conventions of the multi-domain model: cyber modulation
      Φ (eta_off ← eta_off·Φ^σ·Φ^ρ; eta_def ← eta_def·Φ^δ), submarine
      cyber immunity (χ), verdict over naval forces (surface +
      submarine), fire proportional to fractional stock
      (Hausken–Moxnes).

Layers:
    1. Homogeneous (``HomogeneousForce`` / ``run_homogeneous_battle``):
       direct reproduction of Armstrong (2005, 2014), used for
       validation against the published results.
    2. Multi-domain (``UnitGroup`` / ``MultiDomainForce`` /
       ``run_multidomain_battle``): heterogeneous extension with five
       domains and cyber modulation.

Canonical decision: cyber is a PRECONDITION (pre-kinetic). Φ is
computed from the net intensities before the kinetic exchange and
applies in both directions, regardless of the firing order. The
``cyber_follows_sequence=True`` switch enables the alternative reading,
in which the defender's Φ only applies to its return fire.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import numpy as np

__all__ = [
    "EngagementOrder",
    "HomogeneousForce",
    "run_homogeneous_battle",
    "phi",
    "UnitGroup",
    "MultiDomainForce",
    "StochasticResult",
    "run_multidomain_battle",
    "DOMAINS",
    "NAVAL_DOMAINS",
]

# ---------------------------------------------------------------------------
# Domains (package convention)
# ---------------------------------------------------------------------------
DOMAINS = ("s", "u", "a", "c", "y")  # surface, underwater, air, coastal, cyber-EM
NAVAL_DOMAINS = ("s", "u")           # naval verdict: surface + underwater

_EPS = 1e-9


class EngagementOrder(Enum):
    """Engagement order of the salvo (Armstrong, 2014)."""

    SIMULTANEOUS = "simultaneous"
    BLUE_FIRST = "blue_first"
    RED_FIRST = "red_first"


# ---------------------------------------------------------------------------
# Canonical cyber modulation
# ---------------------------------------------------------------------------
def phi(intensity: float, k: float = 1.0, i0: float = 0.0) -> float:
    """Cyber modulation Φ ∈ (0, 1] as a function of the net intensity suffered.

    ``intensity`` <= 0 → no degradation (Φ = 1).
    ``intensity`` > 0  → Φ = 1 − sigmoid(k·(I − I₀)) < 1.
    """
    if intensity <= 0.0:
        return 1.0
    return 1.0 - 1.0 / (1.0 + math.exp(-k * (intensity - i0)))


# ---------------------------------------------------------------------------
# Binomial sampling with fractional stock (Hausken–Moxnes)
# ---------------------------------------------------------------------------
def _binomial_frac(rng: np.random.Generator, units: float, n_per_unit: int,
                   p: float) -> int:
    """Sample fires/intercepts from a possibly fractional stock.

    The integer part fires Binomial(n·⌊u⌋, p); the residual fraction fires
    Binomial(n, p·frac), preserving the mean n·u·p.
    """
    if units <= 0.0 or n_per_unit <= 0 or p <= 0.0:
        return 0
    p = min(p, 1.0)
    whole = int(math.floor(units))
    frac = units - whole
    total = 0
    if whole > 0:
        total += int(rng.binomial(whole * n_per_unit, p))
    if frac > _EPS:
        total += int(rng.binomial(n_per_unit, min(p * frac, 1.0)))
    return total


def _sample_damage(rng: np.random.Generator, n_hits: int, mu_v: float,
                   sigma_v: float, cap: float) -> float:
    """Sum of the damage of ``n_hits`` missiles, Normal(μ, σ) truncated at 0."""
    if n_hits <= 0:
        return 0.0
    if sigma_v <= 0.0:
        return float(min(n_hits * mu_v, cap))
    dmg = np.clip(rng.normal(mu_v, sigma_v, size=n_hits), 0.0, None)
    return float(min(dmg.sum(), cap))


# ===========================================================================
# LAYER 1 -- homogeneous model (Armstrong 2005/2014 validation)
# ===========================================================================
@dataclass
class HomogeneousForce:
    """Homogeneous force in Armstrong's (2005) notation."""

    units: float
    n_off: int
    p_off: float
    n_def: int
    p_def: float
    mu_v: float
    sigma_v: float


def _one_way_homogeneous(rng, shooter_units: float, shooter: HomogeneousForce,
                         target_units: float, target: HomogeneousForce) -> float:
    """One-way salvo; returns the target's loss (in units)."""
    if shooter_units <= _EPS or target_units <= _EPS:
        return 0.0
    off = _binomial_frac(rng, shooter_units, shooter.n_off, shooter.p_off)
    dfd = _binomial_frac(rng, target_units, target.n_def, target.p_def)
    net = max(0, off - dfd)
    return _sample_damage(rng, net, target.mu_v, target.sigma_v, target_units)


def run_homogeneous_battle(blue: HomogeneousForce, red: HomogeneousForce,
                           order: EngagementOrder, n_sim: int = 10_000,
                           seed: Optional[int] = 42):
    """Monte Carlo of one full salvo (out and back) in the given order.

    Returns (blue_losses, red_losses) as arrays of size ``n_sim``.
    """
    rng = np.random.default_rng(seed)
    blue_losses = np.empty(n_sim)
    red_losses = np.empty(n_sim)
    for s in range(n_sim):
        B, A = blue.units, red.units
        if order is EngagementOrder.SIMULTANEOUS:
            dA = _one_way_homogeneous(rng, B, blue, A, red)
            dB = _one_way_homogeneous(rng, A, red, B, blue)
        elif order is EngagementOrder.BLUE_FIRST:
            dA = _one_way_homogeneous(rng, B, blue, A, red)
            dB = _one_way_homogeneous(rng, A - dA, red, B, blue)
        else:
            dB = _one_way_homogeneous(rng, A, red, B, blue)
            dA = _one_way_homogeneous(rng, B - dB, blue, A, red)
        blue_losses[s], red_losses[s] = dB, dA
    return blue_losses, red_losses


# ===========================================================================
# LAYER 2 -- stochastic multi-domain model
# ===========================================================================
@dataclass
class UnitGroup:
    """Homogeneous group of units within a multi-domain force."""

    name: str
    domain: str
    units: float
    n_off: int
    p_off: float
    n_def: int
    p_def: float
    mu_v: float
    sigma_v: float
    sigma_exp: float = 1.0   # σ (scouting) exponent in the Φ modulation
    rho_exp: float = 1.0     # ρ (C2) exponent in the Φ modulation
    delta_exp: float = 1.0   # δ (defence) exponent in the Φ modulation

    def __post_init__(self):
        if self.domain not in DOMAINS:
            raise ValueError(f"invalid domain: {self.domain!r}")


@dataclass
class MultiDomainForce:
    """Force composed of heterogeneous groups across multiple domains."""

    label: str
    groups: List[UnitGroup]
    cyber_offense: float = 0.0

    def initial_stocks(self) -> Dict[str, float]:
        return {g.name: g.units for g in self.groups}

    def naval_strength(self, stocks: Dict[str, float]) -> float:
        return sum(stocks[g.name] for g in self.groups
                   if g.domain in NAVAL_DOMAINS)


def _p_off_effective(group: UnitGroup, phi_val: float) -> float:
    """Canonical offensive modulation; submarines are immune (χ)."""
    if group.domain == "u":
        return group.p_off
    return group.p_off * (phi_val ** group.sigma_exp) * (phi_val ** group.rho_exp)


def _p_def_effective(group: UnitGroup, phi_val: float) -> float:
    """Canonical defensive modulation; submarines are immune (χ)."""
    if group.domain == "u":
        return group.p_def
    return group.p_def * (phi_val ** group.delta_exp)


def _one_way_multidomain(rng, atk: MultiDomainForce, atk_stocks,
                         dfd: MultiDomainForce, dfd_stocks,
                         phi_atk: float, phi_dfd: float) -> Dict[str, float]:
    """One-way multi-domain salvo; allocation proportional to stock."""
    losses = {g.name: 0.0 for g in dfd.groups}
    live = [g for g in dfd.groups if dfd_stocks[g.name] > _EPS]
    if not live:
        return losses
    total_target = sum(dfd_stocks[g.name] for g in live)

    for tg in live:
        share = dfd_stocks[tg.name] / total_target
        off_hits = 0
        for ag in atk.groups:
            p_eff = _p_off_effective(ag, phi_atk) * share
            off_hits += _binomial_frac(rng, atk_stocks[ag.name], ag.n_off, p_eff)
        p_def = _p_def_effective(tg, phi_dfd)
        intercepts = _binomial_frac(rng, dfd_stocks[tg.name], tg.n_def, p_def)
        net = max(0, off_hits - intercepts)
        losses[tg.name] = _sample_damage(rng, net, tg.mu_v, tg.sigma_v,
                                         dfd_stocks[tg.name])
    return losses


@dataclass
class StochasticResult:
    """Aggregate statistics of a Monte Carlo campaign."""

    order: EngagementOrder
    n_sim: int
    blue_naval_final: np.ndarray
    red_naval_final: np.ndarray
    blue_wins: int
    red_wins: int
    draws: int
    blue_losses: np.ndarray = field(default=None)
    red_losses: np.ndarray = field(default=None)

    @property
    def p_blue_win(self) -> float:
        return self.blue_wins / self.n_sim

    @property
    def p_red_win(self) -> float:
        return self.red_wins / self.n_sim

    @property
    def p_draw(self) -> float:
        return self.draws / self.n_sim

    def summary(self) -> Dict[str, float]:
        return {
            "p_blue_win": self.p_blue_win,
            "p_red_win": self.p_red_win,
            "p_draw": self.p_draw,
            "blue_naval_mean": float(self.blue_naval_final.mean()),
            "blue_naval_std": float(self.blue_naval_final.std(ddof=1)),
            "red_naval_mean": float(self.red_naval_final.mean()),
            "red_naval_std": float(self.red_naval_final.std(ddof=1)),
            "blue_loss_mean": float(self.blue_losses.mean()),
            "red_loss_mean": float(self.red_losses.mean()),
        }


def run_multidomain_battle(blue: MultiDomainForce, red: MultiDomainForce,
                           order: EngagementOrder, n_salvos: int = 1,
                           n_sim: int = 5_000, seed: Optional[int] = 7,
                           k_cyber: float = 1.0, i0_cyber: float = 0.0,
                           cyber_follows_sequence: bool = False
                           ) -> StochasticResult:
    """Multi-domain Monte Carlo with engagement order and cyber modulation.

    Parameters
    ----------
    cyber_follows_sequence:
        ``False`` (canonical): cyber as precondition -- Φ applies in both
        directions from the first salvo. ``True``: the Φ degrading the
        second shooter is only applied to its return fire.
    """
    rng = np.random.default_rng(seed)
    phi_blue = phi(red.cyber_offense - blue.cyber_offense, k_cyber, i0_cyber)
    phi_red = phi(blue.cyber_offense - red.cyber_offense, k_cyber, i0_cyber)

    blue_naval0 = blue.naval_strength(blue.initial_stocks())
    red_naval0 = red.naval_strength(red.initial_stocks())

    bnf = np.empty(n_sim)
    rnf = np.empty(n_sim)
    blue_wins = red_wins = draws = 0

    def _apply(stocks, losses):
        for name, v in losses.items():
            stocks[name] = max(0.0, stocks[name] - v)

    for s in range(n_sim):
        bs = blue.initial_stocks()
        rs = red.initial_stocks()
        for salvo_idx in range(n_salvos):
            first_salvo = salvo_idx == 0
            if order is EngagementOrder.SIMULTANEOUS:
                dr = _one_way_multidomain(rng, blue, bs, red, rs, phi_blue, phi_red)
                db = _one_way_multidomain(rng, red, rs, blue, bs, phi_red, phi_blue)
                _apply(rs, dr)
                _apply(bs, db)
            elif order is EngagementOrder.BLUE_FIRST:
                # alternative reading: on the 1st salvo, Red has not yet
                # suffered cyber effects
                pr = 1.0 if (cyber_follows_sequence and first_salvo) else phi_red
                dr = _one_way_multidomain(rng, blue, bs, red, rs, phi_blue, pr)
                _apply(rs, dr)
                db = _one_way_multidomain(rng, red, rs, blue, bs, phi_red, phi_blue)
                _apply(bs, db)
            else:  # RED_FIRST
                pb = 1.0 if (cyber_follows_sequence and first_salvo) else phi_blue
                db = _one_way_multidomain(rng, red, rs, blue, bs, phi_red, pb)
                _apply(bs, db)
                dr = _one_way_multidomain(rng, blue, bs, red, rs, phi_blue, phi_red)
                _apply(rs, dr)
            if (blue.naval_strength(bs) <= _EPS
                    or red.naval_strength(rs) <= _EPS):
                break
        bn, rn = blue.naval_strength(bs), red.naval_strength(rs)
        bnf[s], rnf[s] = bn, rn
        if bn > _EPS and rn <= _EPS:
            blue_wins += 1
        elif rn > _EPS and bn <= _EPS:
            red_wins += 1
        else:
            draws += 1

    return StochasticResult(
        order=order, n_sim=n_sim,
        blue_naval_final=bnf, red_naval_final=rnf,
        blue_wins=blue_wins, red_wins=red_wins, draws=draws,
        blue_losses=blue_naval0 - bnf, red_losses=red_naval0 - rnf,
    )
