"""
naval_salvo.validation.degenerate
==================================

Degenerate-case validators: when the multi-domain heterogeneous model
is restricted to (i) a single domain, (ii) the JPH (2001) degenerate
admissibility, (iii) σ = η = 1 for all pairs, and (iv) ρ = 0, it must
reduce *exactly* to the published Hughes (1995) / JPH (2001) salvo
equations.  These functions encode that reduction analytically and
compare it numerically to the output of the deterministic dynamics
module.

The functions here are *constructors* of canonical scenarios plus
*analytical solvers* that bypass our dynamics engine entirely.  The
tests then run our engine on the same scenario and demand bit-equal
(or near-machine-precision) agreement.

References
----------
- Hughes (1995) NRL 42:267-289, eqs. for ΔA, ΔB.
- Johns, Pilnick & Hughes (2001) NPS-IJWA-01-010, eq. (2.18) and matrix
  form on p. 15-16, Coral Sea worked example p. 27.
- Armstrong (2005) Operations Research 53:830-841, eq. (1)-(2).
- Working document 1.4 §3.4 (degenerate-case recovery contract).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..admissibility import Admissibility
from ..domains import Domain
from ..parameters import (
    DirectionalParameters,
    EngagementParameters,
    PairParameters,
)
from ..state import BattleState, Force, UnitType


# ---------------------------------------------------------------------------
# Hughes (1995) homogeneous baseline
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HughesScenario:
    """
    Parameters for the Hughes (1995) homogeneous salvo battle.

    Notation matches Armstrong (2005), summarised in the sequential-fire
    paper of Armstrong (2014) and reproduced verbatim in the SBPO 2025
    proceedings paper in the project library:

        ΔA = -(βB - yA) / x      subject to  0 ≤ -ΔA ≤ A
        ΔB = -(αA - zB) / w      subject to  0 ≤ -ΔB ≤ B

    where

        A0, B0  -- initial strengths of Red and Blue;
        α, β    -- offensive firepower (well-aimed shots per unit per
                    salvo) of A (= Red) and B (= Blue) respectively;
        z, y    -- defensive firepower (intercepts per unit per salvo)
                    of B and A respectively (note the swap: "z = Blue's
                    interceptors" is what defends against α-shots from
                    Red);
        w, x    -- staying power of B and A respectively (number of
                    hits to put one unit out of action);
        n_salvos -- how many salvo exchanges to simulate.

    All quantities are non-negative; staying powers must be strictly
    positive.

    Mapping to the Hughes/Christiansen "(α, β, a1, a3, b1, b3)" notation:

        α  = α        β  = β
        a1 = x        b1 = w
        a3 = y        b3 = z

    with ΔA = (βB - a3·A)/a1 and ΔB = (αA - b3·B)/b1, modulo signs.
    """

    A0: float
    B0: float
    alpha: float
    beta: float
    z: float                 # Blue's defenders (intercept α-shots)
    y: float                 # Red's defenders (intercept β-shots)
    w: float = 1.0           # Blue staying power
    x: float = 1.0           # Red staying power
    n_salvos: int = 1


def hughes_analytical(scn: HughesScenario) -> tuple[np.ndarray, np.ndarray]:
    """
    Closed-form iteration of the Hughes (1995) salvo equations.

    Returns
    -------
    A_history, B_history : ndarray, shape (n_salvos + 1,)
        Strengths at t = 0, 1, ..., n_salvos.  The first entry is the
        initial strength of each side; subsequent entries are post-salvo
        strengths after 1, 2, ..., n simultaneous exchanges.
    """
    A = float(scn.A0)
    B = float(scn.B0)
    A_hist = [A]
    B_hist = [B]
    for _ in range(scn.n_salvos):
        # Pre-salvo state used by *both* sides simultaneously.
        delta_A = -max(0.0, scn.beta * B - scn.y * A) / scn.x
        delta_B = -max(0.0, scn.alpha * A - scn.z * B) / scn.w
        A = max(0.0, A + delta_A)
        B = max(0.0, B + delta_B)
        A_hist.append(A)
        B_hist.append(B)
    return np.array(A_hist), np.array(B_hist)


def build_hughes_homogeneous_engagement(
    scn: HughesScenario,
    blue_label: str = "Blue",
    red_label: str = "Red",
    domain: Domain = Domain.SURFACE,
) -> tuple[BattleState, EngagementParameters, Admissibility]:
    """
    Build a fully-specified naval_salvo engagement that, when run with
    the deterministic dynamics, must reproduce ``hughes_analytical(scn)``
    bit-for-bit.

    The construction:

    - one unit type per side, all in ``domain`` (default: SURFACE),
    - σ_o = σ_d = η_o = η_d = 1 (no aiming / scouting / training
      degradation),
    - p_offense and p_defense set so that the kernel products
      reproduce the (α, β, y, z) of Hughes / Armstrong,
    - admissibility set to the degenerate "(d, d) = 1" matrix.

    Mapping conventions:

    - Blue is mapped onto "B" of Hughes/Armstrong (Blue has β, z, w).
    - Red  is mapped onto "A" of Hughes/Armstrong (Red has α, y, x).

    With one unit type per side, the per-pair offensive kernel of the
    Red→Blue direction reduces to ``β`` (Red attacking Blue means Blue
    is firing β-shots? -- no: in Hughes the *attack on Red* uses β·B,
    so β is the Blue offensive firepower; equivalently the Red→Blue
    direction's defensive kernel is what intercepts the α-shots).

    Concretely we set, for the Blue→Red direction (Blue attacks Red):

        O_{Blue→Red} = β   (Blue's offensive firepower against Red)
        D_{Blue→Red} = y   (Red's defensive firepower against Blue's shots)

    and for the Red→Blue direction (Red attacks Blue):

        O_{Red→Blue} = α   (Red's offensive firepower against Blue)
        D_{Red→Blue} = z   (Blue's defensive firepower against Red's shots)

    Returns
    -------
    BattleState, EngagementParameters, Admissibility
    """
    # Blue ~ "B" of Hughes (staying power w).
    blue = Force(
        label=blue_label,
        unit_types=[
            UnitType("B", domain, staying_power=scn.w, initial_strength=scn.B0),
        ],
    )
    # Red ~ "A" of Hughes (staying power x).
    red = Force(
        label=red_label,
        unit_types=[
            UnitType("A", domain, staying_power=scn.x, initial_strength=scn.A0),
        ],
    )

    # Blue → Red:  O = β (Blue fires β-shots), D = y (Red intercepts β-shots).
    bar = DirectionalParameters.zeros(blue, red)
    bar.set("B", "A", PairParameters(
        sigma_offense=1.0, eta_offense=1.0, p_offense=scn.beta,
        sigma_defense=1.0, eta_defense=1.0, p_defense=scn.y,
    ))
    # Red → Blue:  O = α (Red fires α-shots), D = z (Blue intercepts α-shots).
    rab = DirectionalParameters.zeros(red, blue)
    rab.set("A", "B", PairParameters(
        sigma_offense=1.0, eta_offense=1.0, p_offense=scn.alpha,
        sigma_defense=1.0, eta_defense=1.0, p_defense=scn.z,
    ))

    ep = EngagementParameters(
        blue=blue, red=red,
        blue_attacks_red=bar,
        red_attacks_blue=rab,
    )
    bs = BattleState(blue=blue, red=red)
    # Use the JPH-degenerate admissibility (only (S, S) = 1).  If the
    # scenario uses a different domain, build a single-cell mask for it.
    if domain is Domain.SURFACE:
        adm = Admissibility.degenerate()
    else:
        M = np.zeros((5, 5), dtype=np.float64)
        M[domain.index, domain.index] = 1.0
        adm = Admissibility.from_array(M)

    return bs, ep, adm
