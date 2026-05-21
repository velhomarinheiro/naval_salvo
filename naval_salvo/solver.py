"""
naval_salvo.solver
==================

Multi-salvo runner.

Step 4 of the implementation plan introduces *campaigns* of multiple
salvos -- the basic unit of analysis the SIGE 2026 paper will report on.
The deterministic dynamics module (`dynamics/deterministic.py`) gives
us one salvo at a time; this module wraps it into a sequence with
trajectory recording, optional dynamic targeting refresh, and
configurable termination conditions.

Design choices
--------------
- **Pulsed-only for now.**  The hybrid EDO + jumps integrator (option
  C of working document 1.4) is the Step 7+ deliverable.  Until then,
  one "step" = one simultaneous salvo, indexed by integer k.
- **Targeting refresh between salvos.**  If the user supplies a
  ``TargetingPolicy`` in the runner config, the runner re-applies it
  at the start of every step *except* the first (the first uses
  whatever σ values the EngagementParameters already carry).  This
  matches the MacKay 2009 / Hausken-Moxnes 2026 semi-dynamic targeting
  semantics.
- **Trajectory recording.**  Returns a ``CampaignTrajectory`` with a
  per-step record of strengths, losses, and raw kernels for both
  sides.  This is what figure-generation code in the paper will pull
  from.
- **Termination.**  Two modes: fixed number of salvos, or stop early
  when one side becomes kinetically combat-ineffective.  Both can be
  combined.

References
----------
- Working document 1.4 §4 (architecture: ``solver.py``).
- Hughes (1995), Armstrong (2005) for the multi-salvo iteration
  semantics this runner generalises.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .admissibility import Admissibility
from .coefficients import apply_targeting_policy
from .cyber import CyberModulator
from .dynamics.deterministic import SalvoResult, salvo_step
from .parameters import EngagementParameters
from .state import BattleState
from .targeting import TargetingPolicy


# ---------------------------------------------------------------------------
# Trajectory containers
# ---------------------------------------------------------------------------


@dataclass
class CampaignTrajectory:
    """
    History of a multi-salvo campaign.

    All arrays are stacked over salvos.  Index ``k`` of the
    ``per_step`` lists refers to the state *after* salvo k+1
    (i.e. ``per_step[0]`` is the post-first-salvo state).

    Attributes
    ----------
    blue_strength_history : ndarray, shape (n_salvos + 1, n_blue_units)
        Row 0 is the initial (pre-first-salvo) strengths; rows 1..n
        are post-salvo strengths.
    red_strength_history : ndarray, shape (n_salvos + 1, n_red_units)
        Same convention.
    per_step_results : list of SalvoResult
        Detailed per-salvo diagnostics.  Length = n_salvos.
    times : ndarray, shape (n_salvos + 1,)
        Salvo timestamps (0, 1, 2, ... by default; arbitrary if the
        caller provides ``time_step``).
    terminated_early : bool
        True if the campaign stopped before reaching ``n_salvos`` due
        to a force becoming kinetically combat-ineffective.
    """

    blue_strength_history: np.ndarray
    red_strength_history: np.ndarray
    per_step_results: list
    times: np.ndarray
    terminated_early: bool

    @property
    def n_completed_salvos(self) -> int:
        """Number of salvos actually run (may be less than requested if
        the campaign ended early)."""
        return len(self.per_step_results)

    def total_strength_history_by_domain(
        self, side: str = "blue"
    ) -> dict:
        """
        Aggregate per-salvo strength totals by domain for one side.

        Returns
        -------
        dict {Domain: ndarray}
            One array per domain, of length ``n_salvos + 1``.

        Useful for the per-domain trajectory figures the paper
        requires.
        """
        from .domains import DOMAIN_ORDER, Domain
        # Pick the right history.
        side_l = side.lower()
        if side_l == "blue":
            hist = self.blue_strength_history
            unit_types = self._blue_unit_types
        elif side_l == "red":
            hist = self.red_strength_history
            unit_types = self._red_unit_types
        else:
            raise ValueError(f"Unknown side {side!r}")
        out = {d: np.zeros(hist.shape[0]) for d in DOMAIN_ORDER}
        for j, ut in enumerate(unit_types):
            out[ut.domain] += hist[:, j]
        return out

    # Unit-type metadata recorded by the solver so the trajectory
    # carries enough information to slice itself by domain after the
    # fact.  Plain attributes; populated by ``run_campaign``.
    _blue_unit_types: list = field(default_factory=list)
    _red_unit_types: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_campaign(
    state: BattleState,
    params: EngagementParameters,
    admissibility: Admissibility,
    *,
    n_salvos: int = 10,
    targeting_policy: Optional[TargetingPolicy] = None,
    cyber_modulator: Optional[CyberModulator] = None,
    stop_on_combat_ineffective: bool = True,
    time_step: float = 1.0,
) -> CampaignTrajectory:
    """
    Run a multi-salvo deterministic campaign.

    Parameters
    ----------
    state : BattleState
        Mutated in place.  After the call, the live forces hold the
        post-final-salvo strengths.
    params : EngagementParameters
        May be replaced internally by an apply_targeting_policy()-
        refreshed copy and/or a cyber-modulated copy at each step.
    admissibility : Admissibility
    n_salvos : int, default 10
        Maximum number of simultaneous salvos to run.
    targeting_policy : TargetingPolicy, optional
        If provided, σ values are recomputed at the start of every
        salvo *after* the first (semi-dynamic targeting in the
        MacKay 2009 / Hausken-Moxnes 2026 sense).  The η, p of every
        pair are preserved throughout.
    cyber_modulator : CyberModulator, optional
        If provided, the Φ modulator is applied at the start of
        *every* salvo (including the first), scaling σ and η of
        every kinetic pair (excluding submarine defenders) by the
        Φ values implied by the current cyber stocks.
    stop_on_combat_ineffective : bool, default True
        If True, terminate the campaign as soon as either side has
        zero kinetic strength across all kinetic domains.
    time_step : float, default 1.0
        Time delta written to ``trajectory.times[k+1] - times[k]``
        and recorded in ``state.salvo_times``.

    Returns
    -------
    CampaignTrajectory

    Notes
    -----
    Order of operations within each salvo step:

        1. (k > 0) Refresh σ via targeting_policy if given.
        2. Apply cyber_modulator (uses *current* cyber stocks).
        3. Run salvo_step.

    Cyber stocks are part of the state and attrite simultaneously
    with kinetic stocks during salvo_step (via the χ matrix's X-X
    cell).  The Φ modulator therefore reads the *post-previous-salvo*
    cyber stocks at step k+1, exactly the canonical reading.
    """
    if n_salvos < 1:
        raise ValueError(f"n_salvos must be >= 1; got {n_salvos}.")

    blue_hist = [state.blue.strength_vector()]
    red_hist = [state.red.strength_vector()]
    times = [state.time]
    per_step_results: list[SalvoResult] = []
    terminated_early = False

    current_params = params
    for k in range(n_salvos):
        if k > 0 and targeting_policy is not None:
            current_params = apply_targeting_policy(
                current_params, admissibility, targeting_policy
            )
        if cyber_modulator is not None:
            current_params = cyber_modulator.apply(current_params)
        state.time = times[-1] + time_step
        result = salvo_step(
            state,
            current_params,
            admissibility,
            apply=True,
            record_time=state.time,
        )
        per_step_results.append(result)
        blue_hist.append(state.blue.strength_vector())
        red_hist.append(state.red.strength_vector())
        times.append(state.time)

        if stop_on_combat_ineffective and state.is_terminated():
            terminated_early = True
            break

    traj = CampaignTrajectory(
        blue_strength_history=np.array(blue_hist),
        red_strength_history=np.array(red_hist),
        per_step_results=per_step_results,
        times=np.array(times),
        terminated_early=terminated_early,
    )
    # Record the unit-type lists so that ``total_strength_history_by_domain``
    # can slice the history without reaching back into BattleState (which
    # the user may continue to mutate).
    traj._blue_unit_types = list(state.blue.unit_types)
    traj._red_unit_types = list(state.red.unit_types)
    return traj
