"""
salvo_mds.py — Multi-domain / multi-salvo stochastic salvo engine (per-unit resolution).

Implements the LAFusion 2026 experiment spec, section 3:
  - exact aggregation of offensive/defensive throughput
  - pooled area defense (interceptors summed at force level)
  - per-hull leaker allocation under UNIFORM (with-replacement) targeting
  - per-hull accumulated damage vs per-type staying power (kill when dmg >= 1)
  - finite magazines (multi-salvo stopping rule)
  - scouting sigma (offense multiplier, applied every salvo) and readiness tau (defense multiplier)
  - engagement order: opening-salvo surprise (blue_first / red_first) then simultaneous

Setting Tmax=1 recovers the single-salvo model exactly (used for validation anchors).
Dependency: numpy only.

--------------------------------------------------------------------------------
FIXED DEFINITIONS (spec sec 9 mandates fixing/documenting these before running):

  * "Effective combat power" (R1/R2/R3) == residual OFFENSIVE throughput of the
    surviving hulls: Sum(offense of alive hulls) / Sum(offense at start).
    Rationale: the paper measures the ability to "attack effectively first" and
    project fires, so combat effectiveness is indexed by residual offensive
    output. Defensive/staying capability is NOT folded into this scalar; it is
    reported separately as the residual-defense fraction (R9 capability-
    degradation profile) so the offense-only convention stays transparent.
    To change the convention, edit CP_WEIGHTS below (weights over o/d/w) — this
    shifts every response and the validation baselines, so treat it as a
    documented modelling decision, not a tuning knob.

  * theta (combat-effectiveness threshold, R3) == 0.30 by default: a side whose
    residual combat power falls to <= 30% of its initial value is "ineffective".
    Victory (R3) == Red ineffective (red_cp <= theta) AND Blue still viable
    (blue_cp > theta). theta is a documented modelling choice; it moves to a
    robustness excursion, not the primary factor set.

  * Tmax (max salvos) default 6; allocation rule default uniform-with-replacement
    (spec sec 3); both are documented settings, not primary factors.
--------------------------------------------------------------------------------
"""
from dataclasses import dataclass
import numpy as np

ENGINE_VERSION = "1.0.0"  # cite in the reproducibility package (spec sec 11)
FER_EPS = 0.01  # loss floor for the FER ratio / log-FER (caps at +/- log(1+1/EPS))

# Effective-combat-power weighting over (offense, defense, staying). The spec's
# offense-only convention is (1, 0, 0); change here to re-index combat power.
CP_WEIGHTS = (1.0, 0.0, 0.0)


@dataclass(frozen=True)
class Platform:
    name: str
    offense: float    # o : ASMs launched per salvo
    defense: float    # d : SAM interceptors available per salvo
    staying: float    # w : hits-to-kill; mean damage per hit u = 1/w
    magazine: int     # number of offensive salvos carried


# --- baseline archetypes (spec section 4, quantity-quality framework) ---
# H dominates every attribute but costs superlinearly (naval construction reality):
# larger hulls carry more offense AND defense; concentration is exponentially expensive.
L_STRIKER = Platform("L", 2, 1, 1, 2)    # missile boat / light corvette: cheap, numerous
M_BALANCED = Platform("M", 3, 2, 2, 3)   # frigate: identical to the Red standard platform
H_ESCORT = Platform("H", 6, 5, 4, 4)     # destroyer/cruiser: dominant, few

def unit_cost(p: Platform, k_c=0.4, k_m=0.5, gamma=1.35) -> float:
    """Cost function (spec section 5): c = k_c*(o+d+w)**gamma + k_m*mag.
    gamma > 1 prices capability concentration superlinearly; gamma is a design factor."""
    cap = p.offense + p.defense + p.staying
    return k_c * (cap ** gamma) + k_m * p.magazine

RED_STD = M_BALANCED  # Red standard platform == the balanced frigate


def make_force(spec):
    """spec: list of (Platform, count). Returns per-hull numpy arrays."""
    o, d, u, mag, typ = [], [], [], [], []
    for plat, n in spec:
        o += [plat.offense] * n
        d += [plat.defense] * n
        u += [1.0 / plat.staying] * n
        mag += [plat.magazine] * n
        typ += [plat.name] * n
    n_tot = len(o)
    o_arr, d_arr = np.array(o, float), np.array(d, float)
    w_arr = 1.0 / np.array(u, float)  # staying = 1/u
    cp0 = CP_WEIGHTS[0] * o_arr + CP_WEIGHTS[1] * d_arr + CP_WEIGHTS[2] * w_arr
    return {
        "o": o_arr,
        "d": d_arr,
        "u": np.array(u, float),
        "cp": cp0,                       # per-hull effective combat power (CP_WEIGHTS)
        "mag": np.array(mag, int),
        "dmg": np.zeros(n_tot, float),
        "alive": np.ones(n_tot, bool),
        "type": np.array(typ),
        "o0_sum": float(o_arr.sum()),    # initial offensive power
        "d0_sum": float(d_arr.sum()),    # initial defensive power (R9)
        "cp0_sum": float(cp0.sum()),     # initial effective combat power (R1/R3)
    }


def _launch(att, sigma, p_o, p_d, defn, tau, rng):
    """Return number of leakers reaching the defender. Decrements shooter magazines."""
    shooters = att["alive"] & (att["mag"] > 0)
    if not shooters.any():
        return 0
    n_off = int(round(sigma * att["o"][shooters].sum()))
    att["mag"][shooters] -= 1
    accurate = rng.binomial(n_off, p_o) if n_off > 0 else 0
    n_def = int(round(tau * defn["d"][defn["alive"]].sum()))
    intercepted = rng.binomial(n_def, p_d) if n_def > 0 else 0
    return max(accurate - intercepted, 0)


def _apply_leaker(defn, h, sd, rng):
    """Apply one leaker to hull h. Returns (delivered, wasted) for this hit."""
    add = max(rng.normal(defn["u"][h], sd), 0.0)
    if not defn["alive"][h]:
        return add, add  # hull already dead -> fully wasted
    needed = 1.0 - defn["dmg"][h]
    if add >= needed:
        defn["dmg"][h] = 1.0
        defn["alive"][h] = False
        return add, add - needed  # overkill spillover wasted
    defn["dmg"][h] += add
    return add, 0.0


def _resolve(defn, leakers, sd, rng, alloc="with_replacement"):
    """Allocate leakers to alive hulls and apply damage.

    alloc = "with_replacement" (spec default, sec 3): each leaker independently
        hits a uniformly-random alive hull -> concentration/overkill possible.
    alloc = "without_replacement" (robustness excursion, sec 10.3): leakers are
        spread to distinct hulls first (a fresh random permutation of the
        currently-alive hulls is drawn each time the bag empties), which
        minimises overkill. Qualitative conclusions and the sign of R5/R6 should
        be unchanged; only R10 (overkill) shifts.

    Returns (delivered, wasted) damage as floats. Wasted = overkill + hits on
    already-dead hulls."""
    if leakers <= 0:
        return 0.0, 0.0
    alive_idx = np.where(defn["alive"])[0]
    if len(alive_idx) == 0:
        return 0.0, 0.0
    delivered = wasted = 0.0
    if alloc == "without_replacement":
        bag = list(rng.permutation(alive_idx))
        for _ in range(leakers):
            if not bag:
                bag = list(rng.permutation(np.where(defn["alive"])[0]))
                if not bag:
                    break  # every hull dead
            h = int(bag.pop())
            dv, ws = _apply_leaker(defn, h, sd, rng)
            delivered += dv
            wasted += ws
    else:
        targets = alive_idx[rng.integers(0, len(alive_idx), size=leakers)]
        for h in targets:
            dv, ws = _apply_leaker(defn, int(h), sd, rng)
            delivered += dv
            wasted += ws
    return delivered, wasted


def _cp_frac(f):
    """Residual EFFECTIVE combat-power fraction (alive CP / initial CP).
    CP is the CP_WEIGHTS combination over offense/defense/staying (default:
    offense-only, per the documented spec-sec-9 convention at the top of file)."""
    if f["cp0_sum"] == 0:
        return 0.0
    return float(f["cp"][f["alive"]].sum()) / f["cp0_sum"]


def _def_frac(f):
    """Residual DEFENSIVE-power fraction (R9 capability-degradation profile):
    alive defensive throughput / initial. Tracked separately from combat power
    so the offense-only CP convention stays transparent."""
    if f["d0_sum"] == 0:
        return 0.0
    return float(f["d"][f["alive"]].sum()) / f["d0_sum"]


def simulate(blue_spec, red_spec, params, rng):
    """One battle. Returns a dict of responses (spec section 9)."""
    B = make_force(blue_spec)
    R = make_force(red_spec)
    order = params.get("order", "simultaneous")
    Tmax = params.get("Tmax", 6)
    theta = params.get("theta", 0.30)
    p_oB, p_oR = params["p_o"], params.get("p_o_red", params["p_o"])
    p_dB, p_dR = params["p_d"], params.get("p_d_red", params["p_d"])
    sB, sR = params["sigma_b"], params["sigma_r"]
    tB, tR = params.get("tau_b", 1.0), params.get("tau_r", 1.0)
    sd = params["sd"]
    alloc = params.get("alloc", "with_replacement")  # sec 10.3 robustness excursion

    delivered_BR = wasted_BR = 0.0  # blue -> red offensive efficiency
    salvos = 0
    for t in range(1, Tmax + 1):
        salvos = t
        if t == 1 and order == "blue_first":
            lk = _launch(B, sB, p_oB, p_dR, R, tR, rng)
            dv, ws = _resolve(R, lk, sd, rng, alloc); delivered_BR += dv; wasted_BR += ws
            lk = _launch(R, sR, p_oR, p_dB, B, tB, rng)
            _resolve(B, lk, sd, rng, alloc)
        elif t == 1 and order == "red_first":
            lk = _launch(R, sR, p_oR, p_dB, B, tB, rng)
            _resolve(B, lk, sd, rng, alloc)
            lk = _launch(B, sB, p_oB, p_dR, R, tR, rng)
            dv, ws = _resolve(R, lk, sd, rng, alloc); delivered_BR += dv; wasted_BR += ws
        else:
            # simultaneous: both launch from start-of-round survivors, then resolve
            lk_b = _launch(B, sB, p_oB, p_dR, R, tR, rng)
            lk_r = _launch(R, sR, p_oR, p_dB, B, tB, rng)
            dv, ws = _resolve(R, lk_b, sd, rng, alloc); delivered_BR += dv; wasted_BR += ws
            _resolve(B, lk_r, sd, rng, alloc)

        blue_cp, red_cp = _cp_frac(B), _cp_frac(R)
        blue_dead = not B["alive"].any()
        red_dead = not R["alive"].any()
        mags_empty = (B["mag"][B["alive"]].sum() == 0) and (R["mag"][R["alive"]].sum() == 0)
        if blue_dead or red_dead or blue_cp <= theta or red_cp <= theta or mags_empty:
            break

    blue_cp, red_cp = _cp_frac(B), _cp_frac(R)
    blue_loss = 1.0 - blue_cp
    red_loss = 1.0 - red_cp
    # R4 (spec sec 9): FER-analog, logit-transformed for comparability.
    # The raw ratio (red_loss/blue_loss) is a ratio of random variables whose
    # per-rep mean is dominated by near-zero denominators (E[X/Y] != E[X]/E[Y]).
    # log-FER = logit of Red's loss-share red_loss/(red_loss+blue_loss); it is
    # symmetric (mirror match -> 0), finite, and averages meaningfully.
    # exp(mean(logfer)) recovers the geometric-mean FER (see monte_carlo).
    fer = (red_loss + FER_EPS) / (blue_loss + FER_EPS)  # raw ratio (kept for reference)
    logfer = float(np.log(fer))
    victory = (red_cp <= theta) and (blue_cp > theta)
    survivors = {str(nm): int(np.sum((B["type"] == nm) & B["alive"])) for nm in np.unique(B["type"])}
    overkill = (wasted_BR / delivered_BR) if delivered_BR > 0 else 0.0
    return {
        "blue_loss": blue_loss, "red_loss": red_loss, "fer": fer, "logfer": logfer,
        "victory": victory, "salvos": salvos, "survivors": survivors,
        "overkill_frac": overkill,
        "blue_ships_lost": int(np.sum(~B["alive"])),
        "red_ships_lost": int(np.sum(~R["alive"])),
        # R9 capability-degradation profile: residual DEFENSIVE power (offense
        # loss is already in blue_loss/red_loss under the offense-only CP rule).
        "blue_def_residual": _def_frac(B),
        "red_def_residual": _def_frac(R),
    }


def monte_carlo(blue_spec, red_spec, params, reps=10000, seed=0):
    """Run many battles; return aggregated responses."""
    rng = np.random.default_rng(seed)
    keys = ["blue_loss", "red_loss", "fer", "logfer", "salvos", "overkill_frac",
            "blue_ships_lost", "red_ships_lost",
            "blue_def_residual", "red_def_residual"]
    acc = {k: np.empty(reps) for k in keys}
    wins = 0
    for i in range(reps):
        r = simulate(blue_spec, red_spec, params, rng)
        for k in keys:
            acc[k][i] = r[k]
        wins += r["victory"]
    out = {f"{k}_mean": float(acc[k].mean()) for k in keys}
    out.update({f"{k}_sd": float(acc[k].std()) for k in keys})
    out["p_victory"] = wins / reps
    # Geometric-mean FER = exp(mean log-FER): the robust R4 point estimate.
    out["fer_geom"] = float(np.exp(out["logfer_mean"]))
    return out
