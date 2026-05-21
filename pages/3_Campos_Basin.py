"""
Page: Campos Basin (configurable multi-domain scenario).

Instead of using the fixed ``BaciaCamposConfig`` (which only exposes
``n_frigates`` / ``submarine_present`` / cyber counts), we build the
scenario directly with the ``EngagementBuilder`` from the ``naval_salvo``
package. This lets the user edit directly in the interface:

- Blue side (Navy):
    * 2 different surface unit types (qty, staying, p_offense, p_defense
      for each type);
    * Submarines -- 1 type, configurable quantity (staying, p_off, p_def);
    * Coastal battery -- qty, staying, p_off, p_def;
    * FPSO -- qty and staying (no offensive power, value asset);
    * Blue Cyber -- qty per sub-type (C2, SEN, WPN, LOG).
- Red side:
    * 2 different surface unit types (qty, staying, p_off, p_def);
    * Strike aviation -- qty, staying, p_off, p_def;
    * Red Cyber -- qty per sub-type.

The page dynamically generates the units, assembles the
``EngagementParameters`` with sensible defaults for the pairs, and runs
``run_campaign``. Shows domain trajectories and final state table.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from naval_salvo import (
    Admissibility,
    ChannelPhi,
    Domain,
    EngagementBuilder,
    Manual,
    StrengthProportional,
    ThreatWeighted,
    Uniform,
    UnitType,
    run_campaign,
)
from naval_salvo.targeting import TargetingPolicy


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Campos Basin -- multi-domain",
    page_icon="⚓",
    layout="wide",
)

st.title("⚓ Multi-domain scenario -- Campos Basin")
st.markdown(
    """
    Configure below the composition of both forces, their capability
    parameters, and the engagement matrices. The simulation runs the
    multi-domain salvo model (extended Hughes 1995) with the canonical
    admissibility matrix and, optionally, the cyber modulator Φ per channel.
    """
)


# ---------------------------------------------------------------------------
# Defaults inspired by the original paper scenario (BACIA_CAMPOS_PARAMETERS).
# Editable at runtime; each class becomes a UnitType only if ``quantity > 0``.
# ---------------------------------------------------------------------------

@dataclass
class ShipClassInputs:
    """Editable inputs for a kinetic unit class.

    The ``p_off`` and ``p_def`` fields represent the **raw power** of the
    platform — before applying composite factors. When building the
    engagement matrix, the effective offensive/defensive power is:

        effective_p_off = p_off × scouting × training
        effective_p_def = p_def × training × alert

    Attacker factors apply to p_off; defender factors apply to p_def.
    There is no cross-coupling (target's alert does not affect the
    opponent's attack). This decomposition follows JPH 2001 eq. 2.18.
    """

    name: str
    quantity: int
    staying: float
    p_off: float
    p_def: float
    scouting: float = 1.0     # σ_scout ∈ [0, 1]   — attacker maintains contact
    training: float = 1.0     # τ_train ∈ [0, 1]   — operator effectiveness
    alert:    float = 1.0     # α_alert ∈ [0, 1]   — defensive alert level


# Cyber: 4 sub-types, all with staying = 1 and p_off against other
# cyber sub-types default. Quantity per sub-type is the only editable
# field (same as the paper).
CYBER_SUBTYPES = ["C2", "SEN", "WPN", "LOG"]
CYBER_DEFAULT_STAYING = 1.0
CYBER_DEFAULT_P_OFF = 0.5      # against other cyber sub-types
CYBER_DEFAULT_P_DEF = 0.2


# ---------------------------------------------------------------------------
# Editable defaults -- "Campos Basin" values from the paper.
# Composite factors (scouting, training, alert) default to 1.0 (neutral
# effect). The user adjusts them to model operational friction, doctrinal
# advantage, surprise, etc.
# ---------------------------------------------------------------------------

BLUE_DEFAULTS = {
    # 2 surface types: main frigate + light corvette/patrol
    "surface_1": ShipClassInputs("Class A Frigate", 2, 3.0, 1.5, 1.0,
                                  scouting=1.0, training=1.0, alert=1.0),
    "surface_2": ShipClassInputs("Corvette/Patrol", 2, 2.0, 1.0, 0.6,
                                  scouting=1.0, training=1.0, alert=1.0),
    "submarine": ShipClassInputs("Conventional submarine", 1, 2.0, 2.0, 0.0,
                                  scouting=1.0, training=1.0, alert=1.0),
    "strike_air": ShipClassInputs("Blue strike aviation", 0, 1.0, 1.8, 0.3,
                                   scouting=1.0, training=1.0, alert=1.0),
    "coastal":   ShipClassInputs("Coastal battery", 1, 4.0, 1.2, 0.4,
                                  scouting=1.0, training=1.0, alert=1.0),
    # FPSO: value asset, high staying, no offensive power.
    "fpso":      ShipClassInputs("FPSO (Pre-Salt)", 4, 6.0, 0.0, 0.0,
                                  scouting=1.0, training=1.0, alert=1.0),
}

RED_DEFAULTS = {
    "surface_1":   ShipClassInputs("Destroyer", 2, 4.0, 2.5, 1.5,
                                    scouting=1.0, training=1.0, alert=1.0),
    "surface_2":   ShipClassInputs("Adv. Frigate", 2, 3.0, 1.8, 1.0,
                                    scouting=1.0, training=1.0, alert=1.0),
    "submarine":   ShipClassInputs("Adversary submarine", 0, 2.0, 2.0, 0.0,
                                    scouting=1.0, training=1.0, alert=1.0),
    "strike_air":  ShipClassInputs("Red strike aviation", 4, 1.0, 2.0, 0.3,
                                    scouting=1.0, training=1.0, alert=1.0),
}


# ---------------------------------------------------------------------------
# Sidebar -- global settings
# ---------------------------------------------------------------------------

st.sidebar.header("Global settings")
n_salvos = st.sidebar.slider(
    "Maximum number of salvos", min_value=1, max_value=30, value=15, step=1,
    help="The campaign may end earlier if a force is neutralised."
)
use_cyber = st.sidebar.checkbox(
    "Enable cyber modulator Φ (channel)", value=True,
    help="Applies the ChannelPhi modulator from the package, which scales "
         "η_offense/η_defense as a function of the cyber strength ratio per "
         "channel (σ, ρ, δ). Submarines are immune."
)
stop_on_termination = st.sidebar.checkbox(
    "End campaign on force collapse", value=True
)

st.sidebar.markdown("---")
st.sidebar.subheader("Targeting policy")
targeting_policy_name = st.sidebar.selectbox(
    "How does each attacker distribute its salvo among admissible targets?",
    options=[
        "StrengthProportional",
        "Uniform",
        "ThreatWeighted",
        "Manual",
    ],
    index=0,
    help=(
        "StrengthProportional: σ proportional to the remaining strength of "
        "the target (MacKay 2009 / Hausken-Moxnes 2026).\n\n"
        "Uniform: equal split among all admissible targets "
        "(Hughes 1995 default).\n\n"
        "ThreatWeighted: σ proportional to editable per-class weights — "
        "useful for prioritising FPSOs.\n\n"
        "Manual: the user defines σ cell by cell in a fixed matrix "
        "(same across all salvos)."
    ),
)
_policy_descriptions = {
    "StrengthProportional":
        "Each salvo is distributed in proportion to the *current* strength "
        "of the target. σ is recomputed before each salvo (semi-dynamic).",
    "Uniform":
        "Each attacker divides its salvo equally among all doctrinally "
        "admissible targets. σ is fixed at the first salvo.",
    "ThreatWeighted":
        "Configure per-class weights in the 'Targeting' tab. "
        "σ is recomputed before each salvo.",
    "Manual":
        "Configure the σ_offense matrices cell by cell in the "
        "'Targeting' tab. σ is fixed across all salvos.",
}
st.sidebar.caption(_policy_descriptions[targeting_policy_name])


# ---------------------------------------------------------------------------
# Helpers for building the widgets of each class.
# ---------------------------------------------------------------------------

def _ship_class_inputs(
    label: str,
    defaults: ShipClassInputs,
    *,
    key_prefix: str,
    allow_zero_quantity: bool = True,
    locked_p_off: bool = False,
    locked_p_def: bool = False,
    help_text: str = "",
) -> ShipClassInputs:
    """Render the widgets for a unit class in an expander.

    Fields collected: name, quantity, staying power, offensive power,
    defensive power, and the 4 composite factors (scouting, training,
    alert, and the complementary alert used as distraction).

    If ``allow_zero_quantity`` is True, quantity 0 omits the class from
    the force (useful for disabling submarine, aviation, second class etc).
    """
    with st.expander(label, expanded=True):
        if help_text:
            st.caption(help_text)
        name = st.text_input(
            "Class name", value=defaults.name, key=f"{key_prefix}_name"
        )
        c1, c2 = st.columns(2)
        quantity = c1.number_input(
            "Quantity", min_value=0 if allow_zero_quantity else 1,
            value=defaults.quantity, step=1, key=f"{key_prefix}_qty"
        )
        staying = c2.number_input(
            "Staying power (ς)", min_value=0.1,
            value=float(defaults.staying), step=0.1, format="%.2f",
            key=f"{key_prefix}_staying",
            help="Number of hits each unit absorbs before being "
                 "rendered combat-ineffective."
        )
        c3, c4 = st.columns(2)
        if locked_p_off:
            p_off = 0.0
            c3.markdown("**Offensive power (p_off):** _0 (no offensive capability)_")
        else:
            p_off = c3.number_input(
                "Raw offensive power (p_off)", min_value=0.0,
                value=float(defaults.p_off), step=0.1, format="%.2f",
                key=f"{key_prefix}_poff",
                help="Hits per unit per salvo against the default target, "
                     "before applying composite factors."
            )
        if locked_p_def:
            p_def = 0.0
            c4.markdown("**Defensive power (p_def):** _0 (no active defence)_")
        else:
            p_def = c4.number_input(
                "Raw defensive power (p_def)", min_value=0.0,
                value=float(defaults.p_def), step=0.1, format="%.2f",
                key=f"{key_prefix}_pdef",
                help="Intercepts per unit per salvo, before applying "
                     "composite factors."
            )

        # ---- Composite factors ----
        st.markdown("**Composite factors** (JPH 2001, eq. 2.18)")
        f1, f2, f3 = st.columns(3)
        scouting = f1.slider(
            "Scouting (σ_scout)", min_value=0.0, max_value=1.0,
            value=float(defaults.scouting), step=0.05,
            key=f"{key_prefix}_scout",
            help="Fraction of time the unit maintains valid cueing "
                 "on the target. Applied to p_off."
        )
        training = f2.slider(
            "Training (τ_train)", min_value=0.0, max_value=1.0,
            value=float(defaults.training), step=0.05,
            key=f"{key_prefix}_train",
            help="Average operator/system effectiveness. Applied to "
                 "p_off and p_def."
        )
        alert = f3.slider(
            "Alert (α_alert)", min_value=0.0, max_value=1.0,
            value=float(defaults.alert), step=0.05,
            key=f"{key_prefix}_alert",
            help="Defensive alert level. Applied only to this unit's "
                 "p_def (multiplies defensive power)."
        )

        return ShipClassInputs(
            name=name, quantity=int(quantity),
            staying=float(staying),
            p_off=float(p_off), p_def=float(p_def),
            scouting=float(scouting),
            training=float(training),
            alert=float(alert),
        )


def _cyber_inputs(
    label: str, key_prefix: str, default_per_subtype: int = 2
) -> dict[str, int]:
    """Render quantities of the 4 cyber sub-types."""
    with st.expander(label, expanded=False):
        st.caption(
            "Cyber stocks per sub-type. C2 = command and control; "
            "SEN = sensors/ISR; WPN = weapon systems; LOG = logistics."
        )
        cols = st.columns(4)
        out: dict[str, int] = {}
        for col, sub in zip(cols, CYBER_SUBTYPES):
            out[sub] = int(col.number_input(
                sub, min_value=0, max_value=20,
                value=default_per_subtype, step=1,
                key=f"{key_prefix}_cyber_{sub}",
            ))
        return out


# ---------------------------------------------------------------------------
# Forms: two tabs (Blue / Red) + one for engagement matrix
# ---------------------------------------------------------------------------

tab_blue, tab_red, tab_engagement, tab_targeting, tab_results = st.tabs(
    [
        "🔵 Blue Force (Navy)",
        "🔴 Red Force",
        "⚔️ Engagement matrix",
        "🎯 Targeting",
        "📊 Results",
    ]
)


# ---- Blue -----------------------------------------------------------------

with tab_blue:
    st.subheader("Blue Force Composition")
    st.info(
        "ℹ️ The **offensive power** and **defensive power** fields "
        "entered here are used to pre-fill the "
        "**⚔️ Engagement matrix** tab. The simulation calculation uses "
        "the values in the matrix at execution time — "
        "if you have refined any cell manually, those specific values "
        "take precedence. Quantities and *staying power*, "
        "however, are read directly from this tab."
    )
    blue_s1 = _ship_class_inputs(
        "Surface -- Type 1",
        BLUE_DEFAULTS["surface_1"],
        key_prefix="blue_s1",
        help_text="First surface unit class (e.g.: frigate)."
    )
    blue_s2 = _ship_class_inputs(
        "Surface -- Type 2",
        BLUE_DEFAULTS["surface_2"],
        key_prefix="blue_s2",
        help_text="Second surface unit class (e.g.: corvette). "
                  "Use quantity 0 to disable."
    )
    blue_sub = _ship_class_inputs(
        "Submarines",
        BLUE_DEFAULTS["submarine"],
        key_prefix="blue_sub",
        help_text="One submarine class; configurable quantity. "
                  "Immune to cyber by model construction."
    )
    blue_air = _ship_class_inputs(
        "Strike aviation",
        BLUE_DEFAULTS["strike_air"],
        key_prefix="blue_air",
        help_text="Strike aircraft/UAVs. Quantity 0 disables this class."
    )
    blue_coastal = _ship_class_inputs(
        "Coastal battery",
        BLUE_DEFAULTS["coastal"],
        key_prefix="blue_coastal",
        help_text="Coastal anti-ship missile battery."
    )
    blue_fpso = _ship_class_inputs(
        "FPSO (value asset)",
        BLUE_DEFAULTS["fpso"],
        key_prefix="blue_fpso",
        locked_p_off=True, locked_p_def=True,
        help_text="Pre-Salt platforms: assets to protect, with no offensive "
                  "or active defensive capability. Staying power only."
    )
    blue_cyber = _cyber_inputs(
        "Blue Cyber (per sub-type)", key_prefix="blue",
        default_per_subtype=0,
    )


# ---- Red ------------------------------------------------------------------

with tab_red:
    st.subheader("Red Force Composition")
    st.info(
        "ℹ️ The **offensive power** and **defensive power** fields "
        "entered here are used to pre-fill the "
        "**⚔️ Engagement matrix** tab. The simulation calculation uses "
        "the values in the matrix at execution time — "
        "if you have refined any cell manually, those specific values "
        "take precedence. Quantities and *staying power*, "
        "however, are read directly from this tab."
    )
    red_s1 = _ship_class_inputs(
        "Surface -- Type 1",
        RED_DEFAULTS["surface_1"],
        key_prefix="red_s1",
        help_text="First adversary surface unit class."
    )
    red_s2 = _ship_class_inputs(
        "Surface -- Type 2",
        RED_DEFAULTS["surface_2"],
        key_prefix="red_s2",
        help_text="Second adversary class. Quantity 0 disables."
    )
    red_sub = _ship_class_inputs(
        "Submarines",
        RED_DEFAULTS["submarine"],
        key_prefix="red_sub",
        help_text="Adversary submarines. Quantity 0 disables. "
                  "Immune to cyber by model construction."
    )
    red_air = _ship_class_inputs(
        "Strike aviation",
        RED_DEFAULTS["strike_air"],
        key_prefix="red_air",
        help_text="Adversary strike aircraft/UAVs."
    )
    red_cyber = _cyber_inputs(
        "Red Cyber (per sub-type)", key_prefix="red",
        default_per_subtype=0,
    )


# ---------------------------------------------------------------------------
# Build the list of active UnitTypes for each side.
# ---------------------------------------------------------------------------

def _build_unit_types(
    classes: list[tuple[ShipClassInputs, Domain, Optional[str]]],
    cyber_counts: dict[str, int],
) -> list[UnitType]:
    """Build active UnitTypes (qty > 0) in the provided order.

    ``classes`` is a list of (inputs, domain, optional-subtype).
    Also includes cyber sub-types with qty > 0 at the end.
    """
    out: list[UnitType] = []
    used_names: set[str] = set()
    for inp, domain, subtype in classes:
        if inp.quantity <= 0:
            continue
        nm = inp.name
        # Ensure uniqueness within the Force.
        base = nm
        k = 2
        while nm in used_names:
            nm = f"{base} ({k})"
            k += 1
        used_names.add(nm)
        out.append(UnitType(
            name=nm, domain=domain,
            staying_power=inp.staying,
            initial_strength=float(inp.quantity),
            subtype=subtype,
        ))
    # Cyber sub-types
    for sub, qty in cyber_counts.items():
        if qty <= 0:
            continue
        nm = f"Cyber-{sub}"
        used_names.add(nm)
        out.append(UnitType(
            name=nm, domain=Domain.CYBER,
            staying_power=CYBER_DEFAULT_STAYING,
            initial_strength=float(qty),
            subtype=sub,
        ))
    return out


# Keeps reference of (effective name, original inputs, role) for later
# use when filling the p_offense / p_defense matrices.
def _resolve_blue_classes() -> list[tuple[ShipClassInputs, Domain, Optional[str], str]]:
    """Return (inputs, domain, subtype, role-key) for each Blue class."""
    return [
        (blue_s1, Domain.SURFACE, None, "blue_s1"),
        (blue_s2, Domain.SURFACE, None, "blue_s2"),
        (blue_sub, Domain.UNDERWATER, None, "blue_sub"),
        (blue_air, Domain.AIR, None, "blue_air"),
        (blue_coastal, Domain.COASTAL, None, "blue_coastal"),
        (blue_fpso, Domain.SURFACE, "pre-salt", "blue_fpso"),
    ]


def _resolve_red_classes() -> list[tuple[ShipClassInputs, Domain, Optional[str], str]]:
    return [
        (red_s1, Domain.SURFACE, None, "red_s1"),
        (red_s2, Domain.SURFACE, None, "red_s2"),
        (red_sub, Domain.UNDERWATER, None, "red_sub"),
        (red_air, Domain.AIR, None, "red_air"),
    ]


# Build the ordered lists and corresponding UnitTypes.
blue_resolved = _resolve_blue_classes()
red_resolved = _resolve_red_classes()

# The builder needs the final name assigned to each UnitType.
def _materialize_force(resolved, cyber_counts):
    classes_for_builder = [(r[0], r[1], r[2]) for r in resolved]
    uts = _build_unit_types(classes_for_builder, cyber_counts)
    # Maps role_key -> final name, for active classes.
    role_to_name: dict[str, str] = {}
    idx = 0
    for inp, _, _, role in resolved:
        if inp.quantity <= 0:
            continue
        role_to_name[role] = uts[idx].name
        idx += 1
    return uts, role_to_name


blue_unit_types, blue_role_to_name = _materialize_force(blue_resolved, blue_cyber)
red_unit_types,  red_role_to_name  = _materialize_force(red_resolved,  red_cyber)


# ---------------------------------------------------------------------------
# Tab: Engagement matrix (p_offense per pair)
# ---------------------------------------------------------------------------

# Defaults are built DYNAMICALLY from the ShipClassInputs of the Blue and
# Red tabs at each rerun, ensuring that changes to the p_offense / p_defense
# fields propagate automatically to the engagement matrix.
#
# Fill rule (JPH 2001 eq. 2.18, multiplicative decomposition):
#
#   p_off[atk → def] = atk.p_off
#                     × atk.scouting               (σ_scout)
#                     × atk.training               (τ_train of attacker)
#
#   p_def[atk → def] = def.p_def
#                     × def.training               (τ_train of defender)
#                     × def.alert                  (α_alert of defender)
#
# Factor interpretation:
#  - scouting (attacker): fraction of time with valid cueing on target.
#  - training: average operator effectiveness, applied separately to
#    each side (attacker uses its own; defender uses its own).
#  - alert (defender): defensive alert level — increases p_def,
#    without double-counting the other side.
#
# Structural exceptions:
#   submarine → aircraft: 0  (null admissibility)

def _make_defaults(
    atk_resolved: list,
    def_resolved: list,
) -> tuple[dict, dict]:
    """Return (p_off_defaults, p_def_defaults) indexed by role-keys.

    Applies the multiplicative decomposition of composite factors to
    the raw p_off and p_def of the classes (JPH 2001 eq. 2.18).
    """
    p_off: dict[tuple[str, str], float] = {}
    p_def: dict[tuple[str, str], float] = {}
    for a_inp, a_dom, _a_sub, a_role in atk_resolved:
        for d_inp, d_dom, _d_sub, d_role in def_resolved:
            key = (a_role, d_role)
            if a_dom is Domain.UNDERWATER and d_dom is Domain.AIR:
                p_off[key] = 0.0
                p_def[key] = 0.0
                continue
            # Offensive side: attacker factors only.
            p_off[key] = (
                float(a_inp.p_off)
                * float(a_inp.scouting)
                * float(a_inp.training)
            )
            # Defensive side: defender factors only.
            p_def[key] = (
                float(d_inp.p_def)
                * float(d_inp.training)
                * float(d_inp.alert)
            )
    return p_off, p_def


# Generate defaults dynamically from CURRENT inputs in the force tabs.
DEFAULT_P_OFF_BAR, DEFAULT_P_DEF_BAR = _make_defaults(
    blue_resolved, red_resolved
)
DEFAULT_P_OFF_RAB, DEFAULT_P_DEF_RAB = _make_defaults(
    red_resolved, blue_resolved
)


def _build_throughput_grid(
    direction: str,
    attacker_roles_active: list[tuple[str, str]],  # (role, final_name)
    defender_roles_active: list[tuple[str, str]],
    defaults_off: dict[tuple[str, str], float],
    defaults_def: dict[tuple[str, str], float],
) -> tuple[dict[tuple[str, str], float], dict[tuple[str, str], float]]:
    """Render data_editor and return {(atk_name, def_name): value} dicts."""
    if not attacker_roles_active or not defender_roles_active:
        st.info(
            f"Direction {direction}: one of the forces has no active units."
        )
        return {}, {}

    atk_role_to_name = dict(attacker_roles_active)
    def_role_to_name = dict(defender_roles_active)

    def _df_from_defaults(defaults: dict[tuple[str, str], float]) -> pd.DataFrame:
        rows = []
        index = []
        for atk_role, atk_name in attacker_roles_active:
            index.append(atk_name)
            row = {}
            for def_role, def_name in defender_roles_active:
                row[def_name] = float(defaults.get((atk_role, def_role), 0.0))
            rows.append(row)
        return pd.DataFrame(rows, index=index)

    st.markdown(f"**{direction}** — attacker (rows) × defender (columns)")
    def _defaults_hash(d: dict) -> str:
        vals = sorted((f"{k[0]}:{k[1]}:{v:.4f}" for k, v in d.items()))
        return str(hash(tuple(vals)) & 0xFFFFFFFF)

    sig = (
        "|".join(n for _, n in attacker_roles_active) + "@" +
        "|".join(n for _, n in defender_roles_active) + "#" +
        _defaults_hash(defaults_off) + _defaults_hash(defaults_def)
    )
    c1, c2 = st.columns(2)
    with c1:
        st.caption("Offensive power p_offense (hits/attacker/salvo)")
        df_off = st.data_editor(
            _df_from_defaults(defaults_off),
            key=f"poff_{direction}_{sig}",
            use_container_width=True,
            num_rows="fixed",
        )
    with c2:
        st.caption("Defensive power p_defense (intercepts/defender/salvo)")
        df_def = st.data_editor(
            _df_from_defaults(defaults_def),
            key=f"pdef_{direction}_{sig}",
            use_container_width=True,
            num_rows="fixed",
        )

    out_off: dict[tuple[str, str], float] = {}
    out_def: dict[tuple[str, str], float] = {}
    for atk_role, atk_name in attacker_roles_active:
        for def_role, def_name in defender_roles_active:
            out_off[(atk_name, def_name)] = float(df_off.loc[atk_name, def_name])
            out_def[(atk_name, def_name)] = float(df_def.loc[atk_name, def_name])
    return out_off, out_def


with tab_engagement:
    st.subheader("Engagement matrix")

    st.markdown(
        """
        This tab contains the parameters that translate **how each platform
        fights** — as opposed to the previous tabs, which only define *how many*
        platforms exist and what their resilience is. There are two quantities
        per ordered pair (attacker → defender):

        - **p_offense** — offensive power: the expected number of
          *effective hits per attacking unit per salvo* against a
          target of the defending class. Compositely incorporates
          the amount of ammunition fired, the weapon system accuracy,
          and the probability of penetrating the target's point defence.
        - **p_defense** — defensive power: the expected number of
          *intercepts per defending unit per salvo* against missiles
          or projectiles of the attacking class. Represents the
          *area-denial* capability of each platform.

        These two parameters, together with the effectiveness coefficients
        η (kept at 1.0 by default) and the fire fraction σ (defined
        by the *targeting* policy), form the **salvo equation kernel**
        for each pair. Loss assignment per salvo follows:

        > **Net losses** ∝ max(0, Σⱼ σ · η · p_offense · Bⱼ
        > − Σⱼ σ · η · p_defense · Aᵢ) / *staying power*

        The `max(0, ·)` sign expresses the fundamental principle of
        the model: unintercepted fire causes losses; fire fully absorbed
        by the defence causes none.
        """
    )

    with st.expander("📐 How to interpret and calibrate values", expanded=True):
        st.markdown(
            """
            #### What each cell represents

            Each cell in the p_offense table corresponds to **one
            attacker row and one defender column**. The value 1.5
            in the cell (Class A Frigate → Destroyer), for example,
            means that each blue frigate fires, in one salvo,
            the equivalent of 1.5 expected hits against a red
            destroyer — before any interception by the destroyer.

            The p_defense table is read inversely: the value 1.0
            in the cell (Class A Frigate, as *defender*, against the
            Destroyer as *attacker*) means that each blue frigate
            intercepts, on average, 1.0 of the missiles fired by each
            red destroyer in its direction.

            ---

            #### Numerical example — symmetric duel

            Consider the simplest pair: 2 Blue Frigates (class A,
            p_offense = 1.5; *staying* = 3.0) against 2 Red Destroyers
            (p_offense = 2.0; *staying* = 4.0), with
            p_defense = 1.0 on both sides and σ = η = 1.

            *First salvo (Red attacks Blue):*

            > Gross fire = 2 × 2.0 = 4.0 hits
            > Blue defence = 2 × 1.0 = 2.0 intercepts
            > Net fire = max(0 ; 4.0 − 2.0) = **2.0 hits**
            > Blue losses = 2.0 / 3.0 ≈ **0.67 frigates**

            *First salvo (Blue attacks Red):*

            > Gross fire = 2 × 1.5 = 3.0 hits
            > Red defence = 2 × 1.5 = 3.0 intercepts
            > Net fire = max(0 ; 3.0 − 3.0) = **0 hits**
            > Red losses = **0**

            The result reveals something immediate: with red defence
            of 1.5 and only 2 frigates, Blue cannot penetrate the
            adversary's protection in that salvo. To change the
            result, the user must either increase Blue's p_offense
            (improve the weapon system), reduce the number of red
            units (targeting), or incorporate the submarine
            (p_offense = 2.0 against ships, with no equivalent
            anti-torpedo defence for Red).

            ---

            #### Numerical example — aviation vs. surface asymmetry

            Red strike aviation has p_offense = 2.5 against
            FPSOs and staying = 1.0. With 4 aircraft and one FPSO with
            staying = 6.0 and no active defence (p_defense = 0):

            > Gross fire = 4 × 2.5 = 10.0 hits
            > Defence = 0
            > FPSO losses per salvo = 10.0 / 6.0 ≈ **1.67 platforms**

            4 FPSOs survive fewer than 3 salvos. If the user adds
            a Blue Frigate intercepting the aviation
            (p_defense = 1.0 against aircraft, with proportional σ):

            > Defence = 1 × 1.0 = 1.0 intercept
            > Net fire = 10.0 − 1.0 = 9.0 hits
            > Losses = 9.0 / 6.0 ≈ **1.5 platforms per salvo**

            The marginal reduction is small because 1 frigate cannot
            saturate 4-aircraft aviation. This result illustrates why
            the model has no *area defence* mechanic: the frigate's
            p_defense does not extend to neighbouring FPSOs — each
            pair is evaluated independently.

            ---

            #### Typical reference ranges

            | Interaction type | p_offense | p_defense |
            |---|---|---|
            | Surface ship vs. surface ship | 1.0 – 3.0 | 0.5 – 2.0 |
            | Submarine vs. surface ship | 1.5 – 3.0 | 0 (low ASW) |
            | Surface ship vs. aircraft | 0.3 – 1.0 | 0.5 – 1.5 |
            | Strike aviation vs. surface ship | 1.5 – 3.0 | 0.1 – 0.5 |
            | Strike aviation vs. FPSO | 2.0 – 4.0 | 0 (fixed target) |
            | Coastal battery vs. surface ship | 1.0 – 2.5 | 0.2 – 0.8 |
            | Submarine vs. aircraft | **0** (null admissibility) | — |

            p_offense values above 4.0 rarely make physical sense
            for individual platforms and normally indicate that the
            parameter is absorbing a scale factor that should be in
            the unit count. Values below 0.1 produce practically
            ineffective fire and can be replaced by 0 with no visible
            impact on results.

            For p_defense, the practical upper bound is the attacker's
            p_offense value: a defence that intercepts more projectiles
            than are fired has no operational meaning — the model
            absorbs this excess via `max(0, ·)`, but the parameter
            loses intuitive meaning. A good rule of thumb is to keep
            p_defense ≤ 0.6 × corresponding attacker's p_offense as
            a starting point for modern platforms at high risk of
            defence saturation.
            """
        )

    with st.expander("⚠️ Notes on zero-admissibility pairs"):
        st.markdown(
            """
            Some pairs are **structurally null** by the canonical
            admissibility matrix and produce no attrition regardless
            of the p_offense entered:

            - **Submarine → Aircraft**: torpedoes cannot reach airborne
              aircraft. Any p_offense value in this pair will be
              multiplied by χ = 0 and will have no effect. The pre-filled
              value of 0.0 reflects this.
            - **Cyber units → Submarine**: by an explicit doctrinal
              decision of the model, the cyber domain does not affect
              submarines. Cyber-submarine pairs are automatically zeroed
              by the engine and do not appear in this table.
            - **Kinetic units → cyber capability**: in the current model,
              kinetic platforms do not directly attrit the opponent's cyber
              capability. Cyber combat occurs only intra-domain
              (cyber vs. cyber) and is configured in the force tabs
              through the sub-type quantities.

            Editing these pairs produces no error, but also no effect —
            the result will be identical to keeping the value at 0.0.
            """
        )

    st.markdown(
        "The values below are pre-filled with the reference table defaults "
        "above and can be edited cell by cell. Pairs involving the cyber "
        "domain are filled automatically and do not appear here."
    )

    blue_active_roles = [
        (r[3], blue_role_to_name[r[3]])
        for r in blue_resolved if r[0].quantity > 0
    ]
    red_active_roles = [
        (r[3], red_role_to_name[r[3]])
        for r in red_resolved if r[0].quantity > 0
    ]

    p_off_bar, p_def_bar = _build_throughput_grid(
        "Blue → Red",
        blue_active_roles, red_active_roles,
        DEFAULT_P_OFF_BAR, DEFAULT_P_DEF_BAR,
    )
    st.markdown("---")
    p_off_rab, p_def_rab = _build_throughput_grid(
        "Red → Blue",
        red_active_roles, blue_active_roles,
        DEFAULT_P_OFF_RAB, DEFAULT_P_DEF_RAB,
    )


# ---------------------------------------------------------------------------
# Add cyber-vs-cyber and cyber-vs-kinetic pairs with reasonable defaults.
# ---------------------------------------------------------------------------

def _add_cyber_pairs(
    p_off: dict[tuple[str, str], float],
    p_def: dict[tuple[str, str], float],
    attackers: list[UnitType],
    defenders: list[UnitType],
) -> None:
    """Fill, in-place, pairs involving the cyber domain with defaults.

    - Cyber vs Cyber: p_off = CYBER_DEFAULT_P_OFF; p_def = CYBER_DEFAULT_P_DEF.
    - Cyber vs kinetic (non-submarine): small p_off (0.1) -- the main effect
      comes via the Φ modulator, not direct attrition. p_def = 0.
    - Cyber vs submarine: 0 (immune).
    - Kinetic vs cyber: 0 (kinetics do not attrit cyber directly in this model).
    """
    for atk in attackers:
        for dfn in defenders:
            key = (atk.name, dfn.name)
            if key in p_off:
                continue
            if atk.domain is Domain.CYBER and dfn.domain is Domain.CYBER:
                p_off[key] = CYBER_DEFAULT_P_OFF
                p_def[key] = CYBER_DEFAULT_P_DEF
            elif atk.domain is Domain.CYBER and dfn.domain is Domain.UNDERWATER:
                p_off[key] = 0.0
                p_def[key] = 0.0
            elif atk.domain is Domain.CYBER:
                p_off[key] = 0.1
                p_def[key] = 0.0
            else:
                # kinetic vs cyber: 0
                p_off[key] = 0.0
                p_def[key] = 0.0


# ---------------------------------------------------------------------------
# Wrapper for Manual with two matrices (one per direction).
#
# The Manual class in the package stores *one* σ matrix. Since
# apply_targeting_policy and EngagementBuilder use the same policy instance
# in both directions (Blue->Red and Red->Blue), we dispatch the correct
# matrix based on the attacker's Force label.
# ---------------------------------------------------------------------------


class _DirectionAwareManual(TargetingPolicy):
    """Manual with separate σ matrices for B→R and R→B.

    Dispatches by the attacker's ``label``. Compatible with
    ``apply_targeting_policy`` and ``run_campaign`` (which call ``compute``
    in both directions).
    """

    def __init__(
        self,
        blue_label: str,
        red_label: str,
        sigma_off_bar: np.ndarray,
        sigma_def_bar: np.ndarray,
        sigma_off_rab: np.ndarray,
        sigma_def_rab: np.ndarray,
    ) -> None:
        self.blue_label = blue_label
        self.red_label = red_label
        # Validation: entries in [0, 1] and finite.
        for name, m in [
            ("sigma_off_bar", sigma_off_bar),
            ("sigma_def_bar", sigma_def_bar),
            ("sigma_off_rab", sigma_off_rab),
            ("sigma_def_rab", sigma_def_rab),
        ]:
            arr = np.asarray(m, dtype=np.float64)
            if np.any(arr < 0.0) or np.any(arr > 1.0) or not np.all(np.isfinite(arr)):
                raise ValueError(
                    f"{name} must have values in [0, 1] and be finite."
                )
        self.sigma_off_bar = np.asarray(sigma_off_bar, dtype=np.float64)
        self.sigma_def_bar = np.asarray(sigma_def_bar, dtype=np.float64)
        self.sigma_off_rab = np.asarray(sigma_off_rab, dtype=np.float64)
        self.sigma_def_rab = np.asarray(sigma_def_rab, dtype=np.float64)

    def compute(self, attacker, defender, admissibility):
        if attacker.label == self.blue_label:
            return self.sigma_off_bar.copy(), self.sigma_def_bar.copy()
        if attacker.label == self.red_label:
            return self.sigma_off_rab.copy(), self.sigma_def_rab.copy()
        raise ValueError(
            f"Unknown attacker: label={attacker.label!r}; "
            f"expected {self.blue_label!r} or {self.red_label!r}."
        )


# ---------------------------------------------------------------------------
# Tab: Targeting -- configure the chosen policy.
# ---------------------------------------------------------------------------

# The policy instance is built here, based on what the user fills in
# this tab (if the policy has parameters). It is consumed in the
# 'Results' tab.
targeting_policy: Optional[TargetingPolicy] = None

with tab_targeting:
    st.subheader(f"Targeting policy: {targeting_policy_name}")

    blue_active_names = [
        blue_role_to_name[r[3]] for r in blue_resolved if r[0].quantity > 0
    ]
    red_active_names = [
        red_role_to_name[r[3]] for r in red_resolved if r[0].quantity > 0
    ]
    # Include cyber (they appear as UnitType with domain CYBER).
    blue_active_names_full = [ut.name for ut in blue_unit_types]
    red_active_names_full = [ut.name for ut in red_unit_types]

    if targeting_policy_name == "StrengthProportional":
        st.info(
            "σ is redistributed before each salvo, proportionally to the "
            "*current* strength of each admissible target. **No parameters "
            "to edit for this policy.**"
        )
        targeting_policy = StrengthProportional()

    elif targeting_policy_name == "Uniform":
        st.info(
            "Each attacker divides its salvo equally among all doctrinally "
            "admissible targets (admissibility > 0). "
            "**No parameters to edit for this policy.**"
        )
        targeting_policy = Uniform()

    elif targeting_policy_name == "ThreatWeighted":
        st.markdown(
            "Define per-class defender weights. Higher weights attract "
            "more fire. Weights do not need to sum to 1 — they are "
            "normalised internally. Useful, e.g., to place high weight "
            "on FPSOs."
        )
        st.caption(
            "Sigma_offense (j → i) ∝ weight(i). "
            "Sigma_defense uses per-attacker weights (all = 1 by default)."
        )

        col_blue, col_red = st.columns(2)

        # Weights on RED defenders (when Blue attacks):
        with col_blue:
            st.markdown(
                "**Weights of Red defenders (targets for Blue)**"
            )
            w_red_defenders: list[float] = []
            for nm in red_active_names_full:
                default = 1.0
                w = st.number_input(
                    nm, min_value=0.0, value=default, step=0.5,
                    format="%.2f",
                    key=f"tw_blue_targets_{nm}",
                    help="Weight of this class as a target for the Blue force."
                )
                w_red_defenders.append(float(w))

        # Weights on BLUE defenders (when Red attacks) -- FPSOs default 3.
        with col_red:
            st.markdown(
                "**Weights of Blue defenders (targets for Red)**"
            )
            w_blue_defenders: list[float] = []
            for nm in blue_active_names_full:
                # High default for FPSO (3.0); rest 1.0.
                default = 3.0 if "FPSO" in nm.upper() else 1.0
                w = st.number_input(
                    nm, min_value=0.0, value=default, step=0.5,
                    format="%.2f",
                    key=f"tw_red_targets_{nm}",
                    help="Weight of this class as a target for the Red force."
                )
                w_blue_defenders.append(float(w))

        class _DirectionAwareThreatWeighted(TargetingPolicy):
            def __init__(self, blue_label, red_label,
                         w_blue_targets, w_red_targets):
                self.blue_label = blue_label
                self.red_label = red_label
                self._tw_bar = ThreatWeighted(
                    offensive_weights=np.array(w_red_targets,
                                               dtype=np.float64),
                )
                self._tw_rab = ThreatWeighted(
                    offensive_weights=np.array(w_blue_targets,
                                               dtype=np.float64),
                )

            def compute(self, attacker, defender, admissibility):
                if attacker.label == self.blue_label:
                    return self._tw_bar.compute(attacker, defender,
                                                 admissibility)
                if attacker.label == self.red_label:
                    return self._tw_rab.compute(attacker, defender,
                                                 admissibility)
                raise ValueError(f"Unknown attacker: {attacker.label!r}")

        targeting_policy = _DirectionAwareThreatWeighted(
            blue_label="Blue", red_label="Red",
            w_blue_targets=w_blue_defenders,
            w_red_targets=w_red_defenders,
        )

    elif targeting_policy_name == "Manual":
        st.markdown(
            "Define the σ_offense matrices cell by cell. Rows sum freely — "
            "it is recommended that each row sums to 1.0 (fraction of the "
            "salvo), but the package accepts values in [0, 1] that do not "
            "sum to 1 (portion of the attacker that actually engages each "
            "target). Pairs with zero admissibility will still be zeroed "
            "by the χ matrix in the engine."
        )
        st.caption(
            "σ_defense is left as zero (no manual defensive allocation). "
            "To model active proportional defence, use StrengthProportional."
        )

        def _sigma_editor(direction: str, atk_names: list[str],
                          def_names: list[str]) -> pd.DataFrame:
            if not atk_names or not def_names:
                st.info(f"{direction}: incomplete forces.")
                return pd.DataFrame()
            n_atk = len(atk_names)
            n_def = len(def_names)
            default = np.full((n_atk, n_def), 1.0 / max(n_def, 1),
                              dtype=np.float64)
            df = pd.DataFrame(default, index=atk_names, columns=def_names)
            sig = "|".join(atk_names) + "@" + "|".join(def_names)
            st.markdown(f"**{direction}**  σ_offense  (rows: "
                        f"attacker; columns: target)")
            return st.data_editor(
                df, key=f"sigma_off_{direction}_{sig}",
                use_container_width=True, num_rows="fixed",
            )

        df_sig_bar = _sigma_editor(
            "Blue → Red",
            blue_active_names_full, red_active_names_full,
        )
        st.markdown("---")
        df_sig_rab = _sigma_editor(
            "Red → Blue",
            red_active_names_full, blue_active_names_full,
        )

        if not df_sig_bar.empty and not df_sig_rab.empty:
            try:
                targeting_policy = _DirectionAwareManual(
                    blue_label="Blue", red_label="Red",
                    sigma_off_bar=df_sig_bar.to_numpy(dtype=np.float64),
                    sigma_def_bar=np.zeros_like(
                        df_sig_bar.to_numpy(dtype=np.float64)
                    ),
                    sigma_off_rab=df_sig_rab.to_numpy(dtype=np.float64),
                    sigma_def_rab=np.zeros_like(
                        df_sig_rab.to_numpy(dtype=np.float64)
                    ),
                )
            except ValueError as e:
                st.error(f"Invalid values in σ matrix: {e}")
                targeting_policy = None


# ---------------------------------------------------------------------------
# Tab: Results
# ---------------------------------------------------------------------------

with tab_results:
    st.subheader("Simulation result")

    # Minimum validations before running (do not use st.stop() to avoid
    # affecting the other tabs).
    blue_total_kinetic = sum(
        inp.quantity for inp, dom, _, _ in blue_resolved if dom.is_kinetic
    )
    red_total_kinetic = sum(
        inp.quantity for inp, dom, _, _ in red_resolved if dom.is_kinetic
    )
    can_run = True
    if blue_total_kinetic == 0:
        st.error("Blue Force needs at least one kinetic unit "
                 "(quantity ≥ 1 for some surface type, submarine, "
                 "or coastal battery).")
        can_run = False
    if red_total_kinetic == 0:
        st.error("Red Force needs at least one kinetic unit "
                 "(surface or strike aviation).")
        can_run = False

    run_clicked = st.button(
        "▶️ Run simulation", type="primary",
        use_container_width=True, disabled=not can_run,
    )
    if run_clicked:

        # Complete the matrices with cyber pairs.
        _add_cyber_pairs(p_off_bar, p_def_bar, blue_unit_types, red_unit_types)
        _add_cyber_pairs(p_off_rab, p_def_rab, red_unit_types, blue_unit_types)

        # Verify that the policy was built (in particular, Manual requires
        # the user to have filled the matrix in the Targeting tab).
        if targeting_policy is None:
            st.error(
                "Targeting policy not available. Check the "
                "🎯 Targeting tab (especially if you chose Manual)."
            )
            st.stop()

        # For the Manual policy, σ is static: we pass it in the builder but
        # do not refresh it between salvos (run_campaign with
        # targeting_policy=None uses the σ embedded in params). Other
        # policies are semi-dynamic and refresh at each salvo.
        is_manual = targeting_policy_name == "Manual"

        # Build the engagement.
        builder = (
            EngagementBuilder()
            .with_blue(blue_unit_types, label="Blue")
            .with_red(red_unit_types,  label="Red")
            .with_throughput_blue_attacks_red(
                p_offense=p_off_bar, p_defense=p_def_bar,
            )
            .with_throughput_red_attacks_blue(
                p_offense=p_off_rab, p_defense=p_def_rab,
            )
            .with_targeting_policy(targeting_policy)
            .with_admissibility(Admissibility.canonical())
        )
        ep = builder.build()

        # Import BattleState from the package to wrap the two Forces.
        from naval_salvo.state import BattleState
        state = BattleState(blue=ep.blue, red=ep.red)

        modulator = ChannelPhi() if use_cyber else None

        with st.spinner("Running multi-domain campaign..."):
            traj = run_campaign(
                state, ep, builder.admissibility,
                n_salvos=n_salvos,
                targeting_policy=None if is_manual else targeting_policy,
                cyber_modulator=modulator,
                stop_on_combat_ineffective=stop_on_termination,
            )

        # ---- Summary ----
        c1, c2, c3 = st.columns(3)
        c1.metric("Salvos executed", traj.n_completed_salvos)
        c2.metric(
            "Early termination",
            "Yes" if traj.terminated_early else "No"
        )
        blue_final = traj.blue_strength_history[-1].sum()
        red_final = traj.red_strength_history[-1].sum()
        winner = "Draw" if (blue_final > 0 and red_final > 0) else (
            "Blue" if red_final == 0 and blue_final > 0 else (
                "Red" if blue_final == 0 and red_final > 0 else "—"
            )
        )
        c3.metric("Surviving side", winner)

        # ---- Charts: trajectory by domain ----
        st.markdown("#### Trajectories by domain")

        def _plot_side(side: str, color_map: dict[Domain, str]) -> go.Figure:
            dom_data = traj.total_strength_history_by_domain(side)
            fig = go.Figure()
            for d in Domain:
                arr = dom_data[d]
                if arr.max() == 0.0:
                    continue
                fig.add_trace(go.Scatter(
                    x=traj.times, y=arr,
                    mode="lines+markers",
                    name=f"{d.value} ({d.name.lower()})",
                    line=dict(color=color_map[d], width=2),
                ))
            fig.update_layout(
                xaxis_title="Salvo k",
                yaxis_title="Total strength in domain",
                height=380,
                margin=dict(l=10, r=10, t=30, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02,
                            xanchor="right", x=1),
            )
            return fig

        color_map = {
            Domain.SURFACE:    "#1f4e79",
            Domain.UNDERWATER: "#3a8c8a",
            Domain.AIR:        "#cc8400",
            Domain.COASTAL:    "#a02020",
            Domain.CYBER:      "#7a52a0",
        }
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Blue Force (Navy)**")
            st.plotly_chart(_plot_side("blue", color_map),
                            use_container_width=True)
        with c2:
            st.markdown("**Red Force**")
            st.plotly_chart(_plot_side("red", color_map),
                            use_container_width=True)

        # ---- Final state table per unit ----
        st.markdown("#### Final state per unit")
        c1, c2 = st.columns(2)

        def _final_df(history: np.ndarray, unit_types: list[UnitType]) -> pd.DataFrame:
            return pd.DataFrame({
                "Unit":       [ut.name for ut in unit_types],
                "Domain":     [ut.domain.value for ut in unit_types],
                "Initial":    [ut.initial_strength for ut in unit_types],
                "Final":      history[-1],
                "Losses":     history[0] - history[-1],
                "% losses":   [
                    (1 - history[-1, j] / history[0, j]) * 100
                    if history[0, j] > 0 else 0.0
                    for j in range(history.shape[1])
                ],
            }).round(2)

        with c1:
            st.markdown("**Blue**")
            st.dataframe(
                _final_df(traj.blue_strength_history, traj._blue_unit_types),
                use_container_width=True, hide_index=True,
            )
        with c2:
            st.markdown("**Red**")
            st.dataframe(
                _final_df(traj.red_strength_history, traj._red_unit_types),
                use_container_width=True, hide_index=True,
            )

        # FPSO highlight (if present)
        fpso_idx = next(
            (j for j, ut in enumerate(traj._blue_unit_types)
             if "FPSO" in ut.name.upper() or
                (ut.subtype and ut.subtype.lower() == "pre-salt")),
            None,
        )
        if fpso_idx is not None:
            initial = traj.blue_strength_history[0, fpso_idx]
            final = traj.blue_strength_history[-1, fpso_idx]
            st.info(
                f"**FPSOs (Pre-Salt):** {final:.1f} of {initial:.0f} "
                f"platforms survived "
                f"({(final / initial * 100 if initial else 0):.1f}%)."
            )
    else:
        st.caption(
            "Click **Run simulation** above to execute the campaign "
            "with the current configuration."
        )
