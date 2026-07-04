# -*- coding: utf-8 -*-
"""Streamlit page -- Stochastic Salvo Equation with Engagement Order.

Stochastic extension of the multi-domain model (Armstrong 2005, 2014):
offensive fires and interceptions follow binomial distributions, damage
per missile follows a normal distribution, and the fire exchange can be
simultaneous or sequential (Blue first / Red first).
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from naval_salvo.stochastic import (
    EngagementOrder, MultiDomainForce, NAVAL_DOMAINS, UnitGroup,
    run_multidomain_battle, phi,
)

st.set_page_config(page_title="Stochastic Salvo", page_icon="🎲",
                   layout="wide")

st.title("🎲 Stochastic Multi-Domain Salvo Equation")
st.markdown(
    "Stochastic extension of the multi-domain model with **engagement order** "
    "(Armstrong, 2005; 2014): offensive fires and interceptions follow "
    "binomial distributions, damage per missile follows a normal "
    "distribution, and the fire exchange can be **simultaneous** or "
    "**sequential** (Blue first / Red first). In sequential mode, return "
    "fire is executed only by the **survivors** of the first salvo."
)
st.markdown(
    "Both forces share the **same platform palette** as the Campos Basin "
    "multi-domain scenario: two surface classes, a submarine (underwater, "
    "cyber-immune), strike aviation, a coastal battery, and an FPSO-type "
    "value asset. Set a slot's quantity to **0** to leave it out of a force."
)

ORDER_LABELS = {
    "Simultaneous": EngagementOrder.SIMULTANEOUS,
    "Blue fires first": EngagementOrder.BLUE_FIRST,
    "Red fires first": EngagementOrder.RED_FIRST,
}

# ---------------------------------------------------------------------------
# Sidebar -- simulation settings
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Simulation")
    n_sim = st.select_slider("Number of Monte Carlo replications",
                             options=[1_000, 2_000, 5_000, 10_000, 20_000],
                             value=5_000)
    n_salvos = st.slider("Maximum number of salvos", 1, 10, 3)
    seed = st.number_input("Random seed", 0, 999_999, 42)
    compare_all = st.checkbox("Compare the three engagement orders",
                              value=True)
    if not compare_all:
        order_label = st.radio("Engagement order",
                               list(ORDER_LABELS.keys()))
    st.divider()
    st.header("📡 Cyber-EM domain")
    cyber_blue = st.slider("Cyber intensity projected by Blue",
                           0.0, 5.0, 0.0, 0.5)
    cyber_red = st.slider("Cyber intensity projected by Red",
                          0.0, 5.0, 0.0, 0.5)
    k_cyber = st.slider("Slope k of the Φ sigmoid", 0.1, 2.0, 0.5, 0.1)
    st.caption("Submarines are immune to cyber degradation (χ).")


DOMAIN_LABELS = {
    "s": "Surface",
    "u": "Underwater",
    "a": "Air",
    "c": "Coastal",
}


def force_editor(defaults: list, key: str) -> list:
    """Force composition editor -- one expander per platform slot.

    Each slot has a fixed domain (matching the Campos Basin palette).
    Setting a slot's quantity to 0 leaves it out of the force. Names are
    de-duplicated within the force so they remain valid stock keys.
    """
    groups = []
    used_names: set[str] = set()
    for i, d in enumerate(defaults):
        dom = d["domain"]
        dom_label = DOMAIN_LABELS[dom]
        title = f"{d['slot']} — {d['name']} ({dom_label})"
        with st.expander(title, expanded=(i == 0)):
            note = f"Domain: **{dom_label}** (`{dom}`)."
            if dom == "u":
                note += " Immune to cyber degradation (χ)."
            if dom not in NAVAL_DOMAINS:
                note += " Support arm — does not count toward the naval verdict."
            st.caption(note)
            c1, c2, c3 = st.columns(3)
            units = c1.number_input("Units", 0.0, 50.0, float(d["units"]),
                                    1.0, key=f"{key}_u{i}")
            n_off = c2.number_input("Offensive missiles/unit", 0, 24,
                                    d["n_off"], key=f"{key}_no{i}")
            p_off = c3.slider("p(hit)", 0.0, 1.0, d["p_off"], 0.01,
                              key=f"{key}_po{i}")
            c4, c5, c6 = st.columns(3)
            n_def = c4.number_input("Interceptions/unit", 0, 24,
                                    d["n_def"], key=f"{key}_nd{i}")
            p_def = c5.slider("p(interception)", 0.0, 1.0, d["p_def"], 0.01,
                              key=f"{key}_pd{i}")
            stay = c6.number_input("Staying power (hits)", 1.0, 10.0,
                                   float(d["staying"]), 0.5,
                                   key=f"{key}_sp{i}")
            sigma_v = st.slider("σ of the damage per missile", 0.0, 0.5,
                                d["sigma_v"], 0.01, key=f"{key}_sv{i}")
            nm = d["name"]
            base, k = nm, 2
            while nm in used_names:
                nm = f"{base} ({k})"
                k += 1
            used_names.add(nm)
            groups.append(UnitGroup(
                name=nm, domain=dom, units=units,
                n_off=n_off, p_off=p_off, n_def=n_def, p_def=p_def,
                mu_v=1.0 / stay, sigma_v=sigma_v,
            ))
    return groups


# ---------------------------------------------------------------------------
# Platform palette -- mirrors the Campos Basin multi-domain scenario:
# two surface classes, a submarine, strike aviation, a coastal battery and
# an FPSO-type value asset. Each side may compose from the full palette;
# defaults follow the Campos Basin scenario (a slot with quantity 0 is
# simply left out of the force).
# ---------------------------------------------------------------------------
BLUE_DEFAULTS = [
    dict(slot="Surface — Type 1", name="Class A Frigate", domain="s", units=2,
         n_off=4, p_off=0.67, n_def=2, p_def=0.67, staying=3.0, sigma_v=0.11),
    dict(slot="Surface — Type 2", name="Corvette/Patrol", domain="s", units=2,
         n_off=2, p_off=0.60, n_def=1, p_def=0.50, staying=2.0, sigma_v=0.11),
    dict(slot="Submarine", name="Conventional submarine", domain="u", units=1,
         n_off=4, p_off=0.80, n_def=0, p_def=0.0, staying=2.0, sigma_v=0.15),
    dict(slot="Strike aviation", name="Blue strike aviation", domain="a", units=0,
         n_off=2, p_off=0.75, n_def=0, p_def=0.0, staying=1.0, sigma_v=0.15),
    dict(slot="Coastal battery", name="Coastal battery", domain="c", units=1,
         n_off=3, p_off=0.65, n_def=0, p_def=0.0, staying=4.0, sigma_v=0.11),
    dict(slot="Value asset (FPSO)", name="FPSO (Pre-Salt)", domain="s", units=4,
         n_off=0, p_off=0.0, n_def=0, p_def=0.0, staying=6.0, sigma_v=0.0),
]

RED_DEFAULTS = [
    dict(slot="Surface — Type 1", name="Destroyer", domain="s", units=2,
         n_off=4, p_off=0.60, n_def=2, p_def=0.60, staying=4.0, sigma_v=0.11),
    dict(slot="Surface — Type 2", name="Adv. Frigate", domain="s", units=2,
         n_off=3, p_off=0.60, n_def=2, p_def=0.55, staying=3.0, sigma_v=0.11),
    dict(slot="Submarine", name="Adversary submarine", domain="u", units=0,
         n_off=4, p_off=0.80, n_def=0, p_def=0.0, staying=2.0, sigma_v=0.15),
    dict(slot="Strike aviation", name="Red strike aviation", domain="a", units=4,
         n_off=2, p_off=0.75, n_def=0, p_def=0.0, staying=1.0, sigma_v=0.15),
    dict(slot="Coastal battery", name="Coastal battery", domain="c", units=0,
         n_off=3, p_off=0.60, n_def=0, p_def=0.0, staying=4.0, sigma_v=0.11),
    dict(slot="Value asset", name="Value asset", domain="s", units=0,
         n_off=0, p_off=0.0, n_def=0, p_def=0.0, staying=6.0, sigma_v=0.0),
]


# ---------------------------------------------------------------------------
# Force composition
# ---------------------------------------------------------------------------
col_b, col_r = st.columns(2)
with col_b:
    st.subheader("🔵 Blue Force")
    blue_groups_all = force_editor(BLUE_DEFAULTS, key="blue")
with col_r:
    st.subheader("🔴 Red Force")
    red_groups_all = force_editor(RED_DEFAULTS, key="red")

# Only slots with a positive quantity join the force.
blue_groups = [g for g in blue_groups_all if g.units > 0]
red_groups = [g for g in red_groups_all if g.units > 0]

blue = MultiDomainForce("Blue", blue_groups, cyber_offense=cyber_blue)
red = MultiDomainForce("Red", red_groups, cyber_offense=cyber_red)

st.caption(
    "The win verdict considers **naval forces only** (surface + underwater). "
    "Air and coastal units contribute fire as supporting arms but do not, by "
    "themselves, decide the outcome."
)

phi_b = phi(cyber_red - cyber_blue, k_cyber)
phi_r = phi(cyber_blue - cyber_red, k_cyber)
if phi_b < 1.0 or phi_r < 1.0:
    st.info(f"Cyber modulation active: Φ(Blue) = {phi_b:.3f} | "
            f"Φ(Red) = {phi_r:.3f}")

# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------
forces_ok = bool(blue_groups) and bool(red_groups)
if not forces_ok:
    st.error("Each force needs at least one platform slot with quantity ≥ 1.")

if st.button("▶️ Run simulation", type="primary", disabled=not forces_ok):
    orders = (list(ORDER_LABELS.items()) if compare_all
              else [(order_label, ORDER_LABELS[order_label])])

    results = {}
    prog = st.progress(0.0)
    for i, (lbl, order) in enumerate(orders):
        results[lbl] = run_multidomain_battle(
            blue, red, order, n_salvos=n_salvos, n_sim=n_sim,
            seed=int(seed), k_cyber=k_cyber,
        )
        prog.progress((i + 1) / len(orders))
    prog.empty()

    # --- Verdict metrics --------------------------------------------------
    st.subheader("Verdict (naval forces: surface + submarine)")
    cols = st.columns(len(results))
    for c, (lbl, r) in zip(cols, results.items()):
        with c:
            st.markdown(f"**{lbl}**")
            st.metric("P(Blue victory)", f"{100 * r.p_blue_win:.1f}%")
            st.metric("P(Red victory)", f"{100 * r.p_red_win:.1f}%")
            st.metric("P(indecisive)", f"{100 * r.p_draw:.1f}%")

    # --- Survivor distributions -------------------------------------------
    st.subheader("Distribution of naval survivors")
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 3.5),
                             sharey=True, squeeze=False)
    for ax, (lbl, r) in zip(axes[0], results.items()):
        bins = np.linspace(0, max(r.blue_naval_final.max(),
                                  r.red_naval_final.max(), 1) + 0.5, 25)
        ax.hist(r.blue_naval_final, bins=bins, alpha=0.6, color="#1f77b4",
                label="Blue")
        ax.hist(r.red_naval_final, bins=bins, alpha=0.6, color="#d62728",
                label="Red")
        ax.set_title(lbl, fontsize=10)
        ax.set_xlabel("Naval survivors")
        ax.legend(fontsize=8)
    axes[0][0].set_ylabel("Frequency")
    fig.tight_layout()
    st.pyplot(fig)

    # --- Summary table ------------------------------------------------------
    st.subheader("Statistics")
    rows = []
    for lbl, r in results.items():
        s = r.summary()
        rows.append({
            "Order": lbl,
            "P(Blue win)": f"{100 * s['p_blue_win']:.1f}%",
            "P(Red win)": f"{100 * s['p_red_win']:.1f}%",
            "Blue naval (mean ± sd)":
                f"{s['blue_naval_mean']:.2f} ± {s['blue_naval_std']:.2f}",
            "Red naval (mean ± sd)":
                f"{s['red_naval_mean']:.2f} ± {s['red_naval_std']:.2f}",
            "Blue mean loss": f"{s['blue_loss_mean']:.2f}",
            "Red mean loss": f"{s['red_loss_mean']:.2f}",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True,
                 hide_index=True)

    if compare_all:
        st.caption(
            "Theoretical property (Armstrong, 2014): firing first is, on "
            "average, at least as good as simultaneous engagement, which is "
            "at least as good as firing second. In individual replications, "
            "a 'lucky' force firing second may still beat an 'unlucky' one "
            "firing first."
        )

st.divider()
with st.expander("📖 Model formulation"):
    st.markdown("**Offensive fires** per attacking group $g$ (stock $U_g$):")
    st.latex(r"\mathrm{Off}_g \sim \mathrm{Binomial}\!\left(n_g\,U_g,\;"
             r"p_g^{\mathrm{off}}\cdot\Phi^{\sigma_g}\Phi^{\rho_g}\right)")
    st.markdown("**Interceptions** of target group $j$:")
    st.latex(r"\mathrm{Def}_j \sim \mathrm{Binomial}\!\left(n_j^{\mathrm{def}}\,"
             r"U_j,\; p_j^{\mathrm{def}}\cdot\Phi^{\delta_j}\right)")
    st.markdown("**Damage** per non-intercepted missile, with losses "
                "truncated at the stock:")
    st.latex(r"\Delta U_j = \min\!\Big(\textstyle\sum_{k=1}^{\mathrm{Net}_j}"
             r"\max(0, v_k),\; U_j\Big),\quad v_k \sim \mathcal{N}(\mu_v,"
             r"\sigma_v^2)")
    st.markdown(
        "In **sequential mode**, the defender's losses are applied before "
        "the return fire, which is executed by the survivors with firepower "
        "proportional to the fractional stock (Hausken–Moxnes). "
        "Submarines are immune to the Φ modulation (χ matrix). The verdict "
        "considers naval forces only (s + u)."
    )
