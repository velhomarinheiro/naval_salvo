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
    EngagementOrder, MultiDomainForce, UnitGroup,
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


def force_editor(label: str, defaults: list, key: str) -> list:
    """Force composition editor (unit groups)."""
    groups = []
    for i, d in enumerate(defaults):
        with st.expander(f"{d['name']}", expanded=(i == 0)):
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
                                   round(1.0 / d["mu_v"], 1), 0.5,
                                   key=f"{key}_sp{i}")
            sigma_v = st.slider("σ of the damage per missile", 0.0, 0.5,
                                d["sigma_v"], 0.01, key=f"{key}_sv{i}")
            groups.append(UnitGroup(
                name=d["name"], domain=d["domain"], units=units,
                n_off=n_off, p_off=p_off, n_def=n_def, p_def=p_def,
                mu_v=1.0 / stay, sigma_v=sigma_v,
            ))
    return groups


# ---------------------------------------------------------------------------
# Force composition
# ---------------------------------------------------------------------------
col_b, col_r = st.columns(2)
with col_b:
    st.subheader("🔵 Blue Force")
    blue_groups = force_editor("Blue", [
        dict(name="Frigates (s)", domain="s", units=4, n_off=4, p_off=0.67,
             n_def=2, p_def=0.67, mu_v=0.33, sigma_v=0.11),
        dict(name="Strike aviation (a)", domain="a", units=2, n_off=2,
             p_off=0.75, n_def=0, p_def=0.0, mu_v=0.5, sigma_v=0.15),
    ], key="blue")
with col_r:
    st.subheader("🔴 Red Force")
    red_groups = force_editor("Red", [
        dict(name="Escorts (s)", domain="s", units=5, n_off=4, p_off=0.60,
             n_def=2, p_def=0.60, mu_v=0.33, sigma_v=0.11),
        dict(name="Submarine (u)", domain="u", units=1, n_off=4, p_off=0.80,
             n_def=0, p_def=0.0, mu_v=0.5, sigma_v=0.15),
    ], key="red")

blue = MultiDomainForce("Blue", blue_groups, cyber_offense=cyber_blue)
red = MultiDomainForce("Red", red_groups, cyber_offense=cyber_red)

phi_b = phi(cyber_red - cyber_blue, k_cyber)
phi_r = phi(cyber_blue - cyber_red, k_cyber)
if phi_b < 1.0 or phi_r < 1.0:
    st.info(f"Cyber modulation active: Φ(Blue) = {phi_b:.3f} | "
            f"Φ(Red) = {phi_r:.3f}")

# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------
if st.button("▶️ Run simulation", type="primary"):
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
