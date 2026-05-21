from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from naval_salvo import (
    Admissibility,
    BattleState,
    DirectionalParameters,
    Domain,
    EngagementParameters,
    Force,
    PairParameters,
    UnitType,
    run_campaign,
)


def build_engagement(A0, B0, alpha, beta, z, y, w, x):
    """Build the Hughes (1995) 1×1 homogeneous engagement.

    Canonical convention (Hughes 1995 / Armstrong 2005):
      A is the Red force; α = offensive power of A; y = defence of A;
                          x = staying power of A.
      B is the Blue force; β = offensive power of B; z = defence of B;
                           w = staying power of B.

    Equations:
      ΔA = -max(0, β·B − y·A) / x
      ΔB = -max(0, α·A − z·B) / w

    That is: each force is defended by its own interceptors —
    y intercepts the β shots from Blue; z intercepts the α shots from Red.
    """
    blue = Force("Blue", [UnitType("B", Domain.SURFACE, staying_power=w, initial_strength=B0)])
    red = Force("Red", [UnitType("A", Domain.SURFACE, staying_power=x, initial_strength=A0)])

    # Blue → Red: Blue fires (β); Red intercepts with its own defence y.
    blue_attacks_red = DirectionalParameters.zeros(blue, red)
    blue_attacks_red.set("B", "A", PairParameters(p_offense=beta, p_defense=y))

    # Red → Blue: Red fires (α); Blue intercepts with its own defence z.
    red_attacks_blue = DirectionalParameters.zeros(red, blue)
    red_attacks_blue.set("A", "B", PairParameters(p_offense=alpha, p_defense=z))

    params = EngagementParameters(
        blue=blue,
        red=red,
        blue_attacks_red=blue_attacks_red,
        red_attacks_blue=red_attacks_blue,
    )
    return BattleState(blue=blue, red=red), params, Admissibility.degenerate()


st.title("Hughes 1995 — homogeneous 1×1 model")
st.markdown("Classic salvo combat model between two homogeneous forces.")
st.latex(
    r"\Delta A = -\frac{\max(0,\ \beta B - yA)}{x}, \quad "
    r"\Delta B = -\frac{\max(0,\ \alpha A - zB)}{w}"
)
st.caption(
    "Convention: A is the Red force (α, y, x); B is the Blue force (β, z, w). "
    "Each force is defended by its own interceptors — *y* "
    "intercepts the β shots from Blue; *z* intercepts the α shots from Red."
)

left, right = st.columns(2)
with left:
    st.subheader("🔴 Red Force — A")
    A0 = st.number_input("A₀ — initial units", 1.0, 200.0, 4.0, 1.0)
    alpha = st.number_input("α — offensive power", 0.0, 100.0, 2.0, 0.1)
    y = st.number_input("y — defence", 0.0, 100.0, 1.0, 0.1,
                        help="Intercepts per Red unit per salvo, "
                             "against the β shots from Blue.")
    x = st.number_input("x — staying power", 0.1, 100.0, 2.0, 0.1)
with right:
    st.subheader("🔵 Blue Force — B")
    B0 = st.number_input("B₀ — initial units", 1.0, 200.0, 4.0, 1.0)
    beta = st.number_input("β — offensive power", 0.0, 100.0, 2.0, 0.1)
    z = st.number_input("z — defence", 0.0, 100.0, 1.0, 0.1,
                        help="Intercepts per Blue unit per salvo, "
                             "against the α shots from Red.")
    w = st.number_input("w — staying power", 0.1, 100.0, 2.0, 0.1)

n_salvos = st.slider("Number of salvos", 1, 20, 5)
stop_early = st.checkbox("Stop when a force is eliminated", value=True)

state, params, adm = build_engagement(A0, B0, alpha, beta, z, y, w, x)
traj = run_campaign(
    state,
    params,
    adm,
    n_salvos=n_salvos,
    stop_on_combat_ineffective=stop_early,
)

history = pd.DataFrame({
    "Salvo": list(range(traj.n_completed_salvos + 1)),
    "Red A": traj.red_strength_history[:, 0],
    "Blue B": traj.blue_strength_history[:, 0],
})

final_A = float(traj.red_strength_history[-1, 0])
final_B = float(traj.blue_strength_history[-1, 0])
mc1, mc2, mc3 = st.columns(3)
mc1.metric("Final A", f"{final_A:.3f}", f"{final_A - A0:+.3f}", delta_color="inverse")
mc2.metric("Final B", f"{final_B:.3f}", f"{final_B - B0:+.3f}", delta_color="inverse")
if final_A > 0 and final_B <= 0:
    outcome = "Red wins"
elif final_B > 0 and final_A <= 0:
    outcome = "Blue wins"
elif final_A <= 0 and final_B <= 0:
    outcome = "Mutual annihilation"
else:
    outcome = "Unresolved"
mc3.metric("Outcome", outcome)

plot_df = history.melt(id_vars="Salvo", var_name="Force", value_name="Residual strength")
fig = px.line(plot_df, x="Salvo", y="Residual strength", color="Force", markers=True)
st.plotly_chart(fig, use_container_width=True)

with st.expander("Trajectory table"):
    st.dataframe(history, use_container_width=True, hide_index=True)

with st.expander("First salvo verification"):
    # Canonical corrected formula:
    # ΔA uses y (Red's own defence), ΔB uses z (Blue's own defence).
    dA = max(0.0, (beta * B0 - y * A0) / x)
    dB = max(0.0, (alpha * A0 - z * B0) / w)
    A1 = max(0.0, A0 - dA)
    B1 = max(0.0, B0 - dB)
    st.write(f"ΔA = max(0, β·B₀ − y·A₀) / x = "
             f"max(0, {beta}·{B0} − {y}·{A0}) / {x} = **{dA:.6f}**")
    st.write(f"Expected A₁ = max(0, A₀ − ΔA) = **{A1:.6f}**")
    st.write(f"ΔB = max(0, α·A₀ − z·B₀) / w = "
             f"max(0, {alpha}·{A0} − {z}·{B0}) / {w} = **{dB:.6f}**")
    st.write(f"Expected B₁ = max(0, B₀ − ΔB) = **{B1:.6f}**")
