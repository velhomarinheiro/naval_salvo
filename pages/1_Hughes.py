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
    """Monta o engajamento homogêneo 1×1 de Hughes (1995).

    Convenção canônica (Hughes 1995 / Armstrong 2005):
      A é a força Vermelha; α = poder ofensivo de A; y = defesa de A;
                            x = staying power de A.
      B é a força Azul;     β = poder ofensivo de B; z = defesa de B;
                            w = staying power de B.

    Equações:
      ΔA = -max(0, β·B − y·A) / x
      ΔB = -max(0, α·A − z·B) / w

    Ou seja: cada força é defendida pelos seus próprios interceptadores
    (y intercepta os tiros β do Azul; z intercepta os tiros α do
    Vermelho).
    """
    blue = Force("Blue", [UnitType("B", Domain.SURFACE, staying_power=w, initial_strength=B0)])
    red = Force("Red", [UnitType("A", Domain.SURFACE, staying_power=x, initial_strength=A0)])

    # Blue → Red: Blue atira (β); Red intercepta com sua própria defesa y.
    blue_attacks_red = DirectionalParameters.zeros(blue, red)
    blue_attacks_red.set("B", "A", PairParameters(p_offense=beta, p_defense=y))

    # Red → Blue: Red atira (α); Blue intercepta com sua própria defesa z.
    red_attacks_blue = DirectionalParameters.zeros(red, blue)
    red_attacks_blue.set("A", "B", PairParameters(p_offense=alpha, p_defense=z))

    params = EngagementParameters(
        blue=blue,
        red=red,
        blue_attacks_red=blue_attacks_red,
        red_attacks_blue=red_attacks_blue,
    )
    return BattleState(blue=blue, red=red), params, Admissibility.degenerate()


st.title("Hughes 1995 — modelo homogêneo 1×1")
st.markdown("Modelo clássico de combate por salvas entre duas forças homogêneas.")
st.latex(
    r"\Delta A = -\frac{\max(0,\ \beta B - yA)}{x}, \quad "
    r"\Delta B = -\frac{\max(0,\ \alpha A - zB)}{w}"
)
st.caption(
    "Convenção: A é a força Vermelha (α, y, x); B é a força Azul (β, z, w). "
    "Cada força é defendida pelos seus próprios interceptadores — *y* "
    "intercepta os tiros β do Azul; *z* intercepta os tiros α do Vermelho."
)

left, right = st.columns(2)
with left:
    st.subheader("🔴 Força Vermelha — A")
    A0 = st.number_input("A₀ — unidades iniciais", 1.0, 200.0, 4.0, 1.0)
    alpha = st.number_input("α — poder ofensivo", 0.0, 100.0, 2.0, 0.1)
    y = st.number_input("y — defesa", 0.0, 100.0, 1.0, 0.1,
                        help="Interceptações por unidade Vermelha por salva, "
                             "contra os tiros β do Azul.")
    x = st.number_input("x — staying power", 0.1, 100.0, 2.0, 0.1)
with right:
    st.subheader("🔵 Força Azul — B")
    B0 = st.number_input("B₀ — unidades iniciais", 1.0, 200.0, 4.0, 1.0)
    beta = st.number_input("β — poder ofensivo", 0.0, 100.0, 2.0, 0.1)
    z = st.number_input("z — defesa", 0.0, 100.0, 1.0, 0.1,
                        help="Interceptações por unidade Azul por salva, "
                             "contra os tiros α do Vermelho.")
    w = st.number_input("w — staying power", 0.1, 100.0, 2.0, 0.1)

n_salvos = st.slider("Número de salvas", 1, 20, 5)
stop_early = st.checkbox("Parar quando uma força for eliminada", value=True)

state, params, adm = build_engagement(A0, B0, alpha, beta, z, y, w, x)
traj = run_campaign(
    state,
    params,
    adm,
    n_salvos=n_salvos,
    stop_on_combat_ineffective=stop_early,
)

history = pd.DataFrame({
    "Salva": list(range(traj.n_completed_salvos + 1)),
    "Vermelha A": traj.red_strength_history[:, 0],
    "Azul B": traj.blue_strength_history[:, 0],
})

final_A = float(traj.red_strength_history[-1, 0])
final_B = float(traj.blue_strength_history[-1, 0])
mc1, mc2, mc3 = st.columns(3)
mc1.metric("A final", f"{final_A:.3f}", f"{final_A - A0:+.3f}", delta_color="inverse")
mc2.metric("B final", f"{final_B:.3f}", f"{final_B - B0:+.3f}", delta_color="inverse")
if final_A > 0 and final_B <= 0:
    outcome = "Vermelha vence"
elif final_B > 0 and final_A <= 0:
    outcome = "Azul vence"
elif final_A <= 0 and final_B <= 0:
    outcome = "Aniquilação mútua"
else:
    outcome = "Não resolvido"
mc3.metric("Resultado", outcome)

plot_df = history.melt(id_vars="Salva", var_name="Força", value_name="Força residual")
fig = px.line(plot_df, x="Salva", y="Força residual", color="Força", markers=True)
st.plotly_chart(fig, use_container_width=True)

with st.expander("Tabela de trajetória"):
    st.dataframe(history, use_container_width=True, hide_index=True)

with st.expander("Verificação da primeira salva"):
    # Fórmula canônica corrigida:
    # ΔA usa y (defesa do próprio Vermelho), ΔB usa z (defesa do próprio Azul).
    dA = max(0.0, (beta * B0 - y * A0) / x)
    dB = max(0.0, (alpha * A0 - z * B0) / w)
    A1 = max(0.0, A0 - dA)
    B1 = max(0.0, B0 - dB)
    st.write(f"ΔA = max(0, β·B₀ − y·A₀) / x = "
             f"max(0, {beta}·{B0} − {y}·{A0}) / {x} = **{dA:.6f}**")
    st.write(f"A₁ esperado = max(0, A₀ − ΔA) = **{A1:.6f}**")
    st.write(f"ΔB = max(0, α·A₀ − z·B₀) / w = "
             f"max(0, {alpha}·{A0} − {z}·{B0}) / {w} = **{dB:.6f}**")
    st.write(f"B₁ esperado = max(0, B₀ − ΔB) = **{B1:.6f}**")
