from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from naval_salvo import (
    BaciaCamposConfig,
    ChannelPhi,
    Domain,
    HauskenPhi,
    SimplePhi,
    StrengthProportional,
    build_bacia_campos,
    run_campaign,
)

st.title("Bacia de Campos — cenário multidomínio")
st.markdown("Cenário aplicado com superfície, submarino, aéreo, FPSOs e domínio cibernético.")

col1, col2 = st.columns(2)
with col1:
    st.subheader("🔵 Azul — defensor")
    n_frigates = st.slider("Fragatas", 0, 6, 1)
    submarine_present = st.checkbox("Submarino presente", value=True)
    mpa_present = st.checkbox("MPA presente", value=True)
    n_fpsos = st.slider("FPSOs", 0, 10, 4)
    blue_cyber = st.slider("Cyber Azul por subtipo", 0, 5, 0)
with col2:
    st.subheader("🔴 Vermelho — atacante")
    n_destroyers = st.slider("Contratorpedeiros", 0, 6, 2)
    strike_air_present = st.checkbox("Aviação de ataque presente", value=True)
    red_cyber = st.slider("Cyber Vermelho por subtipo", 0, 5, 0)

n_salvos = st.slider("Número máximo de salvas", 1, 30, 10)

modulator = None
if blue_cyber > 0 or red_cyber > 0:
    with st.expander("Modulador cibernético Φ", expanded=True):
        family = st.selectbox("Família Φ", ["ChannelPhi — canônico", "SimplePhi", "HauskenPhi"])
        if family.startswith("ChannelPhi"):
            r0 = st.number_input("r₀", 0.1, 10.0, 1.0, 0.1)
            k = st.number_input("k", 1.0, 10.0, 2.0, 0.5)
            modulator = ChannelPhi(r0_sigma=r0, r0_rho=r0, r0_delta=r0, k_sigma=k, k_rho=k, k_delta=k)
        elif family == "SimplePhi":
            modulator = SimplePhi()
        else:
            modulator = HauskenPhi()

cfg = BaciaCamposConfig(
    n_frigates=n_frigates,
    submarine_present=submarine_present,
    mpa_present=mpa_present,
    n_fpsos=n_fpsos,
    n_destroyers=n_destroyers,
    strike_air_present=strike_air_present,
    blue_cyber_per_subtype=blue_cyber,
    red_cyber_per_subtype=red_cyber,
)

state, params, adm = build_bacia_campos(cfg)
traj = run_campaign(
    state,
    params,
    adm,
    n_salvos=n_salvos,
    targeting_policy=StrengthProportional(),
    cyber_modulator=modulator,
    stop_on_combat_ineffective=True,
)

blue_names = [u.name for u in state.blue.unit_types]
red_names = [u.name for u in state.red.unit_types]

kpis = st.columns(5)
def metric_for(name: str, side: str, col):
    if side == "blue":
        names, hist = blue_names, traj.blue_strength_history
    else:
        names, hist = red_names, traj.red_strength_history
    if name in names:
        i = names.index(name)
        col.metric(name, f"{hist[-1, i]:.2f} / {hist[0, i]:.0f}", f"{hist[-1, i]-hist[0, i]:+.2f}", delta_color="inverse")
    else:
        col.metric(name, "—")

metric_for("FPSO", "blue", kpis[0])
metric_for("Frigate", "blue", kpis[1])
metric_for("Submarine", "blue", kpis[2])
metric_for("Destroyer", "red", kpis[3])
metric_for("StrikeAir", "red", kpis[4])

rows = []
for k_i, t in enumerate(traj.times):
    for i, name in enumerate(blue_names):
        rows.append({"Salva": t, "Lado": "Azul", "Unidade": name, "Força": traj.blue_strength_history[k_i, i]})
    for i, name in enumerate(red_names):
        rows.append({"Salva": t, "Lado": "Vermelho", "Unidade": name, "Força": traj.red_strength_history[k_i, i]})
plot_df = pd.DataFrame(rows)
fig = px.line(plot_df, x="Salva", y="Força", color="Unidade", line_dash="Lado", markers=True)
st.plotly_chart(fig, use_container_width=True)

with st.expander("Tabela detalhada"):
    st.dataframe(plot_df.pivot_table(index=["Salva"], columns=["Lado", "Unidade"], values="Força"), use_container_width=True)

st.caption(f"Salvas completadas: {traj.n_completed_salvos}. Encerramento antecipado: {traj.terminated_early}.")
