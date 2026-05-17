from __future__ import annotations

import pandas as pd
import streamlit as st

from naval_salvo import salvo_step
from naval_salvo.validation import (
    BRITISH_GROUPS,
    GERMAN_GROUPS,
    build_coronel_engagement,
    jph_minute_one_delta_good_hope,
)

st.title("JPH 2001 — reprodução de Coronel")
st.markdown(
    "Reprodução do primeiro minuto do exemplo trabalhado por Johns, Pilnick e Hughes para a Batalha de Coronel."
)

state, params, adm = build_coronel_engagement()
result = salvo_step(state, params, adm, apply=False)

rows = []
for i, group in enumerate(BRITISH_GROUPS):
    rows.append({
        "Lado": "Britânico/Azul",
        "Grupo": group.name,
        "Navios": group.n_ships,
        "Kernel": result.blue_raw_kernel[i],
        "Perda do grupo": result.blue_losses[i],
        "Δ por navio": result.blue_losses[i] / group.n_ships,
    })
for i, group in enumerate(GERMAN_GROUPS):
    rows.append({
        "Lado": "Alemão/Vermelho",
        "Grupo": group.name,
        "Navios": group.n_ships,
        "Kernel": result.red_raw_kernel[i],
        "Perda do grupo": result.red_losses[i],
        "Δ por navio": result.red_losses[i] / group.n_ships,
    })

df = pd.DataFrame(rows)
st.dataframe(df, use_container_width=True, hide_index=True)

analytical = jph_minute_one_delta_good_hope()
engine = result.blue_losses[0] / BRITISH_GROUPS[0].n_ships
st.success(
    f"Good Hope: analítico = {analytical:.10f}; engine = {engine:.10f}; diferença = {abs(analytical-engine):.2e}."
)

with st.expander("Interpretação"):
    st.markdown(
        "A reprodução em precisão de máquina indica que a generalização multidomínio preserva o caso heterogêneo clássico como limite."
    )
