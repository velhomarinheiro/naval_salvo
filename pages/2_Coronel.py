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

st.title("JPH 2001 — Coronel reproduction")
st.markdown(
    "Reproduction of the first minute of the worked example by Johns, Pilnick and Hughes for the Battle of Coronel."
)

state, params, adm = build_coronel_engagement()
result = salvo_step(state, params, adm, apply=False)

rows = []
for i, group in enumerate(BRITISH_GROUPS):
    rows.append({
        "Side": "British/Blue",
        "Group": group.name,
        "Ships": group.n_ships,
        "Kernel": result.blue_raw_kernel[i],
        "Group loss": result.blue_losses[i],
        "Δ per ship": result.blue_losses[i] / group.n_ships,
    })
for i, group in enumerate(GERMAN_GROUPS):
    rows.append({
        "Side": "German/Red",
        "Group": group.name,
        "Ships": group.n_ships,
        "Kernel": result.red_raw_kernel[i],
        "Group loss": result.red_losses[i],
        "Δ per ship": result.red_losses[i] / group.n_ships,
    })

df = pd.DataFrame(rows)
st.dataframe(df, use_container_width=True, hide_index=True)

analytical = jph_minute_one_delta_good_hope()
engine = result.blue_losses[0] / BRITISH_GROUPS[0].n_ships
st.success(
    f"Good Hope: analytical = {analytical:.10f}; engine = {engine:.10f}; difference = {abs(analytical-engine):.2e}."
)

with st.expander("Interpretation"):
    st.markdown(
        "The machine-precision reproduction indicates that the multi-domain generalisation preserves the classical heterogeneous case as a limit."
    )
