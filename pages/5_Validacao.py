from __future__ import annotations

import pandas as pd
import streamlit as st

from naval_salvo import salvo_step
from naval_salvo.validation import (
    HughesScenario,
    build_coronel_engagement,
    build_hughes_homogeneous_engagement,
    hughes_analytical,
    jph_minute_one_delta_good_hope,
    BRITISH_GROUPS,
)

st.title("Validação numérica")
st.markdown("Testes canônicos executados em tempo real no app.")

st.subheader("Hughes 1995")
scenarios = [
    ("Balanceado", dict(A0=4, B0=4, alpha=2, beta=2, z=1, y=1, w=2, x=2, n_salvos=1)),
    ("Atacante dominante", dict(A0=4, B0=4, alpha=4, beta=2, z=1, y=1, w=2, x=2, n_salvos=1)),
    ("Defesa saturante", dict(A0=4, B0=4, alpha=1, beta=1, z=5, y=5, w=2, x=2, n_salvos=1)),
]
rows = []
for label, kwargs in scenarios:
    scn = HughesScenario(**kwargs)
    A_an, B_an = hughes_analytical(scn)
    state, params, adm = build_hughes_homogeneous_engagement(scn)
    salvo_step(state, params, adm, apply=True)
    rows.append({
        "Caso": label,
        "A analítico": A_an[1],
        "A engine": state.red.strength_of("A"),
        "ΔA": abs(A_an[1] - state.red.strength_of("A")),
        "B analítico": B_an[1],
        "B engine": state.blue.strength_of("B"),
        "ΔB": abs(B_an[1] - state.blue.strength_of("B")),
    })
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

st.subheader("JPH/Coronel")
state, params, adm = build_coronel_engagement()
out = salvo_step(state, params, adm, apply=False)
analytical = jph_minute_one_delta_good_hope()
engine = out.blue_losses[0] / BRITISH_GROUPS[0].n_ships
st.success(f"Good Hope: analítico = {analytical:.16f}; engine = {engine:.16f}; Δ = {abs(analytical-engine):.2e}.")
