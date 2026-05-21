from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from naval_salvo import phi_sigmoid

st.title("Cyber analysis — Φ function")
st.markdown("Visualisation of the inverse sigmoid used in the canonical cyber modulator.")
st.latex(r"\Phi^p(R)=\frac{1}{1+(R/r_0)^k}")

col1, col2 = st.columns(2)
with col1:
    r0 = st.slider("r₀", 0.1, 5.0, 1.0, 0.1)
with col2:
    k = st.slider("k", 1.0, 10.0, 2.0, 0.5)

R_vals = [i / 100 for i in range(0, 501)]
df = pd.DataFrame({"R": R_vals, "Φ": [phi_sigmoid(R, r0=r0, k=k) for R in R_vals]})
fig = px.line(df, x="R", y="Φ", markers=False)
fig.add_hline(y=0.5, line_dash="dot")
fig.add_vline(x=r0, line_dash="dot")
st.plotly_chart(fig, use_container_width=True)

st.markdown(
    f"With r₀ = {r0:.1f}, half-degradation occurs when R = {r0:.1f}. "
    f"With k = {k:.1f}, the transition is {'smooth' if k < 2 else 'moderate' if k < 4 else 'sharp'}."
)
