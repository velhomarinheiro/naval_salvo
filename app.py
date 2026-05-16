from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="Equação de Salva Multidomínio",
    page_icon="⚓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("⚓ Calculadora de Equação de Salva Multidomínio")
st.markdown(
    """
    Aplicação Streamlit para explorar o pacote `naval_salvo`:

    - modelo clássico de Hughes (1995);
    - reprodução JPH/Coronel;
    - cenário multidomínio da Bacia de Campos;
    - análise cibernética e validação numérica.

    Use o menu lateral para navegar pelas páginas.
    """
)

st.info(
    "Esta versão foi reorganizada para deploy no Streamlit Community Cloud: "
    "o pacote local `naval_salvo/` fica na raiz, e as páginas ficam na pasta `pages/`."
)
