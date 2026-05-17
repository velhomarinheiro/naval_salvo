from __future__ import annotations

import streamlit as st

st.title("Sobre o app")
st.markdown(
    """
    Esta aplicação foi reorganizada para funcionar no Streamlit Community Cloud.

    ## Estrutura usada

    ```text
    app.py
    requirements.txt
    pages/
    naval_salvo/
    paper_artifacts/
    ```

    ## Observações de deploy

    - O arquivo principal é `app.py`.
    - O pacote `naval_salvo` fica na raiz do repositório.
    - As páginas ficam em `pages/`, usando o mecanismo nativo multipage do Streamlit.
    - Evitou-se importar todas as páginas no `app.py`, reduzindo erros de importação na inicialização.
    """
)
