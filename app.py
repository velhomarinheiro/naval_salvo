from pathlib import Path
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
PAGES_DIR = BASE_DIR / "pages"

st.set_page_config(
    page_title="Equação de Salva Multidomínio",
    page_icon="⚓",
    layout="wide",
    initial_sidebar_state="expanded",
)

pages = {
    "Início": [
        st.Page(PAGES_DIR / "0_Inicio.py", title="Página Inicial", icon="🏠"),
    ],
    "Modelo": [
        st.Page(PAGES_DIR / "1_Hughes.py", title="Hughes 1995", icon="⚓"),
        st.Page(
            PAGES_DIR / "3_Bacia_de_Campos.py",
            title="Cenário de Modelo Multidomínio",
            icon="🌊",
        ),
        st.Page(PAGES_DIR / "4_Cyber.py", title="Cyber", icon="🛰️"),
        st.Page(PAGES_DIR / "8_Estocastico.py", title="Salva Estocástica", icon="🎲"),
    ],
    "Documentação": [
        st.Page(PAGES_DIR / "5_Validacao.py", title="Validação", icon="✅"),
        st.Page(PAGES_DIR / "6_Sobre.py", title="Sobre", icon="ℹ️"),
        st.Page(PAGES_DIR / "7_Calculadora_Offline.py", title="Calculadora Offline", icon="📥"),
    ],
}

pg = st.navigation(pages, position="sidebar")
pg.run()
