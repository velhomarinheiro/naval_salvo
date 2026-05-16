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

# Diagnóstico temporário
if not PAGES_DIR.exists():
    st.error(f"Pasta pages não encontrada em: {PAGES_DIR}")
    st.write("Arquivos encontrados na raiz:")
    st.write([p.name for p in BASE_DIR.iterdir()])
    st.stop()

st.sidebar.success("Menu carregado")

pages = {
    "Modelo": [
        st.Page(PAGES_DIR / "1_Hughes.py", title="Hughes 1995", icon="⚓"),
        st.Page(PAGES_DIR / "2_Coronel.py", title="JPH / Coronel", icon="📘"),
        st.Page(PAGES_DIR / "3_Bacia_de_Campos.py", title="Bacia de Campos", icon="🌊"),
        st.Page(PAGES_DIR / "4_Cyber.py", title="Cyber", icon="🛰️"),
    ],
    "Validação": [
        st.Page(PAGES_DIR / "5_Validacao.py", title="Validação", icon="✅"),
        st.Page(PAGES_DIR / "6_Sobre.py", title="Sobre", icon="ℹ️"),
    ],
}

pg = st.navigation(pages, position="sidebar")
pg.run()
