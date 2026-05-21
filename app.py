from pathlib import Path
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
PAGES_DIR = BASE_DIR / "pages"

st.set_page_config(
    page_title="Multi-Domain Salvo Equation",
    page_icon="⚓",
    layout="wide",
    initial_sidebar_state="expanded",
)

pages = {
    "Home": [
        st.Page(PAGES_DIR / "0_Home.py", title="Home Page", icon="🏠"),
    ],
    "Model": [
        st.Page(PAGES_DIR / "1_Hughes.py", title="Hughes 1995", icon="⚓"),
        st.Page(
            PAGES_DIR / "3_Campos_Basin.py",
            title="Multi-Domain Model Scenario",
            icon="🌊",
        ),
        st.Page(PAGES_DIR / "4_Cyber.py", title="Cyber", icon="🛰️"),
    ],
    "Documentation": [
        st.Page(PAGES_DIR / "5_Validation.py", title="Validation", icon="✅"),
        st.Page(PAGES_DIR / "6_About.py", title="About", icon="ℹ️"),
        st.Page(PAGES_DIR / "7_Offline_Calculator.py", title="Offline Calculator", icon="📥"),
    ],
}

pg = st.navigation(pages, position="sidebar")
pg.run()
