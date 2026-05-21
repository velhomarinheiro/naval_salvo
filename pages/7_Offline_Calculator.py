"""
"Offline Calculator" page of the Multi-Domain Salvo Equation application.

Presents the Excel spreadsheet version of the calculator and the PDF
user guide with usage instructions. Files must be placed in the
offline_assets/ folder at the repository root.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st


BASE_DIR = Path(__file__).resolve().parent.parent
ASSETS_DIR = BASE_DIR / "offline_assets"

EXCEL_FILE = ASSETS_DIR / "calculadora_offline.xlsx"
PDF_FILE = ASSETS_DIR / "cartilha_calculadora_offline.pdf"


st.title("Offline Calculator")
st.caption("Excel spreadsheet version for local use of the Salvo Equation")

st.markdown("---")

st.header("About the offline version")

st.markdown(
    """
    This page provides an **offline** version of the Salvo Equation
    calculator, in **Excel spreadsheet** format. The spreadsheet allows
    basic simulations to be carried out without requiring internet
    connectivity or access to the Streamlit application.

    The offline calculator applies the model with some **simplifications**
    relative to the online version. It was designed for educational use,
    exploratory exercises, war games, preliminary force composition
    studies, and sensitivity analysis support.

    The spreadsheet comes with a **user guide**, in PDF format, with
    instructions for filling in the fields, interpreting the results,
    and the main limitations of the tool.
    """
)

st.info(
    "The offline version is complementary to the online application. For "
    "more complete analyses, especially involving multi-domain scenarios, "
    "cyber effects, and dynamic targeting selection policies, it is "
    "recommended to use the interactive pages of the app."
)

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Excel spreadsheet")

    st.markdown(
        """
        The spreadsheet contains a simplified implementation of the
        calculator, allowing basic force parameters, offensive power,
        defensive power, and *staying power* to be adjusted.

        **Format:** `.xlsx`
        """
    )

    if EXCEL_FILE.exists():
        st.download_button(
            label="Download offline calculator (.xlsx)",
            data=EXCEL_FILE.read_bytes(),
            file_name="calculadora_offline.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    else:
        st.warning(
            "Spreadsheet file not yet found. "
            "Add the file to `offline_assets/calculadora_offline.xlsx`."
        )

with col2:
    st.subheader("User guide")

    st.markdown(
        """
        The user guide provides instructions for using the spreadsheet,
        including:

        - description of input fields;
        - recommended filling sequence;
        - reading the results;
        - limitations of the offline version;
        - interpretation caveats.

        **Format:** `.pdf`
        """
    )

    if PDF_FILE.exists():
        st.download_button(
            label="Download user guide (.pdf)",
            data=PDF_FILE.read_bytes(),
            file_name="cartilha_calculadora_offline.pdf",
            mime="application/pdf",
        )
    else:
        st.warning(
            "User guide file not yet found. "
            "Add the file to `offline_assets/cartilha_calculadora_offline.pdf`."
        )

st.markdown("---")

st.header("Limitations of the offline version")

st.markdown(
    """
    The offline calculator is deliberately simpler than the online
    application. In particular, it may not include all features available
    in the multi-domain panel, such as:

    - full cross-domain admissibility matrix;
    - cyber modulation Φ;
    - dynamic targeting selection update between salvos;
    - interactive visualisations;
    - automatic reproduction of validation and sensitivity reports.

    Therefore, results should be interpreted as analytical support and
    not as operational forecasts.
    """
)
