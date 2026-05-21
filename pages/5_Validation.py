from pathlib import Path

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = BASE_DIR / "paper_artifacts"


def load_csv(filename: str, fallback: pd.DataFrame) -> pd.DataFrame:
    path = ARTIFACTS_DIR / filename
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception:
            return fallback
    return fallback


def show_image(filename: str, caption: str) -> None:
    path = ARTIFACTS_DIR / filename
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.info(f"Figure not found in paper_artifacts/: {filename}")


def download_file(filename: str, label: str, mime: str) -> None:
    path = ARTIFACTS_DIR / filename
    if path.exists():
        st.download_button(label=label, data=path.read_bytes(), file_name=filename, mime=mime)
    else:
        st.caption(f"File not found: {filename}")


hughes_one_salvo = pd.DataFrame(
    [
        ["Balanced", 4, 2.0, 2.0, "0.00e+00", 4, 2.0, 2.0, "0.00e+00"],
        ["Attacker dominant", 4, 2.0, 2.0, "0.00e+00", 4, 0.0, 0.0, "0.00e+00"],
        ["Defender dominant", 4, 2.0, 2.0, "0.00e+00", 4, 4.0, 4.0, "0.00e+00"],
        ["Asymmetric forces", 6, 3.0, 3.0, "0.00e+00", 3, 0.0, 0.0, "0.00e+00"],
        ["Saturating defence", 4, 4.0, 4.0, "0.00e+00", 4, 4.0, 4.0, "0.00e+00"],
    ],
    columns=["Case", "A_pre", "A_post analytical", "A_post engine", "Delta_A", "B_pre", "B_post analytical", "B_post engine", "Delta_B"],
)

hughes_multisalvo = pd.DataFrame(
    [
        [0, 6.0, 6.0, "0.00e+00", 4.0, 4.0, "0.00e+00"],
        [1, 1.0, 1.0, "0.00e+00", 3.0, 3.0, "0.00e+00"],
        [2, 0.0, 0.0, "0.00e+00", 3.0, 3.0, "0.00e+00"],
        [3, 0.0, 0.0, "0.00e+00", 3.0, 3.0, "0.00e+00"],
        [4, 0.0, 0.0, "0.00e+00", 3.0, 3.0, "0.00e+00"],
    ],
    columns=["k", "A analytical", "A engine", "Delta_A", "B analytical", "B engine", "Delta_B"],
)

coronel = pd.DataFrame(
    [
        ["Good Hope + Monmouth", 2, 1.605, 0.120960, 0.075364, 0.037682, 0.037682, "0.00e+00"],
        ["Glasgow", 1, 1.230, 0.051960, 0.042244, 0.042244, 0.042244, "0.00e+00"],
        ["Scharnhorst + Gneisenau", 2, 1.665, 0.000000, 0.000000, 0.000000, 0.000000, "0.00e+00"],
        ["Leipzig + Dresden", 2, 1.115, 0.000000, 0.000000, 0.000000, 0.000000, "0.00e+00"],
    ],
    columns=["Target group", "n_ships", "Staying power", "Total kernel", "Group loss", "Delta engine", "Delta analytical", "Delta"],
)

s1 = pd.DataFrame(
    [
        [0, "Y", 2, 0.00, 0.00, 0.00, 2.00, 3.30],
        [1, "Y", 3, 0.00, 0.00, 0.00, 1.78, 2.08],
        [2, "Y", 3, 0.46, 1.83, 0.49, 0.00, 0.00],
        [3, "Y", 2, 1.02, 3.00, 0.61, 0.00, 0.00],
        [4, "Y", 2, 1.34, 4.00, 0.67, 0.00, 0.00],
        [0, "N", 2, 0.00, 0.00, "-", 2.00, 3.30],
        [1, "N", 2, 0.00, 0.00, "-", 2.00, 2.20],
        [2, "N", 3, 0.00, 0.00, "-", 1.09, 0.29],
        [3, "N", 2, 0.60, 3.00, "-", 0.00, 0.00],
        [4, "N", 2, 0.92, 4.00, "-", 0.00, 0.00],
    ],
    columns=["n_frig", "sub?", "salvos", "FPSO_f", "Frig_f", "Sub_f", "Dest_f", "StrAir_f"],
)

s2 = pd.DataFrame(
    [
        ["Y", 0, 0, "No cyber", 3, 0.00, 0.00, 0.00, 1.78, 2.08, 0.00, 0.00],
        ["Y", 1, 1, "Symmetric (1,1)", 10, 0.00, 0.00, 0.00, 2.00, 3.12, 0.00, 3.72],
        ["Y", 2, 2, "Symmetric (2,2)", 11, 0.00, 0.00, 0.00, 2.00, 2.77, 0.00, 7.13],
        ["Y", 3, 3, "Symmetric (3,3)", 9, 0.00, 0.00, 0.00, 2.00, 2.56, 0.00, 10.60],
        ["Y", 1, 2, "Red asymmetric (1,2)", 3, 0.00, 0.00, 0.00, 2.00, 3.72, 0.00, 8.00],
        ["Y", 1, 3, "Red asymmetric (1,3)", 3, 0.00, 0.00, 0.00, 2.00, 3.85, 0.00, 12.00],
        ["Y", 0, 2, "Red dominant (0,2)", 2, 0.00, 0.00, 0.00, 2.00, 4.00, 0.00, 8.00],
        ["Y", 0, 3, "Red dominant (0,3)", 2, 0.00, 0.00, 0.00, 2.00, 4.00, 0.00, 12.00],
        ["Y", 2, 1, "Blue asymmetric (2,1)", 4, 3.79, 1.00, 0.58, 0.00, 0.00, 8.00, 0.00],
        ["Y", 3, 0, "Blue dominant (3,0)", 2, 4.00, 1.00, 0.74, 0.00, 0.00, 12.00, 0.00],
        ["N", 0, 0, "No cyber", 2, 0.00, 0.00, "-", 2.00, 2.20, 0.00, 0.00],
        ["N", 1, 1, "Symmetric (1,1)", 9, 0.00, 0.00, "-", 2.00, 3.16, 0.00, 3.73],
        ["N", 2, 2, "Symmetric (2,2)", 10, 0.00, 0.00, "-", 2.00, 2.83, 0.00, 7.19],
        ["N", 3, 3, "Symmetric (3,3)", 9, 0.00, 0.00, "-", 2.00, 2.61, 0.00, 10.66],
        ["N", 1, 2, "Red asymmetric (1,2)", 3, 0.00, 0.00, "-", 2.00, 3.73, 0.00, 8.00],
        ["N", 1, 3, "Red asymmetric (1,3)", 2, 0.00, 0.00, "-", 2.00, 3.86, 0.00, 12.00],
        ["N", 0, 2, "Red dominant (0,2)", 2, 0.00, 0.00, "-", 2.00, 4.00, 0.00, 8.00],
        ["N", 0, 3, "Red dominant (0,3)", 2, 0.00, 0.00, "-", 2.00, 4.00, 0.00, 12.00],
        ["N", 2, 1, "Blue asymmetric (2,1)", 5, 3.78, 1.00, "-", 0.00, 0.00, 7.67, 0.00],
        ["N", 3, 0, "Blue dominant (3,0)", 2, 4.00, 1.00, "-", 0.00, 0.00, 12.00, 0.00],
    ],
    columns=["sub?", "X_B", "X_R", "Configuration", "salvos", "FPSO_f", "Frig_f", "Sub_f", "Dest_f", "StrAir_f", "X_B_f", "X_R_f"],
)

s3a = pd.DataFrame(
    [[r0, k, 0.5000, 2, 0.00, 0.00, 0.00, 2.00, 4.00] for r0 in [0.5, 1.0, 2.0] for k in [1.0, 2.0, 4.0]],
    columns=["r0", "k", "Phi@parity", "salvos", "FPSO_f", "Frig_f", "Sub_f", "Dest_f", "StrAir_f"],
)

s3b = pd.DataFrame(
    [
        [0.5, 1.0, 15, 0.94, 0.56, 0.00, 2.00, 3.25],
        [0.5, 2.0, 15, 1.16, 0.68, 0.00, 2.00, 3.31],
        [0.5, 4.0, 15, 1.28, 0.74, 0.00, 2.00, 3.35],
        [1.0, 1.0, 13, 0.00, 0.00, 0.00, 2.00, 3.10],
        [1.0, 2.0, 10, 0.00, 0.00, 0.00, 2.00, 3.12],
        [1.0, 4.0, 7, 0.00, 0.00, 0.00, 2.00, 3.14],
        [2.0, 1.0, 7, 0.00, 0.00, 0.00, 2.00, 2.85],
        [2.0, 2.0, 4, 0.00, 0.00, 0.00, 2.00, 2.64],
        [2.0, 4.0, 4, 0.00, 0.00, 0.00, 1.91, 2.36],
    ],
    columns=["r0", "k", "salvos", "FPSO_f", "Frig_f", "Sub_f", "Dest_f", "StrAir_f"],
)

phi_reference = pd.DataFrame(
    [
        ["r0=0.5, k=2", 1.000, 0.500, 0.200, 0.059, 0.010, 0.000],
        ["r0=1.0, k=2", 1.000, 0.800, 0.500, 0.200, 0.038, 0.000],
        ["r0=2.0, k=2", 1.000, 0.941, 0.800, 0.500, 0.138, 0.000],
        ["r0=1.0, k=1", 1.000, 0.667, 0.500, 0.333, 0.167, 0.000],
        ["r0=1.0, k=4", 1.000, 0.941, 0.500, 0.059, 0.002, 0.000],
    ],
    columns=["Parameter", "R=0", "R=0.5", "R=1.0", "R=2.0", "R=5.0", "R=infinity"],
)

s5 = pd.DataFrame(
    [[0, 0.6695, 0.5282], [1, 0.6695, 0.0000], [2, 0.6695, 0.0000], [3, 0.6695, 0.0000]],
    columns=["Red cyber per subtype", "Residual submarine", "Residual frigate"],
)

st.title("Validation and Sensitivity Report")
st.caption("naval_salvo package - computational implementation of the Multi-Domain Salvo Equation")

st.markdown(
    """
This page consolidates the numerical validation results and sensitivity analysis of the
`naval_salvo` package. The report combines the reproduction of classical cases from the
salvo combat literature with experiments in the multi-domain Campos Basin scenario.
"""
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Package tests", "288")
col2.metric("Maximum delta", "0.00e+00")
col3.metric("Escort threshold", ">= 2 frigates")
col4.metric("Sub residual 1st salvo", "0.6695")

st.divider()

tab_valid, tab_sens, tab_figs, tab_sintese, tab_down = st.tabs(
    ["Canonical validation", "Sensitivity", "Figures", "Operational synthesis", "Downloads"]
)

with tab_valid:
    st.header("Part I - Numerical formula validation")
    st.markdown(
        """
The validation verifies whether the multi-domain formulation preserves the classical cases
when restricted to the original assumptions. The degenerate admissibility matrix reduces the
model to the Hughes (1995) 1v1 homogeneous case, and restriction to a single kinetic domain
preserves the intra-domain heterogeneous case of Johns, Pilnick and Hughes (2001).
"""
    )

    st.subheader("1.1 Hughes 1995 recovery - one salvo")
    st.dataframe(hughes_one_salvo, use_container_width=True, hide_index=True)

    st.subheader("1.2 Hughes 1995 recovery - multiple salvos")
    st.dataframe(hughes_multisalvo, use_container_width=True, hide_index=True)

    st.subheader("1.3 JPH 2001 reproduction - Coronel")
    st.dataframe(coronel, use_container_width=True, hide_index=True)

    st.success(
        "In all validation regimes presented, the difference between the engine and the analytical value is zero at machine precision."
    )

with tab_sens:
    st.header("Part II - Sensitivity analysis of the Campos Basin scenario")

    st.subheader("2.1 Escort sizing vs. submarine")
    st.markdown(
        """
This experiment varies the number of defending frigates between 0 and 4, with the submarine
present or absent, keeping the cyber component off.
"""
    )
    st.dataframe(s1, use_container_width=True, hide_index=True)

    st.markdown(
        """
Main patterns: with 0 or 1 frigate, FPSO survival is zero; with 2 or more frigates,
survival becomes positive and grows with escort sizing. The presence of the submarine
increases the resistance of the Blue force and contributes to attrition of the Red force
in specific configurations.
"""
    )

    st.subheader("2.2 Cyber matrix vs. submarine")
    st.dataframe(s2, use_container_width=True, hide_index=True)

    st.markdown(
        """
Cyber parity prolongs the campaign, while strong cyber asymmetries accelerate the collapse
of the dominated side. Under Blue cyber dominance, the engagement outcome reverses: Blue
preserves its surface assets and eliminates Red in a few salvos.
"""
    )

    st.subheader("2.3 Sensitivity to r0 and k parameters of the Phi family")
    st.markdown("**Red cyber dominance:**")
    st.dataframe(s3a, use_container_width=True, hide_index=True)
    st.markdown("**Cyber parity:**")
    st.dataframe(s3b, use_container_width=True, hide_index=True)
    st.markdown("**Reference values of the Phi(R) function:**")
    st.dataframe(phi_reference, use_container_width=True, hide_index=True)

    st.subheader("2.4 Submarine cyber immunity")
    st.dataframe(s5, use_container_width=True, hide_index=True)
    st.info(
        "The submarine maintains the same residual fraction after the first salvo for all levels of Red cyber pressure."
    )

with tab_figs:
    st.header("Figures generated by the reproduction pipeline")
    st.markdown("The figures below are loaded automatically when the `paper_artifacts/` folder is in the repository.")

    show_image("fig1_admissibility.png", "Figure 1 - 5x5 admissibility matrix")
    show_image("fig2_trajectory_no_cyber.png", "Figure 2 - Domain trajectories without cyber")
    show_image("fig3_cyber_comparison.png", "Figure 3 - Cyber scenario comparison")
    show_image("fig4_frigate_sensitivity.png", "Figure 4 - Sensitivity to number of frigates")
    show_image("fig5_submarine_sensitivity.png", "Figure 5 - Sensitivity to submarine")
    show_image("fig6_phi_sigmoid_curves.png", "Figure 6 - Phi functional family curves")
    show_image("fig7_cyber_subtype_breakdown.png", "Figure 7 - Cyber subtype breakdown")

with tab_sintese:
    st.header("Part III - Operational synthesis")
    st.markdown(
        """
The implementation literally reproduces the classical cases from the salvo combat literature
in which the proposed model is a limiting case. Building on this basis, the Campos Basin
experiments indicate four main results:

1. There is a defensibility threshold associated with the number of escort frigates.
2. Cyber parity tends to prolong the conflict, functioning as a mutual denial mechanism.
3. Cyber asymmetry can accelerate the collapse of the dominated side.
4. The submarine exhibits structural immunity to cyber modulation in the model.

The main methodological reading is that the multi-domain extension preserves the classical
models, but allows exploration of interactions that do not appear in the single-domain model:
cross-domain admissibility, dynamic targeting, cyber capability, and differential asset preservation.
"""
    )

    st.warning(
        "Important limitation: the canonical Hughes/JPH equation does not model area defence between escort and protected platform. Thus, FPSO survival results should be read as estimates of a pessimistic scenario or one without area defence coverage."
    )

with tab_down:
    st.header("Artifact downloads")
    st.markdown("The buttons appear when the corresponding files are present in `paper_artifacts/`.")

    download_file("table1_parameters.csv", "Download Table 1 - Parameters", "text/csv")
    download_file("table2_sensitivity_matrix.csv", "Download Table 2 - Sensitivity", "text/csv")
    download_file("table3_jph_recovery.csv", "Download Table 3 - Validation", "text/csv")
    download_file("paper_sections_5_and_6.md", "Download sections 5 and 6 - Markdown", "text/markdown")
    download_file(
        "paper_sections_5_and_6.docx",
        "Download sections 5 and 6 - Word",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )
