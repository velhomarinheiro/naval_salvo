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
        st.info(f"Figura nao encontrada em paper_artifacts/: {filename}")


def download_file(filename: str, label: str, mime: str) -> None:
    path = ARTIFACTS_DIR / filename
    if path.exists():
        st.download_button(label=label, data=path.read_bytes(), file_name=filename, mime=mime)
    else:
        st.caption(f"Arquivo nao encontrado: {filename}")


hughes_one_salvo = pd.DataFrame(
    [
        ["Balanceado", 4, 2.0, 2.0, "0.00e+00", 4, 2.0, 2.0, "0.00e+00"],
        ["Atacante dominante", 4, 2.0, 2.0, "0.00e+00", 4, 0.0, 0.0, "0.00e+00"],
        ["Defensor dominante", 4, 2.0, 2.0, "0.00e+00", 4, 4.0, 4.0, "0.00e+00"],
        ["Forcas assimetricas", 6, 3.0, 3.0, "0.00e+00", 3, 0.0, 0.0, "0.00e+00"],
        ["Defesa saturante", 4, 4.0, 4.0, "0.00e+00", 4, 4.0, 4.0, "0.00e+00"],
    ],
    columns=["Caso", "A_pre", "A_post analitico", "A_post engine", "Delta_A", "B_pre", "B_post analitico", "B_post engine", "Delta_B"],
)

hughes_multisalvo = pd.DataFrame(
    [
        [0, 6.0, 6.0, "0.00e+00", 4.0, 4.0, "0.00e+00"],
        [1, 1.0, 1.0, "0.00e+00", 3.0, 3.0, "0.00e+00"],
        [2, 0.0, 0.0, "0.00e+00", 3.0, 3.0, "0.00e+00"],
        [3, 0.0, 0.0, "0.00e+00", 3.0, 3.0, "0.00e+00"],
        [4, 0.0, 0.0, "0.00e+00", 3.0, 3.0, "0.00e+00"],
    ],
    columns=["k", "A analitico", "A engine", "Delta_A", "B analitico", "B engine", "Delta_B"],
)

coronel = pd.DataFrame(
    [
        ["Good Hope + Monmouth", 2, 1.605, 0.120960, 0.075364, 0.037682, 0.037682, "0.00e+00"],
        ["Glasgow", 1, 1.230, 0.051960, 0.042244, 0.042244, 0.042244, "0.00e+00"],
        ["Scharnhorst + Gneisenau", 2, 1.665, 0.000000, 0.000000, 0.000000, 0.000000, "0.00e+00"],
        ["Leipzig + Dresden", 2, 1.115, 0.000000, 0.000000, 0.000000, 0.000000, "0.00e+00"],
    ],
    columns=["Grupo alvo", "n_navios", "Staying power", "Kernel total", "Loss grupo", "Delta engine", "Delta analitico", "Delta"],
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
    columns=["n_frig", "sub?", "salvas", "FPSO_f", "Frig_f", "Sub_f", "Dest_f", "StrAir_f"],
)

s2 = pd.DataFrame(
    [
        ["Y", 0, 0, "Sem cyber", 3, 0.00, 0.00, 0.00, 1.78, 2.08, 0.00, 0.00],
        ["Y", 1, 1, "Simetrico (1,1)", 10, 0.00, 0.00, 0.00, 2.00, 3.12, 0.00, 3.72],
        ["Y", 2, 2, "Simetrico (2,2)", 11, 0.00, 0.00, 0.00, 2.00, 2.77, 0.00, 7.13],
        ["Y", 3, 3, "Simetrico (3,3)", 9, 0.00, 0.00, 0.00, 2.00, 2.56, 0.00, 10.60],
        ["Y", 1, 2, "Assimetrico vermelho (1,2)", 3, 0.00, 0.00, 0.00, 2.00, 3.72, 0.00, 8.00],
        ["Y", 1, 3, "Assimetrico vermelho (1,3)", 3, 0.00, 0.00, 0.00, 2.00, 3.85, 0.00, 12.00],
        ["Y", 0, 2, "Dominio vermelho (0,2)", 2, 0.00, 0.00, 0.00, 2.00, 4.00, 0.00, 8.00],
        ["Y", 0, 3, "Dominio vermelho (0,3)", 2, 0.00, 0.00, 0.00, 2.00, 4.00, 0.00, 12.00],
        ["Y", 2, 1, "Assimetrico azul (2,1)", 4, 3.79, 1.00, 0.58, 0.00, 0.00, 8.00, 0.00],
        ["Y", 3, 0, "Dominio azul (3,0)", 2, 4.00, 1.00, 0.74, 0.00, 0.00, 12.00, 0.00],
        ["N", 0, 0, "Sem cyber", 2, 0.00, 0.00, "-", 2.00, 2.20, 0.00, 0.00],
        ["N", 1, 1, "Simetrico (1,1)", 9, 0.00, 0.00, "-", 2.00, 3.16, 0.00, 3.73],
        ["N", 2, 2, "Simetrico (2,2)", 10, 0.00, 0.00, "-", 2.00, 2.83, 0.00, 7.19],
        ["N", 3, 3, "Simetrico (3,3)", 9, 0.00, 0.00, "-", 2.00, 2.61, 0.00, 10.66],
        ["N", 1, 2, "Assimetrico vermelho (1,2)", 3, 0.00, 0.00, "-", 2.00, 3.73, 0.00, 8.00],
        ["N", 1, 3, "Assimetrico vermelho (1,3)", 2, 0.00, 0.00, "-", 2.00, 3.86, 0.00, 12.00],
        ["N", 0, 2, "Dominio vermelho (0,2)", 2, 0.00, 0.00, "-", 2.00, 4.00, 0.00, 8.00],
        ["N", 0, 3, "Dominio vermelho (0,3)", 2, 0.00, 0.00, "-", 2.00, 4.00, 0.00, 12.00],
        ["N", 2, 1, "Assimetrico azul (2,1)", 5, 3.78, 1.00, "-", 0.00, 0.00, 7.67, 0.00],
        ["N", 3, 0, "Dominio azul (3,0)", 2, 4.00, 1.00, "-", 0.00, 0.00, 12.00, 0.00],
    ],
    columns=["sub?", "X_B", "X_R", "Configuracao", "salvas", "FPSO_f", "Frig_f", "Sub_f", "Dest_f", "StrAir_f", "X_B_f", "X_R_f"],
)

s3a = pd.DataFrame(
    [[r0, k, 0.5000, 2, 0.00, 0.00, 0.00, 2.00, 4.00] for r0 in [0.5, 1.0, 2.0] for k in [1.0, 2.0, 4.0]],
    columns=["r0", "k", "Phi@parity", "salvas", "FPSO_f", "Frig_f", "Sub_f", "Dest_f", "StrAir_f"],
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
    columns=["r0", "k", "salvas", "FPSO_f", "Frig_f", "Sub_f", "Dest_f", "StrAir_f"],
)

phi_reference = pd.DataFrame(
    [
        ["r0=0.5, k=2", 1.000, 0.500, 0.200, 0.059, 0.010, 0.000],
        ["r0=1.0, k=2", 1.000, 0.800, 0.500, 0.200, 0.038, 0.000],
        ["r0=2.0, k=2", 1.000, 0.941, 0.800, 0.500, 0.138, 0.000],
        ["r0=1.0, k=1", 1.000, 0.667, 0.500, 0.333, 0.167, 0.000],
        ["r0=1.0, k=4", 1.000, 0.941, 0.500, 0.059, 0.002, 0.000],
    ],
    columns=["Parametro", "R=0", "R=0.5", "R=1.0", "R=2.0", "R=5.0", "R=infinito"],
)

s5 = pd.DataFrame(
    [[0, 0.6695, 0.5282], [1, 0.6695, 0.0000], [2, 0.6695, 0.0000], [3, 0.6695, 0.0000]],
    columns=["Cyber vermelho por subtipo", "Sub residual", "Frig residual"],
)

st.title("Relatorio de Validacao e Sensibilidade")
st.caption("Pacote naval_salvo - implementacao computacional da Equacao de Salva Multidominio")

st.markdown(
    """
Esta pagina consolida os resultados de validacao numerica e analise de sensibilidade do pacote
`naval_salvo`. O relatorio combina a reproducao de casos classicos da literatura de combate por
salvas com experimentos no cenario multidominio da Bacia de Campos.
"""
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Testes do pacote", "288")
col2.metric("Delta maximo", "0.00e+00")
col3.metric("Limiar de escolta", ">= 2 fragatas")
col4.metric("Sub residual 1a salva", "0.6695")

st.divider()

tab_valid, tab_sens, tab_figs, tab_sintese, tab_down = st.tabs(
    ["Validacao canonica", "Sensibilidade", "Figuras", "Sintese operacional", "Downloads"]
)

with tab_valid:
    st.header("Parte I - Validacao numerica da formula")
    st.markdown(
        """
A validacao verifica se a formulacao multidominio preserva os casos classicos quando restrita
as hipoteses originais. A matriz de admissibilidade degenerada reduz o modelo ao caso
homogeneo 1v1 de Hughes (1995), e a restricao a um unico dominio cinetico preserva o caso
heterogeneo intra-dominio de Johns, Pilnick e Hughes (2001).
"""
    )

    st.subheader("1.1 Recuperacao Hughes 1995 - uma salva")
    st.dataframe(hughes_one_salvo, use_container_width=True, hide_index=True)

    st.subheader("1.2 Recuperacao Hughes 1995 - multiplas salvas")
    st.dataframe(hughes_multisalvo, use_container_width=True, hide_index=True)

    st.subheader("1.3 Reproducao JPH 2001 - Coronel")
    st.dataframe(coronel, use_container_width=True, hide_index=True)

    st.success(
        "Em todos os regimes de validacao apresentados, a diferenca entre o engine e o valor analitico e zero em precisao de maquina."
    )

with tab_sens:
    st.header("Parte II - Analise de sensibilidade do cenario Bacia de Campos")

    st.subheader("2.1 Dimensionamento da escolta x submarino")
    st.markdown(
        """
Este experimento varia o numero de fragatas defensoras entre 0 e 4, com o submarino presente
ou ausente, mantendo o componente cibernetico desligado.
"""
    )
    st.dataframe(s1, use_container_width=True, hide_index=True)

    st.markdown(
        """
Padroes principais: com 0 ou 1 fragata, a sobrevivencia das FPSOs e nula; com 2 ou mais
fragatas, a sobrevivencia passa a ser positiva e cresce com o dimensionamento da escolta.
A presenca do submarino aumenta a resistencia da forca Azul e contribui para a atricao da
forca Vermelha em configuracoes especificas.
"""
    )

    st.subheader("2.2 Matriz cyber x submarino")
    st.dataframe(s2, use_container_width=True, hide_index=True)

    st.markdown(
        """
A paridade cibernetica prolonga a campanha, enquanto assimetrias ciberneticas fortes aceleram
o colapso do lado dominado. Sob dominio cibernetico Azul, o resultado do encontro se inverte:
Blue preserva os ativos de superficie e elimina Red em poucas salvas.
"""
    )

    st.subheader("2.3 Sensibilidade aos parametros r0 e k da familia Phi")
    st.markdown("**Dominio cibernetico vermelho:**")
    st.dataframe(s3a, use_container_width=True, hide_index=True)
    st.markdown("**Paridade cibernetica:**")
    st.dataframe(s3b, use_container_width=True, hide_index=True)
    st.markdown("**Valores de referencia da funcao Phi(R):**")
    st.dataframe(phi_reference, use_container_width=True, hide_index=True)

    st.subheader("2.4 Imunidade cibernetica do submarino")
    st.dataframe(s5, use_container_width=True, hide_index=True)
    st.info(
        "O submarino mantem a mesma fracao residual apos a primeira salva para todos os niveis de pressao cibernetica vermelha."
    )

with tab_figs:
    st.header("Figuras geradas pelo pipeline de reproducao")
    st.markdown("As figuras abaixo sao carregadas automaticamente quando a pasta `paper_artifacts/` esta no repositorio.")

    show_image("fig1_admissibility.png", "Figura 1 - Matriz de admissibilidade 5x5")
    show_image("fig2_trajectory_no_cyber.png", "Figura 2 - Trajetorias por dominio sem cyber")
    show_image("fig3_cyber_comparison.png", "Figura 3 - Comparacao de cenarios ciberneticos")
    show_image("fig4_frigate_sensitivity.png", "Figura 4 - Sensibilidade ao numero de fragatas")
    show_image("fig5_submarine_sensitivity.png", "Figura 5 - Sensibilidade ao submarino")
    show_image("fig6_phi_sigmoid_curves.png", "Figura 6 - Curvas da familia funcional Phi")
    show_image("fig7_cyber_subtype_breakdown.png", "Figura 7 - Breakdown dos subtipos ciberneticos")

with tab_sintese:
    st.header("Parte III - Sintese operacional")
    st.markdown(
        """
A implementacao reproduz literalmente os casos classicos da literatura de combate por salvas
nos quais o modelo proposto e um caso limite. A partir dessa base, os experimentos da Bacia de
Campos indicam quatro resultados principais:

1. Existe um limiar de defensibilidade associado ao numero de fragatas de escolta.
2. A paridade cibernetica tende a alongar o conflito, funcionando como mecanismo de denegacao mutua.
3. A assimetria cibernetica pode acelerar o colapso do lado dominado.
4. O submarino apresenta imunidade estrutural a modulacao cibernetica no modelo.

A principal leitura metodologica e que a extensao multidominio preserva os modelos classicos,
mas permite explorar interacoes que nao aparecem no modelo monodominio: admissibilidade entre
dominios, targeting dinamico, capacidade cibernetica e preservacao diferencial de ativos.
"""
    )

    st.warning(
        "Limite importante: a equacao canonica de Hughes/JPH nao modela defesa de area entre escolta e plataforma protegida. Assim, os resultados de sobrevivencia das FPSOs devem ser lidos como estimativas de um cenario pessimista ou sem cobertura de defesa de area."
    )

with tab_down:
    st.header("Downloads dos artefatos")
    st.markdown("Os botoes aparecem quando os arquivos correspondentes estao presentes em `paper_artifacts/`.")

    download_file("table1_parameters.csv", "Baixar Tabela 1 - Parametros", "text/csv")
    download_file("table2_sensitivity_matrix.csv", "Baixar Tabela 2 - Sensibilidade", "text/csv")
    download_file("table3_jph_recovery.csv", "Baixar Tabela 3 - Validacao", "text/csv")
    download_file("paper_sections_5_and_6.md", "Baixar secoes 5 e 6 - Markdown", "text/markdown")
    download_file(
        "paper_sections_5_and_6.docx",
        "Baixar secoes 5 e 6 - Word",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )
