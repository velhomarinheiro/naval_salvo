from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parents[1] if len(Path(__file__).resolve().parents) > 1 else Path.cwd()
ARTIFACTS_DIR = BASE_DIR / "paper_artifacts"

st.title("Relatório de Validação e Sensibilidade")
st.caption("Pacote `naval_salvo` — implementação computacional da Equação de Salva Multidomínio")

st.markdown(
    """
Esta página consolida os principais resultados do relatório de validação numérica e análise de
sensibilidade do pacote `naval_salvo`. A validação é organizada em dois blocos: reprodução dos
casos clássicos da literatura de combate por salvas e exploração sistemática do cenário
multidomínio aplicado à Bacia de Campos.

Os resultados abaixo foram estruturados para funcionar mesmo quando os arquivos `.csv` e as figuras
não estiverem disponíveis no deploy. Quando a pasta `paper_artifacts/` existir no repositório, a página
também tentará carregar automaticamente as tabelas e figuras geradas pelo pipeline de reprodução.
"""
)


def _read_csv_or_none(filename: str) -> pd.DataFrame | None:
    path = ARTIFACTS_DIR / filename
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception as exc:  # pragma: no cover - UI fallback
            st.warning(f"Não foi possível carregar `{filename}`: {exc}")
    return None


def _show_image_if_exists(filename: str, caption: str) -> None:
    path = ARTIFACTS_DIR / filename
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.info(f"Figura `{filename}` não encontrada em `paper_artifacts/`.")


def _download_if_exists(filename: str, label: str, mime: str = "text/csv") -> None:
    path = ARTIFACTS_DIR / filename
    if path.exists():
        st.download_button(
            label=label,
            data=path.read_bytes(),
            file_name=filename,
            mime=mime,
        )


# -----------------------------------------------------------------------------
# Tabelas fixas/fallback
# -----------------------------------------------------------------------------

hughes_one_salvo = pd.DataFrame(
    [
        ["Balanceado (α=β=2, ς=2)", 4, 2.0, 2.0, "0.00e+00", 4, 2.0, 2.0, "0.00e+00"],
        ["Atacante dominante (α=4)", 4, 2.0, 2.0, "0.00e+00", 4, 0.0, 0.0, "0.00e+00"],
        ["Defensor dominante (z=3)", 4, 2.0, 2.0, "0.00e+00", 4, 4.0, 4.0, "0.00e+00"],
        ["Forças assimétricas (A=6, B=3, β=4)", 6, 3.0, 3.0, "0.00e+00", 3, 0.0, 0.0, "0.00e+00"],
        ["Defesa saturante (z=y=5)", 4, 4.0, 4.0, "0.00e+00", 4, 4.0, 4.0, "0.00e+00"],
    ],
    columns=[
        "Caso", "A_pre", "A_post analítico", "A_post engine", "Δ_A",
        "B_pre", "B_post analítico", "B_post engine", "Δ_B",
    ],
)

hughes_multisalvo = pd.DataFrame(
    [
        [0, 6.0, 6.0, "0.00e+00", 4.0, 4.0, "0.00e+00"],
        [1, 1.0, 1.0, "0.00e+00", 3.0, 3.0, "0.00e+00"],
        [2, 0.0, 0.0, "0.00e+00", 3.0, 3.0, "0.00e+00"],
        [3, 0.0, 0.0, "0.00e+00", 3.0, 3.0, "0.00e+00"],
        [4, 0.0, 0.0, "0.00e+00", 3.0, 3.0, "0.00e+00"],
    ],
    columns=["k", "A analítico", "A engine", "Δ_A", "B analítico", "B engine", "Δ_B"],
)

coronel = pd.DataFrame(
    [
        ["Good Hope + Monmouth (britânico)", 2, 1.605, 0.120960, 0.075364, 0.037682, 0.037682, "0.00e+00"],
        ["Glasgow (britânico)", 1, 1.230, 0.051960, 0.042244, 0.042244, 0.042244, "0.00e+00"],
        ["Scharnhorst + Gneisenau (alemão)", 2, 1.665, 0.000000, 0.000000, 0.000000, 0.000000, "0.00e+00"],
        ["Leipzig + Dresden (alemão)", 2, 1.115, 0.000000, 0.000000, 0.000000, 0.000000, "0.00e+00"],
    ],
    columns=["Grupo (alvo)", "n_navios", "ς/navio", "Kernel total", "Loss grupo", "ΔA/navio engine", "ΔA/navio analítico", "Δ"],
)

s1 = pd.DataFrame(
    [
        [0, "Y", 2, 0.00, 0.00, 0.00, 2.00, 3.30],
        [1, "Y", 3, 0.00, 0.00, 0.00, 1.78, 2.08],
        [2, "Y", 3, 0.46, 1.83, 0.49, 0.00, 0.00],
        [3, "Y", 2, 1.02, 3.00, 0.61, 0.00, 0.00],
        [4, "Y", 2, 1.34, 4.00, 0.67, 0.00, 0.00],
        [0, "N", 2, 0.00, 0.00, "—", 2.00, 3.30],
        [1, "N", 2, 0.00, 0.00, "—", 2.00, 2.20],
        [2, "N", 3, 0.00, 0.00, "—", 1.09, 0.29],
        [3, "N", 2, 0.60, 3.00, "—", 0.00, 0.00],
        [4, "N", 2, 0.92, 4.00, "—", 0.00, 0.00],
    ],
    columns=["n_frig", "sub?", "salvas", "FPSO_f", "Frig_f", "Sub_f", "Dest_f", "StrAir_f"],
)

s2 = pd.DataFrame(
    [
        ["Y", 0, 0, "Sem cyber", 3, 0.00, 0.00, 0.00, 1.78, 2.08, 0.00, 0.00],
        ["Y", 1, 1, "Simétrico (1,1)", 10, 0.00, 0.00, 0.00, 2.00, 3.12, 0.00, 3.72],
        ["Y", 2, 2, "Simétrico (2,2)", 11, 0.00, 0.00, 0.00, 2.00, 2.77, 0.00, 7.13],
        ["Y", 3, 3, "Simétrico (3,3)", 9, 0.00, 0.00, 0.00, 2.00, 2.56, 0.00, 10.60],
        ["Y", 1, 2, "Assim. Vermelho (1,2)", 3, 0.00, 0.00, 0.00, 2.00, 3.72, 0.00, 8.00],
        ["Y", 1, 3, "Assim. Vermelho (1,3)", 3, 0.00, 0.00, 0.00, 2.00, 3.85, 0.00, 12.00],
        ["Y", 0, 2, "Domínio Vermelho (0,2)", 2, 0.00, 0.00, 0.00, 2.00, 4.00, 0.00, 8.00],
        ["Y", 0, 3, "Domínio Vermelho (0,3)", 2, 0.00, 0.00, 0.00, 2.00, 4.00, 0.00, 12.00],
        ["Y", 2, 1, "Assim. Azul (2,1)", 4, 3.79, 1.00, 0.58, 0.00, 0.00, 8.00, 0.00],
        ["Y", 3, 0, "Domínio Azul (3,0)", 2, 4.00, 1.00, 0.74, 0.00, 0.00, 12.00, 0.00],
        ["N", 0, 0, "Sem cyber", 2, 0.00, 0.00, "—", 2.00, 2.20, 0.00, 0.00],
        ["N", 1, 1, "Simétrico (1,1)", 9, 0.00, 0.00, "—", 2.00, 3.16, 0.00, 3.73],
        ["N", 2, 2, "Simétrico (2,2)", 10, 0.00, 0.00, "—", 2.00, 2.83, 0.00, 7.19],
        ["N", 3, 3, "Simétrico (3,3)", 9, 0.00, 0.00, "—", 2.00, 2.61, 0.00, 10.66],
        ["N", 1, 2, "Assim. Vermelho (1,2)", 3, 0.00, 0.00, "—", 2.00, 3.73, 0.00, 8.00],
        ["N", 1, 3, "Assim. Vermelho (1,3)", 2, 0.00, 0.00, "—", 2.00, 3.86, 0.00, 12.00],
        ["N", 0, 2, "Domínio Vermelho (0,2)", 2, 0.00, 0.00, "—", 2.00, 4.00, 0.00, 8.00],
        ["N", 0, 3, "Domínio Vermelho (0,3)", 2, 0.00, 0.00, "—", 2.00, 4.00, 0.00, 12.00],
        ["N", 2, 1, "Assim. Azul (2,1)", 5, 3.78, 1.00, "—", 0.00, 0.00, 7.67, 0.00],
        ["N", 3, 0, "Domínio Azul (3,0)", 2, 4.00, 1.00, "—", 0.00, 0.00, 12.00, 0.00],
    ],
    columns=["sub?", "X_B", "X_R", "Configuração", "salvas", "FPSO_f", "Frig_f", "Sub_f", "Dest_f", "StrAir_f", "X_B_f", "X_R_f"],
)

s3a = pd.DataFrame(
    [[r0, k, 0.5000, 2, 0.00, 0.00, 0.00, 2.00, 4.00] for r0 in [0.5, 1.0, 2.0] for k in [1.0, 2.0, 4.0]],
    columns=["r₀", "k", "Φ@parity", "salvas", "FPSO_f", "Frig_f", "Sub_f", "Dest_f", "StrAir_f"],
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
    columns=["r₀", "k", "salvas", "FPSO_f", "Frig_f", "Sub_f", "Dest_f", "StrAir_f"],
)

phi_reference = pd.DataFrame(
    [
        ["(r₀=0.5, k=2)", 1.000, 0.500, 0.200, 0.059, 0.010, 0.000],
        ["(r₀=1.0, k=2)", 1.000, 0.800, 0.500, 0.200, 0.038, 0.000],
        ["(r₀=2.0, k=2)", 1.000, 0.941, 0.800, 0.500, 0.138, 0.000],
        ["(r₀=1.0, k=1)", 1.000, 0.667, 0.500, 0.333, 0.167, 0.000],
        ["(r₀=1.0, k=4)", 1.000, 0.941, 0.500, 0.059, 0.002, 0.000],
    ],
    columns=["Φ(R) com r₀, k", "R=0", "R=0.5", "R=1.0", "R=2.0", "R=5.0", "R=∞"],
)

s5 = pd.DataFrame(
    [
        [0, 0.6695, 0.5282],
        [1, 0.6695, 0.0000],
        [2, 0.6695, 0.0000],
        [3, 0.6695, 0.0000],
    ],
    columns=["Cyber Vermelho por sub-tipo", "Sub residual", "Frig residual"],
)

# -----------------------------------------------------------------------------
# Métricas de destaque
# -----------------------------------------------------------------------------

col1, col2, col3, col4 = st.columns(4)
col1.metric("Testes do pacote", "288")
col2.metric("Δ máximo na validação", "0.00e+00")
col3.metric("Limiar de escolta", "≥ 2 fragatas")
col4.metric("Sub residual 1ª salva", "0.6695")

st.divider()

tabs = st.tabs(
    [
        "1. Validação canônica",
        "2. Sensibilidade",
        "3. Trajetórias e figuras",
        "4. Síntese operacional",
        "5. Downloads",
    ]
)

with tabs[0]:
    st.header("Parte I — Validação numérica da fórmula")

    st.markdown(
        """
A validação verifica se a formulação multidomínio preserva os casos clássicos quando
restrita às hipóteses originais. Em particular, a matriz de admissibilidade degenerada
reduz o modelo ao caso homogêneo 1v1 de Hughes (1995), enquanto a restrição a um
único domínio cinético preserva o caso heterogêneo intra-domínio de Johns, Pilnick e
Hughes (2001).
"""
    )

    st.subheader("1.1 Recuperação do caso degenerado de Hughes (1995)")
    st.markdown(
        """
A formulação multidomínio reduz-se ao modelo homogêneo quando apenas o par
superfície × superfície permanece ativo na matriz χ. Foram testados cinco regimes
paramétricos para cobrir situações balanceadas, dominância ofensiva, dominância
defensiva, assimetria de força e defesa saturante.
"""
    )
    st.dataframe(hughes_one_salvo, use_container_width=True, hide_index=True)

    st.markdown("**Recuperação Hughes 1995 multi-salva.**")
    st.dataframe(hughes_multisalvo, use_container_width=True, hide_index=True)

    st.success(
        "A diferença máxima entre engine e valor analítico é zero literal nos regimes testados."
    )

    st.subheader("1.2 Reprodução do exemplo trabalhado de Coronel — JPH 2001")
    st.markdown(
        """
O exemplo de Coronel foi reproduzido cell-by-cell para o primeiro minuto da batalha,
com dois grupos britânicos e dois grupos alemães. No minuto inicial, os britânicos estão
silenciosos; Scharnhorst e Gneisenau dividem fogo contra Good Hope e Monmouth;
Leipzig e Dresden engajam Glasgow.
"""
    )
    st.dataframe(coronel, use_container_width=True, hide_index=True)

    with st.expander("Verificação aritmética dos valores analíticos"):
        st.latex(r"\Delta A_{Good\ Hope} = \frac{2\cdot(0.028\cdot 2.16\cdot 0.5\cdot 1.0)}{1.605} = 0.0376822429906542")
        st.latex(r"\Delta A_{Glasgow} = \frac{2\cdot(0.012\cdot 2.165\cdot 1.0)}{1.230} = 0.0422439024390244")
        st.markdown("Os grupos alemães recebem ΔB = 0 porque os britânicos estão silenciosos no minuto 1.")

    st.subheader("1.3 Síntese da validação")
    st.markdown(
        """
A implementação satisfaz duas propriedades de correção:

1. reduz-se ao caso homogêneo de Hughes 1995 sob admissibilidade degenerada;
2. reduz-se ao caso heterogêneo intra-domínio de Johns–Pilnick–Hughes 2001 sob
   admissibilidade restrita ao domínio cinético atuante.

Em ambos os casos, a diferença numérica entre o engine e os valores analíticos é zero
em precisão de máquina. Isso indica que as extensões propostas — cinco domínios,
modulação Φ e realocação proporcional κ — são aditivas e não alteram o comportamento
do modelo nos regimes clássicos.
"""
    )

with tabs[1]:
    st.header("Parte II — Análise de sensibilidade do cenário Bacia de Campos")
    st.markdown(
        """
A análise de sensibilidade percorre quatro eixos: dimensionamento da escolta de superfície,
presença do submarino, capacidade cibernética e parâmetros da família funcional Φ.
Os experimentos usam política de mira `StrengthProportional` para o defensor,
`ThreatWeighted` para o atacante, matriz de admissibilidade canônica com χ=0.5 e duração
máxima de 15 salvas.
"""
    )

    st.subheader("2.1 Sensibilidade I — Dimensionamento da escolta × submarino")
    st.dataframe(s1, use_container_width=True, hide_index=True)
    st.markdown(
        """
**Padrões observados:**

- Com 0–1 fragatas, a sobrevivência das FPSOs é zero, independentemente do submarino.
- Com duas ou mais fragatas, os FPSOs preservam-se em fração crescente.
- A presença do submarino aumenta a sobrevivência das FPSOs e estende a duração da campanha.
- Configurações com duas ou mais fragatas conseguem impor atrição decisiva à força Vermelha.
"""
    )

    st.subheader("2.2 Sensibilidade II — Matriz cibernético × submarino")
    st.dataframe(s2, use_container_width=True, hide_index=True)
    st.markdown(
        """
**Padrões observados:**

- A paridade cibernética prolonga o conflito, mas não necessariamente altera o vencedor.
- A assimetria cibernética Vermelha produz um modo de resolução rápida, com preservação dos meios Vermelhos.
- O domínio cibernético Azul preserva integralmente os FPSOs e elimina a força Vermelha em duas salvas.
- O efeito do submarino aparece de forma mais clara nas trajetórias do que apenas no valor final da campanha.
"""
    )

    st.subheader("2.3 Sensibilidade III — Parâmetros (r₀, k) da família Φ")
    left, right = st.columns(2)
    with left:
        st.markdown("**Sob domínio cibernético Vermelho — X_B=0, X_R=2**")
        st.dataframe(s3a, use_container_width=True, hide_index=True)
    with right:
        st.markdown("**Sob paridade cibernética — X_B=1, X_R=1**")
        st.dataframe(s3b, use_container_width=True, hide_index=True)

    st.markdown("**Curvas funcionais de referência para Φ(R).**")
    st.dataframe(phi_reference, use_container_width=True, hide_index=True)

    st.markdown(
        """
Sob assimetria cibernética extrema, o desfecho é robusto às escolhas razoáveis de
(r₀, k). Sob paridade, r₀ torna-se o parâmetro mais influente: valores baixos produzem
maior degradação mútua e prolongam a campanha, podendo preservar parcialmente as FPSOs.
"""
    )

    st.subheader("2.4 Demonstração da imunidade cibernética do submarino")
    st.dataframe(s5, use_container_width=True, hide_index=True)
    st.markdown(
        """
A fração residual do submarino após a primeira salva permanece exatamente constante sob
pressão cibernética Vermelha crescente. A fragata, por contraste, vai a zero a partir de
qualquer capacidade cibernética Vermelha positiva.
"""
    )

with tabs[2]:
    st.header("Parte III — Trajetórias e figuras")
    st.markdown(
        """
Quando disponíveis, as figuras geradas pelo script `reproduce_paper.py` são carregadas a
partir da pasta `paper_artifacts/`. Caso as imagens não apareçam, gere os artefatos localmente
e envie a pasta para o GitHub.
"""
    )

    fig_col1, fig_col2 = st.columns(2)
    with fig_col1:
        _show_image_if_exists("fig1_admissibility.png", "Figura 1 — Matriz de admissibilidade χ 5×5")
        _show_image_if_exists("fig3_cyber_comparison.png", "Figura 3 — Comparação dos regimes cibernéticos")
        _show_image_if_exists("fig5_submarine_sensitivity.png", "Figura 5 — Sensibilidade e imunidade do submarino")
        _show_image_if_exists("fig7_cyber_subtype_breakdown.png", "Figura 7 — Subtipos cibernéticos")
    with fig_col2:
        _show_image_if_exists("fig2_trajectory_no_cyber.png", "Figura 2 — Trajetória baseline sem cibernético")
        _show_image_if_exists("fig4_frigate_sensitivity.png", "Figura 4 — Sensibilidade ao número de fragatas")
        _show_image_if_exists("fig6_phi_sigmoid_curves.png", "Figura 6 — Curvas da família Φ")

    st.info("Comando de reprodução dos artefatos: `python reproduce_paper.py --out paper_artifacts/`.")

with tabs[3]:
    st.header("Parte IV — Síntese operacional")

    st.subheader("Achado 1 — Limiar de defensibilidade")
    st.markdown(
        """
O cenário sem cyber sugere que duas fragatas constituem o limiar mínimo para sobrevivência
não nula das FPSOs. A presença do submarino melhora a sobrevivência dos ativos protegidos
e aumenta a duração da campanha.
"""
    )

    st.subheader("Achado 2 — Paridade cibernética prolonga o conflito")
    st.markdown(
        """
A paridade cibernética tende a estender o confronto para 9–11 salvas, contra 2–3 salvas
no cenário sem cyber. O efeito é interpretado como forma de denegação mútua, embora não
altere necessariamente o vencedor nos parâmetros de referência.
"""
    )

    st.subheader("Achado 3 — Imunidade estrutural do submarino")
    st.markdown(
        """
O submarino mantém fração residual constante após a primeira salva sob todos os níveis de
pressão cibernética Vermelha testados. Essa propriedade deriva da imposição estrutural de
Φ ≡ 1 para defensores no domínio submarino.
"""
    )

    st.subheader("Achado 4 — Cyber como operador estrutural")
    st.markdown(
        """
Sob domínio cibernético Azul, o resultado do encontro inverte-se: a força Azul preserva suas
unidades de superfície, incluindo FPSOs, e elimina a força Vermelha em duas salvas. Isso
indica que a modulação Φ não atua apenas como amplificador marginal, mas pode reordenar
qualitativamente o desfecho do cenário.
"""
    )

    st.warning(
        "Limitação importante: a formulação canônica Hughes/JPH não modela defesa de área entre escolta e plataforma protegida. "
        "Assim, os resultados de sobrevivência das FPSOs devem ser lidos como um cenário conservador/pessimista, não como previsão doutrinária definitiva."
    )

with tabs[4]:
    st.header("Downloads e reprodução")
    st.markdown(
        """
Se os arquivos estiverem disponíveis em `paper_artifacts/`, os botões abaixo permitem baixar
as tabelas e o relatório gerados pelo pipeline de reprodução.
"""
    )

    st.subheader("Tabelas")
    _download_if_exists("table1_parameters.csv", "Baixar Tabela 1 — Parâmetros")
    _download_if_exists("table2_sensitivity_matrix.csv", "Baixar Tabela 2 — Matriz de sensibilidade")
    _download_if_exists("table3_jph_recovery.csv", "Baixar Tabela 3 — Recuperação JPH/Hughes")

    st.subheader("Relatório")
    _download_if_exists(
        "paper_sections_5_and_6.docx",
        "Baixar seções 5 e 6 em DOCX",
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )
    _download_if_exists("paper_sections_5_and_6.md", "Baixar seções 5 e 6 em Markdown", mime="text/markdown")

    st.subheader("Comando de reprodução")
    st.code("python reproduce_paper.py --out paper_artifacts/", language="bash")

    st.markdown(
        """
