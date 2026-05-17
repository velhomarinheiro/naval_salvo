"""
Página "Calculadora Offline" do aplicativo Equação de Salva Multidomínio.

Apresenta a versão em planilha Excel da calculadora e a cartilha em PDF
com instruções de uso. Os arquivos devem ser colocados na pasta
offline_assets/ na raiz do repositório.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st


BASE_DIR = Path(__file__).resolve().parent.parent
ASSETS_DIR = BASE_DIR / "offline_assets"

EXCEL_FILE = ASSETS_DIR / "calculadora_offline.xlsx"
PDF_FILE = ASSETS_DIR / "cartilha_calculadora_offline.pdf"


st.title("Calculadora Offline")
st.caption("Versão em planilha Excel para uso local da Equação de Salva")

st.markdown("---")

st.header("Sobre a versão offline")

st.markdown(
    """
    Esta página disponibiliza uma versão **offline** da calculadora da
    Equação de Salva, em formato de **planilha Excel**. A planilha permite
    realizar simulações básicas sem depender de conexão com a internet ou
    do acesso ao aplicativo Streamlit.

    A calculadora offline aplica o modelo com algumas **simplificações**
    em relação à versão online. Ela foi pensada para uso didático,
    exercícios exploratórios, jogos de guerra, estudos preliminares de
    composição de força e apoio à análise de sensibilidade.

    A planilha acompanha uma **cartilha do usuário**, em formato PDF, com
    instruções para preenchimento dos campos, interpretação dos resultados
    e principais limitações da ferramenta.
    """
)

st.info(
    "A versão offline é complementar ao aplicativo online. Para análises "
    "mais completas, especialmente envolvendo cenários multidomínio, "
    "efeitos cibernéticos e políticas dinâmicas de seleção de alvos, "
    "recomenda-se utilizar as páginas interativas do app."
)

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Planilha Excel")

    st.markdown(
        """
        A planilha contém uma implementação simplificada da calculadora,
        permitindo alterar parâmetros básicos de força, poder ofensivo,
        poder defensivo e *staying power*.

        **Formato:** `.xlsx`
        """
    )

    if EXCEL_FILE.exists():
        st.download_button(
            label="Baixar calculadora offline (.xlsx)",
            data=EXCEL_FILE.read_bytes(),
            file_name="calculadora_offline.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    else:
        st.warning(
            "Arquivo da planilha ainda não encontrado. "
            "Adicione o arquivo em `offline_assets/calculadora_offline.xlsx`."
        )

with col2:
    st.subheader("Cartilha do usuário")

    st.markdown(
        """
        A cartilha apresenta orientações de uso da planilha, incluindo:

        - descrição dos campos de entrada;
        - sequência recomendada de preenchimento;
        - leitura dos resultados;
        - limitações da versão offline;
        - cuidados de interpretação.

        **Formato:** `.pdf`
        """
    )

    if PDF_FILE.exists():
        st.download_button(
            label="Baixar cartilha do usuário (.pdf)",
            data=PDF_FILE.read_bytes(),
            file_name="cartilha_calculadora_offline.pdf",
            mime="application/pdf",
        )
    else:
        st.warning(
            "Arquivo da cartilha ainda não encontrado. "
            "Adicione o arquivo em `offline_assets/cartilha_calculadora_offline.pdf`."
        )

st.markdown("---")

st.header("Limitações da versão offline")

st.markdown(
    """
    A calculadora offline é deliberadamente mais simples que o aplicativo
    online. Em especial, ela pode não incluir todos os recursos disponíveis
    no painel multidomínio, como:

    - matriz completa de admissibilidade entre domínios;
    - modulação cibernética Φ;
    - atualização dinâmica da seleção de alvos entre salvas;
    - visualizações interativas;
    - reprodução automática dos relatórios de validação e sensibilidade.

    Por isso, os resultados devem ser interpretados como apoio à análise e
    não como previsão operacional.
    """
)
