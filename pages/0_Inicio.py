"""
Página inicial do aplicativo "Equação de Salva Multidomínio".

Esta é a landing page do app Streamlit. Ela descreve o projeto,
suas finalidades e o que o usuário encontrará em cada uma das demais
páginas acessíveis pelo menu lateral.
"""

from __future__ import annotations

import streamlit as st


# ---------------------------------------------------------------------------
# Cabeçalho
# ---------------------------------------------------------------------------

st.title("⚓ Equação de Salva Multidomínio - v. 1.0")
st.caption(
    "Ferramenta interativa de exploração para análises navais "
    "multidomínio — versão experimental - publicada em 17/05/2026"
)

st.markdown("---")


# ---------------------------------------------------------------------------
# Apresentação do projeto
# ---------------------------------------------------------------------------

st.header("Apresentação do projeto")

st.markdown(
    """
    Este aplicativo é uma **ferramenta interativa** para a exploração
    da *Equação de Salva Multidomínio* aplicada a cenários navais. Ele
    permite ao usuário variar parâmetros de composição de força,
    coeficientes ofensivos e defensivos, políticas de seleção de alvos
    e efeitos cibernéticos, e observar as trajetórias resultantes ao
    longo de sucessivas salvas.

    A versão atual é **experimental**. Os modelos implementados ainda
    estão em fase de calibração, e os valores numéricos pré-preenchidos
    nas páginas seguem ordens de grandeza razoáveis, mas não constituem
    parâmetros validados para situações reais. A ferramenta deve ser
    entendida como **um instrumento de apoio à reflexão**, não como
    um previsor de resultados de combate.

    O propósito principal do aplicativo é servir como auxílio em:

    - análises de **estratégia e tática naval**, especialmente
      explorando o efeito relativo de diferentes parâmetros sobre a
      sobrevivência de ativos e a duração do engajamento;
    - **jogos de guerra** e exercícios doutrinários, oferecendo uma
      base quantitativa para discutir trade-offs de composição de força;
    - estudos de **planejamento de força**, examinando, por exemplo,
      o impacto marginal de adicionar uma plataforma, ampliar a
      escolta ou incorporar capacidades cibernéticas.

    O aplicativo apoia-se em uma tradição consolidada de modelagem em
    pesquisa operacional militar. As principais referências que
    fundamentam os modelos implementados são:

    - **Hughes (1995)** — formulação original da equação de salva
      homogênea.
    - **Johns, Pilnick e Hughes (2001)** — generalização heterogênea
      do modelo de Hughes, com múltiplos tipos de unidades por força.
    - **Armstrong (2005, 2013, 2014)** — extensões do modelo para
      letalidade, fogo de área e trocas sequenciais de salvas.
    - **MacKay (2009)** e **Hausken & Moxnes (2026)** — modelos com
      realocação proporcional de fogo e taxas de morte variáveis.
    - **Lucas e McGunnigle (2003)** — discussão sobre a utilidade e os
      limites de modelos simples de combate naval.

    O aplicativo também incorpora um capítulo próprio: **a Equação de
    Salva Multidomínio**, que estende as formulações clássicas para
    cinco domínios — superfície, subsuperfície, ar, costa e cibernético —
    com uma matriz de admissibilidade entre domínios e um modulador
    cibernético multiplicativo. Esta extensão está descrita em maior
    detalhe na aba **Sobre**.
    """
)

st.info(
    "💡 **Sobre a interpretação dos resultados.** Cada simulação "
    "produz uma trajetória deterministicamente exata para os "
    "parâmetros informados, mas a sensibilidade do resultado a esses "
    "parâmetros é elevada. O valor analítico da ferramenta está em "
    "comparar configurações entre si, não em fixar previsões "
    "absolutas sobre um confronto específico."
)


# ---------------------------------------------------------------------------
# Páginas disponíveis
# ---------------------------------------------------------------------------

st.markdown("---")
st.header("Páginas disponíveis")

st.markdown(
    """
    O menu lateral dá acesso às seguintes páginas. Cada uma é
    independente e pode ser usada isoladamente.
    """
)


st.subheader("Hughes 1995 — modelo clássico homogêneo")

st.markdown(
    """
    Nesta página o usuário trabalha com o **modelo clássico de salvas
    homogêneo** proposto por Wayne P. Hughes Jr. em 1995. Cada força é
    representada por uma única classe de unidade, caracterizada por sua
    quantidade inicial, poder ofensivo, poder defensivo e *staying power*.

    A equação calcula, para cada salva, as perdas de cada lado em função
    do diferencial entre o fogo recebido e o fogo interceptado. É o modelo
    mais enxuto da família e serve como introdução didática ao raciocínio
    das equações de salva e como caso limite de validação dos modelos mais
    complexos.
    """
)


st.subheader("Cenário de Modelo Multidomínio")

st.markdown(
    """
    Esta página implementa um **cenário multidomínio configurável**
    inspirado na defesa de uma área de plataformas de produção de petróleo
    e gás, à semelhança da Bacia de Campos. O cenário representa um
    problema operacional para uma marinha de águas costeiras: proteger
    ativos de elevado valor econômico e estratégico contra uma força hostil
    atuando em múltiplos domínios.

    O usuário pode configurar unidades de superfície, submarinos, bateria
    costeira, FPSOs, aviação de ataque, capacidades cibernéticas, matriz
    de engajamento e políticas de seleção de alvos.

    Internamente, o modelo aplica uma **matriz de admissibilidade entre
    domínios**, com valores em três níveis: primária, marginal calibrável
    e nula. Submarinos são tratados como imunes ao domínio cibernético,
    por escolha doutrinária explícita do modelo.
    """
)


st.subheader("Cyber — análise da modulação cibernética")

st.markdown(
    """
    Esta página é dedicada à **análise do efeito da modulação cibernética
    Φ** sobre os parâmetros do modelo. Em vez de tratar o domínio
    cibernético apenas como fonte de atrição direta, o modelo o representa
    como um *modulador multiplicativo* dos coeficientes cinéticos
    ofensivos e defensivos.

    A intuição central é que uma vantagem cibernética pode degradar a
    capacidade oponente de mirar, lançar e defender, sem necessariamente
    destruir unidades físicas.
    """
)


st.subheader("Salva Estocástica — troca Monte Carlo com ordem de engajamento")

st.markdown(
    """
    Esta página apresenta a **versão estocástica** do modelo multidomínio,
    seguindo Armstrong (2005, 2014). Em vez de uma única trajetória exata,
    cada engajamento é simulado muitas vezes: os fogos ofensivos e as
    interceptações são sorteados de **distribuições binomiais** e o dano
    por míssil não interceptado de uma **distribuição normal**. O resultado
    é uma *distribuição* de desfechos — probabilidades de vitória para cada
    lado e a dispersão de unidades sobreviventes — em vez de um único valor
    determinístico.

    A troca de fogo pode ser **simultânea** ou **sequencial** (Azul atira
    primeiro / Vermelho atira primeiro). No modo sequencial, o fogo de
    retorno é executado apenas pelos **sobreviventes** da primeira salva, o
    que captura o valor da iniciativa: atirar primeiro é, em média, ao menos
    tão bom quanto atirar simultaneamente, que por sua vez é ao menos tão
    bom quanto atirar em segundo.

    Ambas as forças podem ser compostas a partir da **mesma paleta de
    plataformas** do cenário multidomínio — duas classes de superfície, um
    submarino, aviação de ataque, uma bateria costeira e um ativo de valor
    tipo FPSO — e as convenções canônicas são preservadas, incluindo a
    modulação ciber Φ e a imunidade ciber dos submarinos.
    """
)


st.subheader("Validação — verificação numérica e sensibilidade")

st.markdown(
    """
    A página de validação reúne o **relatório de validação numérica** do
    engine implementado. Mostra a reprodução, com precisão de máquina, de
    dois casos canônicos da literatura: o engajamento homogêneo de Hughes
    (1995) e a Batalha de Coronel, a partir do exemplo trabalhado de
    Johns, Pilnick e Hughes (2001).

    Além das validações, a página oferece **análises de sensibilidade**
    sobre os principais parâmetros do modelo multidomínio.
    """
)


st.subheader("Sobre — informações conceituais, autoria e referências")

st.markdown(
    """
    A página **Sobre** reúne a descrição conceitual da Equação de Salva
    Multidomínio, as escolhas de modelagem, as informações de autoria e
    contexto institucional do projeto, além das referências bibliográficas
    completas.
    """
)


# ---------------------------------------------------------------------------
# Rodapé
# ---------------------------------------------------------------------------

st.markdown("---")
st.caption(
    "Versão experimental. Os modelos e parâmetros aqui implementados "
    "estão em fase de calibração e não substituem análises operacionais "
    "formais. Use as páginas pelo menu lateral."
)
