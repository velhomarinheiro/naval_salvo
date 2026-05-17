"""
Página "Sobre" do aplicativo.

Reúne em um único lugar: descrição conceitual do modelo, autoria,
contexto institucional, declaração de uso de ferramentas de IA como
apoio, e referências bibliográficas.
"""

from __future__ import annotations

import streamlit as st


# ---------------------------------------------------------------------------
# Cabeçalho
# ---------------------------------------------------------------------------

st.title("Sobre")
st.caption(
    "Descrição conceitual, autoria, declaração de uso de IA e "
    "referências bibliográficas"
)

st.markdown("---")


# ---------------------------------------------------------------------------
# Descrição conceitual
# ---------------------------------------------------------------------------

st.header("Descrição conceitual do modelo")

st.markdown(
    """
    A *Equação de Salva Multidomínio* implementada neste aplicativo é
    uma extensão das equações clássicas de salva navais (Hughes 1995;
    Johns, Pilnick e Hughes 2001) para um ambiente operacional em que
    a interação ocorre simultaneamente em cinco domínios:

    - **superfície** (S) — navios de combate, USVs, plataformas de
      produção e helicópteros embarcados, estes últimos absorvidos
      como capacidade da plataforma-mãe;
    - **subsuperfície** (U) — submarinos, UUVs e minas submarinas;
    - **ar** (A) — aeronaves tripuladas e não-tripuladas;
    - **costa** (C) — baterias costeiras de mísseis anti-navio,
      artilharia costeira e minas costeiras;
    - **cibernético-eletromagnético** (X) — efeitos cibernéticos e
      de guerra eletrônica, decompostos em quatro sub-tipos
      funcionais (C2, sensores, armas e logística).

    A interação entre domínios é mediada por uma **matriz de
    admissibilidade** 5×5, que codifica em três níveis quais pares
    atacante-defensor são doutrinariamente possíveis: primária (1),
    marginal calibrável (χ ∈ [0, 1]) e estruturalmente nula (0). Por
    exemplo, um torpedo lançado por um submarino não atinge uma
    aeronave em voo (interação nula), enquanto um helicóptero
    operando sonar mergulhador contra um submarino é uma interação
    marginal cuja efetividade depende da calibração de χ.

    O domínio cibernético recebe um tratamento distinto dos demais.
    Em vez de atuar predominantemente por atrição direta — o que
    seria pouco realista —, ele atua como **modulador multiplicativo
    Φ** dos coeficientes ofensivos e defensivos dos pares cinéticos.
    Quando uma força tem vantagem cibernética sobre a oponente, seus
    sensores, sistemas de armas, redes de comando e logística operam
    em maior eficiência relativa, e os coeficientes da força oponente
    são correspondentemente degradados. A função Φ é uma sigmoide
    parametrizada pela razão de força cibernética entre os dois lados
    em cada canal.

    Os submarinos são tratados como **imunes ao domínio cibernético**
    por escolha doutrinária explícita do modelo (isolamento por
    profundidade e silêncio de emissão). Esta escolha é discutível,
    e a sua flexibilização é um ponto natural para evolução futura
    do aplicativo.

    O regime de combate é **pulsado simultâneo**: a cada salva, os
    dois lados calculam suas perdas com base no estado pré-salva e
    aplicam o resultado em conjunto. Esta é a convenção adotada por
    Hughes (1995), Johns-Pilnick-Hughes (2001) e Armstrong (2005). A
    variante de trocas sequenciais (Armstrong 2014) não está
    implementada nesta versão.
    """
)


# ---------------------------------------------------------------------------
# Autoria
# ---------------------------------------------------------------------------

st.markdown("---")
st.header("Autoria")

col_info, col_contact = st.columns([2, 1])

with col_info:
    st.markdown(
        """
        **Alberto Ferreira Filho**

        Oficial Superior da Marinha do Brasil. Mestre em Defesa e
        Estudos Estratégicos pelo *U.S. Naval War College*. Doutorando
        em Administração Pública na Escola Brasileira de Administração
        Pública e de Empresas (EBAPE) da Fundação Getulio Vargas.

        Este aplicativo é um desdobramento aplicado de seus interesses
        de pesquisa nas interseções entre pesquisa operacional, estudos
        estratégicos e modelagem quantitativa de combate naval. A
        ferramenta é oferecida como contribuição ao debate acadêmico
        e doutrinário sobre o emprego de modelos de salvas em
        análises navais multidomínio.
        """
    )

with col_contact:
    st.markdown(
        """
        **Contato**

        ✉️ ferreirafilhoalberto@gmail.com

        🔗 [LinkedIn](https://www.linkedin.com/in/alberto-ferreira-filho-0b21a71a3)
        """
    )


# ---------------------------------------------------------------------------
# Declaração de uso de IA
# ---------------------------------------------------------------------------

st.markdown("---")
st.header("Uso de ferramentas de inteligência artificial")

st.markdown(
    """
    No desenvolvimento deste aplicativo foram empregadas ferramentas
    de inteligência artificial generativa — em particular **Claude**
    (Anthropic) e **ChatGPT** (OpenAI) — como **apoio** ao trabalho
    do autor nas seguintes atividades:

    - revisão e organização das **formulações matemáticas** do modelo,
      incluindo a sistematização da notação, a verificação de
      consistência entre as várias equações da família, e a
      conferência de passagens algébricas;
    - **redação e revisão dos códigos em Python** que compõem o
      pacote `naval_salvo` e as páginas Streamlit deste aplicativo,
      com ênfase em estruturação do código, documentação, validação
      numérica e correção de inconsistências.

    O emprego dessas ferramentas seguiu uma lógica de **apoio, não de
    delegação**: as escolhas de modelagem, a seleção das referências,
    a interpretação dos resultados e a responsabilidade sobre o
    conteúdo final são integralmente do autor. As ferramentas foram
    usadas como assistentes de produtividade, à maneira de um
    revisor técnico ou de um par disponível para discussão, sem
    substituir o julgamento humano sobre as decisões substantivas
    do projeto.

    Esta declaração é feita em conformidade com as boas práticas
    contemporâneas de transparência sobre o uso de IA em produção
    acadêmica e técnica.
    """
)


# ---------------------------------------------------------------------------
# Referências
# ---------------------------------------------------------------------------

st.markdown("---")
st.header("Referências bibliográficas")

st.markdown(
    """
    **Família clássica das equações de salva**

    - Hughes, W. P. (1995). A Salvo Model of Warships in Missile
      Combat Used to Evaluate Their Staying Power. *Naval Research
      Logistics*, 42(2), 267–289.
    - Johns, M. E.; Pilnick, S. E.; Hughes, W. P. (2001).
      *Heterogeneous Salvo Model for the Navy After Next* (NPS-IJWA-
      01-010). Naval Postgraduate School.
    - Armstrong, M. J. (2005). A Stochastic Salvo Model for Naval
      Surface Combat. *Operations Research*, 53(5), 830–841.
    - Armstrong, M. J. (2013). The Salvo Combat Model with Area Fire.
      *Naval Research Logistics*, 60(8), 652–660.
    - Armstrong, M. J. (2014). The Salvo Combat Model with a
      Sequential Exchange of Fire. *Journal of the Operational
      Research Society*, 65(10), 1593–1601.

    **Realocação de fogo e modelos de Lanchester estendidos**

    - MacKay, N. J. (2009). Lanchester combat models. *Journal of the
      Operational Research Society*, 60, 1421–1427.
    - Hausken, K.; Moxnes, J. F. (2026). A multi-equipment
      Lanchester model with proportional reallocation. *Annals of
      Operations Research*, 357, 1003–1019.

    **Discussão metodológica e aplicações**

    - Lucas, T. W.; McGunnigle, J. E. (2003). When is model
      complexity too much? Illustrating the benefits of simple models
      with Hughes' salvo equations. *Naval Research Logistics*, 50(3),
      197–217.

    **Aplicações e calibrações auxiliares utilizadas no projeto**

    - Beall, T. R. (1990). *Naval Gunnery and Naval Salvo Combat
      Data*. Dados originais empregados nas reproduções históricas
      (Coronel, Coral Sea, Savo Island) implementadas na aba de
      validação.
    - Christiansen, K. P. (2008). *Fitting Salvo Equations to Naval
      Combat Data*. Naval Postgraduate School.
    - Casola, K. (2017). *Optimisation of Naval Gun Firing Patterns
      for Engagement of Manoeuvring Surface Targets*. Naval
      Postgraduate School.
    - Vasankari, L. (2024). *Littoral Naval Warfare with Multi-Agent
      Reinforcement Learning*. Dissertação de mestrado.

    **Trabalho próprio do autor**

    - Araujo, T. M. P. de C.; Ferreira Filho, A.; Santos, M. dos;
      Gomes, C. F. S.; Fróes, B. E. (2025). Apresentação de um
      aplicativo web para auxílio no cálculo de equações de salva.
      *Anais do LVII Simpósio Brasileiro de Pesquisa Operacional*,
      Gramado, RS.
    """
)


# ---------------------------------------------------------------------------
# Rodapé
# ---------------------------------------------------------------------------

st.markdown("---")
st.caption(
    "Versão experimental — sugestões, críticas e propostas de "
    "colaboração são bem-vindas pelos canais de contato listados acima."
)
