import streamlit as st

st.title("Equação de Salva Multidomínio")

st.markdown("""
## Apresentação do projeto

Este aplicativo implementa uma ferramenta interativa para exploração da Equação de Salva Multidomínio aplicada a cenários navais.

### Abas disponíveis

**Hughes 1995**  
Modelo clássico homogêneo de combate por salvas.

**Cenário de Modelo Multidomínio**  
Simulação do cenário Bacia de Campos, com forças de superfície, submarinas, aéreas, costeiras e cibernéticas.

**Cyber**  
Análise dos efeitos da modulação cibernética Φ sobre os parâmetros do modelo.

**Validação**  
Relatório de validação numérica e análise de sensibilidade.

**Sobre**  
Informações conceituais, autoria e referências do projeto.
""")
