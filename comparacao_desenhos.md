# Qual desenho adotar no artigo? — comparação dos dois experimentos

Dois experimentos de composição de força foram executados sobre o **mesmo motor**,
o mesmo conjunto de 10 composições, o mesmo regime neutro e as mesmas três escalas
orçamentárias. Diferem em uma coisa: **como o orçamento dos dois lados se relaciona**.

| | **Experimento A** — paridade ±10% | **Experimento B** — paridade estrita |
|---|---|---|
| Orçamento | Vermelho fixo; Azul = ρ × Vermelho, ρ ∈ {0,9; 1,0; 1,1} | **ρ = 1 sempre** (mesmo orçamento dos dois lados) |
| Células por escala | 300 (10 × 10 × 3 ρ) | 100 (10 × 10) |
| Réplicas por célula | 5.000 | **20.000** |
| Batalhas totais (3 escalas) | 4,5 M | 6,0 M |
| IC95 típico por célula | ±0,07 a ±0,13 log-FER | **±0,016 a ±0,033 log-FER** |
| Erros-padrão no CSV | não | **sim** (`logfer_se`, `p_victory_se`) |
| Responde "orçamento vs design?" | **sim** | não (não há variação de orçamento) |
| Responde "que composição vence?" | sim, com ruído | **sim, com precisão** |
| Arquivos | `cross_composition_results.csv`, `cross10_*`, `cross20_*` | `parity5_*`, `parity10_*`, `parity20_*` |

---

## 1. Os dois concordam — isto não é uma escolha entre resultados conflitantes

Comparei os nove confrontos puros (3 pares × 3 escalas) entre os dois experimentos,
usando o subconjunto ρ=1 do Experimento A:

| Confronto | A (ρ=1, 5k reps) | B (20k reps) | diferença | dentro do IC? |
|---|---|---|---|---|
| 5×M L vs M | −0,607 | −0,537 ± 0,035 | +0,070 | sim |
| 5×M L vs H | +0,168 | +0,268 ± 0,046 | +0,100 | sim |
| 5×M M vs H | +0,782 | +0,790 ± 0,039 | +0,008 | sim |
| 10×M L vs M | +0,153 | +0,172 ± 0,020 | +0,020 | sim |
| 10×M L vs H | +1,613 | +1,606 ± 0,029 | −0,007 | sim |
| 10×M M vs H | +1,900 | +1,879 ± 0,038 | −0,021 | sim |
| 20×M L vs M | +0,536 | +0,536 ± 0,012 | −0,000 | sim |
| 20×M L vs H | +2,355 | +2,352 ± 0,017 | −0,003 | sim |
| 20×M M vs H | +2,448 | +2,493 ± 0,028 | +0,045 | sim |

**Todas as nove diferenças estão dentro do intervalo de confiança**; a maior é de
0,100 log-FER, justamente onde o Experimento A era menos preciso. A correlação de
Spearman entre os rankings de composição é **0,988 nas três escalas**.

**Consequência prática:** você não corre risco de "escolher o resultado errado".
Qualquer um dos dois sustenta as mesmas conclusões sobre composição de força. A
escolha é sobre **qual pergunta o artigo faz** e **quanta precisão você quer
reportar**.

---

## 2. O que cada um oferece que o outro não oferece

**Só o Experimento A responde:** *quanto vale um desvio orçamentário de ±10%
comparado com a escolha da composição?* Resultado: em 5×M os dois efeitos
empatam (0,20 vs 0,22); em 10×M e 20×M o orçamento passa a dominar (0,21 vs 0,15;
0,16 vs 0,11). É também o único que expõe a **granularidade de aquisição** —
o efeito não-monotônico de +10% de orçamento conforme ele completa ou não a compra
de um casco inteiro.

**Só o Experimento B oferece:**
- **Precisão 4× maior** e erros-padrão por célula no CSV — permite afirmar quais
  confrontos estão *decididos* (96–97% dos 90 pares a 95% de confiança) e marcar
  os indecididos nas figuras.
- **Atribuição causal limpa:** com orçamento idêntico, nenhuma diferença de
  resultado pode ser atribuída a dinheiro. É o desenho que corresponde
  literalmente à pergunta "qual composição é melhor?".
- **O cruzamento quantidade–equilíbrio medido, não estimado.** No Experimento A a
  inversão L-vs-M entre 5×M e 10×M (−0,61 → +0,15) tinha IC da ordem de ±0,35 no
  piloto — era sugestiva. No B ela é −0,54 ± 0,03 → +0,17 ± 0,02, a dezenas de
  erros-padrão de zero.
- **Um achado que só apareceu com precisão:** em 10×M, **L vence M no confronto
  direto** (+0,17 ± 0,02) mas **M tem média maior contra o campo inteiro** (0,163
  vs 0,153). Confronto direto e desempenho médio são perguntas diferentes e a
  resposta diverge — no Experimento A isso estava submerso no ruído.

---

## 3. Recomendação

**Adote o Experimento B (paridade estrita) como resultado principal e mantenha o
A como excursão.** Não são alternativas concorrentes: são a análise principal e
uma análise de sensibilidade que responde uma pergunta adicional.

Razões, em ordem de peso:

1. **A pergunta do artigo é sobre composição.** Paridade estrita isola exatamente
   essa variável. Todo revisor entende "mesmo orçamento, quem vence?" sem
   ressalvas; "orçamento variando ±10% agregado em médias" exige explicar por que
   a média sobre ρ é uma quantidade interpretável.
2. **Precisão reportável.** Com erros-padrão por célula você pode afirmar
   significância confronto a confronto — e, depois da revisão do pacote do seu
   coautor, isso deixou de ser um detalhe: a fragilidade da FIG5 do artigo
   principal foi exatamente uma estatística sem intervalo. Um segundo resultado
   com o mesmo problema seria um alvo fácil na revisão.
3. **Custa quase nada manter o A.** Ele já está rodado; entra como uma seção de
   robustez de meia página que acrescenta a comparação orçamento-vs-design e a
   granularidade de aquisição — dois pontos genuinamente interessantes que o B
   não pode fazer.

**Estrutura sugerida:** resultados principais a partir do B (matrizes das três
escalas, 3×3 puro com ICs, curva de escala); uma subseção de robustez a partir do
A com o trade-off orçamento-vs-design e a granularidade; e a observação de que os
dois desenhos concordam (Spearman 0,988) como evidência de robustez do próprio
achado.

**Se precisar escolher apenas um** (limite de páginas): fique com o **B**. As
conclusões sobre composição são idênticas, a precisão é muito melhor, e a única
perda é o trade-off orçamento-vs-design — que pode ser mencionado em uma frase
como trabalho relacionado, remetendo ao pacote suplementar.

---

## 4. Conclusões (idênticas nos dois desenhos)

1. **Não há ciclo pedra-papel-tesoura** entre frotas puras em nenhuma escala; a
   ordenação é estrita.
2. **A hierarquia depende da escala do engajamento.** M é a melhor frota pura em
   5×M; L assume em 10×M e 20×M. O ponto de inversão está entre 5×M e 10×M.
3. **A frota pesada colapsa monotonicamente com a escala:** L-vs-H sobe de +0,27
   para +2,35 do menor ao maior orçamento. Em 10×M e 20×M o H puro é a composição
   inimiga mais frágil da matriz. Poucos cascos não escalam contra salvas grandes
   — saturação da defesa agrupada com concentração do dano recebido.
4. **A heterogeneidade não tem prêmio** em paridade e regime neutro: nenhuma
   composição mista é a melhor resposta contra nenhuma composição, em nenhuma
   escala. (Consistente com a checagem de alocação do experimento principal, onde
   o sinal do R6 inverte sob teto orçamentário rígido.)
5. **A composição decide cada vez menos conforme a escala cresce** — a amplitude
   entre a melhor e a pior composição cai para ~metade entre 5×M e 20×M.

---

## 5. Advertências que valem para os dois

- **Regime único.** Os fatores de processo estão fixos no centro neutro, sem
  vantagem de fusão. A interação composição × Δσ é a extensão natural (e a ponte
  para o artigo LAFusion principal).
- **Empates crescem com a escala.** Frotas maiores terminam mais vezes sem
  decisão, o que comprime P(vitória) em termos absolutos. Comparações entre
  escalas devem usar log-FER; nunca níveis absolutos de P(vitória).
- **Alocação inteira sob teto orçamentário** (`best_integer_fleet`) é usada em
  ambos: `round()` por plataforma deixaria as misturas estourarem o orçamento em
  até +20% e decidiria os confrontos por arredondamento. Esta é uma contribuição
  metodológica do trabalho e merece sua própria subseção.
