# Simulação de Configuração de Forças (Cross-Composition)
## Documento explicativo para redação de artigo

**Base de dados:** `cross_composition_results.csv` (300 células × 44 colunas)
**Figuras:** `fig6_cross_matrix.png`, `fig7_budget_sensitivity.png`
**Código:** `cross_composition.py` (experimento) + `salvo_mds.py` (motor, v1.0.0)
**Reprodução em um comando:** `make cross` (ou `python3 cross_composition.py --reps 5000 --jobs 4`)

---

## 1. Pergunta de pesquisa

Dado um orçamento aproximadamente igual dos dois lados (paridade ±10%), **como a
composição da força — a mistura de plataformas leves (L), médias (M) e pesadas
(H) — determina o resultado de um engajamento naval de salvas de mísseis, quando
AMBOS os lados escolhem sua composição?**

Perguntas derivadas, todas respondidas pela base de dados:

1. Existe uma composição **dominante** em paridade, ou uma estrutura cíclica
   tipo pedra-papel-tesoura entre frotas puras (L vence H, H vence M, M vence L)?
2. Qual é a **melhor resposta** de Azul a cada composição de Vermelho?
3. Quanto vale uma **vantagem orçamentária de ±10%** em comparação com a
   escolha da composição?
4. Como a **granularidade de aquisição** (frotas inteiras, cascos indivisíveis)
   distorce a conversão de orçamento em poder de combate?

### Relação com o experimento LAFusion principal

O experimento principal (data farming, `farm.py`) fixa Vermelho em 5×M e varre
fatores de processo (σ, τ, ρ, p_o, p_d, sd) com o design NOAB. Esta simulação
**libera a hipótese de Vermelho fixo**: os dois lados são construídos por
frações de orçamento sobre os mesmos três arquétipos, e todas as combinações se
enfrentam. Em contrapartida, os fatores de processo ficam **fixos num centro
neutro** (sem vantagem informacional), para que a composição — e somente ela —
dirija o resultado. Os dois experimentos são complementares: um mapeia o efeito
do regime com design fixo de um lado; o outro mapeia o espaço de design × design
num regime fixo.

---

## 2. O motor de simulação (`salvo_mds.py`, versão 1.0.0)

Modelo de salvas estocástico, **multi-salvo**, com resolução **por casco**
(per-hull), estendendo o modelo de salvas de Hughes/Armstrong:

- **Estado por casco:** tipo k, dano acumulado ∈ [0,1) (morte quando ≥1),
  magazine (salvas restantes), vivo/morto. Dano médio por míssil que atinge:
  u_k = 1/w_k (w = staying power).
- **Por rodada:** cada lado lança `N_off = round(σ·Σ o_k)` mísseis (atiradores
  vivos com magazine>0, magazine decrementado); acertos ~ Binomial(N_off, p_o);
  interceptações agrupadas no nível da força (`N_def = round(τ·Σ d_k)`,
  ~ Binomial(N_def, p_d)); vazadores = max(acertos − interceptações, 0).
- **Alocação uniforme com reposição:** cada vazador atinge um casco vivo
  aleatório; dano ~ Normal(u_k, sd) truncado em 0; dano além da saturação do
  casco é contabilizado como **overkill** (desperdício).
- **Ordem de engajamento:** surpresa só na salva de abertura (blue_first /
  red_first); demais rodadas simultâneas. Nesta simulação: **simultâneo sempre**.
- **Término:** aniquilação, poder de combate ≤ θ, magazines vazios, ou T_max.

**Validação do motor** (suíte `test_validation.py`, 7 testes automatizados):
- Reproduz o exemplo 6-on-3 de **Armstrong (2011)** — perda agregada média
  0,678 vs 0,679 publicado (erro 0,1%), dp 0,464 vs 0,467 — a camada de
  agregação/vazadores é fiel ao modelo de salvas estocástico verificado.
- **Simetria de espelho**: frota idêntica dos dois lados → perdas
  estatisticamente iguais, log-FER ≈ 0.
- Swing de concentração-vs-dispersão de Tiah (2007) reproduzido em sinal.

### Definições fixadas (documentadas antes da rodada)

| Definição | Valor | Observação |
|---|---|---|
| Poder de combate efetivo | ofensa residual: Σ o_k vivos / Σ o_k iniciais | `CP_WEIGHTS=(1,0,0)`; defesa residual reportada à parte |
| θ (limiar de inefetividade) | 0,30 | vitória = adversário ≤ θ **e** próprio > θ |
| T_max | 6 salvas | |
| Alocação | uniforme com reposição | excursão sem-reposição disponível no motor |

---

## 3. Arquétipos e função de custo

Framework quantidade–qualidade (cascos maiores carregam mais de tudo, mas a
concentração de capacidade é precificada superlinearmente):

| Arquétipo | Símbolo | Ofensa o | Defesa d | Staying w | Magazine | Custo unitário (γ=1,35) |
|---|---|---|---|---|---|---|
| Lancha/corveta leve | **L** | 2 | 1 | 1 | 2 | 3,6 |
| Fragata | **M** | 3 | 2 | 2 | 3 | 7,0 |
| Contratorpedeiro/cruzador | **H** | 6 | 5 | 4 | 4 | 17,5 |

Custo: `c = 0,4·(o+d+w)^1,35 + 0,5·mag`. Orçamento de referência = valor da
força Vermelha padrão do experimento principal: **35,2** (= 5×M). Throughput
ofensivo por unidade de custo: L (0,56) > M (0,43) > H (0,34) — quantidade
compra saturação de ataque; qualidade compra defesa concentrada, magazines
profundos e menos pontos de mira.

---

## 4. Alocação orçamentária inteira — o problema e a solução

**Problema (achado metodológico em si).** A rotina do experimento principal
(`force_from_shares`) usa `round()` por plataforma: `n_k = round(orçamento·s_k/c_k)`.
Com frotas de 2–10 cascos, isso permite que misturas **estourem o orçamento em
até +20%** — o centroide (⅓,⅓,⅓) com orçamento 35,2 arredonda para 3L+2M+1H =
custo 42,3. Num experimento com Vermelho fixo e ρ contínuo, isso é ruído
absorvível; num **confronto direto em paridade**, o brinde de arredondamento —
não a tática — decidiria o resultado. Testes-piloto desta simulação com `round()`
mostraram o centroide "dominando tudo" por esse artefato.

**Solução (`best_integer_fleet`).** O espaço de frotas inteiras é minúsculo
(≤ ~150 combinações), então a alocação é resolvida **por enumeração exata**:
entre todas as frotas (n_L, n_M, n_H) com **custo ≤ orçamento (teto rígido,
nunca estoura)**, escolhe-se a que minimiza a distância L2 entre as frações de
valor realizadas (gasto_k/orçamento) e as frações-alvo; empates resolvidos por
maior gasto. Orçamento não gasto conta como déficit de fração, maximizando a
utilização sujeita à fidelidade da mistura. Determinístico, ótimo e transparente.

**Frotas realizadas** (composição : frota : utilização do orçamento):

| Mistura | ρ=0,9 | ρ=1,0 | ρ=1,1 |
|---|---|---|---|
| L | 8L (91%) | 9L (92%) | 10L (93%) |
| M | 4M (89%) | **5M (100%)** | 5M (91%) |
| H | **1H (55%)** | 2H (99%) | 2H (90%) |
| LM | 4L+2M (90%) | 5L+2M (91%) | 4L+3M (92%) |
| LH | 3L+1H (89%) | 4L+1H (91%) | 5L+1H (92%) |
| MH | 2M+1H (100%) | 2M+1H (90%) | 3M+1H (100%) |
| C (⅓,⅓,⅓) | 1L+1M+1H (89%) | 2L+1M+1H (90%) | 3L+1M+1H (91%) |
| L+ (⅔,⅙,⅙) | 6L+1M (90%) | 7L+1M (92%) | 7L+1M (83%) |
| M+ (⅙,⅔,⅙) | 1L+3M (78%) | 2L+3M (80%) | 2L+4M (91%) |
| H+ (⅙,⅙,⅔) | 1L+1M+1H (89%) | 2L+1M+1H (90%) | 2L+1M+1H (82%) |

Utilização média 89%; mínimo 55% (H puro com ρ=0,9: um segundo H não cabe em
31,7). A tabela é dado de pesquisa: mostra a **quantização de aquisição** que
qualquer orçamento discreto impõe. Note que M puro em ρ=1,0 realiza exatamente
os 5×M do experimento principal (âncora preservada).

---

## 5. Desenho experimental

- **Composições (10, simplex-centroid):** 3 vértices puros (L, M, H), 3 pontos
  médios de aresta (LM, LH, MH), centroide (C), 3 pontos interiores (L+, M+, H+
  = ⅔ numa plataforma, ⅙ nas demais). **Mesmo conjunto para Azul e Vermelho.**
- **Cruzamento completo:** 10 × 10 = 100 confrontos de composição.
- **Orçamento:** Vermelho = 35,2 sempre; Azul = ρ·35,2 com **ρ ∈ {0,9; 1,0; 1,1}**
  (paridade ±10%). Total 300 células.
- **Regime neutro fixo** (isola o efeito da composição): simultâneo,
  σ_B = σ_R = 0,7, τ = 0,75 (ambos), p_o = 0,75, p_d = 0,75, sd = 0,15,
  T_max = 6, θ = 0,30. Sem vantagem informacional de nenhum lado.
- **Replicações:** 5.000 batalhas Monte Carlo por célula → **1.500.000 batalhas**.
- **Sementes:** master seed 20260731; sub-semente por célula via
  `numpy SeedSequence.spawn` (registrada na coluna `seed` — cada célula é
  regenerável individualmente).

---

## 6. Dicionário de dados (`cross_composition_results.csv`, 300 linhas × 44 colunas)

### Identificação da célula
| Coluna | Significado |
|---|---|
| `blue_mix`, `red_mix` | rótulo da composição (L, M, H, LM, LH, MH, C, L+, M+, H+) |
| `blue_i`, `red_i` | índice da composição (0–9, na ordem acima) |
| `rho` | razão orçamentária Azul/Vermelho (0,9 / 1,0 / 1,1) |
| `s_L_b`, `s_M_b`, `s_H_b` | frações-alvo de orçamento de Azul |
| `s_L_r`, `s_M_r`, `s_H_r` | frações-alvo de orçamento de Vermelho |
| `blue_hulls`, `red_hulls` | número total de cascos da frota realizada |
| `blue_spend`, `red_spend` | custo efetivamente gasto pela alocação inteira |
| `blue_util`, `red_util` | utilização do orçamento (gasto/orçamento) |
| `seed` | sub-semente Monte Carlo da célula |

### Respostas (média e desvio-padrão sobre as 5.000 réplicas: sufixos `_mean`/`_sd`)
| Coluna | Significado |
|---|---|
| `blue_loss`, `red_loss` | perda de poder de combate (1 − ofensa residual/inicial) ∈ [0,1] |
| `p_victory` | P(vitória de Azul): Vermelho ≤ θ **e** Azul > θ no término (sem `_sd`) |
| `logfer` | **log-FER** (razão de troca logarítmica; >0 favorece Azul; espelho → 0). Métrica R4 recomendada |
| `fer_geom` | FER geométrico = exp(média do log-FER) (sem `_sd`) |
| `fer` | ⚠️ razão bruta por réplica — **média-de-razões enviesada; NÃO usar** (mantida só por completude) |
| `salvos` | salvas até a decisão (tempo de batalha) |
| `overkill_frac` | fração de dano desperdiçado (Azul→Vermelho): saturação/quantização |
| `blue_ships_lost`, `red_ships_lost` | cascos perdidos (inteiros) |
| `blue_damage`, `red_damage` | dano agregado contínuo recebido, em navios (comparável a Armstrong) |
| `blue_def_residual`, `red_def_residual` | fração da capacidade defensiva sobrevivente (perfil de degradação) |

---

## 7. Verificações internas (validade da rodada)

1. **Diagonal-espelho:** nas 10 células com mesma composição e ρ=1, |log-FER|
   máximo = **0,053** (esperado ≈ 0) — o motor não favorece nenhum lado.
2. **Antissimetria:** máx |logfer(i,j) + logfer(j,i)| = **0,184** em ρ=1 —
   compatível com ruído Monte Carlo puro (5.000 reps).

---

## 8. Resultados principais (com números)

### 8.1 Não há ciclo pedra-papel-tesoura; M domina entre as puras
3×3 puro em paridade (log-FER, Azul vs Vermelho): M vence L (+0,57), M vence H
(+0,78), L vence H (+0,17). Ordem estrita: **M > L > H**. A plataforma
balanceada — exatamente o Vermelho padrão do experimento principal — é a
composição pura mais forte no regime neutro, o que **justifica retroativamente
a escolha de 5×M como benchmark** do experimento principal.

### 8.2 Ranking completo e "dureza" como Vermelho (ρ=1)
- Como Azul (média de P(vitória) contra todos os Vermelhos):
  M=0,26 > L=0,24 > L+=0,16 > H=0,14 > LM=0,13 > LH=0,08 > H+=0,07 > C=0,07 > MH=0,05 > M+=0,04.
- Como Vermelho (média de P(vitória de Azul) contra ele; menor = mais duro):
  M=0,03 < L=0,06 < LM=0,07 < H=0,09 < L+=0,10 < LH=0,10 < C=0,13 < H+=0,13 < MH=0,19 < M+=0,34.
- M é simultaneamente o melhor atacante médio e o defensor mais duro.

### 8.3 Melhor resposta: M contra quase tudo; L contra M
Contra 8 das 10 composições de Vermelho a melhor resposta é M. Contra o próprio
M (e contra MH), a melhor resposta muda para **L** — dispersão: muitos cascos
diluem a alocação uniforme do oponente forte e saturam a defesa agrupada dele.
P(vitória) da melhor resposta contra Red=M é só 0,08: em paridade e regime
neutro, o confronto contra o benchmark é quase indecidível — como deveria ser.

### 8.4 Orçamento ±10% ≈ escolha de composição
Média de P(vitória): 0,03 (ρ=0,9) → 0,12 (ρ=1,0) → 0,23 (ρ=1,1); efeito do
orçamento ≈ 0,20. A escolha da composição em paridade fixa abrange 0,22
(M=0,26 vs M+=0,04). **Errar o design custa o mesmo que ceder 10% de orçamento.**

### 8.5 Orçamento só vira combate quando fecha um casco (granularidade)
O ganho de +10% é fortemente não-monotônico por composição (fig7): dispara onde
os 10% completam a compra de um casco (MH passa a 3M+1H, 100% de utilização;
M+ salta de 80% para 91%) e é nulo ou negativo onde piora o encaixe inteiro
(H+ em ρ=1,1 utiliza só 82%; H puro em ρ=0,9 desaba para 1 casco, 55%).
Fenômeno de aquisição que modelos contínuos/agregados não capturam — argumento
central para a resolução per-unit.

---

## 9. Figuras

- **fig6_cross_matrix.png** — matriz 10×10 em paridade. Cor = log-FER
  (divergente azul↔vermelho, ponto médio neutro em 0 = espelho; segura para
  daltonismo). Texto na célula = P(vitória de Azul). Contorno preto = melhor
  resposta de Azul por coluna (por Vermelho). Diagonal pontilhada = espelhos.
  A quase-neutralidade visível da diagonal é a verificação de simetria.
- **fig7_budget_sensitivity.png** — P(vitória) média por composição de Azul,
  uma linha por ρ. O zigue-zague da linha ρ=1,1 **é dado, não ruído**: é a
  granularidade de aquisição (§8.5).

---

## 10. Limitações e extensões naturais

1. **Regime único.** Os fatores de processo estão fixos no centro neutro. As
   conclusões valem para esse regime; a interação composição × vantagem de
   fusão (Δσ ≠ 0) é a extensão mais interessante — basta varrer `NEUTRAL` em
   `cross_composition.py` (ex.: repetir a matriz com σ_B = 0,9 vs σ_R = 0,6).
2. **Granularidade intrínseca.** Com orçamento ≈ 2–10 cascos, parte do efeito
   de composição é efeito de encaixe inteiro. Isso é reportado (utilização por
   célula) e é um achado em si, mas para separar os dois efeitos pode-se
   re-rodar com orçamento de referência multiplicado (ex.: 3×35,2), onde a
   quantização encolhe.
3. **Alocação uniforme.** Sem targeting preferencial (atacar cascos de alto
   valor primeiro) — extensão futura do motor (spec do experimento principal,
   diferida à sequência). A excursão sem-reposição está disponível
   (`params["alloc"]="without_replacement"`).
4. **Empates abundantes.** No regime neutro, muitas batalhas terminam sem
   vencedor (ambos > θ ou magazines vazios) — P(vitória) baixa em módulo. O
   log-FER complementa como métrica contínua de vantagem.
5. **O `fer_mean` da base é enviesado** (média-de-razões); usar `logfer_mean`
   / `fer_geom` (ver dicionário).

---

## 11. Reprodutibilidade

- **Ambiente:** Python 3.11; versões fixadas em `requirements-lock.txt`
  (numpy 2.4.6, pandas 3.0.3, scipy 1.17.1, scikit-learn 1.9.0,
  matplotlib 3.11.0, pytest 9.1.1). Motor `salvo_mds.ENGINE_VERSION = "1.0.0"`.
- **Sementes:** master 20260731; por célula na coluna `seed`.
- **Comandos:**
  ```bash
  pip install -r requirements-lock.txt
  pytest test_validation.py -v          # valida o motor (7 testes)
  make cross                            # regenera base + figuras + sumário
  ```
- **Verificação independente de uma célula:** qualquer linha do CSV pode ser
  regenerada isoladamente com `monte_carlo(blue, red, NEUTRAL, reps=5000,
  seed=<coluna seed>)`, reconstruindo as frotas com `best_integer_fleet`.

---

## 12. Esqueleto sugerido do artigo e claims defensáveis

1. **Introdução** — decisão de composição de força (high–low mix) quando ambos
   os lados desenham; lacuna: modelos de salvas tratam o adversário como fixo.
2. **Modelo** — §2–3 deste documento (motor validado contra Armstrong 2011).
3. **Método** — §4–5: alocação inteira exata sob teto (contribuição
   metodológica própria: o artefato do arredondamento e sua correção merecem
   uma subseção — é um alerta geral para estudos de mistura com frotas pequenas).
4. **Resultados** — §8 na ordem: dominância de M (8.1–8.2), estrutura de melhor
   resposta (8.3), orçamento vs design (8.4), granularidade (8.5).
5. **Discussão** — implicações para planejamento de força com orçamento
   limitado; por que dispersão (L) emerge como resposta ao benchmark forte;
   ligação com Hughes (saturação de defesas) e com a literatura high–low mix.
6. **Limitações/futuro** — §10 (interação com vantagem de fusão = ponte para o
   artigo LAFusion principal).

**Claims que a base sustenta diretamente** (todos com números em §8):
(i) inexistência de ciclo entre frotas puras no regime neutro; (ii) dominância
da plataforma balanceada em paridade; (iii) dispersão como melhor resposta ao
oponente balanceado; (iv) equivalência aproximada entre ±10% de orçamento e a
amplitude total da escolha de composição; (v) não-monotonicidade do valor do
orçamento por granularidade de aquisição.

---

## 13. Cenário ampliado — orçamento de referência 10×M (base primária recomendada)

Para reduzir a quantização inteira que domina o comparativo no orçamento
pequeno (§10.2), o experimento foi re-executado com **orçamento de referência =
10 × custo da fragata M ≈ 70,3** (o dobro da base da spec), mantendo a função de
custo (γ=1,35), o regime neutro, ρ ∈ {0,9; 1,0; 1,1}, 5.000 reps/célula e o
mesmo esquema de sementes. **Base:** `cross10_results.csv` (mesmas 44 colunas);
figuras `cross10_matrix.png`, `cross10_budget_sensitivity.png`; sumário
`cross10_summary.md`. Reprodução: `python3 cross_composition.py --reps 5000
--jobs 4 --budget-m 10`.

**Frotas em paridade:** 19L / 10M / 4H puras; utilização média sobe de 89% para
**96%** (mín. 83% vs 55%) — o ±10% de orçamento agora move as frotas de forma
quase contínua (L: 17→19→21; M: 9→10→11). Checks: diagonal-espelho |log-FER|
máx 0,072; antissimetria 0,144.

**Resultados no orçamento ampliado (paridade):**

| Métrica | 5×M (35,2) | 10×M (70,3) |
|---|---|---|
| 3×3 puro (log-FER) | M>L>H (M vs L +0,57; M vs H +0,78) | **L≈M ≫ H** (L vs M +0,15; M vs H +1,90; L vs H +1,61) |
| Ranking Azul (topo) | M=0,26; L=0,24 | M=0,16; L=0,16; … C último (0,01) |
| Vermelho mais duro / mais frágil | M (0,03) / M+ (0,34) | L (0,02) / **H (0,19)** |
| Efeito ±10% orçamento vs amplitude do design | 0,20 vs 0,22 (≈ empate) | **0,21 vs 0,15 (orçamento vence)** |

**Novos achados que a escala revela:**

1. **A frota pesada colapsa em escala (saturação).** Com 4H contra 19L/10M, o
   H puro passa a ser o Vermelho mais frágil da matriz (P(vitória de Azul)
   contra ele = 0,19; L vs H: P=0,51, log-FER +1,61) e um dos piores Azuis. No
   orçamento pequeno, o H era competitivo porque as salvas adversárias eram
   pequenas; ao dobrar a escala, o volume ofensivo satura a defesa agrupada dos
   poucos cascos — o resultado clássico do modelo de salvas emergindo da
   resolução per-unit. Poucos pontos de mira não escalam.
2. **Quantidade e equilíbrio convergem no topo.** L e M empatam como melhores
   composições (log-FER L vs M ≈ +0,15, quase espelho); a hierarquia M>L do
   orçamento pequeno era parcialmente efeito de granularidade.
3. **Heterogeneidade continua sem prêmio em paridade neutra:** o centroide é a
   pior composição Azul (0,01) — consistente com a checagem de robustez do R6
   (alocação capped) do experimento principal.
4. **Em escala, orçamento pesa mais que design:** o efeito de ±10% de orçamento
   (+0,21) supera a amplitude total da escolha de composição (0,15) — inversão
   do resultado do orçamento pequeno, onde design ≈ orçamento. A granularidade
   ainda aparece pontualmente: para o H puro, +10% em ρ=1,1 não compra nenhum
   casco extra (segue 4H, utilização cai 99%→90%) e o ganho é nulo — o "degrau"
   visível em `cross10_budget_sensitivity.png`.

**Nota de redação:** o contraste 5×M vs 10×M é, em si, um resultado central do
artigo — a dependência de escala do trade-off quantidade–qualidade (H viável em
frotas pequenas, dominado em frotas médias) e a transição "design importa ≈
orçamento" → "orçamento importa mais" são as duas curvas que só a resolução
per-unit com alocação inteira honesta consegue exibir.

---

## 14. Curva de escala completa — três pontos (5×M / 10×M / 20×M)

Terceiro ponto executado com orçamento **20×M ≈ 140,7** (frotas puras em
paridade: 39L / 20M / 8H; utilização média 97%, mín. 90%; checks: diagonal-
espelho 0,042, antissimetria 0,083 — os mais estreitos dos três pontos, como
esperado com frotas maiores). Bases: `cross20_results.csv`; consolidação:
`scale_curve.py` → `fig_scale_curve.png` + `scale_curve_summary.md`.
Reprodução: `python3 cross_composition.py --reps 5000 --jobs 4 --budget-m 20 &&
python3 scale_curve.py`.

| Métrica (paridade) | 5×M | 10×M | 20×M | Tendência |
|---|---|---|---|---|
| log-FER **L vs M** | −0,61 | +0,15 | +0,54 | M vence → **L vence** (cruzamento ≈ 10×M) |
| log-FER **L vs H** | +0,17 | +1,61 | +2,36 | colapso do H aprofunda monotonicamente |
| log-FER **M vs H** | +0,78 | +1,90 | +2,45 | idem |
| Amplitude da escolha de design | 0,22 | 0,15 | 0,11 | design importa cada vez menos |
| Efeito de ±10% de orçamento | +0,20 | +0,21 | +0,16 | ≈ estável (comprime a 20× pelos empates) |
| Melhor Azul / Vermelho mais frágil | M / M+ | M / H | **L / H** | quantidade assume o topo |
| Utilização (média / mín.) | 89% / 55% | 96% / 83% | 97% / 90% | quantização → desprezível |

**Os três resultados de escala do artigo:**

1. **O colapso da frota pesada é monotônico e acelera.** L-vs-H vai de +0,17
   (quase empate, 9L vs 2H) a +2,36 (massacre, 39L vs 8H). Mecanismo:
   o volume ofensivo cresce com a escala, e a defesa agrupada de poucos cascos
   satura — cada casco H adicional acrescenta defesa linearmente, mas o
   adversário acrescenta mísseis na mesma proporção; o que muda é que a frota
   numerosa dilui os vazadores entre mais alvos enquanto o H concentra o dano
   recebido. Capitais são viáveis apenas em engajamentos pequenos.
2. **Existe um cruzamento quantidade–equilíbrio (~10×M).** M vence L em frotas
   pequenas (−0,61), empata em médias (+0,15) e perde em grandes (+0,54) — o
   ponto ótimo do high–low mix desloca-se para a ponta *low* conforme o
   engajamento cresce.
3. **Design encolhe, orçamento persiste.** A amplitude da escolha de composição
   cai pela metade (0,22 → 0,11) enquanto o efeito de ±10% de orçamento
   permanece na mesma ordem (+0,20 → +0,16): em frotas grandes, massa
   orçamentária domina a decisão de mistura — convergência ao comportamento
   lanchesteriano/contínuo, com a ressalva de que a 20×M o regime neutro
   torna-se altamente indeciso (P(vitória) comprime; o log-FER segue
   informativo e é a métrica recomendada nessa escala).

---

## 15. Desenho de PARIDADE ESTRITA (ρ = 1 sempre) nas três escalas

**Motivação.** Nas §5/§13/§14 o orçamento variava ±10%, o que mistura dois
efeitos (composição e orçamento). Aqui os dois lados têm **sempre o mesmo
orçamento**, isolando a composição como única causa do resultado. Como a
dimensão ρ desaparece, as réplicas foram concentradas: **20.000 batalhas por
confronto** (contra 5.000 antes), 100 confrontos por escala, 3 escalas =
**6.000.000 de batalhas**.

**Bases:** `parity5_results.csv`, `parity10_results.csv`, `parity20_results.csv`
(agora com `logfer_se`, `p_victory_se` e `reps` por célula). Consolidação:
`parity_scale.py` → `parity_scale_matrix.png`, `parity_scale_curve.png`,
`parity_scale_summary.md`. Reprodução:
`python3 cross_composition.py --reps 20000 --jobs 4 --budget-m {5,10,20}
--rhos 1.0 --out-prefix parity{5,10,20} && python3 parity_scale.py`.

**Precisão e verificação.** IC95 mediano por célula: **±0,033 (5×M), ±0,025
(10×M), ±0,016 (20×M)**. Diagonal-espelho máx. 0,031 / 0,029 / 0,022 e
antissimetria máx. 0,062 / 0,058 / 0,053 — todas dentro do IC de uma célula, ou
seja, o motor é simétrico dentro da precisão de Monte Carlo. **96–97% dos 90
pares não-espelho são decididos a 95%** (as células indecididas estão marcadas
com um ponto na figura das matrizes).

### 3×3 puro, com intervalos (log-FER ± IC95)

| Confronto | 5×M | 10×M | 20×M |
|---|---|---|---|
| **L vs M** | **−0,54 ± 0,03** | **+0,17 ± 0,02** | **+0,54 ± 0,01** |
| **L vs H** | +0,27 ± 0,05 | +1,61 ± 0,03 | +2,35 ± 0,02 |
| **M vs H** | +0,79 ± 0,04 | +1,88 ± 0,04 | +2,49 ± 0,03 |
| Pura mais forte | **M** | **L** | **L** |

### Resultados

1. **O cruzamento quantidade–equilíbrio agora é medido, não estimado.** L-vs-M
   passa de −0,54 ± 0,03 (M vence) para +0,17 ± 0,02 (L vence) entre 5×M e
   10×M — ambos a muitos desvios-padrão de zero. O ponto de inversão fica
   **entre 5×M e 10×M**, e a vantagem de L segue crescendo (+0,54 em 20×M).
2. **O colapso do H é monotônico e maciço:** L-vs-H sobe +0,27 → +1,61 → +2,35;
   M-vs-H sobe +0,79 → +1,88 → +2,49. Em 10×M e 20×M o H puro é o Vermelho mais
   frágil da matriz (P(vitória de Azul) contra ele = 0,192 e 0,148, contra
   0,021 e 0,001 do L). Poucos cascos não escalam contra salvas grandes.
3. **Confronto direto ≠ desempenho médio (não-transitividade parcial).** Em
   10×M, **L vence M no confronto direto** (+0,17 ± 0,02), mas **M tem média
   maior contra o campo inteiro** (0,163 vs 0,153). São perguntas diferentes:
   "que frota bate aquela frota?" e "que frota se sai melhor contra um
   adversário desconhecido?" — e a resposta muda. Em 20×M as duas convergem
   para L (0,114 vs 0,086). Vale explicitar no artigo qual das duas leituras
   se está reportando.
4. **A composição decide cada vez menos com a escala** (painel B da curva,
   ambas as medidas indexadas ao valor de 5×M): a amplitude da média de
   P(vitória) cai para 0,69 e 0,51 do valor inicial, e o |log-FER| médio sobre
   os 90 pares cai para 0,66 e 0,68. As duas medidas concordam na direção; a
   segunda estabiliza entre 10×M e 20×M enquanto a primeira segue caindo —
   coerente com o adensamento de empates (ver advertência abaixo), que comprime
   P(vitória) mas não a razão de troca.
5. **A melhor resposta é quase sempre L ou M, nunca H nem misturas.** Em 20×M,
   L é a melhor resposta contra 6 das 10 composições e M contra as outras 4;
   nenhuma composição mista é a melhor resposta contra nada em nenhuma escala.
   No regime neutro e em paridade, a heterogeneidade não tem prêmio — o
   centroide fica em penúltimo (10×M) e no meio-baixo da tabela (20×M).

**Advertência de leitura (empates em escala):** no regime neutro, frotas
maiores terminam mais frequentemente sem decisão (atrito mútuo acima de θ com
magazines exauridos) — a P(vitória) absoluta cai nos três pontos por essa via.
As comparações desta seção usam log-FER (imune a esse deslocamento) ou
diferenças de P(vitória) dentro da mesma escala, nunca níveis absolutos entre
escalas.
