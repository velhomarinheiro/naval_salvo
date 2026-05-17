# Equação de Salva Multidominio — Seções 5 e 6 (rascunho)

> **Nota ao pesquisador.** Este rascunho de §5 e §6 foi produzido pela
> Fase 2 da implementação a partir dos artefatos numéricos gerados
> pelo pacote `naval_salvo` (288 testes passando, validação JPH/Hughes
> em precisão de máquina).  Os números, figuras e tabelas estão
> ancorados nos arquivos em `paper_artifacts/`.  Trate-o como ponto
> de partida; refine a redação para o estilo SBPO desejado e ajuste
> os argumentos onde julgar necessário.  Pontos sinalizados com
> **[VERIFICAR]** merecem revisão sua antes da submissão.

---

## 5. Validação numérica e resultados

A formulação proposta foi implementada em pacote Python aberto
(`naval_salvo`, com 288 testes unitários e de integração) e validada
em três níveis de exigência: recuperação exata do modelo
homogêneo de Hughes [Hughes, 1995], recuperação cell-by-cell do
exemplo trabalhado de Coronel em Johns, Pilnick e Hughes [Johns
et al., 2001], e análise de sensibilidade no cenário de Defesa da
Bacia de Campos.

### 5.1 Recuperação de casos clássicos

A propriedade estrutural enunciada na §3.5 — preservação do modelo
heterogêneo de Johns, Pilnick e Hughes como caso limite — foi
verificada numericamente em duas situações canônicas (Tabela 3).

**Tabela 3 — Recuperação numérica em precisão de máquina**

| Cenário                                 | Métrica           | Analítico   | Engine    | Δ          |
| --------------------------------------- | ----------------- | ----------- | --------- | ---------- |
| Hughes 1995 (1 salva, balanceado)       | Força A pós-salva | 2.000000    | 2.000000  | 0.00e+00   |
| Hughes 1995 (1 salva, balanceado)       | Força B pós-salva | 2.000000    | 2.000000  | 0.00e+00   |
| JPH 2001 Coronel (1ª min., Good Hope)   | ΔA por navio      | 0.037682    | 0.037682  | 0.00e+00   |

A reprodução do exemplo trabalhado de Coronel é particularmente
estrita: a equação (2) avaliada para ΔA₁ no primeiro minuto da
batalha (Scharnhorst e Gneisenau engajando metade de seu fogo
contra Good Hope, ε = 0,028, β = 2,16, ψ = 0,5, ς = 1,605) produz o
valor analítico 2 · (0,028 · 2,16 · 0,5)/1,605 ≈ 0,03768; o valor
calculado pela implementação (com a equação (9) restrita ao único
domínio cinético *s*) coincide a esse valor com erro numérico
nulo. A reprodução é portanto não aproximada, mas exata em
precisão de máquina, indicando que a generalização proposta é
estritamente conservativa em relação ao corpo clássico de
resultados de combate por mísseis.

### 5.2 Cenário central — Defesa da Bacia de Campos

O cenário ilustrativo central segue a especificação de §3:
componente Azul (MB) com uma fragata classe Tamandaré-equivalente,
um submarino da classe Riachuelo, uma aeronave de patrulha
marítima e quatro plataformas FPSO; componente Vermelho com dois
contratorpedeiros e uma esquadrilha de aviação de ataque.
Capacidades cibernéticas variam entre os experimentos. Os
parâmetros calibrados estão dispostos na Tabela 1 com sua origem
documentada na literatura [Christiansen, 2008; Casola, 2017]. A
matriz de admissibilidade adotada (χ = 0,5 para os pares cinéticos
marginais) é a apresentada na Figura 1.

A análise abrangeu trinta e duas configurações combinando: (i)
zero a três fragatas defensoras; (ii) presença/ausência do
submarino; (iii) capacidade cibernética simétrica, assimétrica
Vermelha, dominante Vermelha e dominante Azul; e (iv) parâmetros
(r₀, k) do modulador Φ canônico.

#### 5.2.1 Sensibilidade ao dimensionamento da escolta

A relação entre número de fragatas e sobrevivência dos ativos do
Pré-Sal é monotônica (Figura 4a): zero ou uma fragata implicam
perda total dos quatro FPSOs em duas a três salvas; duas fragatas
preservam aproximadamente metade de uma plataforma em média; e
três ou quatro fragatas preservam de uma a três plataformas. A
contribuição marginal do submarino, embora mais discreta no
desfecho final dos FPSOs (linha azul ligeiramente acima da linha
âmbar na Figura 4a), torna-se nítida na duração da campanha
(Figura 4b): a presença do submarino estende em cerca de uma salva
o tempo até que a força Azul perca capacidade cinética agregada,
oferecendo janela operacional adicional para resposta.

[VERIFICAR] **Limitação documentada.** A formulação canônica de
Hughes/JPH assume que cada unidade defende exclusivamente a si
mesma, sem mecanismo de cobertura de defesa de área entre escolta
e plataforma protegida — a fragata não estende o seu CIWS para a
FPSO adjacente. As cifras de sobrevivência das plataformas devem
ser interpretadas, portanto, como cota superior do efeito
adversário, e a extensão do modelo para acomodar defesa de área
é trabalho prioritário em fase subsequente.

#### 5.2.2 Modulação cibernética: o achado principal

A Figura 3 sintetiza o resultado mais conspícuo do estudo: sob
quatro configurações cibernéticas distintas, a curva de
sobrevivência cinética agregada de cada lado exibe regimes
qualitativamente distintos.

- **Sem cibernético** (cinza): conflito resolve-se em três salvas
  com colapso total da força Azul; força Vermelha sai com sua
  componente de superfície intacta (1,78/2 contratorpedeiros) e
  ataque aéreo parcialmente atrítado (≈ 2,5/4).
- **Cibernético simétrico** (teal): ambos os lados sofrem
  degradação multiplicativa equivalente em σ, ρ e δ; o conflito
  alonga-se para dez salvas e nenhum dos lados elimina rapidamente
  o outro. A distribuição final de força total Vermelha (≈ 4,8) é
  superior à do caso sem cibernético.
- **Domínio cibernético Vermelho** (vermelho): η_offense Azul é
  multiplicado por valores baixos de Φ; a força Azul colapsa em
  duas salvas, e a força Vermelha preserva sua composição cinética
  praticamente intacta (componente cinética total = 6,0).
- **Domínio cibernético Azul** (azul): a simetria do caso anterior
  produz desfecho oposto — Azul preserva sua componente cinética e
  Vermelho é aniquilado.

A leitura comparativa traz duas inferências de interesse
operacional. Primeiro, a paridade cibernética não é equivalente à
ausência de cibernético: a paridade *prolonga* o conflito mas não
o resolve a favor de qualquer lado, sugerindo que investimento
cibernético simétrico bilateral atua como denegação mútua de
área operacional. Segundo, a assimetria cibernética é decisiva:
o lado dominante preserva sua componente cinética de modo robusto,
enquanto o lado dominado é eliminado mais rapidamente do que sob
ausência de cibernético, dado que sua resposta cinética é
amputada antes que possa surtir efeito.

#### 5.2.3 Imunidade cibernética do submarino

A propriedade enunciada em §4.3 — preservação estrutural do
domínio submarino frente à modulação Φ — manifesta-se
quantitativamente como invariância da fração residual do
submarino após a primeira salva, qualquer que seja a capacidade
cibernética Vermelha (Figura 5b: barras teal constantes em 0,67
para níveis cibernéticos 0, 1, 2 e 3). Em contraste, a fração
residual da fragata, no mesmo período, decai de 0,53 (sem
cibernético) para 0 (qualquer cibernético Vermelho ≥ 1), um
efeito que se traduz no painel (a) como colapso da curva tracejada
vermelha em uma única salva.

A leitura doutrinária dessa propriedade é direta: em
ambientes de contestação cibernética significativa, o submarino é
o único ativo cuja eficácia *operacional* — não apenas física —
é estruturalmente preservada. Esse achado tem relevância imediata
para o planejamento da Esquadra brasileira no horizonte de
incorporação dos submarinos da classe Riachuelo e do submarino
com propulsão nuclear, e ilustra o tipo de inferência analítica
que a formulação multi-domínio possibilita e que formulações
mono-domínio ou cibernética-agregada não capturariam.

#### 5.2.4 Forma funcional de Φ — sensibilidade aos parâmetros calibráveis

A escolha da família funcional Φ^p(R) = 1 / [1 + (R/r₀)^k] para a
modulação cibernética (equação 12) implica dois parâmetros
calibráveis por canal: r₀ e k. A Figura 6 ilustra a sensibilidade
qualitativa dessas escolhas. O parâmetro r₀ controla o ponto de
meia-degradação: para r₀ = 0,5, a meia-degradação Φ = 0,5 ocorre
em razão de força cibernética R^p = 0,5 (atacante cibernético com
metade da capacidade do alvo já produz 50% de degradação); para
r₀ = 2,0, a mesma meia-degradação requer R^p = 2,0 (atacante
precisa do dobro de capacidade). O parâmetro k controla a
nitidez do joelho da sigmoide: k = 1 produz transição suave;
k = 4 produz transição quase abrupta em torno de r₀.

A calibração de (r₀, k) é portanto chave para a interpretação
operacional: r₀ baixo modela ambientes em que pequenas vantagens
cibernéticas têm efeito desproporcional (situação típica em
domínios em que defesas eletrônicas estão pouco preparadas);
r₀ alto modela ambientes endurecidos, em que apenas vantagem
significativa converte-se em degradação efetiva. O presente
trabalho adotou r₀ = k = padrões operacionalmente neutros (r₀ =
1, k = 2) para os experimentos da §5.2.2; a sensibilidade dos
desfechos a essa escolha é análise prevista para trabalho futuro.

### 5.3 Síntese da matriz de sensibilidade

A Tabela 2 condensa a matriz combinada submarino × cibernético
× duração da campanha:

**Tabela 2 — Matriz de sensibilidade combinada**

| Sub | Cyber Azul | Cyber Vermelho | Configuração            | Salvas | FPSO | Frag | Sub  | Dest |
| --- | ---------- | -------------- | ----------------------- | ------ | ---- | ---- | ---- | ---- |
| N   | 0          | 0              | Sem cyber               | 2      | 0,00 | 0,00 | —    | 2,00 |
| N   | 2          | 2              | Simétrico (2,2)         | 10     | 0,00 | 0,00 | —    | 2,00 |
| N   | 1          | 2              | Assimétrico Vermelho    | 3      | 0,00 | 0,00 | —    | 2,00 |
| N   | 0          | 3              | Domínio Vermelho (0,3)  | 2      | 0,00 | 0,00 | —    | 2,00 |
| Y   | 0          | 0              | Sem cyber               | 3      | 0,00 | 0,00 | 0,00 | 1,78 |
| Y   | 2          | 2              | Simétrico (2,2)         | 11     | 0,00 | 0,00 | 0,00 | 2,00 |
| Y   | 1          | 2              | Assimétrico Vermelho    | 3      | 0,00 | 0,00 | 0,00 | 2,00 |
| Y   | 0          | 3              | Domínio Vermelho (0,3)  | 2      | 0,00 | 0,00 | 0,00 | 2,00 |

Três regularidades emergem dos números acima.  Primeira, o impacto
do submarino na atrição do contratorpedeiro Vermelho desaparece
sob qualquer pressão cibernética assimétrica Vermelha — embora
o submarino sobreviva (imunidade), seu efeito ofensivo sobre o
contratorpedeiro é mediado pela η_offense modulada de outras
unidades Azuis cuja eficácia colapsa, e o submarino sozinho não
consegue saturar a defesa Vermelha. Segunda, o regime cibernético
simétrico estende a campanha em um fator de cinco a seis salvas
em ambos os casos com e sem submarino — confirmando o caráter de
denegação mútua mencionado em §5.2.2. Terceira, a presença do
submarino é a única configuração cinética que infringe atrição
mensurável aos contratorpedeiros (1,78 vs 2,00) — sob ausência
de cibernético — refletindo a admissibilidade marginal *u → s* na
matriz χ.

---

## 6. Conclusões

Este trabalho desenvolve uma extensão multi-domínio do modelo de
salva heterogêneo de Johns, Pilnick e Hughes [Johns et al., 2001]
que (i) acomoda explicitamente cinco domínios operacionais —
superfície, submarino, aéreo, costeiro e cibernético-eletromagnético —
por meio de matriz de admissibilidade 5×5 com três níveis de
codificação; (ii) preserva a natureza pulsada das salvas via
arquitetura híbrida combinando equações diferenciais ordinárias
entre salvas com saltos discretos nos instantes de salva; (iii)
integra a estrutura de realocação proporcional de Hausken e Moxnes
[Hausken & Moxnes, 2026] no fator κ que generaliza a fração de
mira da formulação original; e (iv) introduz tratamento específico
para o domínio cibernético-eletromagnético via família de
moduladores Φ que escalam multiplicativamente os parâmetros
compostos σ, ρ e δ dos demais domínios.

A propriedade estrutural mais conseqüente da formulação proposta
é a preservação estrita do modelo heterogêneo original como caso
limite (§3.5), validada numericamente em precisão de máquina para
dois cenários canônicos: a configuração simples de Hughes [Hughes,
1995] e a reprodução cell-by-cell do exemplo trabalhado de Coronel
em Johns, Pilnick e Hughes [Johns et al., 2001]. Esse resultado
garante que a generalização não desfaz nenhum resultado clássico
da literatura de salvas: extensões em domínios, em modulação
cibernética e em realocação proporcional são adicionadas sem custo
de retrocompatibilidade analítica.

A análise quantitativa do cenário ilustrativo de Defesa da Bacia
de Campos produziu três achados de relevância operacional. Primeiro,
em ambientes de paridade cibernética bilateral o conflito não se
resolve no horizonte de dez a doze salvas, sugerindo que o
investimento cibernético simétrico atua como mecanismo de denegação
mútua de área operacional — observação cuja interpretação
estratégica merece desenvolvimento em estudos subseqüentes.
Segundo, em regimes de assimetria cibernética significativa, o
lado dominante preserva sua componente cinética enquanto o
dominado colapsa em ritmo *acelerado* em relação à ausência de
cibernético — a degradação cibernética não desloca apenas a
distribuição final, mas a temporalidade da campanha. Terceiro, a
imunidade estrutural do submarino à modulação cibernética (§4.3)
materializa-se como invariância da sua fração residual após a
primeira salva, qualquer que seja a pressão cibernética Vermelha
(Figura 5b): em ambientes de contestação cibernética acentuada, o
submarino é o único ativo Azul cuja efetividade operacional —
e não apenas física — é preservada por construção.

Esse último achado tem relevância imediata para o planejamento
de longo prazo da Esquadra brasileira, no horizonte de incorporação
dos submarinos da classe Riachuelo e do submarino com propulsão
nuclear (PROSUB). Em ambientes de contestação eletromagnética e
cibernética cada vez mais densa — característica do domínio
marítimo contemporâneo — a capacidade submarina deixa de ser apenas
um instrumento de denegação de área e converte-se em ativo
*estruturalmente preservado* de projeção de poder, cuja
contribuição à defesa de áreas estratégicas como a Bacia de
Campos não é redutível à de meios de superfície ou aéreos.

### 6.1 Limitações e trabalho futuro

Três limitações da formulação atual merecem registro explícito.

**Cobertura de defesa de área.** A equação canônica de Hughes/JPH
assume que cada unidade defende exclusivamente a si mesma, sem
mecanismo de área defense entre escolta e plataforma protegida
[VERIFICAR — esta limitação é estrutural ao modelo de salva e não
trivial de remover]. As cifras de sobrevivência das plataformas
FPSO devem portanto ser interpretadas como cota superior do efeito
adversário; uma extensão prioritária do modelo é a introdução de
mecanismo de área defense que permita que o CIWS de uma fragata
contribua para a defesa das plataformas adjacentes — uma alteração
não-trivial da equação central que constituirá objeto de trabalho
imediato.

**Calibração paramétrica.** Os parâmetros usados nos experimentos
são derivados da literatura aberta [Christiansen, 2008; Casola, 2017]
e devem ser entendidos como ilustrativos. Refinamento via fontes
doutrinárias da Marinha do Brasil é o passo natural antes de
extrair conclusões doutrinárias robustas.

**Casos estocástico e seqüencial.** A formulação atual cobre o
caso determinístico simultâneo. Extensões para versão estocástica
[Armstrong, 2005] e seqüencial [Armstrong, 2014] são compatíveis
com a arquitetura proposta e constituirão trabalho subseqüente.

**Validação histórica.** A reprodução do exemplo de Coronel,
embora numericamente exata, é restrita a um único minuto da
batalha. Reprodução de batalhas inteiras (Coral Sea, Savo Island)
e de cenários multi-domínio sinteticamente construídos é trajetória
natural da pesquisa.

### 6.2 Reprodutibilidade

A implementação completa, incluindo testes unitários (288 casos,
todos verdes) e o script de reprodução das figuras e tabelas
deste artigo, está disponível como pacote Python aberto em
[VERIFICAR — endereço do repositório a ser inserido]. Todas as
figuras e tabelas podem ser regeneradas a partir de um único
comando (`python reproduce_paper.py`).
