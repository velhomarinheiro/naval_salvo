"""
Página: Bacia de Campos (cenário multidomínio configurável).

Reformulação da página original: em vez de usar o ``BaciaCamposConfig``
fixo (que só expõe ``n_frigates`` / ``submarine_present`` / contagens de
cyber), construímos o cenário diretamente com o ``EngagementBuilder``
do pacote ``naval_salvo``.  Isso permite ao usuário editar, na própria
interface:

- Lado Azul (MB):
    * 02 tipos diferentes de unidades de superfície (qtd, staying,
      p_offense, p_defense para cada tipo);
    * Submarinos -- 01 tipo, qtd configurável (staying, p_off, p_def);
    * Bateria costeira -- qtd, staying, p_off, p_def;
    * FPSO -- qtd e staying (sem poder ofensivo, como ativo de valor);
    * Cyber Azul -- qtd por sub-tipo (C2, SEN, WPN, LOG).
- Lado Vermelho:
    * 02 tipos diferentes de unidades de superfície (qtd, staying,
      p_off, p_def);
    * Aviação de ataque -- qtd, staying, p_off, p_def;
    * Cyber Vermelho -- qtd por sub-tipo.

A página gera dinamicamente as unidades, monta o ``EngagementParameters``
com defaults sensatos para os pares e roda ``run_campaign``.  Mostra
trajetórias por domínio e tabela final.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from naval_salvo import (
    Admissibility,
    ChannelPhi,
    Domain,
    EngagementBuilder,
    Manual,
    StrengthProportional,
    ThreatWeighted,
    Uniform,
    UnitType,
    run_campaign,
)
from naval_salvo.targeting import TargetingPolicy


# ---------------------------------------------------------------------------
# Configuração da página
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Bacia de Campos -- multidomínio",
    page_icon="⚓",
    layout="wide",
)

st.title("⚓ Cenário multidomínio -- Bacia de Campos")
st.markdown(
    """
    Configure abaixo a composição das duas forças, seus parâmetros de
    capacidade, e as matrizes de engajamento. A simulação roda o modelo
    de salva multidomínio (Hughes 1995 estendido) com a matriz de
    admissibilidade canônica e, opcionalmente, o modulador cibernético
    Φ por canal.
    """
)


# ---------------------------------------------------------------------------
# Defaults inspirados no cenário original do paper (BACIA_CAMPOS_PARAMETERS).
# Editáveis em runtime; cada classe vira um UnitType só se ``quantity > 0``.
# ---------------------------------------------------------------------------

@dataclass
class ShipClassInputs:
    """Inputs editáveis para uma classe de unidade kinética.

    Os campos ``p_off`` e ``p_def`` representam o **poder bruto** da
    plataforma — antes de aplicar os fatores compostos. Na construção
    da matriz de engajamento, o poder ofensivo/defensivo efetivo é

        p_off_efetivo = p_off × scouting × treinamento
        p_def_efetivo = p_def × treinamento × alerta

    Os fatores do atacante são aplicados ao p_off; os do defensor, ao
    p_def. Não há acoplamento cruzado (alerta do alvo não afeta o
    ataque do oponente). Esta decomposição segue JPH 2001 eq. 2.18.
    """

    name: str
    quantity: int
    staying: float
    p_off: float
    p_def: float
    scouting: float = 1.0     # σ_scout ∈ [0, 1]   — atacante mantém contato
    training: float = 1.0     # τ_train ∈ [0, 1]   — eficácia do operador
    alert:    float = 1.0     # α_alert ∈ [0, 1]   — nível de alerta defensivo


# Cyber: 4 sub-tipos, todos com staying = 1 e p_off contra outros
# sub-tipos cyber default.  Quantidade por sub-tipo é o único campo
# editável (igual ao paper).
CYBER_SUBTYPES = ["C2", "SEN", "WPN", "LOG"]
CYBER_DEFAULT_STAYING = 1.0
CYBER_DEFAULT_P_OFF = 0.5      # contra outros sub-tipos cyber
CYBER_DEFAULT_P_DEF = 0.2


# ---------------------------------------------------------------------------
# Defaults editáveis no formulário -- valores "Bacia de Campos" do paper.
# Os fatores compostos (scouting, treinamento, alerta) ficam em 1.0 por
# default (efeito neutro). O usuário ajusta caso queira modelar fricção
# operacional, vantagem doutrinária, surpresa, etc.
# ---------------------------------------------------------------------------

BLUE_DEFAULTS = {
    # 2 tipos de superfície: fragata principal + corveta/patrulha leve
    "surface_1": ShipClassInputs("Fragata classe A", 2, 3.0, 1.5, 1.0,
                                  scouting=1.0, training=1.0, alert=1.0),
    "surface_2": ShipClassInputs("Corveta/Patrulha", 2, 2.0, 1.0, 0.6,
                                  scouting=1.0, training=1.0, alert=1.0),
    "submarine": ShipClassInputs("Submarino convencional", 1, 2.0, 2.0, 0.0,
                                  scouting=1.0, training=1.0, alert=1.0),
    "strike_air": ShipClassInputs("Aviação de ataque Azul", 0, 1.0, 1.8, 0.3,
                                   scouting=1.0, training=1.0, alert=1.0),
    "coastal":   ShipClassInputs("Bateria costeira", 1, 4.0, 1.2, 0.4,
                                  scouting=1.0, training=1.0, alert=1.0),
    # FPSO: ativo de valor, alta staying, sem poder ofensivo.
    "fpso":      ShipClassInputs("FPSO (Pré-Sal)", 4, 6.0, 0.0, 0.0,
                                  scouting=1.0, training=1.0, alert=1.0),
}

RED_DEFAULTS = {
    "surface_1":   ShipClassInputs("Destróier", 2, 4.0, 2.5, 1.5,
                                    scouting=1.0, training=1.0, alert=1.0),
    "surface_2":   ShipClassInputs("Fragata adv.", 2, 3.0, 1.8, 1.0,
                                    scouting=1.0, training=1.0, alert=1.0),
    "submarine":   ShipClassInputs("Submarino adversário", 0, 2.0, 2.0, 0.0,
                                    scouting=1.0, training=1.0, alert=1.0),
    "strike_air":  ShipClassInputs("Aviação de ataque Vermelha", 4, 1.0, 2.0, 0.3,
                                    scouting=1.0, training=1.0, alert=1.0),
}


# ---------------------------------------------------------------------------
# Sidebar -- configurações globais
# ---------------------------------------------------------------------------

st.sidebar.header("Configurações globais")
n_salvos = st.sidebar.slider(
    "Número máximo de salvas", min_value=1, max_value=30, value=15, step=1,
    help="A campanha pode terminar antes se uma força for neutralizada."
)
use_cyber = st.sidebar.checkbox(
    "Ativar modulador cibernético Φ (canal)", value=True,
    help="Aplica o modulador ChannelPhi do pacote, que escala η_offense/"
         "η_defense em função da razão de força cibernética por canal "
         "(σ, ρ, δ). Submarinos são imunes."
)
stop_on_termination = st.sidebar.checkbox(
    "Encerrar campanha ao colapso de uma das forças", value=True
)

st.sidebar.markdown("---")
st.sidebar.subheader("Política de targeting")
targeting_policy_name = st.sidebar.selectbox(
    "Como cada atacante reparte sua salva entre os alvos admissíveis?",
    options=[
        "StrengthProportional",
        "Uniform",
        "ThreatWeighted",
        "Manual",
    ],
    index=0,
    help=(
        "StrengthProportional: σ proporcional à força remanescente do "
        "alvo (MacKay 2009 / Hausken-Moxnes 2026).\n\n"
        "Uniform: divisão igual entre todos os alvos admissíveis "
        "(Hughes 1995 default).\n\n"
        "ThreatWeighted: σ proporcional a pesos editáveis por classe "
        "defensora -- útil para priorizar FPSOs.\n\n"
        "Manual: o usuário define σ célula a célula em uma matriz "
        "fixa (a mesma em todas as salvas)."
    ),
)
_policy_descriptions = {
    "StrengthProportional":
        "Cada salva é distribuída em proporção à força *atual* do alvo. "
        "σ é recalculada antes de cada salva (semi-dinâmica).",
    "Uniform":
        "Cada atacante divide sua salva igualmente entre os alvos "
        "doutrinariamente admissíveis. σ é fixada na primeira salva.",
    "ThreatWeighted":
        "Configure pesos por classe defensora na aba 'Targeting'. "
        "σ é recalculada antes de cada salva.",
    "Manual":
        "Configure as matrizes σ_offense célula a célula na aba "
        "'Targeting'. σ é fixa em todas as salvas.",
}
st.sidebar.caption(_policy_descriptions[targeting_policy_name])


# ---------------------------------------------------------------------------
# Helpers para construir os widgets de cada classe.
# ---------------------------------------------------------------------------

def _ship_class_inputs(
    label: str,
    defaults: ShipClassInputs,
    *,
    key_prefix: str,
    allow_zero_quantity: bool = True,
    locked_p_off: bool = False,
    locked_p_def: bool = False,
    help_text: str = "",
) -> ShipClassInputs:
    """Renderiza os widgets de uma classe de unidade num expander.

    Campos coletados: nome, quantidade, staying power, poder ofensivo,
    poder defensivo, e os 4 fatores compostos (scouting, treinamento,
    alerta, e o complementar de alerta usado como distração).

    Se ``allow_zero_quantity`` for True, quantidade 0 omite a classe da
    força (útil para desativar submarino, aviação, segunda classe etc).
    """
    with st.expander(label, expanded=True):
        if help_text:
            st.caption(help_text)
        name = st.text_input(
            "Nome da classe", value=defaults.name, key=f"{key_prefix}_name"
        )
        c1, c2 = st.columns(2)
        quantity = c1.number_input(
            "Quantidade", min_value=0 if allow_zero_quantity else 1,
            value=defaults.quantity, step=1, key=f"{key_prefix}_qty"
        )
        staying = c2.number_input(
            "Staying power (ς)", min_value=0.1,
            value=float(defaults.staying), step=0.1, format="%.2f",
            key=f"{key_prefix}_staying",
            help="Número de hits que cada unidade absorve antes de ser "
                 "neutralizada."
        )
        c3, c4 = st.columns(2)
        if locked_p_off:
            p_off = 0.0
            c3.markdown("**Poder ofensivo (p_off):** _0 (sem capacidade ofensiva)_")
        else:
            p_off = c3.number_input(
                "Poder ofensivo bruto (p_off)", min_value=0.0,
                value=float(defaults.p_off), step=0.1, format="%.2f",
                key=f"{key_prefix}_poff",
                help="Hits por unidade por salva, contra o alvo padrão, "
                     "antes da aplicação dos fatores compostos."
            )
        if locked_p_def:
            p_def = 0.0
            c4.markdown("**Poder defensivo (p_def):** _0 (sem defesa ativa)_")
        else:
            p_def = c4.number_input(
                "Poder defensivo bruto (p_def)", min_value=0.0,
                value=float(defaults.p_def), step=0.1, format="%.2f",
                key=f"{key_prefix}_pdef",
                help="Interceptações por unidade por salva, antes da "
                     "aplicação dos fatores compostos."
            )

        # ---- Fatores compostos ----
        st.markdown("**Fatores compostos** (JPH 2001, eq. 2.18)")
        f1, f2, f3 = st.columns(3)
        scouting = f1.slider(
            "Scouting (σ_scout)", min_value=0.0, max_value=1.0,
            value=float(defaults.scouting), step=0.05,
            key=f"{key_prefix}_scout",
            help="Fração de tempo em que a unidade mantém cueing válido "
                 "sobre o alvo. Aplicado ao p_off."
        )
        training = f2.slider(
            "Treinamento (τ_train)", min_value=0.0, max_value=1.0,
            value=float(defaults.training), step=0.05,
            key=f"{key_prefix}_train",
            help="Eficácia média do operador/sistema. Aplicado a p_off "
                 "e p_def."
        )
        alert = f3.slider(
            "Alerta (α_alert)", min_value=0.0, max_value=1.0,
            value=float(defaults.alert), step=0.05,
            key=f"{key_prefix}_alert",
            help="Nível de alerta defensivo. Aplicado apenas ao p_def "
                 "desta unidade (multiplica o poder defensivo)."
        )

        return ShipClassInputs(
            name=name, quantity=int(quantity),
            staying=float(staying),
            p_off=float(p_off), p_def=float(p_def),
            scouting=float(scouting),
            training=float(training),
            alert=float(alert),
        )


def _cyber_inputs(
    label: str, key_prefix: str, default_per_subtype: int = 2
) -> dict[str, int]:
    """Renderiza quantidades dos 4 sub-tipos cyber."""
    with st.expander(label, expanded=False):
        st.caption(
            "Stocks cibernéticos por sub-tipo. C2 = comando e controle; "
            "SEN = sensores/ISR; WPN = sistemas de armas; LOG = logística."
        )
        cols = st.columns(4)
        out: dict[str, int] = {}
        for col, sub in zip(cols, CYBER_SUBTYPES):
            out[sub] = int(col.number_input(
                sub, min_value=0, max_value=20,
                value=default_per_subtype, step=1,
                key=f"{key_prefix}_cyber_{sub}",
            ))
        return out


# ---------------------------------------------------------------------------
# Formulários: duas abas (Azul / Vermelho) + uma para matriz de engajamento
# ---------------------------------------------------------------------------

tab_blue, tab_red, tab_engagement, tab_targeting, tab_results = st.tabs(
    [
        "🔵 Força Azul (MB)",
        "🔴 Força Vermelha",
        "⚔️ Matriz de engajamento",
        "🎯 Targeting",
        "📊 Resultados",
    ]
)


# ---- Azul -----------------------------------------------------------------

with tab_blue:
    st.subheader("Composição da Força Azul")
    st.info(
        "ℹ️ Os campos de **poder ofensivo** e **poder defensivo** "
        "informados aqui são usados para pré-preencher a aba "
        "**⚔️ Matriz de engajamento**. O cálculo da simulação utiliza "
        "os valores que estiverem na matriz no momento da execução — "
        "se você refinou alguma célula manualmente, esses valores "
        "específicos têm precedência. Quantidades e *staying power*, "
        "ao contrário, são lidos diretamente desta aba."
    )
    blue_s1 = _ship_class_inputs(
        "Superfície -- Tipo 1",
        BLUE_DEFAULTS["surface_1"],
        key_prefix="blue_s1",
        help_text="Primeira classe de unidades de superfície (ex.: fragata)."
    )
    blue_s2 = _ship_class_inputs(
        "Superfície -- Tipo 2",
        BLUE_DEFAULTS["surface_2"],
        key_prefix="blue_s2",
        help_text="Segunda classe de unidades de superfície (ex.: corveta). "
                  "Use quantidade 0 para desativar."
    )
    blue_sub = _ship_class_inputs(
        "Submarinos",
        BLUE_DEFAULTS["submarine"],
        key_prefix="blue_sub",
        help_text="Uma classe de submarinos; quantidade configurável. "
                  "Imunes ao cyber por construção do modelo."
    )
    blue_air = _ship_class_inputs(
        "Aviação de ataque",
        BLUE_DEFAULTS["strike_air"],
        key_prefix="blue_air",
        help_text="Aeronaves/UAVs de ataque ao mar. Quantidade 0 desativa "
                  "esta classe."
    )
    blue_coastal = _ship_class_inputs(
        "Bateria costeira",
        BLUE_DEFAULTS["coastal"],
        key_prefix="blue_coastal",
        help_text="Bateria costeira de mísseis anti-navio."
    )
    blue_fpso = _ship_class_inputs(
        "FPSO (ativo de valor)",
        BLUE_DEFAULTS["fpso"],
        key_prefix="blue_fpso",
        locked_p_off=True, locked_p_def=True,
        help_text="Plataformas do Pré-Sal: alvos a proteger, sem capacidade "
                  "ofensiva ou defensiva ativa. Apenas staying power."
    )
    blue_cyber = _cyber_inputs(
        "Cyber Azul (por sub-tipo)", key_prefix="blue",
        default_per_subtype=0,
    )


# ---- Vermelho -------------------------------------------------------------

with tab_red:
    st.subheader("Composição da Força Vermelha")
    st.info(
        "ℹ️ Os campos de **poder ofensivo** e **poder defensivo** "
        "informados aqui são usados para pré-preencher a aba "
        "**⚔️ Matriz de engajamento**. O cálculo da simulação utiliza "
        "os valores que estiverem na matriz no momento da execução — "
        "se você refinou alguma célula manualmente, esses valores "
        "específicos têm precedência. Quantidades e *staying power*, "
        "ao contrário, são lidos diretamente desta aba."
    )
    red_s1 = _ship_class_inputs(
        "Superfície -- Tipo 1",
        RED_DEFAULTS["surface_1"],
        key_prefix="red_s1",
        help_text="Primeira classe de unidades de superfície adversárias."
    )
    red_s2 = _ship_class_inputs(
        "Superfície -- Tipo 2",
        RED_DEFAULTS["surface_2"],
        key_prefix="red_s2",
        help_text="Segunda classe adversária. Quantidade 0 desativa."
    )
    red_sub = _ship_class_inputs(
        "Submarinos",
        RED_DEFAULTS["submarine"],
        key_prefix="red_sub",
        help_text="Submarinos adversários. Quantidade 0 desativa. "
                  "Imunes ao cyber por construção do modelo."
    )
    red_air = _ship_class_inputs(
        "Aviação de ataque",
        RED_DEFAULTS["strike_air"],
        key_prefix="red_air",
        help_text="Aeronaves/UAVs adversários de ataque ao mar."
    )
    red_cyber = _cyber_inputs(
        "Cyber Vermelho (por sub-tipo)", key_prefix="red",
        default_per_subtype=0,
    )


# ---------------------------------------------------------------------------
# Monta a lista de UnitType ativos para cada lado.
# ---------------------------------------------------------------------------

def _build_unit_types(
    classes: list[tuple[ShipClassInputs, Domain, Optional[str]]],
    cyber_counts: dict[str, int],
) -> list[UnitType]:
    """Constrói os UnitType ativos (qtd > 0) na ordem fornecida.

    ``classes`` é uma lista de (inputs, domínio, subtype-opcional).
    Inclui também os sub-tipos cibernéticos com qtd > 0 ao final.
    """
    out: list[UnitType] = []
    used_names: set[str] = set()
    for inp, domain, subtype in classes:
        if inp.quantity <= 0:
            continue
        nm = inp.name
        # Garante unicidade dentro da Force.
        base = nm
        k = 2
        while nm in used_names:
            nm = f"{base} ({k})"
            k += 1
        used_names.add(nm)
        out.append(UnitType(
            name=nm, domain=domain,
            staying_power=inp.staying,
            initial_strength=float(inp.quantity),
            subtype=subtype,
        ))
    # Sub-tipos cyber
    for sub, qty in cyber_counts.items():
        if qty <= 0:
            continue
        nm = f"Cyber-{sub}"
        used_names.add(nm)
        out.append(UnitType(
            name=nm, domain=Domain.CYBER,
            staying_power=CYBER_DEFAULT_STAYING,
            initial_strength=float(qty),
            subtype=sub,
        ))
    return out


# Mantém referência de (nome efetivo, inputs originais, papel) para usar
# depois na hora de preencher as matrizes p_offense / p_defense.
def _resolve_blue_classes() -> list[tuple[ShipClassInputs, Domain, Optional[str], str]]:
    """Devolve (inputs, domínio, subtype, role-key) para cada classe Azul."""
    return [
        (blue_s1, Domain.SURFACE, None, "blue_s1"),
        (blue_s2, Domain.SURFACE, None, "blue_s2"),
        (blue_sub, Domain.UNDERWATER, None, "blue_sub"),
        (blue_air, Domain.AIR, None, "blue_air"),
        (blue_coastal, Domain.COASTAL, None, "blue_coastal"),
        (blue_fpso, Domain.SURFACE, "pre-salt", "blue_fpso"),
    ]


def _resolve_red_classes() -> list[tuple[ShipClassInputs, Domain, Optional[str], str]]:
    return [
        (red_s1, Domain.SURFACE, None, "red_s1"),
        (red_s2, Domain.SURFACE, None, "red_s2"),
        (red_sub, Domain.UNDERWATER, None, "red_sub"),
        (red_air, Domain.AIR, None, "red_air"),
    ]


# Constrói as listas ordenadas e os UnitType correspondentes.
blue_resolved = _resolve_blue_classes()
red_resolved = _resolve_red_classes()

# Para o builder precisamos do nome final atribuído a cada UnitType.
def _materialize_force(resolved, cyber_counts):
    classes_for_builder = [(r[0], r[1], r[2]) for r in resolved]
    uts = _build_unit_types(classes_for_builder, cyber_counts)
    # Mapeia role_key -> nome final, para classes ativas.
    role_to_name: dict[str, str] = {}
    idx = 0
    for inp, _, _, role in resolved:
        if inp.quantity <= 0:
            continue
        role_to_name[role] = uts[idx].name
        idx += 1
    return uts, role_to_name


blue_unit_types, blue_role_to_name = _materialize_force(blue_resolved, blue_cyber)
red_unit_types,  red_role_to_name  = _materialize_force(red_resolved,  red_cyber)


# ---------------------------------------------------------------------------
# Aba: Matriz de engajamento (p_offense por par)
# ---------------------------------------------------------------------------

# Os defaults são construídos DINAMICAMENTE a partir dos ShipClassInputs
# das abas Azul e Vermelho a cada rerun, garantindo que mudanças nos
# campos de p_offense / p_defense das abas de força se propaguem
# automaticamente para a matriz de engajamento.
#
# Regra de preenchimento (JPH 2001 eq. 2.18, decomposição multiplicativa):
#
#   p_off[atk → def] = atk.p_off
#                     × atk.scouting               (σ_scout)
#                     × atk.training               (τ_train do atacante)
#
#   p_def[atk → def] = def.p_def
#                     × def.training               (τ_train do defensor)
#                     × def.alert                  (α_alert do defensor)
#
# A interpretação dos fatores:
#  - scouting (atacante): fração do tempo com cueing válido sobre o alvo.
#  - training: eficácia média do operador, aplicada separadamente em
#    cada lado (atacante usa o seu próprio; defensor usa o dele).
#  - alert (defensor): nível de alerta defensivo — aumenta o p_def,
#    sem dupla contagem do outro lado.
#
# Exceções estruturais:
#   submarino → aeronave : 0  (admissibilidade nula)

def _make_defaults(
    atk_resolved: list,
    def_resolved: list,
) -> tuple[dict, dict]:
    """Devolve (p_off_defaults, p_def_defaults) indexados por role-keys.

    Aplica a decomposição multiplicativa dos fatores compostos sobre
    o p_off e p_def brutos das classes (JPH 2001 eq. 2.18).
    """
    p_off: dict[tuple[str, str], float] = {}
    p_def: dict[tuple[str, str], float] = {}
    for a_inp, a_dom, _a_sub, a_role in atk_resolved:
        for d_inp, d_dom, _d_sub, d_role in def_resolved:
            key = (a_role, d_role)
            if a_dom is Domain.UNDERWATER and d_dom is Domain.AIR:
                p_off[key] = 0.0
                p_def[key] = 0.0
                continue
            # Lado ofensivo: fatores do atacante apenas.
            p_off[key] = (
                float(a_inp.p_off)
                * float(a_inp.scouting)
                * float(a_inp.training)
            )
            # Lado defensivo: fatores do defensor apenas.
            p_def[key] = (
                float(d_inp.p_def)
                * float(d_inp.training)
                * float(d_inp.alert)
            )
    return p_off, p_def


# Gera os defaults dinamicamente com base nos inputs ATUAIS das abas.
DEFAULT_P_OFF_BAR, DEFAULT_P_DEF_BAR = _make_defaults(
    blue_resolved, red_resolved
)
DEFAULT_P_OFF_RAB, DEFAULT_P_DEF_RAB = _make_defaults(
    red_resolved, blue_resolved
)


def _build_throughput_grid(
    direction: str,
    attacker_roles_active: list[tuple[str, str]],  # (role, final_name)
    defender_roles_active: list[tuple[str, str]],
    defaults_off: dict[tuple[str, str], float],
    defaults_def: dict[tuple[str, str], float],
) -> tuple[dict[tuple[str, str], float], dict[tuple[str, str], float]]:
    """Render data_editor and return {(atk_name, def_name): value} dicts."""
    if not attacker_roles_active or not defender_roles_active:
        st.info(
            f"Direção {direction}: uma das forças não tem unidades ativas."
        )
        return {}, {}

    atk_role_to_name = dict(attacker_roles_active)
    def_role_to_name = dict(defender_roles_active)

    def _df_from_defaults(defaults: dict[tuple[str, str], float]) -> pd.DataFrame:
        rows = []
        index = []
        for atk_role, atk_name in attacker_roles_active:
            index.append(atk_name)
            row = {}
            for def_role, def_name in defender_roles_active:
                row[def_name] = float(defaults.get((atk_role, def_role), 0.0))
            rows.append(row)
        return pd.DataFrame(rows, index=index)

    st.markdown(f"**{direction}** — atacante (linhas) × defensor (colunas)")
    # A sig inclui: nomes das unidades (detecta mudança de composição)
    # E um hash dos valores dos defaults (detecta mudança de p_off/p_def
    # nas abas de força, mesmo que os nomes não mudem).
    def _defaults_hash(d: dict) -> str:
        vals = sorted((f"{k[0]}:{k[1]}:{v:.4f}" for k, v in d.items()))
        return str(hash(tuple(vals)) & 0xFFFFFFFF)

    sig = (
        "|".join(n for _, n in attacker_roles_active) + "@" +
        "|".join(n for _, n in defender_roles_active) + "#" +
        _defaults_hash(defaults_off) + _defaults_hash(defaults_def)
    )
    c1, c2 = st.columns(2)
    with c1:
        st.caption("Poder ofensivo p_offense (hits/atacante/salva)")
        df_off = st.data_editor(
            _df_from_defaults(defaults_off),
            key=f"poff_{direction}_{sig}",
            use_container_width=True,
            num_rows="fixed",
        )
    with c2:
        st.caption("Poder defensivo p_defense (interceptações/defensor/salva)")
        df_def = st.data_editor(
            _df_from_defaults(defaults_def),
            key=f"pdef_{direction}_{sig}",
            use_container_width=True,
            num_rows="fixed",
        )

    out_off: dict[tuple[str, str], float] = {}
    out_def: dict[tuple[str, str], float] = {}
    for atk_role, atk_name in attacker_roles_active:
        for def_role, def_name in defender_roles_active:
            out_off[(atk_name, def_name)] = float(df_off.loc[atk_name, def_name])
            out_def[(atk_name, def_name)] = float(df_def.loc[atk_name, def_name])
    return out_off, out_def


with tab_engagement:
    st.subheader("Matriz de engajamento")

    st.markdown(
        """
        Esta aba reúne os parâmetros que traduzem **como cada plataforma
        combate** — em oposição às abas anteriores, que apenas definem
        *quantas* plataformas existem e qual é a sua resistência. São
        duas grandezas por par ordenado (atacante → defensor):

        - **p_offense** — poder ofensivo: o número esperado de
          *acertos efetivos por unidade atacante por salva* contra um
          alvo da classe defensora. Incorpora, de forma composta, a
          quantidade de munição lançada, a precisão do sistema de armas
          e a probabilidade de penetrar a defesa pontual do alvo.
        - **p_defense** — poder defensivo: o número esperado de
          *interceptações por unidade defensora por salva* contra
          mísseis ou projéteis da classe atacante. Representa a
          capacidade de *área-denial* de cada plataforma.

        Esses dois parâmetros, junto com os coeficientes de efetividade
        η (mantidos em 1,0 por padrão) e a fração de fogo σ (definida
        pela política de *targeting*), compõem o **kernel da equação de
        salva** para cada par. A atribuição de perdas a cada salva
        segue:

        > **Perdas líquidas** ∝ max(0, Σⱼ σ · η · p_offense · Bⱼ
        > − Σⱼ σ · η · p_defense · Aᵢ) / *staying power*

        O sinal de `max(0, ·)` expressa o princípio fundamental do
        modelo: fogo não-interceptado causa perdas; fogo totalmente
        absorvido pela defesa não causa nenhuma.
        """
    )

    with st.expander("📐 Como interpretar e calibrar os valores", expanded=True):
        st.markdown(
            """
            #### O que cada célula representa

            Cada célula da tabela de p_offense corresponde a **uma
            linha de atacante e uma coluna de defensor**. O valor 1,5
            na célula (Fragata classe A → Destróier), por exemplo,
            significa que cada fragata azul lança, em uma salva,
            o equivalente a 1,5 acertos esperados sobre um destróier
            vermelho — antes de qualquer interceptação pelo destróier.

            Já a tabela de p_defense é lida ao contrário: o valor 1,0
            na célula (Fragata classe A, como *defensora*, contra o
            Destróier como *atacante*) significa que cada fragata azul
            intercepta, em média, 1,0 dos mísseis lançados por cada
            destróier vermelho em sua direção.

            ---

            #### Exemplo numérico — duelo simétrico

            Considere o par mais simples: 2 Fragatas Azuis (classe A,
            p_offense = 1,5; *staying* = 3,0) contra 2 Destróieres
            Vermelhos (p_offense = 2,0; *staying* = 4,0), com
            p_defense = 1,0 em ambos os lados e σ = η = 1.

            *Primeira salva (Vermelho ataca Azul):*

            > Fogo bruto = 2 × 2,0 = 4,0 acertos  
            > Defesa azul = 2 × 1,0 = 2,0 interceptações  
            > Fogo líquido = max(0 ; 4,0 − 2,0) = **2,0 hits**  
            > Perdas Azuis = 2,0 / 3,0 ≈ **0,67 fragatas**

            *Primeira salva (Azul ataca Vermelho):*

            > Fogo bruto = 2 × 1,5 = 3,0 acertos  
            > Defesa vermelha = 2 × 1,5 = 3,0 interceptações  
            > Fogo líquido = max(0 ; 3,0 − 3,0) = **0 hits**  
            > Perdas Vermelhas = **0**

            O resultado revela algo imediato: com a defesa vermelha
            de 1,5 e apenas 2 fragatas, o Azul não consegue penetrar
            a proteção do adversário nessa salva. Para mudar o
            resultado, o usuário deve ou aumentar p_offense do Azul
            (melhorar o sistema de armas), ou reduzir a quantidade de
            unidades vermelhas (targeting), ou incorporar o submarino
            (p_offense = 2,0 contra navios, sem equivalente na defesa
            antitorpedo do Vermelho).

            ---

            #### Exemplo numérico — assimetria aviação × superfície

            A aviação de ataque vermelha tem p_offense = 2,5 contra
            FPSOs e staying = 1,0. Com 4 aeronaves e uma FPSO com
            staying = 6,0 e sem defesa ativa (p_defense = 0):

            > Fogo bruto = 4 × 2,5 = 10,0 acertos  
            > Defesa = 0  
            > Perdas FPSO por salva = 10,0 / 6,0 ≈ **1,67 plataformas**

            4 FPSOs sobrevivem menos de 3 salvas. Se o usuário
            acrescentar uma Fragata Azul interceptando a aviação
            (p_defense = 1,0 contra aeronaves, com σ proporcional):

            > Defesa = 1 × 1,0 = 1,0 interceptação  
            > Fogo líquido = 10,0 − 1,0 = 9,0 hits  
            > Perdas = 9,0 / 6,0 ≈ **1,5 plataformas por salva**

            A redução marginal é pequena porque 1 fragata não
            consegue saturar a aviação de 4 aeronaves. Esse
            resultado ilustra por que o modelo não tem mecânica de
            *area defense*: p_defense da fragata não se estende às
            FPSOs vizinhas — cada par é avaliado independentemente.

            ---

            #### Faixas de referência usuais

            | Tipo de interação | p_offense | p_defense |
            |---|---|---|
            | Navio de superfície vs. navio de superfície | 1,0 – 3,0 | 0,5 – 2,0 |
            | Submarino vs. navio de superfície | 1,5 – 3,0 | 0 (ASW baixo) |
            | Navio de superfície vs. aeronave | 0,3 – 1,0 | 0,5 – 1,5 |
            | Aviação de ataque vs. navio de superfície | 1,5 – 3,0 | 0,1 – 0,5 |
            | Aviação de ataque vs. FPSO | 2,0 – 4,0 | 0 (alvo fixo) |
            | Bateria costeira vs. navio de superfície | 1,0 – 2,5 | 0,2 – 0,8 |
            | Submarino vs. aeronave | **0** (admissibilidade nula) | — |

            Valores de p_offense acima de 4,0 raramente fazem sentido
            físico para plataformas individuais e normalmente indicam
            que o parâmetro está absorvendo um fator de escala que
            deveria estar na quantidade de unidades. Valores abaixo
            de 0,1 produzem fogo praticamente inefetivo e podem ser
            substituídos por 0 sem impacto visível nos resultados.

            Para p_defense, o limite superior prático é o valor de
            p_offense do atacante: uma defesa que intercepta mais
            projéteis do que são lançados não tem significado
            operacional — o modelo absorve esse excesso pelo
            `max(0, ·)`, mas o parâmetro perde sentido intuitivo.
            Uma boa regra de bolso é manter p_defense ≤ 0,6 ×
            p_offense do atacante correspondente como ponto de
            partida para plataformas modernas em alto risco de
            saturação de defesa.
            """
        )

    with st.expander("⚠️ Avisos sobre pares com admissibilidade zero"):
        st.markdown(
            """
            Alguns pares são **estruturalmente nulos** pela matriz de
            admissibilidade canônica e não produzem atrito
            independentemente do p_offense informado:

            - **Submarino → Aeronave**: torpedos não atingem aeronaves
              em voo. Qualquer valor de p_offense nesse par será
              multiplicado por χ = 0 e não terá efeito. O valor 0,0
              pré-preenchido reflete isso.
            - **Unidades cibernéticas → Submarino**: por decisão
              doutrinária do modelo, o domínio cibernético não atinge
              submarinos. Os pares cyber-submarino são zerados
              automaticamente pelo engine e não aparecem nesta tabela.
            - **Unidades cinéticas → capacidade cibernética**: no
              modelo atual, plataformas cinéticas não atritam
              diretamente a capacidade cibernética do oponente. O
              confronto cibernético ocorre apenas intra-domínio
              (cyber vs. cyber) e é configurado na aba de forças
              através das quantidades por sub-tipo.

            Editar esses pares não produz erro, mas também não
            produz efeito — o resultado será idêntico ao de manter
            o valor em 0,0.
            """
        )

    st.markdown(
        "Os valores abaixo são pré-preenchidos com os defaults da "
        "tabela de referência acima e podem ser editados célula a "
        "célula. Pares envolvendo o domínio cibernético são "
        "preenchidos automaticamente e não aparecem aqui."
    )

    blue_active_roles = [
        (r[3], blue_role_to_name[r[3]])
        for r in blue_resolved if r[0].quantity > 0
    ]
    red_active_roles = [
        (r[3], red_role_to_name[r[3]])
        for r in red_resolved if r[0].quantity > 0
    ]

    p_off_bar, p_def_bar = _build_throughput_grid(
        "Azul → Vermelho",
        blue_active_roles, red_active_roles,
        DEFAULT_P_OFF_BAR, DEFAULT_P_DEF_BAR,
    )
    st.markdown("---")
    p_off_rab, p_def_rab = _build_throughput_grid(
        "Vermelho → Azul",
        red_active_roles, blue_active_roles,
        DEFAULT_P_OFF_RAB, DEFAULT_P_DEF_RAB,
    )


# ---------------------------------------------------------------------------
# Adiciona pares cyber-vs-cyber e cyber-vs-kinético com defaults razoáveis.
# ---------------------------------------------------------------------------

def _add_cyber_pairs(
    p_off: dict[tuple[str, str], float],
    p_def: dict[tuple[str, str], float],
    attackers: list[UnitType],
    defenders: list[UnitType],
) -> None:
    """Preenche, in-place, os pares envolvendo o domínio cyber com defaults.

    - Cyber vs Cyber: p_off = CYBER_DEFAULT_P_OFF; p_def = CYBER_DEFAULT_P_DEF.
    - Cyber vs kinético (não submarino): p_off pequeno (0.1) -- o efeito
      principal vem via modulador Φ, não da atrição direta. p_def = 0.
    - Cyber vs submarino: 0 (imune).
    - Kinético vs cyber: 0 (não atritam cyber diretamente neste modelo).
    """
    for atk in attackers:
        for dfn in defenders:
            key = (atk.name, dfn.name)
            if key in p_off:
                continue
            if atk.domain is Domain.CYBER and dfn.domain is Domain.CYBER:
                p_off[key] = CYBER_DEFAULT_P_OFF
                p_def[key] = CYBER_DEFAULT_P_DEF
            elif atk.domain is Domain.CYBER and dfn.domain is Domain.UNDERWATER:
                p_off[key] = 0.0
                p_def[key] = 0.0
            elif atk.domain is Domain.CYBER:
                p_off[key] = 0.1
                p_def[key] = 0.0
            else:
                # kinético vs cyber: 0
                p_off[key] = 0.0
                p_def[key] = 0.0


# ---------------------------------------------------------------------------
# Wrapper para Manual com duas matrizes (uma por direção).
#
# A classe Manual do pacote guarda *uma* matriz σ.  Como apply_targeting_
# policy e EngagementBuilder usam a mesma instância da política nas duas
# direções (Blue->Red e Red->Blue), precisamos despachar a matriz certa
# em função da identidade do atacante.  Faço isso por comparação do
# label da Force (Force.label == "Blue" / "Red").
# ---------------------------------------------------------------------------


class _DirectionAwareManual(TargetingPolicy):
    """Manual com matrizes σ separadas para B→R e R→B.

    Despacha pela ``label`` do atacante. Compatível com
    ``apply_targeting_policy`` e ``run_campaign`` (que chamam ``compute``
    nas duas direções).
    """

    def __init__(
        self,
        blue_label: str,
        red_label: str,
        sigma_off_bar: np.ndarray,
        sigma_def_bar: np.ndarray,
        sigma_off_rab: np.ndarray,
        sigma_def_rab: np.ndarray,
    ) -> None:
        self.blue_label = blue_label
        self.red_label = red_label
        # Validações: entradas em [0, 1] e finitas.
        for name, m in [
            ("sigma_off_bar", sigma_off_bar),
            ("sigma_def_bar", sigma_def_bar),
            ("sigma_off_rab", sigma_off_rab),
            ("sigma_def_rab", sigma_def_rab),
        ]:
            arr = np.asarray(m, dtype=np.float64)
            if np.any(arr < 0.0) or np.any(arr > 1.0) or not np.all(np.isfinite(arr)):
                raise ValueError(
                    f"{name} deve ter valores em [0, 1] e finitos."
                )
        self.sigma_off_bar = np.asarray(sigma_off_bar, dtype=np.float64)
        self.sigma_def_bar = np.asarray(sigma_def_bar, dtype=np.float64)
        self.sigma_off_rab = np.asarray(sigma_off_rab, dtype=np.float64)
        self.sigma_def_rab = np.asarray(sigma_def_rab, dtype=np.float64)

    def compute(self, attacker, defender, admissibility):
        if attacker.label == self.blue_label:
            return self.sigma_off_bar.copy(), self.sigma_def_bar.copy()
        if attacker.label == self.red_label:
            return self.sigma_off_rab.copy(), self.sigma_def_rab.copy()
        raise ValueError(
            f"Atacante desconhecido: label={attacker.label!r}; "
            f"esperava {self.blue_label!r} ou {self.red_label!r}."
        )


# ---------------------------------------------------------------------------
# Aba: Targeting -- configura a política escolhida.
# ---------------------------------------------------------------------------

# A instância da política é construída aqui, com base no que o usuário
# preencher nesta aba (se a política tiver parâmetros). É consumida na
# aba 'Resultados'.
targeting_policy: Optional[TargetingPolicy] = None

with tab_targeting:
    st.subheader(f"Política de targeting: {targeting_policy_name}")

    blue_active_names = [
        blue_role_to_name[r[3]] for r in blue_resolved if r[0].quantity > 0
    ]
    red_active_names = [
        red_role_to_name[r[3]] for r in red_resolved if r[0].quantity > 0
    ]
    # Inclui cyber (eles aparecem como UnitType com domínio CYBER).
    blue_active_names_full = [ut.name for ut in blue_unit_types]
    red_active_names_full = [ut.name for ut in red_unit_types]

    if targeting_policy_name == "StrengthProportional":
        st.info(
            "σ é redistribuída antes de cada salva, proporcionalmente à "
            "força *atual* de cada alvo admissível. **Não há parâmetros "
            "para editar nesta política.**"
        )
        targeting_policy = StrengthProportional()

    elif targeting_policy_name == "Uniform":
        st.info(
            "Cada atacante divide sua salva igualmente entre todos os "
            "alvos doutrinariamente admissíveis (admissibilidade > 0). "
            "**Não há parâmetros para editar nesta política.**"
        )
        targeting_policy = Uniform()

    elif targeting_policy_name == "ThreatWeighted":
        st.markdown(
            "Defina pesos por classe defensora. Pesos maiores atraem "
            "mais fogo. Os pesos não precisam somar 1 — são "
            "normalizados internamente. Útil, p.ex., para colocar peso "
            "alto nos FPSOs."
        )
        st.caption(
            "Sigma_offense (j → i) ∝ peso(i). "
            "Sigma_defense usa pesos por atacante (todos = 1 por default)."
        )

        col_blue, col_red = st.columns(2)

        # Pesos sobre defensores VERMELHOS (quando Azul ataca):
        with col_blue:
            st.markdown(
                "**Pesos das defensoras Vermelhas (alvos para o Azul)**"
            )
            w_red_defenders: list[float] = []
            for nm in red_active_names_full:
                default = 1.0
                w = st.number_input(
                    nm, min_value=0.0, value=default, step=0.5,
                    format="%.2f",
                    key=f"tw_blue_targets_{nm}",
                    help="Peso desta classe como alvo para a força Azul."
                )
                w_red_defenders.append(float(w))

        # Pesos sobre defensores AZUIS (quando Vermelho ataca) -- FPSOs default 3.
        with col_red:
            st.markdown(
                "**Pesos das defensoras Azuis (alvos para o Vermelho)**"
            )
            w_blue_defenders: list[float] = []
            for nm in blue_active_names_full:
                # Default alto para FPSO (3.0); restante 1.0.
                default = 3.0 if "FPSO" in nm.upper() else 1.0
                w = st.number_input(
                    nm, min_value=0.0, value=default, step=0.5,
                    format="%.2f",
                    key=f"tw_red_targets_{nm}",
                    help="Peso desta classe como alvo para a força Vermelha."
                )
                w_blue_defenders.append(float(w))

        # Importante: ThreatWeighted é uma política simétrica em código:
        # ela aplica os mesmos pesos *nos dois usos*, com o vetor cujo
        # tamanho bate com o defensor.  Como os defensores têm tamanhos
        # diferentes nas duas direções, precisamos de DUAS instâncias e
        # passá-las como _DirectionAwareThreatWeighted.  Para
        # simplificar, criamos um wrapper que despacha.

        class _DirectionAwareThreatWeighted(TargetingPolicy):
            def __init__(self, blue_label, red_label,
                         w_blue_targets, w_red_targets):
                self.blue_label = blue_label
                self.red_label = red_label
                self._tw_bar = ThreatWeighted(
                    offensive_weights=np.array(w_red_targets,
                                               dtype=np.float64),
                )
                self._tw_rab = ThreatWeighted(
                    offensive_weights=np.array(w_blue_targets,
                                               dtype=np.float64),
                )

            def compute(self, attacker, defender, admissibility):
                if attacker.label == self.blue_label:
                    return self._tw_bar.compute(attacker, defender,
                                                 admissibility)
                if attacker.label == self.red_label:
                    return self._tw_rab.compute(attacker, defender,
                                                 admissibility)
                raise ValueError(f"Atacante desconhecido: {attacker.label!r}")

        targeting_policy = _DirectionAwareThreatWeighted(
            blue_label="Blue", red_label="Red",
            w_blue_targets=w_blue_defenders,
            w_red_targets=w_red_defenders,
        )

    elif targeting_policy_name == "Manual":
        st.markdown(
            "Defina as matrizes σ_offense célula a célula. As linhas "
            "somam livremente — recomenda-se que cada linha some 1.0 "
            "(fração da salva), mas o pacote aceita valores em [0, 1] "
            "sem somar 1 (parcela do atacante que efetivamente "
            "engaja cada alvo). Os pares com admissibilidade zero "
            "ainda serão zerados pela própria matriz χ no engine."
        )
        st.caption(
            "σ_defense é deixada como zero (sem alocação defensiva "
            "manual). Caso queira modelar defesa ativa proporcional, "
            "use StrengthProportional."
        )

        def _sigma_editor(direction: str, atk_names: list[str],
                          def_names: list[str]) -> pd.DataFrame:
            if not atk_names or not def_names:
                st.info(f"{direction}: forças incompletas.")
                return pd.DataFrame()
            n_atk = len(atk_names)
            n_def = len(def_names)
            default = np.full((n_atk, n_def), 1.0 / max(n_def, 1),
                              dtype=np.float64)
            df = pd.DataFrame(default, index=atk_names, columns=def_names)
            sig = "|".join(atk_names) + "@" + "|".join(def_names)
            st.markdown(f"**{direction}**  σ_offense  (linhas: "
                        f"atacante; colunas: alvo)")
            return st.data_editor(
                df, key=f"sigma_off_{direction}_{sig}",
                use_container_width=True, num_rows="fixed",
            )

        df_sig_bar = _sigma_editor(
            "Azul → Vermelho",
            blue_active_names_full, red_active_names_full,
        )
        st.markdown("---")
        df_sig_rab = _sigma_editor(
            "Vermelho → Azul",
            red_active_names_full, blue_active_names_full,
        )

        if not df_sig_bar.empty and not df_sig_rab.empty:
            try:
                targeting_policy = _DirectionAwareManual(
                    blue_label="Blue", red_label="Red",
                    sigma_off_bar=df_sig_bar.to_numpy(dtype=np.float64),
                    sigma_def_bar=np.zeros_like(
                        df_sig_bar.to_numpy(dtype=np.float64)
                    ),
                    sigma_off_rab=df_sig_rab.to_numpy(dtype=np.float64),
                    sigma_def_rab=np.zeros_like(
                        df_sig_rab.to_numpy(dtype=np.float64)
                    ),
                )
            except ValueError as e:
                st.error(f"Valores inválidos na matriz σ: {e}")
                targeting_policy = None


# ---------------------------------------------------------------------------
# Aba: Resultados
# ---------------------------------------------------------------------------

with tab_results:
    st.subheader("Resultado da simulação")

    # Validações mínimas antes de rodar (não usam st.stop() para não
    # afetar as outras abas).
    blue_total_kinetic = sum(
        inp.quantity for inp, dom, _, _ in blue_resolved if dom.is_kinetic
    )
    red_total_kinetic = sum(
        inp.quantity for inp, dom, _, _ in red_resolved if dom.is_kinetic
    )
    can_run = True
    if blue_total_kinetic == 0:
        st.error("A Força Azul precisa de pelo menos uma unidade cinética "
                 "(quantidade ≥ 1 em algum tipo de superfície, submarino "
                 "ou bateria costeira).")
        can_run = False
    if red_total_kinetic == 0:
        st.error("A Força Vermelha precisa de pelo menos uma unidade "
                 "cinética (superfície ou aviação de ataque).")
        can_run = False

    run_clicked = st.button(
        "▶️ Rodar simulação", type="primary",
        use_container_width=True, disabled=not can_run,
    )
    if run_clicked:

        # Completa as matrizes com os pares cyber.
        _add_cyber_pairs(p_off_bar, p_def_bar, blue_unit_types, red_unit_types)
        _add_cyber_pairs(p_off_rab, p_def_rab, red_unit_types, blue_unit_types)

        # Verifica que a política foi construída (em particular, Manual
        # exige que o usuário tenha preenchido a matriz na aba Targeting).
        if targeting_policy is None:
            st.error(
                "Política de targeting não disponível. Verifique a aba "
                "🎯 Targeting (especialmente se escolheu Manual)."
            )
            st.stop()

        # Para a política Manual, σ é estática: passamos no builder mas
        # não a refrescamos entre salvas (run_campaign com targeting_
        # policy=None usa as σ embutidas em params). Demais políticas
        # são semi-dinâmicas e refrescam a cada salva.
        is_manual = targeting_policy_name == "Manual"

        # Monta o engagement.
        builder = (
            EngagementBuilder()
            .with_blue(blue_unit_types, label="Blue")
            .with_red(red_unit_types,  label="Red")
            .with_throughput_blue_attacks_red(
                p_offense=p_off_bar, p_defense=p_def_bar,
            )
            .with_throughput_red_attacks_blue(
                p_offense=p_off_rab, p_defense=p_def_rab,
            )
            .with_targeting_policy(targeting_policy)
            .with_admissibility(Admissibility.canonical())
        )
        ep = builder.build()

        # Importa o BattleState do pacote para envolver as duas Forces.
        from naval_salvo.state import BattleState
        state = BattleState(blue=ep.blue, red=ep.red)

        modulator = ChannelPhi() if use_cyber else None

        with st.spinner("Rodando campanha multidomínio..."):
            traj = run_campaign(
                state, ep, builder.admissibility,
                n_salvos=n_salvos,
                targeting_policy=None if is_manual else targeting_policy,
                cyber_modulator=modulator,
                stop_on_combat_ineffective=stop_on_termination,
            )

        # ---- Sumário ----
        c1, c2, c3 = st.columns(3)
        c1.metric("Salvas executadas", traj.n_completed_salvos)
        c2.metric(
            "Encerramento antecipado",
            "Sim" if traj.terminated_early else "Não"
        )
        blue_final = traj.blue_strength_history[-1].sum()
        red_final = traj.red_strength_history[-1].sum()
        winner = "Empate" if (blue_final > 0 and red_final > 0) else (
            "Azul" if red_final == 0 and blue_final > 0 else (
                "Vermelho" if blue_final == 0 and red_final > 0 else "—"
            )
        )
        c3.metric("Lado remanescente", winner)

        # ---- Gráficos: trajetória por domínio ----
        st.markdown("#### Trajetórias por domínio")

        def _plot_side(side: str, color_map: dict[Domain, str]) -> go.Figure:
            dom_data = traj.total_strength_history_by_domain(side)
            fig = go.Figure()
            for d in Domain:
                arr = dom_data[d]
                if arr.max() == 0.0:
                    continue
                fig.add_trace(go.Scatter(
                    x=traj.times, y=arr,
                    mode="lines+markers",
                    name=f"{d.value} ({d.name.lower()})",
                    line=dict(color=color_map[d], width=2),
                ))
            fig.update_layout(
                xaxis_title="Salva k",
                yaxis_title="Força total no domínio",
                height=380,
                margin=dict(l=10, r=10, t=30, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02,
                            xanchor="right", x=1),
            )
            return fig

        color_map = {
            Domain.SURFACE:    "#1f4e79",
            Domain.UNDERWATER: "#3a8c8a",
            Domain.AIR:        "#cc8400",
            Domain.COASTAL:    "#a02020",
            Domain.CYBER:      "#7a52a0",
        }
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Força Azul (MB)**")
            st.plotly_chart(_plot_side("blue", color_map),
                            use_container_width=True)
        with c2:
            st.markdown("**Força Vermelha**")
            st.plotly_chart(_plot_side("red", color_map),
                            use_container_width=True)

        # ---- Tabela final por unidade ----
        st.markdown("#### Estado final por unidade")
        c1, c2 = st.columns(2)

        def _final_df(history: np.ndarray, unit_types: list[UnitType]) -> pd.DataFrame:
            return pd.DataFrame({
                "Unidade":    [ut.name for ut in unit_types],
                "Domínio":    [ut.domain.value for ut in unit_types],
                "Inicial":    [ut.initial_strength for ut in unit_types],
                "Final":      history[-1],
                "Perdas":     history[0] - history[-1],
                "% perdas":   [
                    (1 - history[-1, j] / history[0, j]) * 100
                    if history[0, j] > 0 else 0.0
                    for j in range(history.shape[1])
                ],
            }).round(2)

        with c1:
            st.markdown("**Azul**")
            st.dataframe(
                _final_df(traj.blue_strength_history, traj._blue_unit_types),
                use_container_width=True, hide_index=True,
            )
        with c2:
            st.markdown("**Vermelho**")
            st.dataframe(
                _final_df(traj.red_strength_history, traj._red_unit_types),
                use_container_width=True, hide_index=True,
            )

        # FPSO em destaque (se presente)
        fpso_idx = next(
            (j for j, ut in enumerate(traj._blue_unit_types)
             if "FPSO" in ut.name.upper() or
                (ut.subtype and ut.subtype.lower() == "pre-salt")),
            None,
        )
        if fpso_idx is not None:
            initial = traj.blue_strength_history[0, fpso_idx]
            final = traj.blue_strength_history[-1, fpso_idx]
            st.info(
                f"**FPSOs (Pré-Sal):** {final:.1f} de {initial:.0f} "
                f"plataformas sobreviveram "
                f"({(final / initial * 100 if initial else 0):.1f}%)."
            )
    else:
        st.caption(
            "Clique em **Rodar simulação** acima para executar a campanha "
            "com a configuração atual."
        )
