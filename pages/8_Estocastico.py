# -*- coding: utf-8 -*-
"""Página Streamlit — Equação de Salva Estocástica com Ordem de Engajamento.

Extensão estocástica do modelo multidomínio (Armstrong 2005, 2014): fogos
ofensivos e interceptações seguem distribuições binomiais, o dano por
míssil segue distribuição normal, e a troca de fogo pode ser simultânea
ou sequencial (Azul primeiro / Vermelho primeiro).
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from naval_salvo.stochastic import (
    EngagementOrder, MultiDomainForce, NAVAL_DOMAINS, UnitGroup,
    run_multidomain_battle, phi,
)

st.set_page_config(page_title="Salva Estocástica", page_icon="🎲",
                   layout="wide")

st.title("🎲 Equação de Salva Estocástica Multidomínio")
st.markdown(
    "Extensão estocástica do modelo multidomínio com **ordem de engajamento** "
    "(Armstrong, 2005; 2014): fogos ofensivos e interceptações seguem "
    "distribuições binomiais, o dano por míssil segue distribuição normal, "
    "e a troca de fogo pode ser **simultânea** ou **sequencial** "
    "(Azul primeiro / Vermelho primeiro). No modo sequencial, o fogo de "
    "retorno é executado apenas pelos **sobreviventes** da primeira salva."
)
st.markdown(
    "Ambas as forças compartilham a **mesma paleta de plataformas** do "
    "cenário multidomínio da Bacia de Campos: duas classes de superfície, "
    "um submarino (submarino, imune a ciber), aviação de ataque, uma "
    "bateria costeira e um ativo de valor tipo FPSO. Ajuste a quantidade de "
    "um slot para **0** para deixá-lo de fora de uma força."
)

ORDER_LABELS = {
    "Simultâneo": EngagementOrder.SIMULTANEOUS,
    "Azul ataca primeiro": EngagementOrder.BLUE_FIRST,
    "Vermelho ataca primeiro": EngagementOrder.RED_FIRST,
}

# ---------------------------------------------------------------------------
# Sidebar — configuração da simulação
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Simulação")
    n_sim = st.select_slider("Nº de replicações Monte Carlo",
                             options=[1_000, 2_000, 5_000, 10_000, 20_000],
                             value=5_000)
    n_salvos = st.slider("Nº máximo de salvas", 1, 10, 3)
    seed = st.number_input("Semente aleatória", 0, 999_999, 42)
    compare_all = st.checkbox("Comparar as três ordens de engajamento",
                              value=True)
    if not compare_all:
        order_label = st.radio("Ordem de engajamento",
                               list(ORDER_LABELS.keys()))
    st.divider()
    st.header("📡 Domínio Ciber-EM")
    cyber_blue = st.slider("Intensidade ciber projetada por Azul",
                           0.0, 5.0, 0.0, 0.5)
    cyber_red = st.slider("Intensidade ciber projetada por Vermelho",
                          0.0, 5.0, 0.0, 0.5)
    k_cyber = st.slider("Inclinação k da sigmoide Φ", 0.1, 2.0, 0.5, 0.1)
    st.caption("Submarinos são imunes à degradação ciber (χ).")


DOMAIN_LABELS = {
    "s": "Superfície",
    "u": "Submarino",
    "a": "Aéreo",
    "c": "Costeiro",
}


def force_editor(defaults: list, key: str) -> list:
    """Editor de composição de força — um expander por slot de plataforma.

    Cada slot tem um domínio fixo (espelhando a paleta da Bacia de Campos).
    Ajustar a quantidade de um slot para 0 o deixa fora da força. Nomes são
    deduplicados dentro da força para permanecerem chaves de estoque válidas.
    """
    groups = []
    used_names: set[str] = set()
    for i, d in enumerate(defaults):
        dom = d["domain"]
        dom_label = DOMAIN_LABELS[dom]
        title = f"{d['slot']} — {d['name']} ({dom_label})"
        with st.expander(title, expanded=(i == 0)):
            note = f"Domínio: **{dom_label}** (`{dom}`)."
            if dom == "u":
                note += " Imune à degradação ciber (χ)."
            if dom not in NAVAL_DOMAINS:
                note += " Arma de apoio — não conta para o veredito naval."
            st.caption(note)
            c1, c2, c3 = st.columns(3)
            units = c1.number_input("Unidades", 0.0, 50.0, float(d["units"]),
                                    1.0, key=f"{key}_u{i}")
            n_off = c2.number_input("Mísseis ofensivos/unid.", 0, 24,
                                    d["n_off"], key=f"{key}_no{i}")
            p_off = c3.slider("p(acerto)", 0.0, 1.0, d["p_off"], 0.01,
                              key=f"{key}_po{i}")
            c4, c5, c6 = st.columns(3)
            n_def = c4.number_input("Interceptações/unid.", 0, 24,
                                    d["n_def"], key=f"{key}_nd{i}")
            p_def = c5.slider("p(interceptação)", 0.0, 1.0, d["p_def"], 0.01,
                              key=f"{key}_pd{i}")
            stay = c6.number_input("Staying power (acertos)", 1.0, 10.0,
                                   float(d["staying"]), 0.5,
                                   key=f"{key}_sp{i}")
            sigma_v = st.slider("σ do dano por míssil", 0.0, 0.5,
                                d["sigma_v"], 0.01, key=f"{key}_sv{i}")
            nm = d["name"]
            base, k = nm, 2
            while nm in used_names:
                nm = f"{base} ({k})"
                k += 1
            used_names.add(nm)
            groups.append(UnitGroup(
                name=nm, domain=dom, units=units,
                n_off=n_off, p_off=p_off, n_def=n_def, p_def=p_def,
                mu_v=1.0 / stay, sigma_v=sigma_v,
            ))
    return groups


# ---------------------------------------------------------------------------
# Paleta de plataformas — espelha o cenário multidomínio da Bacia de Campos:
# duas classes de superfície, um submarino, aviação de ataque, uma bateria
# costeira e um ativo de valor tipo FPSO. Cada lado pode compor a partir da
# paleta completa; os valores-padrão seguem o cenário da Bacia de Campos (um
# slot com quantidade 0 simplesmente fica de fora da força).
# ---------------------------------------------------------------------------
BLUE_DEFAULTS = [
    dict(slot="Superfície — Tipo 1", name="Fragata Classe A", domain="s", units=2,
         n_off=4, p_off=0.67, n_def=2, p_def=0.67, staying=3.0, sigma_v=0.11),
    dict(slot="Superfície — Tipo 2", name="Corveta/Patrulha", domain="s", units=2,
         n_off=2, p_off=0.60, n_def=1, p_def=0.50, staying=2.0, sigma_v=0.11),
    dict(slot="Submarino", name="Submarino convencional", domain="u", units=1,
         n_off=4, p_off=0.80, n_def=0, p_def=0.0, staying=2.0, sigma_v=0.15),
    dict(slot="Aviação de ataque", name="Aviação de ataque Azul", domain="a", units=0,
         n_off=2, p_off=0.75, n_def=0, p_def=0.0, staying=1.0, sigma_v=0.15),
    dict(slot="Bateria costeira", name="Bateria costeira", domain="c", units=1,
         n_off=3, p_off=0.65, n_def=0, p_def=0.0, staying=4.0, sigma_v=0.11),
    dict(slot="Ativo de valor (FPSO)", name="FPSO (Pré-Sal)", domain="s", units=4,
         n_off=0, p_off=0.0, n_def=0, p_def=0.0, staying=6.0, sigma_v=0.0),
]

RED_DEFAULTS = [
    dict(slot="Superfície — Tipo 1", name="Contratorpedeiro", domain="s", units=2,
         n_off=4, p_off=0.60, n_def=2, p_def=0.60, staying=4.0, sigma_v=0.11),
    dict(slot="Superfície — Tipo 2", name="Fragata avançada", domain="s", units=2,
         n_off=3, p_off=0.60, n_def=2, p_def=0.55, staying=3.0, sigma_v=0.11),
    dict(slot="Submarino", name="Submarino adversário", domain="u", units=0,
         n_off=4, p_off=0.80, n_def=0, p_def=0.0, staying=2.0, sigma_v=0.15),
    dict(slot="Aviação de ataque", name="Aviação de ataque Vermelha", domain="a", units=4,
         n_off=2, p_off=0.75, n_def=0, p_def=0.0, staying=1.0, sigma_v=0.15),
    dict(slot="Bateria costeira", name="Bateria costeira", domain="c", units=0,
         n_off=3, p_off=0.60, n_def=0, p_def=0.0, staying=4.0, sigma_v=0.11),
    dict(slot="Ativo de valor", name="Ativo de valor", domain="s", units=0,
         n_off=0, p_off=0.0, n_def=0, p_def=0.0, staying=6.0, sigma_v=0.0),
]


# ---------------------------------------------------------------------------
# Composição das forças
# ---------------------------------------------------------------------------
col_b, col_r = st.columns(2)
with col_b:
    st.subheader("🔵 Força Azul")
    blue_groups_all = force_editor(BLUE_DEFAULTS, key="blue")
with col_r:
    st.subheader("🔴 Força Vermelha")
    red_groups_all = force_editor(RED_DEFAULTS, key="red")

# Apenas slots com quantidade positiva entram na força.
blue_groups = [g for g in blue_groups_all if g.units > 0]
red_groups = [g for g in red_groups_all if g.units > 0]

blue = MultiDomainForce("Azul", blue_groups, cyber_offense=cyber_blue)
red = MultiDomainForce("Vermelho", red_groups, cyber_offense=cyber_red)

st.caption(
    "O veredito de vitória considera **apenas as forças navais** "
    "(superfície + submarino). Unidades aéreas e costeiras contribuem com "
    "fogo como armas de apoio, mas não decidem, por si sós, o resultado."
)

phi_b = phi(cyber_red - cyber_blue, k_cyber)
phi_r = phi(cyber_blue - cyber_red, k_cyber)
if phi_b < 1.0 or phi_r < 1.0:
    st.info(f"Modulação ciber ativa: Φ(Azul) = {phi_b:.3f} | "
            f"Φ(Vermelho) = {phi_r:.3f}")

# ---------------------------------------------------------------------------
# Execução
# ---------------------------------------------------------------------------
forces_ok = bool(blue_groups) and bool(red_groups)
if not forces_ok:
    st.error("Cada força precisa de ao menos um slot de plataforma com "
             "quantidade ≥ 1.")

if st.button("▶️ Executar simulação", type="primary", disabled=not forces_ok):
    orders = (list(ORDER_LABELS.items()) if compare_all
              else [(order_label, ORDER_LABELS[order_label])])

    results = {}
    prog = st.progress(0.0)
    for i, (lbl, order) in enumerate(orders):
        results[lbl] = run_multidomain_battle(
            blue, red, order, n_salvos=n_salvos, n_sim=n_sim,
            seed=int(seed), k_cyber=k_cyber,
        )
        prog.progress((i + 1) / len(orders))
    prog.empty()

    # --- Métricas de veredito -------------------------------------------
    st.subheader("Veredito (forças navais: superfície + submarino)")
    cols = st.columns(len(results))
    for c, (lbl, r) in zip(cols, results.items()):
        with c:
            st.markdown(f"**{lbl}**")
            st.metric("P(vitória Azul)", f"{100 * r.p_blue_win:.1f}%")
            st.metric("P(vitória Vermelho)", f"{100 * r.p_red_win:.1f}%")
            st.metric("P(indecisivo)", f"{100 * r.p_draw:.1f}%")

    # --- Distribuições de sobreviventes ---------------------------------
    st.subheader("Distribuição de sobreviventes navais")
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 3.5),
                             sharey=True, squeeze=False)
    for ax, (lbl, r) in zip(axes[0], results.items()):
        bins = np.linspace(0, max(r.blue_naval_final.max(),
                                  r.red_naval_final.max(), 1) + 0.5, 25)
        ax.hist(r.blue_naval_final, bins=bins, alpha=0.6, color="#1f77b4",
                label="Azul")
        ax.hist(r.red_naval_final, bins=bins, alpha=0.6, color="#d62728",
                label="Vermelho")
        ax.set_title(lbl, fontsize=10)
        ax.set_xlabel("Sobreviventes navais")
        ax.legend(fontsize=8)
    axes[0][0].set_ylabel("Frequência")
    fig.tight_layout()
    st.pyplot(fig)

    # --- Tabela-resumo ----------------------------------------------------
    st.subheader("Estatísticas")
    rows = []
    for lbl, r in results.items():
        s = r.summary()
        rows.append({
            "Ordem": lbl,
            "P(vit. Azul)": f"{100 * s['p_blue_win']:.1f}%",
            "P(vit. Verm.)": f"{100 * s['p_red_win']:.1f}%",
            "Naval Azul (média ± dp)":
                f"{s['blue_naval_mean']:.2f} ± {s['blue_naval_std']:.2f}",
            "Naval Verm. (média ± dp)":
                f"{s['red_naval_mean']:.2f} ± {s['red_naval_std']:.2f}",
            "Perda média Azul": f"{s['blue_loss_mean']:.2f}",
            "Perda média Verm.": f"{s['red_loss_mean']:.2f}",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True,
                 hide_index=True)

    if compare_all:
        st.caption(
            "Propriedade teórica (Armstrong, 2014): atirar primeiro é, em "
            "média, ao menos tão bom quanto o engajamento simultâneo, que é "
            "ao menos tão bom quanto atirar em segundo. Em replicações "
            "individuais, uma força 'sortuda' atirando em segundo pode "
            "superar uma 'azarada' atirando primeiro."
        )

st.divider()
with st.expander("📖 Formulação do modelo"):
    st.markdown("**Fogos ofensivos** por grupo atacante $g$ (estoque $U_g$):")
    st.latex(r"\mathrm{Off}_g \sim \mathrm{Binomial}\!\left(n_g\,U_g,\;"
             r"p_g^{\mathrm{off}}\cdot\Phi^{\sigma_g}\Phi^{\rho_g}\right)")
    st.markdown("**Interceptações** do grupo-alvo $j$:")
    st.latex(r"\mathrm{Def}_j \sim \mathrm{Binomial}\!\left(n_j^{\mathrm{def}}\,"
             r"U_j,\; p_j^{\mathrm{def}}\cdot\Phi^{\delta_j}\right)")
    st.markdown("**Dano** por míssil não interceptado, com perdas truncadas "
                "no estoque:")
    st.latex(r"\Delta U_j = \min\!\Big(\textstyle\sum_{k=1}^{\mathrm{Net}_j}"
             r"\max(0, v_k),\; U_j\Big),\quad v_k \sim \mathcal{N}(\mu_v,"
             r"\sigma_v^2)")
    st.markdown(
        "No **modo sequencial**, as perdas do defensor são aplicadas antes "
        "do fogo de retorno, executado pelos sobreviventes com poder de "
        "fogo proporcional ao estoque fracionário (Hausken–Moxnes). "
        "Submarinos são imunes à modulação Φ (matriz χ). O veredito "
        "considera apenas as forças navais (s + u)."
    )
