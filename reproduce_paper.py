"""
reproduce_paper.py
==================

Single-command generator of all figures and tables for the SBPO 2026
paper (Equação de Salva Multidominio).

Usage
-----

    python reproduce_paper.py [--out OUTPUT_DIR]

Default OUTPUT_DIR is ``./paper_artifacts``.

The script emits:

- ``fig1_admissibility.png``    Heat-map of the 5×5 admissibility matrix
                                 with the three encoding levels visible.
- ``fig2_trajectory_no_cyber.png``  Per-domain trajectory under the
                                 baseline scenario.
- ``fig3_cyber_comparison.png`` Headline: with vs without cyber (Blue
                                 kinetic survival).
- ``fig4_frigate_sensitivity.png``  FPSO survival vs n_frigates.
- ``fig5_submarine_sensitivity.png``  Salvo count vs submarine present.
- ``fig6_phi_sigmoid_curves.png``  Φ(R) curves for varying r0, k.
- ``fig7_cyber_subtype_breakdown.png``  Cyber stocks by sub-type over
                                          time.
- ``table1_parameters.csv``     Calibrated parameter inventory.
- ``table2_sensitivity_matrix.csv``  Combined cyber × submarine matrix.
- ``table3_jph_recovery.csv``   Numerical JPH/Coronel reproduction
                                 for validation.

All figures use a consistent visual style (Seaborn-like grid, sober
colours suitable for a defence-OR audience, no chartjunk).
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

import naval_salvo as ns
from naval_salvo import (
    Admissibility,
    BACIA_CAMPOS_PARAMETERS,
    BaciaCamposConfig,
    ChannelPhi,
    Domain,
    StrengthProportional,
    build_bacia_campos,
    canonical_matrix,
    phi_sigmoid,
    run_campaign,
    salvo_step,
)
from naval_salvo.validation import (
    HughesScenario,
    build_coronel_engagement,
    build_hughes_homogeneous_engagement,
    coronel_minute_one_targeting,
    hughes_analytical,
    jph_minute_one_delta_good_hope,
    BRITISH_GROUPS,
)


# ---------------------------------------------------------------------------
# Visual style
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "legend.fontsize": 9,
    "legend.frameon": False,
    "figure.dpi": 130,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})

# Sober, accessible colour scheme.
COLORS = {
    "blue": "#1f4e79",
    "red":  "#a02020",
    "amber": "#cc8400",
    "teal": "#3a8c8a",
    "grey": "#666666",
    "domain": {
        Domain.SURFACE:    "#1f4e79",
        Domain.UNDERWATER: "#3a8c8a",
        Domain.AIR:        "#cc8400",
        Domain.COASTAL:    "#a02020",
        Domain.CYBER:      "#7a52a0",
    },
}


# ---------------------------------------------------------------------------
# Figure 1 -- Admissibility matrix
# ---------------------------------------------------------------------------


def figure_admissibility(out_path: Path, chi: float = 0.5) -> None:
    """5×5 admissibility heat-map with annotated cell values."""
    M = canonical_matrix(chi=chi)
    domain_codes = [d.value for d in Domain]

    fig, ax = plt.subplots(figsize=(5.0, 4.5))
    im = ax.imshow(M, cmap="YlGnBu", vmin=0.0, vmax=1.0, aspect="equal")
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels(domain_codes, fontsize=11)
    ax.set_yticklabels(domain_codes, fontsize=11)
    ax.set_xlabel("Defensor (alvo)", fontsize=11, labelpad=8)
    ax.set_ylabel("Atacante", fontsize=11, labelpad=8)
    ax.set_title(
        f"Matriz de admissibilidade $\\mathbb{{1}}^{{(d',d)}}$  "
        f"(χ = {chi})",
        fontsize=11
    )

    # Annotate each cell.
    for i in range(5):
        for j in range(5):
            value = M[i, j]
            if value == 0.0:
                txt = "0"
            elif value == 1.0:
                txt = "1"
            else:
                txt = f"{value:.2f}"
            color = "white" if value > 0.55 else "black"
            ax.text(j, i, txt, ha="center", va="center",
                    color=color, fontsize=11)

    # Highlight Bacia de Campos pairs (S↔S, S↔A, A↔S, A↔U,
    # X↔[S,A,C], etc.) with a thin border.
    bacia_pairs = [
        (Domain.SURFACE.index, Domain.SURFACE.index),
        (Domain.SURFACE.index, Domain.AIR.index),
        (Domain.AIR.index,     Domain.SURFACE.index),
        (Domain.AIR.index,     Domain.UNDERWATER.index),
        (Domain.UNDERWATER.index, Domain.SURFACE.index),
    ]
    for (i, j) in bacia_pairs:
        rect = plt.Rectangle((j - 0.48, i - 0.48), 0.96, 0.96,
                             fill=False, edgecolor="#cc0000", lw=1.5,
                             linestyle=":")
        ax.add_patch(rect)

    cbar = fig.colorbar(im, ax=ax, shrink=0.78)
    cbar.set_label("$\\mathbb{1}^{(d',d)}$", fontsize=10)
    fig.savefig(out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2 -- Per-domain trajectory under baseline (no cyber)
# ---------------------------------------------------------------------------


def figure_trajectory_baseline(out_path: Path) -> None:
    """Per-domain stack of strengths over salvos in the baseline."""
    cfg = BaciaCamposConfig(
        n_frigates=2, submarine_present=True,
        blue_cyber_per_subtype=0, red_cyber_per_subtype=0,
    )
    state, ep, adm = build_bacia_campos(cfg)
    traj = run_campaign(state, ep, adm, n_salvos=15,
                        targeting_policy=StrengthProportional())

    blue_dom = traj.total_strength_history_by_domain("blue")
    red_dom = traj.total_strength_history_by_domain("red")
    times = traj.times

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), sharey=False)
    for ax, dom_data, side_label in [
        (axes[0], blue_dom, "MB (Azul)"),
        (axes[1], red_dom,  "Adversário (Vermelho)"),
    ]:
        for d in [Domain.SURFACE, Domain.UNDERWATER,
                  Domain.AIR, Domain.COASTAL, Domain.CYBER]:
            arr = dom_data[d]
            if arr.max() == 0.0:
                continue  # don't plot empty series
            ax.plot(times, arr, marker="o", markersize=4,
                    color=COLORS["domain"][d], linewidth=1.8,
                    label=f"{d.value} ({d.name.lower()})")
        ax.set_xlabel("Salva  $k$")
        ax.set_ylabel("Força total no domínio")
        ax.set_title(side_label)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.legend(loc="upper right")

    fig.suptitle("Trajetórias por domínio — cenário baseline (sem cyber)",
                 fontsize=12, y=1.02)
    fig.savefig(out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3 -- Cyber comparison (with vs without)
# ---------------------------------------------------------------------------


def figure_cyber_comparison(out_path: Path) -> None:
    """Headline figure: Blue total-kinetic survival with vs without cyber."""

    def _run(blue_cyber: int, red_cyber: int,
             modulator) -> ns.CampaignTrajectory:
        cfg = BaciaCamposConfig(
            n_frigates=1, submarine_present=True,
            blue_cyber_per_subtype=blue_cyber,
            red_cyber_per_subtype=red_cyber,
        )
        state, ep, adm = build_bacia_campos(cfg)
        return run_campaign(state, ep, adm, n_salvos=15,
                            targeting_policy=StrengthProportional(),
                            cyber_modulator=modulator)

    runs = [
        ("Sem cibernético",         _run(0, 0, None),                 COLORS["grey"]),
        ("Cibernético simétrico",   _run(2, 2, ChannelPhi()),         COLORS["teal"]),
        ("Domínio cibernético Vermelho", _run(0, 3, ChannelPhi()),    COLORS["red"]),
        ("Domínio cibernético Azul", _run(3, 0, ChannelPhi()),        COLORS["blue"]),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))

    # Panel A: Blue total kinetic stock over time.
    ax = axes[0]
    for label, traj, color in runs:
        blue_kin = sum(
            traj.total_strength_history_by_domain("blue")[d]
            for d in [Domain.SURFACE, Domain.UNDERWATER,
                      Domain.AIR, Domain.COASTAL]
        )
        ax.plot(traj.times, blue_kin, marker="o", markersize=4,
                color=color, linewidth=1.8, label=label)
    ax.set_xlabel("Salva  $k$")
    ax.set_ylabel("Força cinética total Azul (MB)")
    ax.set_title("(a) Sobrevivência cinética Azul")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right", fontsize=8)

    # Panel B: Red total kinetic stock over time.
    ax = axes[1]
    for label, traj, color in runs:
        red_kin = sum(
            traj.total_strength_history_by_domain("red")[d]
            for d in [Domain.SURFACE, Domain.UNDERWATER,
                      Domain.AIR, Domain.COASTAL]
        )
        ax.plot(traj.times, red_kin, marker="o", markersize=4,
                color=color, linewidth=1.8, label=label)
    ax.set_xlabel("Salva  $k$")
    ax.set_ylabel("Força cinética total Vermelha")
    ax.set_title("(b) Sobrevivência cinética Vermelha")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right", fontsize=8)

    fig.suptitle(
        "Impacto da modulação cibernética (Φ canônico, paper eqs. 12-13)",
        fontsize=12, y=1.02,
    )
    fig.savefig(out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4 -- Frigate sensitivity (FPSO survival)
# ---------------------------------------------------------------------------


def figure_frigate_sensitivity(out_path: Path) -> None:
    """FPSO survival vs n_frigates, with and without submarine."""
    n_frigates_range = list(range(0, 5))
    fpso_with_sub = []
    fpso_no_sub = []
    salvos_with_sub = []
    salvos_no_sub = []

    for nf in n_frigates_range:
        for has_sub, fpso_list, salvos_list in [
            (True,  fpso_with_sub, salvos_with_sub),
            (False, fpso_no_sub,   salvos_no_sub),
        ]:
            cfg = BaciaCamposConfig(
                n_frigates=nf, submarine_present=has_sub,
                blue_cyber_per_subtype=0, red_cyber_per_subtype=0,
            )
            state, ep, adm = build_bacia_campos(cfg)
            traj = run_campaign(state, ep, adm, n_salvos=20,
                                targeting_policy=StrengthProportional())
            blue_names = [ut.name for ut in traj._blue_unit_types]
            fpso_idx = blue_names.index("FPSO") if "FPSO" in blue_names else None
            fpso_surv = (
                traj.blue_strength_history[-1, fpso_idx]
                if fpso_idx is not None else 0.0
            )
            fpso_list.append(fpso_surv)
            salvos_list.append(traj.n_completed_salvos)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))

    ax = axes[0]
    ax.plot(n_frigates_range, fpso_with_sub, marker="o", markersize=7,
            color=COLORS["blue"], linewidth=2,
            label="com submarino")
    ax.plot(n_frigates_range, fpso_no_sub, marker="s", markersize=7,
            color=COLORS["amber"], linewidth=2,
            label="sem submarino")
    ax.set_xlabel("Número de fragatas defensoras")
    ax.set_ylabel("FPSOs sobreviventes (de 4)")
    ax.set_title("(a) Sobrevivência dos ativos do Pré-Sal")
    ax.set_xticks(n_frigates_range)
    ax.set_ylim(-0.1, 4.2)
    ax.legend(loc="upper left")

    ax = axes[1]
    ax.plot(n_frigates_range, salvos_with_sub, marker="o", markersize=7,
            color=COLORS["blue"], linewidth=2,
            label="com submarino")
    ax.plot(n_frigates_range, salvos_no_sub, marker="s", markersize=7,
            color=COLORS["amber"], linewidth=2,
            label="sem submarino")
    ax.set_xlabel("Número de fragatas defensoras")
    ax.set_ylabel("Salvas até término")
    ax.set_title("(b) Duração da campanha")
    ax.set_xticks(n_frigates_range)
    ax.legend(loc="upper left")

    fig.suptitle("Sensibilidade ao dimensionamento da escolta de superfície",
                 fontsize=12, y=1.02)
    fig.savefig(out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 5 -- Submarine sensitivity
# ---------------------------------------------------------------------------


def figure_submarine_sensitivity(out_path: Path) -> None:
    """
    The story this figure tells: under varying Red cyber pressure,
    the submarine's residual fraction *after one salvo* is invariant
    (paper §4.3 immunity), while the frigate's residual fraction
    collapses to zero.  We visualise this with two panels:

    (a) Trajectories of submarine and frigate under no cyber
        vs Red cyber dominance -- the qualitative immunity comparison.
    (b) Salvo-1 residual fraction of submarine vs frigate as a
        function of Red cyber pressure -- the quantitative story.
    """
    cyber_levels = [0, 1, 2, 3]   # Red cyber per sub-type

    # ---- Panel (b) data ----
    sub_residual_s1 = []      # submarine residual fraction after 1 salvo
    frig_residual_s1 = []     # frigate residual fraction after 1 salvo

    for cred in cyber_levels:
        cfg = BaciaCamposConfig(
            n_frigates=1, submarine_present=True,
            blue_cyber_per_subtype=0, red_cyber_per_subtype=cred,
        )
        state, ep, adm = build_bacia_campos(cfg)
        modulator = ChannelPhi() if cred > 0 else None
        traj = run_campaign(state, ep, adm, n_salvos=1,
                            targeting_policy=StrengthProportional(),
                            cyber_modulator=modulator,
                            stop_on_combat_ineffective=False)
        bnames = [ut.name for ut in traj._blue_unit_types]
        sub_idx = bnames.index("Submarine")
        frig_idx = bnames.index("Frigate")
        sub_residual_s1.append(
            traj.blue_strength_history[1, sub_idx] /
            traj.blue_strength_history[0, sub_idx]
        )
        frig_residual_s1.append(
            traj.blue_strength_history[1, frig_idx] /
            traj.blue_strength_history[0, frig_idx]
        )

    # ---- Panel (a) data: trajectories for cred=0 vs cred=3 ----
    traj_data = {}
    for cred in [0, 3]:
        cfg = BaciaCamposConfig(
            n_frigates=1, submarine_present=True,
            blue_cyber_per_subtype=0, red_cyber_per_subtype=cred,
        )
        state, ep, adm = build_bacia_campos(cfg)
        modulator = ChannelPhi() if cred > 0 else None
        traj = run_campaign(state, ep, adm, n_salvos=4,
                            targeting_policy=StrengthProportional(),
                            cyber_modulator=modulator,
                            stop_on_combat_ineffective=False)
        bnames = [ut.name for ut in traj._blue_unit_types]
        traj_data[cred] = {
            "times": traj.times,
            "sub":  traj.blue_strength_history[:, bnames.index("Submarine")],
            "frig": traj.blue_strength_history[:, bnames.index("Frigate")],
        }

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))

    # Panel (a): trajectories.
    ax = axes[0]
    ax.plot(traj_data[0]["times"], traj_data[0]["sub"],
            marker="o", markersize=6, color=COLORS["teal"],
            linewidth=2, label="Submarino — sem cyber")
    ax.plot(traj_data[3]["times"], traj_data[3]["sub"],
            marker="o", markersize=6, color=COLORS["teal"],
            linewidth=2, linestyle="--",
            label="Submarino — cyber Vermelho dominante")
    ax.plot(traj_data[0]["times"], traj_data[0]["frig"],
            marker="s", markersize=6, color=COLORS["red"],
            linewidth=2, label="Fragata — sem cyber")
    ax.plot(traj_data[3]["times"], traj_data[3]["frig"],
            marker="s", markersize=6, color=COLORS["red"],
            linewidth=2, linestyle="--",
            label="Fragata — cyber Vermelho dominante")
    ax.set_xlabel("Salva  $k$")
    ax.set_ylabel("Força residual")
    ax.set_title("(a) Trajetórias: imune vs. modulado")
    ax.set_xlim(left=0)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="upper right", fontsize=8)

    # Panel (b): salvo-1 residual fraction vs cyber pressure.
    x = np.array(cyber_levels)
    width = 0.35
    ax = axes[1]
    ax.bar(x - width/2, sub_residual_s1, width,
           color=COLORS["teal"], label="Submarino (imune ao Φ)")
    ax.bar(x + width/2, frig_residual_s1, width,
           color=COLORS["red"], label="Fragata (modulada por Φ)")
    ax.set_xlabel("Capacidade cibernética Vermelha (por sub-tipo)")
    ax.set_ylabel("Fração residual após 1ª salva")
    ax.set_title("(b) Robustez vs. pressão cibernética")
    ax.set_xticks(x)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right")
    # Annotate the submarine bars with their values to emphasise
    # the constancy (immunity).
    for xi, val in zip(x, sub_residual_s1):
        ax.text(xi - width/2, val + 0.02, f"{val:.2f}",
                ha="center", fontsize=8, color=COLORS["teal"])

    fig.suptitle(
        "Imunidade cibernética do submarino (paper §4.3): "
        "preservação estrutural sob pressão cibernética",
        fontsize=12, y=1.02,
    )
    fig.savefig(out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 6 -- Φ sigmoid curves (paper eq. 12)
# ---------------------------------------------------------------------------


def figure_phi_sigmoid(out_path: Path) -> None:
    """Φ(R) under varying r0 and k -- shows the sigmoid family."""
    R_grid = np.linspace(0, 5, 200)
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8))

    # Left panel: vary r0 with k=2.
    ax = axes[0]
    for r0, color in [(0.5, COLORS["blue"]), (1.0, COLORS["teal"]),
                      (2.0, COLORS["amber"])]:
        phi = [phi_sigmoid(R, r0=r0, k=2.0) for R in R_grid]
        ax.plot(R_grid, phi, color=color, linewidth=2,
                label=f"$r_0 = {r0}$")
    ax.axhline(0.5, color=COLORS["grey"], linestyle=":", linewidth=1)
    ax.set_xlabel("Razão cibernética $R^p$")
    ax.set_ylabel("$\\Phi^p(R)$")
    ax.set_title("(a) Sensibilidade a $r_0$  (com $k=2$)")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()

    # Right panel: vary k with r0=1.
    ax = axes[1]
    for k, color in [(1.0, COLORS["blue"]), (2.0, COLORS["teal"]),
                     (4.0, COLORS["amber"])]:
        phi = [phi_sigmoid(R, r0=1.0, k=k) for R in R_grid]
        ax.plot(R_grid, phi, color=color, linewidth=2,
                label=f"$k = {k}$")
    ax.axhline(0.5, color=COLORS["grey"], linestyle=":", linewidth=1)
    ax.axvline(1.0, color=COLORS["grey"], linestyle=":", linewidth=1)
    ax.set_xlabel("Razão cibernética $R^p$")
    ax.set_ylabel("$\\Phi^p(R)$")
    ax.set_title("(b) Sensibilidade a $k$  (com $r_0=1$)")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()

    fig.suptitle(
        "Família funcional $\\Phi^p(R) = 1 / [1 + (R/r_0)^k]$  "
        "(paper, equação 12)",
        fontsize=12, y=1.02,
    )
    fig.savefig(out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 7 -- Cyber sub-type breakdown over time
# ---------------------------------------------------------------------------


def figure_cyber_subtype_breakdown(out_path: Path) -> None:
    """Trajetória por sub-tipo cibernético, ambas as forças."""
    cfg = BaciaCamposConfig(
        n_frigates=1, submarine_present=True,
        blue_cyber_per_subtype=2, red_cyber_per_subtype=2,
    )
    state, ep, adm = build_bacia_campos(cfg)
    traj = run_campaign(state, ep, adm, n_salvos=10,
                        targeting_policy=StrengthProportional(),
                        cyber_modulator=ChannelPhi())
    blue_names = [ut.name for ut in traj._blue_unit_types]
    red_names = [ut.name for ut in traj._red_unit_types]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), sharey=True)
    for ax, names, hist, side in [
        (axes[0], blue_names, traj.blue_strength_history, "MB (Azul)"),
        (axes[1], red_names,  traj.red_strength_history,  "Adversário"),
    ]:
        cyber_indices = {
            ("Cyber-C2", "$y_\\delta$"):     [i for i, n in enumerate(names)
                                              if n == "Cyber-C2"],
            ("Cyber-SEN", "$y_\\sigma$"):   [i for i, n in enumerate(names)
                                              if n == "Cyber-SEN"],
            ("Cyber-WPN", "$y_\\rho$"):     [i for i, n in enumerate(names)
                                              if n == "Cyber-WPN"],
            ("Cyber-LOG", "$y_{def}$"):     [i for i, n in enumerate(names)
                                              if n == "Cyber-LOG"],
        }
        colors = [COLORS["blue"], COLORS["teal"],
                  COLORS["amber"], COLORS["red"]]
        for ((tech_name, latex), idxs), color in zip(
                cyber_indices.items(), colors):
            if not idxs:
                continue
            arr = hist[:, idxs[0]]
            ax.plot(traj.times, arr, marker="o", markersize=4,
                    color=color, linewidth=1.8,
                    label=f"{latex} ({tech_name.split('-')[1]})")
        ax.set_xlabel("Salva  $k$")
        ax.set_ylabel("Força do sub-tipo cibernético")
        ax.set_title(side)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.legend(loc="upper right", fontsize=8)

    fig.suptitle(
        "Atrição intra-domínio cibernético por sub-tipo  "
        "(equação 14 do paper)",
        fontsize=12, y=1.02,
    )
    fig.savefig(out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def _write_csv(path: Path, header: list, rows: list[list]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def table_parameters(out_path: Path) -> None:
    """Inventário de parâmetros calibrados (Tabela 1 do paper)."""
    header = ["Tipo de unidade", "Domínio", "ς (staying)",
              "p_offense", "p_defense", "η (eficácia)"]
    domain_for = {
        "Frigate":   "S (superfície)",
        "Submarine": "U (subsuperfície)",
        "MPA":       "A (aérea)",
        "FPSO":      "S (superfície, sub-tipo Pré-Sal)",
        "Destroyer": "S (superfície)",
        "StrikeAir": "A (aérea)",
        "Cyber-C2":  "X (ciber, $y_\\delta$)",
        "Cyber-SEN": "X (ciber, $y_\\sigma$)",
        "Cyber-WPN": "X (ciber, $y_\\rho$)",
        "Cyber-LOG": "X (ciber, $y_{def}$)",
    }
    rows = []
    for name, (s, p_off, p_def, eta) in BACIA_CAMPOS_PARAMETERS.items():
        rows.append([name, domain_for.get(name, "?"),
                     s, p_off, p_def, eta])
    _write_csv(out_path, header, rows)


def table_sensitivity_matrix(out_path: Path) -> None:
    """Matriz de sensibilidade combinada (cyber × submarino)."""
    rows = []
    header = ["Sub", "Cyber Azul", "Cyber Vermelho", "Modulador",
              "Salvas", "FPSO", "Fragatas", "Submarino", "Destroyer"]

    runs = [
        ("N", 0, 0, None,         "Sem cyber"),
        ("N", 2, 2, ChannelPhi(), "Simétrico (2,2)"),
        ("N", 1, 2, ChannelPhi(), "Assimétrico Vermelho (1,2)"),
        ("N", 0, 3, ChannelPhi(), "Domínio Vermelho (0,3)"),
        ("Y", 0, 0, None,         "Sem cyber"),
        ("Y", 2, 2, ChannelPhi(), "Simétrico (2,2)"),
        ("Y", 1, 2, ChannelPhi(), "Assimétrico Vermelho (1,2)"),
        ("Y", 0, 3, ChannelPhi(), "Domínio Vermelho (0,3)"),
    ]

    for sub_str, blue_x, red_x, modulator, label in runs:
        cfg = BaciaCamposConfig(
            n_frigates=1,
            submarine_present=(sub_str == "Y"),
            blue_cyber_per_subtype=blue_x,
            red_cyber_per_subtype=red_x,
        )
        state, ep, adm = build_bacia_campos(cfg)
        traj = run_campaign(state, ep, adm, n_salvos=15,
                            targeting_policy=StrengthProportional(),
                            cyber_modulator=modulator)
        bnames = [ut.name for ut in traj._blue_unit_types]
        rnames = [ut.name for ut in traj._red_unit_types]

        def _final(side_hist, names, target):
            if target in names:
                return float(side_hist[-1, names.index(target)])
            return None

        fpso = _final(traj.blue_strength_history, bnames, "FPSO")
        frig = _final(traj.blue_strength_history, bnames, "Frigate")
        sub  = _final(traj.blue_strength_history, bnames, "Submarine")
        dest = _final(traj.red_strength_history, rnames, "Destroyer")

        rows.append([
            sub_str, blue_x, red_x, label,
            traj.n_completed_salvos,
            f"{fpso:.2f}" if fpso is not None else "-",
            f"{frig:.2f}" if frig is not None else "-",
            f"{sub:.2f}"  if sub  is not None else "-",
            f"{dest:.2f}" if dest is not None else "-",
        ])

    _write_csv(out_path, header, rows)


def table_jph_recovery(out_path: Path) -> None:
    """Validação numérica: recuperação JPH/Hughes."""
    header = ["Cenário", "Métrica", "Valor analítico",
              "Valor do engine", "Δ (machine-precision)"]
    rows = []

    # Hughes 1995 homogeneous, balanced single salvo.
    scn = HughesScenario(A0=4, B0=4, alpha=2, beta=2,
                         z=1, y=1, w=2, x=2, n_salvos=1)
    A_an, B_an = hughes_analytical(scn)
    bs, ep, adm = build_hughes_homogeneous_engagement(scn)
    salvo_step(bs, ep, adm, apply=True)
    rows.append([
        "Hughes 1995 (1 salva, balanceado)",
        "Força A pós-salva",
        f"{A_an[1]:.6f}",
        f"{bs.red.strength_of('A'):.6f}",
        f"{abs(A_an[1] - bs.red.strength_of('A')):.2e}",
    ])
    rows.append([
        "Hughes 1995 (1 salva, balanceado)",
        "Força B pós-salva",
        f"{B_an[1]:.6f}",
        f"{bs.blue.strength_of('B'):.6f}",
        f"{abs(B_an[1] - bs.blue.strength_of('B')):.2e}",
    ])

    # JPH 2001 Coronel minute-1.
    bs_c, ep_c, adm_c = build_coronel_engagement(
        coronel_minute_one_targeting()
    )
    out = salvo_step(bs_c, ep_c, adm_c, apply=False)
    expected_per_ship = jph_minute_one_delta_good_hope()
    actual_per_ship = out.blue_losses[0] / BRITISH_GROUPS[0].n_ships
    rows.append([
        "JPH 2001 Coronel (1ª minuto, Good Hope)",
        "ΔA por navio",
        f"{expected_per_ship:.6f}",
        f"{actual_per_ship:.6f}",
        f"{abs(expected_per_ship - actual_per_ship):.2e}",
    ])

    _write_csv(out_path, header, rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproduce all paper figures and tables from a single command."
    )
    parser.add_argument("--out", default="paper_artifacts",
                        help="Output directory")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("Generating figures and tables in:", out.resolve())
    print()

    # Figures.
    print("[1/7] Figure 1 -- Admissibility matrix...")
    figure_admissibility(out / "fig1_admissibility.png")

    print("[2/7] Figure 2 -- Per-domain trajectory (baseline)...")
    figure_trajectory_baseline(out / "fig2_trajectory_no_cyber.png")

    print("[3/7] Figure 3 -- Cyber comparison (headline)...")
    figure_cyber_comparison(out / "fig3_cyber_comparison.png")

    print("[4/7] Figure 4 -- Frigate sensitivity...")
    figure_frigate_sensitivity(out / "fig4_frigate_sensitivity.png")

    print("[5/7] Figure 5 -- Submarine sensitivity...")
    figure_submarine_sensitivity(out / "fig5_submarine_sensitivity.png")

    print("[6/7] Figure 6 -- Phi sigmoid curves...")
    figure_phi_sigmoid(out / "fig6_phi_sigmoid_curves.png")

    print("[7/7] Figure 7 -- Cyber sub-type breakdown...")
    figure_cyber_subtype_breakdown(out / "fig7_cyber_subtype_breakdown.png")

    # Tables.
    print()
    print("Generating tables...")
    table_parameters(out / "table1_parameters.csv")
    table_sensitivity_matrix(out / "table2_sensitivity_matrix.csv")
    table_jph_recovery(out / "table3_jph_recovery.csv")

    print()
    print("Done.  Artifacts:")
    for p in sorted(out.iterdir()):
        size = p.stat().st_size
        print(f"  {p.name:50s} {size:>10,} bytes")


if __name__ == "__main__":
    main()
