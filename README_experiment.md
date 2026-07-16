# LAFusion 2026 — Reproducibility Package

*The Operational Value of Information Fusion in Naval Salvo Combat: A Multi-Salvo
Data-Farming Study of Scouting Advantage, Force Design, and Engagement Posture.*

This is the self-contained data-farming experiment (spec sections 3–11). It is
independent of the Streamlit app in `naval_salvo/`; it depends only on the
numpy stack.

## One-command reproduction (spec sec 11)

```bash
python3 -m pip install -r requirements-lock.txt   # or: make deps
make all        # validation tests -> pilot farm -> figures/tables
make paper      # paper scale: 10,000 reps x 128 process points
```

`make all` runs, in order: the section-10 validation regression suite, the
data-farming run (design → simulation → metamodels), and the figure/table
generation. Every number is regenerable from fixed seeds.

## Reproducibility guarantees

| Item | Where |
|---|---|
| Engine version | `salvo_mds.ENGINE_VERSION == "1.0.0"` |
| Environment pin | `requirements-lock.txt` (Python 3.11.15) |
| Master seed | `farm.MASTER_SEED == 20260731` (submission-window open) |
| Per-design-point sub-seeds | recorded in `farm_design.csv` (`seed` column) |
| Deterministic design | mixture is fixed; process design is seeded LHS (or the NOB matrix via `--nob-path`) |

## Files

| File | Role |
|---|---|
| `salvo_mds.py` | Multi-salvo, per-hull stochastic engine (spec sec 3). Fixed definitions documented in the module header. |
| `farm.py` | Design generation (mixture × process × order), farming runner, metamodels (partition tree + OLS). Writes `farm_design.csv`, `farm_results.csv`. |
| `analyze.py` | Paper figures 1–5 + `tables_summary.md`. |
| `validate_demo.py` | Readable validation + demonstration narrative. |
| `test_validation.py` | Automated section-10 regression suite (7 tests). |
| `Makefile` | One-command reproduction targets. |
| `requirements-lock.txt` | Pinned environment. |

## Fixed modelling definitions (spec sec 9)

- **Effective combat power** (R1/R2/R3) = residual **offensive** throughput of
  surviving hulls, `Σ alive offense / Σ initial offense`. Set by
  `salvo_mds.CP_WEIGHTS = (1, 0, 0)` (weights over offense/defense/staying).
  Defensive capability is *not* folded in; it is reported separately as the
  residual-defense fraction (R9).
- **θ = 0.30** (`FIXED["theta"]`): a side at ≤30% of initial combat power is
  ineffective. **Victory (R3)** = Red ineffective **and** Blue viable.
- **T_max = 6** (`FIXED["Tmax"]`); **allocation** = uniform-with-replacement
  (spec sec 3). Both are documented settings, moved to robustness excursions,
  not primary factors.

## Response set (spec sec 9) → columns in `farm_results.csv`

| # | Response | Column(s) |
|---|---|---|
| R1 | Blue combat-power loss | `blue_loss_mean` |
| R2 | Red neutralization fraction | `red_loss_mean` |
| R3 | P(Blue victory) | `p_victory` |
| R4 | FER-analog (logit-transformed) | `logfer_mean`, `fer_geom` |
| R7 | Outcome variance | `blue_loss_sd` |
| R8 | Salvos-to-decision | `salvos_mean` |
| R9 | Capability-degradation profile | `blue_def_residual_mean`, `red_def_residual_mean`, `*_ships_lost_mean` |
| R10 | Overkill / wasted-damage fraction | `overkill_frac_mean` |

> **R4 note.** The raw `fer_mean` (mean of `red_loss/blue_loss` per rep) is a
> biased mean-of-ratios — it reads ~19 even in a perfectly symmetric mirror
> match. R4 uses the logit-transformed **log-FER**; `exp(logfer_mean) = fer_geom`
> is the geometric-mean FER (≈1 at parity). Do not report `fer_mean`.

## Figures (spec sections 1 & 9) — `analyze.py`

| Figure | File | Claim it supports |
|---|---|---|
| Fig 1 | `fig1_fusion_compounding.png` | Fusion value compounds with battle duration, then saturates (contribution #3). |
| Fig 2 | `fig2_partition_tree.png` | Partition-tree drivers of Red combat-power loss (sec 8 toolkit). |
| Fig 3 | `fig3_mixture_ternary.png` | P(victory) over the force-design simplex (value of heterogeneity, R6). |
| Fig 4 | `fig4_heterogeneity.png` | Matched-pair value of heterogeneity by engagement order (R6). |
| Fig 5 | `fig5_fusion_exchange.png` | Information–force exchange rate: when Δσ substitutes for budget (R5, contribution #2). |
| Tables | `tables_summary.md` | Headline numbers for the paper text. |

## Validation excursions (spec sec 10) — `test_validation.py`

| Test | Excursion |
|---|---|
| `test_symmetry_mirror_match` | Mirror-match symmetry (sec 6): losses symmetric, log-FER ≈ 0. |
| `test_raw_fer_is_biased_but_logfer_is_not` | Justifies the R4 log-FER transform. |
| `test_tmax1_single_salvo` | `T_max=1` collapses to a single exchange. |
| `test_armstrong_anchor` | Armstrong (2011) homogeneous anchor (10.1). **Placeholder inputs** — flip `ARMSTRONG_CALIBRATED` once Armstrong's published benchmark parameters are supplied; currently locks a regression baseline + scenario-independent properties. |
| `test_tiah_swing` | Tiah concentration→dispersion swing (10.2): FER<1 → FER>1. Exact 0.54/2.0 needs Tiah's inputs; the sign of the swing is the check. |
| `test_fusion_value_compounds` | Fusion value grows with duration (contribution #3). |
| `test_allocation_robustness` | Uniform without-replacement (10.3): overkill drops, R5/R6 signs unchanged. |

## Open calibration items (paper-author inputs)

1. **Armstrong 2.64 anchor** — supply Armstrong (2011) benchmark inputs in
   `test_validation.ARMSTRONG_INPUTS`, then set `ARMSTRONG_CALIBRATED = True`.
2. **NOB design** — for the published run, pass the SEED Center mixed NOB matrix
   via `farm.py --nob-path ...` (LHS is the dependency-free stand-in).
