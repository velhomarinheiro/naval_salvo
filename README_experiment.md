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
| `gamma_excursion.py` | Cost-convexity (γ) robustness excursion (spec sec 5 → paper sec 5.5). |
| `extract_noab_design.py` | Deterministic NOAB design extraction from the SEED Center workbook. |
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
| — | Aggregate continuous damage (Armstrong-comparable loss, ships) | `blue_damage_mean`, `red_damage_mean` |

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
| `test_armstrong_anchor` | Armstrong (2011) anchor (10.1), **calibrated** to the sec-5.1 6-on-3 illustrative example: aggregate continuous loss reproduces Armstrong's simulation (mean 0.679, sd 0.467) to <2%. Per-hull integer kills stay quantized far below it — the kill-quantization layer this model adds. (The spec's "2.64" is not in Armstrong 2011; 6-on-3 is the paper's actual published anchor.) |
| `test_tiah_swing` | Tiah concentration→dispersion swing (10.2): FER<1 → FER>1. Exact 0.54/2.0 needs Tiah's inputs; the sign of the swing is the check. |
| `test_fusion_value_compounds` | Fusion value grows with duration (contribution #3). |
| `test_allocation_robustness` | Uniform without-replacement (10.3): overkill drops, R5/R6 signs unchanged. |

## Robustness excursions

- **Cost convexity γ (spec sec 5 → paper sec 5.5)** — `make gamma` (or
  `python3 gamma_excursion.py`). Sweeps γ ∈ [1.15, 1.55] at the process centre
  and reports the budget-optimal (s_L, s_M, s_H) at each γ, locating the
  **tipping γ** where the optimum flips from quality (H-heavy) to quantity
  (L-heavy). Writes `gamma_excursion.csv` + `gamma_excursion.png`. γ only
  reprices Blue's platforms; the budget stays anchored to the γ=1.35 Red
  reference so the high-low mix decision is isolated.

  Pilot finding (P(victory) objective, process centre, ρ=1.1, 6k reps):
  quality (H-heavy) is optimal through γ≈1.40, and the optimum flips to
  quantity (L-heavy) at a **tipping γ ≈ 1.45** — so at the baseline γ=1.35 the
  concentrated-quality design is still favoured, and it takes a markedly higher
  convexity penalty to tip the high-low mix toward many cheap hulls. Regenerate
  at paper scale before quoting in §5.5.

## NOAB process design (paper-scale)

The LHS in `build_process_design` is the reproducible pilot stand-in; the
published run uses the NPS SEED Center **NOAB mixed design**
(`NOAB_Mixed_Designs_v4.xlsx`, v4). Provenance and extraction are scripted:

```bash
python3 extract_noab_design.py   # workbook -> nob_design_raw.csv (deterministic)
```

- **Source:** sheet `CodedValues up to 75 factors` → a **128-design-point** NOAB
  mixed design (the "up to 300 factors" family is 512 points; 7 factors don't
  need it). Columns 51–75 are the 25 continuous factors (`cont1..cont25`), each
  a balanced permutation of the integers **1..128** (coded range `[1, 128]`).
- **Selection:** the continuous factors are generic placeholders with no
  meaningful order and the design is near-orthogonal (max |pairwise corr| among
  `cont1..cont7` ≈ **0.004**), so we take `cont1..cont7` and map them in order to
  `sigma_b, sigma_r, tau, rho, p_o, p_d, sd` (documented in `extract_noab_design.py`).

`load_nob_design()` also accepts a pre-scaled real-unit export
(`already_scaled=True`, handoff Path A) if the workbook's data-entry UI is used
under Excel instead. `build_process_design` auto-runs `check_design()` on any
loaded NOB design (point count, per-factor ranges, max pairwise correlation) and
warns if it is not near-orthogonal. Paper-scale run:

```bash
python3 farm.py --reps 10000 --process-points 128 \
    --nob-path nob_design_raw.csv --nob-coded-lo 1 --nob-coded-hi 128 \
    --jobs 4 --out-prefix farm_paperscale
# then regenerate figures from the paper-scale results:
python3 analyze.py farm_paperscale_results.csv
```

Or in one step: `make paper-noab JOBS=4`.

`--jobs N` parallelises over design points (identical results to serial — each
point is self-seeded). Design = 10 mixture × 128 process × 3 orders = 3,840
points × 10,000 reps ≈ 38.4M battles.

## Open items (paper-author inputs)

1. **Armstrong anchor** — ✅ calibrated to the 6-on-3 illustrative example of
   Armstrong (2011), which reproduces the paper's published simulation output
   (0.679 / 0.467) to within 2%. Note for the write-up: the spec's "2.64" is not
   found in Armstrong (2011); the 6-on-3 example is the verifiable anchor.
2. **NOAB design workbook** — ✅ integrated. `NOAB_Mixed_Designs_v4.xlsx` is
   provided locally (the NPS host `nps.edu` is blocked by this environment's
   egress policy); `extract_noab_design.py` regenerates the design CSV from it
   deterministically. The workbook and the derived CSVs are not versioned (see
   `.gitignore`); the extraction script + provenance above make them regenerable.
