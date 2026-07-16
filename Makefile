# Makefile — one-command reproduction for the LAFusion 2026 experiment
# (spec sec 11). Pipeline: validate -> design+simulate+metamodels -> figures/tables.
#
#   make all           # regression tests + pilot farm + figures (fast, reproducible)
#   make paper         # tests + paper-scale farm (10k reps, 128 pts) + figures
#   make test          # just the spec-sec-10 validation regression suite
#   make demo          # human-readable validation narrative
#   make farm          # data-farming run (override REPS/PTS below)
#   make figures       # paper figures + tables from farm_results.csv
#   make clean         # remove generated run artifacts
#
# Override run scale, e.g.:  make farm REPS=2000 PTS=64

PY      ?= python3
REPS    ?= 500      # Monte Carlo battles per design point (pilot default)
PTS     ?= 33       # space-filling process-design points (pilot default)
PREFIX  ?= farm

RESULTS := $(PREFIX)_results.csv

.PHONY: all paper test demo farm figures clean deps

all: test farm figures ## full reproducible pilot pipeline

paper: ## paper-scale run (10,000 reps x 128 process points)
	$(MAKE) test
	$(MAKE) farm REPS=10000 PTS=128
	$(MAKE) figures

test: ## spec sec-10 validation regression suite
	$(PY) -m pytest test_validation.py -v

demo: ## readable validation + demonstration narrative
	$(PY) validate_demo.py

farm: ## design + simulate + metamodels -> $(PREFIX)_design.csv, $(PREFIX)_results.csv
	$(PY) farm.py --reps $(REPS) --process-points $(PTS) --out-prefix $(PREFIX)

figures: $(RESULTS) ## paper figures (fig1..fig5) + tables_summary.md
	$(PY) analyze.py

$(RESULTS):
	$(MAKE) farm

deps: ## install the pinned reproducibility environment
	$(PY) -m pip install -r requirements-lock.txt

clean: ## remove generated run artifacts (keep source + design seeds)
	rm -f $(PREFIX)_design.csv $(PREFIX)_results.csv \
	      fig1_fusion_compounding.png fig2_partition_tree.png \
	      fig3_mixture_ternary.png fig4_heterogeneity.png fig5_fusion_exchange.png \
	      tables_summary.md
