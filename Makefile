SHELL := /bin/bash

VENV ?= .venv
SYSTEM_PYTHON ?= python3
PYTHON ?= $(if $(wildcard $(VENV)/bin/python),$(VENV)/bin/python,$(SYSTEM_PYTHON))
PIP ?= $(VENV)/bin/pip
DOCKER_IMAGE ?= spiral-lshade
DOCKER_RUN ?= docker run --rm -it -v $(CURDIR):/app -w /app $(DOCKER_IMAGE)
TARGET ?= help

# Defaults mirror the commands shown in README.md. Override them as needed:
ENG_DIMS ?= 10 20
ENG_FIDS ?= 1-16
ENG_RUNS ?= 30
ENG_ALGS ?= NLSHADE-RSP LSHADE JADE jSO Spiral-LSHADE CMAES SciPyDE LBFGSB PSO GWO MealpyGA SSA
ENG_BUDGET ?= 4000
ENG_TOL ?= 1e-8
ENG_SEED ?= 0
ENG_OUTDIR ?= results_cec2

CEC14_DIMS ?= 20
CEC14_FIDS ?= 1-30
CEC14_RUNS ?= 30
CEC14_ALGS ?= NLSHADE-RSP LSHADE JADE jSO Spiral-LSHADE CMAES SciPyDE LBFGSB PSO GWO MealpyGA SSA
CEC14_BUDGET ?= 4000
CEC14_TOL ?= 1e-8
CEC14_SEED ?= 0
CEC14_OUTDIR ?= results_cec14_20D

BBOB_DIMS ?= 2 3 5 10 20
BBOB_FUNCTIONS ?= 1-24
BBOB_INSTANCES ?= 1-15
BBOB_ALGS ?= NLSHADE-RSP LSHADE JADE jSO
BBOB_BUDGET ?= 4000
BBOB_SEED ?= 42
BBOB_OUTDIR ?= exdata

STATS_SCRIPT ?= cec2022_full_stats.py
STATS_CSV ?= cec2014_results.csv
STATS_ERT ?= cec2014_ERT_summary.csv
STATS_OUT ?= appendix_figs
STATS_DIMS ?= 20
STATS_PENALTY ?= 2.0

MEAN_STD_SCRIPT ?= cec2022_mean_std.py
MEAN_STD_CSV ?= cec2014_results.csv
MEAN_STD_OUT ?= appendix_figs
MEAN_STD_METRIC ?= err

COMPARE_SCRIPT ?= compare.py
COMPARE_ARGS ?=

REGEN_APPENDIX_SCRIPT ?= scripts/regenerate_appendix_figs.py
REGEN_APPENDIX_ARGS ?=

RUN_PY_CHECK = @if ! command -v $(PYTHON) >/dev/null 2>&1; then \
		echo "Python interpreter '$(PYTHON)' not found. Run 'make install' first or override PYTHON=python3"; \
		exit 1; \
	fi

.PHONY: help venv install clean run-eng run-cec14-20d run-bbob run-compare run-stats run-mean-std regen-appendix docker-build docker-shell docker-make docker-%

help:
	@echo "Spiral-LSHADE helper targets"
	@echo "  make install            # create $(VENV) and install requirements"
	@echo "  make run-eng            # engineering design benchmark sweep"
	@echo "  make run-cec14-20d      # CEC-2014 20D sweep"
	@echo "  make run-bbob           # COCO/BBOB sweep"
	@echo "  make run-compare        # launch compare.py (COCO/BBOB post-processing)"
	@echo "  make run-stats          # generate ERT + performance profiles"
	@echo "  make run-mean-std       # generate mean/std plots"
	@echo "  make regen-appendix     # rebuild appendix_figs from stored CSVs"
	@echo "  make docker-build       # build the Docker image ($(DOCKER_IMAGE))"
	@echo "  make docker-shell       # start an interactive shell inside the Docker image"
	@echo "  make docker-make TARGET=<name>  # run a Make target inside the Docker image"
	@echo "  make docker-<target>    # shortcut to run any target inside Docker (e.g. docker-run-eng)"
	@echo "Variables such as PYTHON, ENG_* or CEC14_* can be overridden at invocation time."

venv:
	$(SYSTEM_PYTHON) -m venv $(VENV)

install: venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

clean:
	rm -rf $(VENV) __pycache__ */__pycache__

run-eng:
	$(RUN_PY_CHECK)
	$(PYTHON) -m slo_bench.cec_run3 \
		--suite eng --dims $(ENG_DIMS) --fids $(ENG_FIDS) --runs $(ENG_RUNS) \
		--algs $(ENG_ALGS) \
		--budget-mult $(ENG_BUDGET) --target-tol $(ENG_TOL) \
		--seed0 $(ENG_SEED) --outdir $(ENG_OUTDIR)

run-cec14-20d:
	$(RUN_PY_CHECK)
	$(PYTHON) -m slo_bench.cec_run2 \
		--suite cec2014 --dims $(CEC14_DIMS) --fids $(CEC14_FIDS) --runs $(CEC14_RUNS) \
		--algs $(CEC14_ALGS) \
		--budget-mult $(CEC14_BUDGET) --target-tol $(CEC14_TOL) \
		--seed0 $(CEC14_SEED) --outdir $(CEC14_OUTDIR)

run-bbob:
	$(RUN_PY_CHECK)
	$(PYTHON) -m slo_bench.bbob_run \
		--dims $(BBOB_DIMS) \
		--functions $(BBOB_FUNCTIONS) \
		--instances $(BBOB_INSTANCES) \
		--algs $(BBOB_ALGS) \
		--budget-mult $(BBOB_BUDGET) --seed $(BBOB_SEED) \
		--outdir $(BBOB_OUTDIR)

run-compare:
	$(RUN_PY_CHECK)
	$(PYTHON) $(COMPARE_SCRIPT) $(COMPARE_ARGS)

run-stats:
	$(RUN_PY_CHECK)
	$(PYTHON) $(STATS_SCRIPT) \
		--csv $(STATS_CSV) \
		--ert $(STATS_ERT) \
		--out $(STATS_OUT) \
		--dims $(STATS_DIMS) \
		--perf-profile-penalty $(STATS_PENALTY)

run-mean-std:
	$(RUN_PY_CHECK)
	$(PYTHON) $(MEAN_STD_SCRIPT) \
		--csv $(MEAN_STD_CSV) \
		--out $(MEAN_STD_OUT) \
		--metric $(MEAN_STD_METRIC)

regen-appendix:
	$(RUN_PY_CHECK)
	$(PYTHON) $(REGEN_APPENDIX_SCRIPT) $(REGEN_APPENDIX_ARGS)

docker-build:
	docker build -t $(DOCKER_IMAGE) .

docker-shell:
	$(DOCKER_RUN) /bin/bash

docker-make:
	$(DOCKER_RUN) make $(TARGET)

docker-%:
	$(DOCKER_RUN) make $*
