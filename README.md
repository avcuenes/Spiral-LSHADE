# Spiral-LSHADE: Spiral-Guided L-SHADE with Eigen-DE and Stall Rescues

**Spiral-LSHADE** is a bound-constrained black-box optimizer for benchmarks and engineering design.  
It extends **L-SHADE** (success-history DE + linear population-size reduction) with:
- **Spiral drift** (structured, geometry-aware diversification around the incumbent),
- **Stall rescue** (budget-capped Nelder–Mead, triggered only on stagnation),
- **Optional Eigen-DE** crossover for ill-conditioned landscapes.

👉 Article title: **Spiral-Guided Success-History Differential Evolution for Black-Box Optimization**

---

## Features

- **L-SHADE core**
  - Success-history parameter adaptation
  - Linear population-size reduction
  - External archive
- **Eigen-DE crossover**  
  - Periodic / stall-triggered  
  - Rotation-invariant exploitation
- **Isotropic spiral drift**  
  - Decaying perturbations on incumbent best  
  - Disabled near basin
- **Rescues**
  - Fallback `rand/1` mutation  
  - Opportunistic Nelder–Mead polishing  
  - In-place restarts
- **Safeguards**
  - Early stop with target tolerance  
  - Stall and diversity guards  
  - Reflection to preserve feasibility

---

## Installation

### 1. Clone this repository
```bash
git clone https://github.com/<your-username>/SLO-HBYRID.git
cd SLO-HBYRID
```

### 2. Create a virtual enviroment

```bash
python3 -m venv .venv
source .venv/bin/activate    # Linux/macOS

pip install -r requirements.txt
```

### Makefile shortcuts

A root-level `Makefile` wraps the commands shown below and handles the default `.venv` virtual environment for you.

- `make help` – list all available helper targets.
- `make install` – create `.venv` (using `python3 -m venv`) and install `requirements.txt`.
- `make run-eng` – run the engineering-design sweep (`slo_bench.cec_run3`).
- `make run-cec14-20d` – run the 20D CEC-2014 sweep (`slo_bench.cec_run2`).
- `make run-bbob` – run the COCO/BBOB sweep (`slo_bench.bbob_run`).
- `make run-stats` / `make run-mean-std` – reproduce the statistics/plot commands at the end of this README.

All parameters are exposed as variables (e.g. `ENG_DIMS`, `CEC14_OUTDIR`, `BBOB_SEED`, etc.), so you can override them inline:

```bash
make run-bbob BBOB_DIMS="5 10" BBOB_SEED=99
make run-cec14-20d PYTHON=.venv/bin/python CEC14_OUTDIR=results_cec14_custom
```

### Test algorithms

#### Engineering design (selected problems)
```bash
python3 -m slo_bench.cec_run3 \
  --suite eng --dims 10 20 --fids 1-16 --runs 30 \
  --algs NLSHADE-RSP LSHADE JADE jSO SLO_HBYRID CMAES SciPyDE LBFGSB PSO GWO MealpyGA SSA \
  --budget-mult 4000 --target-tol 1e-8 \
  --seed0 0 --outdir results_cec2
```
#### CEC-2014 (20D)
```bash
python3 -m slo_bench.cec_run2 \
  --suite cec2014 --dims 20 --fids 1-30 --runs 30 \
  --algs NLSHADE-RSP LSHADE JADE jSO SLO_HBYRID CMAES SciPyDE LBFGSB PSO GWO MealpyGA SSA \
  --budget-mult 4000 --target-tol 1e-8 \
  --seed0 0 --outdir results_cec14_20D
```

#### COCO/BBOB (2–20D, 24 functions)
```bash
python3 -m slo_bench.bbob_run \
  --dims 2 3 5 10 20 \
  --functions 1-24 \
  --instances 1-15 \
  --algs NLSHADE-RSP LSHADE JADE jSO \
  --budget-mult 4000 --seed 42 \
  --outdir exdata
```


```bash
# ERT and performance profiles
python3 cec2022_full_stats.py \
  --csv cec2014_results.csv \
  --ert cec2014_ERT_summary.csv \
  --out appendix_figs \
  --dims 20 --perf-profile-penalty 2.0

# Mean/std plots
python3 cec2022_mean_std.py \
  --csv cec2014_results.csv \
  --out appendix_figs \
  --metric err
```

### Re-generating figures from stored CSVs
If you already have populated `results_*` folders (e.g., from the archives in this
repository) you can rebuild every `appendix_figs` directory without rerunning any
benchmark sweeps:

```bash
python scripts/regenerate_appendix_figs.py
```

The helper auto-detects every `results_*` directory that contains the plotting
scripts and invokes them with the correct CSV/ERT files. Pass explicit folders to
limit the work (for example `python scripts/regenerate_appendix_figs.py results_cec22_10D`)
or run with `--dry-run` to simply review the commands that would be executed.

## Docker workflow

A lightweight `Dockerfile` is provided for fully reproducible runs. Build it with Docker or via the Makefile helper:

```bash
make docker-build            # builds the image tagged 'spiral-lshade'
make docker-shell            # drops into /app inside the container with the repo mounted
make docker-make TARGET=run-eng   # run any Make target inside the container
make docker-run-eng          # convenience wrapper for 'make run-eng' inside Docker
make docker-regen-appendix   # regenerate appendix_figs via Docker (pattern: docker-<target>)
```

Inside the container you can invoke the same `make run-*` targets shown above (dependencies are already installed). The repository directory is bind-mounted, so generated result files persist on the host. Any Make target can be run inside Docker by prefixing it with `docker-`, which simply forwards to `docker run … make <target>`.


## Project Structures
```
SLO-HBYRID/
│
├── .venv/                  # virtual environment (optional)
├── exdata/                 # example benchmark outputs
├── ppdata/                 # processed benchmark data
├── results_cec14_10D/      # stored results (CEC-2014, 10D)
├── results_cec14_20D/
├── results_cec22_10D/
├── results_cec22_20D/
├── results_eng/
│
├── slo_bench/              # main benchmarking package
│   ├── bbob_run.py
│   ├── cec_run.py
│   ├── cec_run2.py
│   ├── cec_run3.py
│   ├── cec22_analyzer.py
│   ├── make_figs.py
│   ├── plot_bench_seaborn.py
│   ├── problems.py
│   ├── spiral_lshade.py
│   ├── summarize_eng_results.py
│   └── compare.py
│
├── LICENSE
├── README.md
├── requirements.txt
└── .gitignore

```
