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
