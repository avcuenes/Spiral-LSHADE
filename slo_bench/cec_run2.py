# -*- coding: utf-8 -*-
"""
slo_bench.cec_run
-----------------
CEC-2014 / CEC-2022 benchmark runner with multiple optimizers.

CSV per-run log  →  <outdir>/<suite>_results.csv
CSV ERT summary →  <outdir>/<suite>_ERT_summary.csv

Example
-------
python3 -m slo_bench.cec_run2 \
  --suite cec2022 --dims 10 --fids 1-12 --runs 10 \
  --algs Spiral-LSHADE IPOP-CMAES BIPOP-CMAES SciPyDE LBFGSB jSO L_SHADE \
  --budget-mult 20000 --target-tol 1e-8 \
  --seed0 0 --outdir results_cec
"""
from __future__ import annotations
import argparse, csv, os, sys, time, importlib, warnings
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple, Optional

import numpy as np
# --- NumPy 2.x backwards-compat for older libs ---
if not hasattr(np, "float"):   np.float   = float   # noqa
if not hasattr(np, "int"):     np.int     = int     # noqa
if not hasattr(np, "bool"):    np.bool    = bool    # noqa
if not hasattr(np, "object"):  np.object  = object  # noqa

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))



# ---------------------------------------------------------------------
#  Your Spiral-LSHADE hybrid (required for Spiral-LSHADE)
# ---------------------------------------------------------------------
try:
    from spiral_lshade import SpiralLSHADEParams, _slo_lshade_core
    HAVE_SLO = True
except Exception:
    HAVE_SLO = False

# ---------------------------------------------------------------------
#  Optional third-party libraries (lazy used)
# ---------------------------------------------------------------------
try:    import cma;                                               HAVE_CMA    = True
except Exception:                                                 HAVE_CMA    = False

try:
    from scipy.optimize import differential_evolution, minimize;  HAVE_SCIPY  = True
except Exception:                                                 HAVE_SCIPY  = False

try:
    import pybobyqa;                                              HAVE_PYBOBYQA = True
except Exception:                                                 HAVE_PYBOBYQA = False

try:
    import pyade.jso   as pyade_jso
    import pyade.lshade as pyade_lshade
    HAVE_PYADE = True
except Exception:
    HAVE_PYADE = False

try:    import minionpy as mpy;                                   HAVE_MINION = True
except Exception:                                                 HAVE_MINION = False

# OPFUNU – CEC suites
try:    from opfunu.cec_based import cec2014 as ofu2014;          HAVE_2014   = True
except Exception:                                                 HAVE_2014   = False
try:    from opfunu.cec_based import cec2022 as ofu2022;          HAVE_2022   = True
except Exception:                                                 HAVE_2022   = False


# =====================================================================
#  Helpers
# =====================================================================
def set_seed(seed: int):
    np.random.seed(int(seed))

def project_box_open(x, lb, ub):
    x  = np.asarray(x,  dtype=np.float64)
    lb = np.asarray(lb, dtype=np.float64)
    ub = np.asarray(ub, dtype=np.float64)
    rng = ub - lb
    eps = np.maximum(1e-8 * np.maximum(rng, 1.0), 1e-12)
    eps = np.minimum(eps, 0.45 * np.maximum(rng, 1e-30))
    return np.minimum(ub - eps, np.maximum(lb + eps, x))

@dataclass
class Problem:
    fid: int; name: str; dim: int
    lower: np.ndarray; upper: np.ndarray
    fopt: float
    f: Callable[[np.ndarray], float]

def _to_array(v, default, size):
    if hasattr(v, "__len__") and not isinstance(v, (str, bytes)):
        arr = np.asarray(v, dtype=float)
        if arr.size == 1:      return np.full(size, float(arr.item()))
        if arr.size == size:   return arr.astype(float)
    return np.full(size, float(default))

# ---------------------------------------------------------------------
#  Problem factory
# ---------------------------------------------------------------------
def build_problem(suite: str, fid: int, dim: int) -> Problem:
    suite = suite.lower()
    if suite == "cec2014":
        if not HAVE_2014:
            raise RuntimeError("opfunu cec2014 not available")
        fmap = {i: getattr(ofu2014, f"F{i}2014") for i in range(1, 31)}
    elif suite == "cec2022":
        if not HAVE_2022:
            raise RuntimeError("opfunu cec2022 not available")
        fmap = {i: getattr(ofu2022, f"F{i}2022") for i in range(1, 13)}
    else:
        raise ValueError("suite must be 'cec2014' or 'cec2022'")

    if fid not in fmap:
        raise ValueError(f"fid {fid} not in suite {suite}")
    fcls = fmap[fid]

    # constructor kwargs vary across OPFUNU versions
    fobj = None
    for k in ("ndim", "dimension", "problem_size", "n_dimensions", "dim"):
        try:
            fobj = fcls(**{k: dim})
            break
        except TypeError:
            continue
    if fobj is None:
        try:
            fobj = fcls(dim)
        except TypeError as e:
            raise RuntimeError(f"Cannot construct {fcls.__name__} for dim={dim}: {e}")

    lower = _to_array(getattr(fobj, "lb", -100.0), -100.0, dim)
    upper = _to_array(getattr(fobj, "ub",  100.0),  100.0, dim)
    fopt  = float(getattr(fobj, "f_bias", getattr(fobj, "f_global", 0.0)))

    def f(x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        if hasattr(fobj, "evaluate"):
            return float(fobj.evaluate(x))
        return float(fobj(x))

    return Problem(fid=fid, name=f"F{fid:02d}_{suite.upper()}",
                   dim=dim, lower=lower, upper=upper, fopt=fopt, f=f)

# ---------------------------------------------------------------------
#  Evaluation counter with early-hit
# ---------------------------------------------------------------------
class EvalCounter:
    def __init__(self, f: Callable[[np.ndarray], float],
                 fopt: float, target_tol: float):
        self.f = f
        self.fopt = float(fopt)
        self.target_tol = float(target_tol)
        self.nfe  = 0
        self.best = float("inf")
        self.hit  = False
    def __call__(self, x: np.ndarray) -> float:
        y = float(self.f(x)); self.nfe += 1
        if y < self.best:
            self.best = y
            if abs(self.best - self.fopt) <= self.target_tol:
                self.hit = True
        return y

# =====================================================================
#  Algorithm adapters
# =====================================================================
# --- SLO-HBYRID (your hybrid) ----------------------------------------
def run_spiral_lshade_hybrid(f, lb, ub, budget, seed, kw=None):
    if not HAVE_SLO:
        raise RuntimeError("spiral_lshade module not available")
    if isinstance(kw, SpiralLSHADEParams):
        kw = {"params": kw}
    kw  = dict(kw) if kw else {}
    p   = kw.get("params", SpiralLSHADEParams())
    rng = np.random.default_rng(int(kw.get("rng_seed", seed)))
    best_f, nfe = _slo_lshade_core(f, np.asarray(lb), np.asarray(ub),
                                   int(budget), rng, p)
    return float(best_f), int(nfe)

def run_Spiral-LSHADE(prob: Problem, budget: int, seed: int) -> Tuple[float, int]:
    evalf = EvalCounter(lambda z: prob.f(project_box_open(z, prob.lower, prob.upper)),
                        prob.fopt, 1e-8)
    f_best, nfe = run_spiral_lshade_hybrid(evalf, prob.lower, prob.upper,
                                           budget, seed, kw={})
    return float(f_best), int(nfe)

# --- CMA-ES (vanilla) -------------------------------------------------
def run_cmaes(prob: Problem, budget: int, seed: int):
    if not HAVE_CMA: raise RuntimeError("cma not installed")
    evalf = EvalCounter(lambda z: prob.f(project_box_open(z, prob.lower, prob.upper)),
                        prob.fopt, 1e-8)
    x0 = (prob.lower + prob.upper) / 2.0
    sigma0 = 0.3 * float(np.max(prob.upper - prob.lower))
    opts = dict(bounds=[prob.lower, prob.upper],
                maxfevals=int(budget), seed=int(seed),
                verbose=-9, CMA_active=True)
    _, f_best, _, nfe, *_ = cma.fmin(lambda x: evalf(np.asarray(x, float)),
                                     x0, sigma0, options=opts)
    return float(f_best), int(nfe)

# --- IPOP-CMA-ES (manual restarts with doubling popsize) --------------
def run_ipop_cmaes(prob: Problem, budget: int, seed: int):
    if not HAVE_CMA:
        raise RuntimeError("cma not installed")
    set_seed(seed)

    evalf = EvalCounter(lambda z: prob.f(project_box_open(np.asarray(z, float), prob.lower, prob.upper)),
                        prob.fopt, 1e-8)

    x0 = (prob.lower + prob.upper) / 2.0
    sigma0 = 0.3 * float(np.max(prob.upper - prob.lower))

    base_pop = int(4 + np.floor(3 * np.log(prob.dim)))
    popsize = max(4, base_pop)

    f_best = float("inf")
    restart_id = 0

    while evalf.nfe < budget:
        remaining = int(budget - evalf.nfe)
        if remaining <= 0:
            break

        opts = dict(
            bounds=[prob.lower, prob.upper],
            maxfevals=remaining,
            seed=int(seed + restart_id),
            verbose=-9,
            CMA_active=True,
            popsize=int(popsize)
        )

        xopt, f_restart, _, _nfe, *_ = cma.fmin(
            lambda x: evalf(np.asarray(x, float)),
            x0, sigma0, options=opts
        )
        if float(f_restart) < f_best:
            f_best = float(f_restart)
        restart_id += 1
        popsize *= 2
        x0 = project_box_open(
            x0 + 0.05 * (np.random.rand(prob.dim) - 0.5), prob.lower, prob.upper
        )

    return float(f_best), int(evalf.nfe)

# --- BIPOP-CMA-ES (alternate small/large pop sizes) -------------------
def run_bipop_cmaes(prob: Problem, budget: int, seed: int):
    if not HAVE_CMA:
        raise RuntimeError("cma not installed")
    set_seed(seed)

    evalf = EvalCounter(lambda z: prob.f(project_box_open(np.asarray(z, float), prob.lower, prob.upper)),
                        prob.fopt, 1e-8)

    x0 = (prob.lower + prob.upper) / 2.0
    sigma_base = 0.3 * float(np.max(prob.upper - prob.lower))
    f_best = float("inf")

    base_pop = int(4 + np.floor(3 * np.log(prob.dim)))
    restart_id = 0

    while evalf.nfe < budget:
        remaining = int(budget - evalf.nfe)
        if remaining <= 0:
            break

        if restart_id % 2 == 0:
            popmult = 0.5
        else:
            popmult = 2 ** (restart_id // 2 + 1)

        opts = dict(
            bounds=[prob.lower, prob.upper],
            maxfevals=remaining,
            seed=int(seed + restart_id),
            verbose=-9,
            CMA_active=True,
            popsize=max(4, int(base_pop * popmult))
        )

        xopt, f_restart, _, _nfe, *_ = cma.fmin(
            lambda x: evalf(np.asarray(x, float)),
            x0, sigma_base, options=opts
        )
        f_best = min(f_best, float(f_restart))
        restart_id += 1
        x0 = project_box_open(
            x0 + 0.05 * (np.random.rand(prob.dim) - 0.5), prob.lower, prob.upper
        )

    return float(f_best), int(evalf.nfe)

# --- SciPy Differential Evolution ------------------------------------
def run_scipy_de(prob: Problem, budget: int, seed: int):
    if not HAVE_SCIPY:
        raise RuntimeError("scipy not installed")
    evalf = EvalCounter(lambda z: prob.f(project_box_open(z, prob.lower, prob.upper)),
                        prob.fopt, 1e-8)
    dim, pop = prob.dim, 15
    maxiter = max(1, budget // (pop * dim))
    res = differential_evolution(
        lambda x: evalf(np.asarray(x, float)),
        bounds=list(zip(prob.lower, prob.upper)),
        popsize=pop, maxiter=maxiter, seed=seed,
        polish=False, updating='deferred', workers=1)
    return float(res.fun), int(evalf.nfe)

# --- SciPy L-BFGS-B ---------------------------------------------------
def run_lbfgsb(prob: Problem, budget: int, seed: int):
    if not HAVE_SCIPY:
        raise RuntimeError("scipy not installed")
    evalf = EvalCounter(lambda z: prob.f(project_box_open(z, prob.lower, prob.upper)),
                        prob.fopt, 1e-8)
    x0 = prob.lower + np.random.default_rng(seed).random(prob.dim) * (prob.upper - prob.lower)
    minimize(lambda x: evalf(np.asarray(x, float)), x0, method="L-BFGS-B",
             bounds=list(zip(prob.lower, prob.upper)),
             options=dict(maxfun=int(budget), disp=False))
    return float(evalf.best), int(evalf.nfe)

# --- Py-BOBYQA (multi-start, budget-sliced, robust) -------------------
def run_pybobyqa(prob: Problem, budget: int, seed: int):
    if not HAVE_PYBOBYQA:
        raise RuntimeError("pybobyqa not installed")

    set_seed(seed)

    evalf = EvalCounter(
        lambda z: prob.f(project_box_open(np.asarray(z, float), prob.lower, prob.upper)),
        prob.fopt, 1e-8
    )

    rng = np.random.default_rng(seed)
    dim = int(prob.dim)
    box_rng = np.maximum(prob.upper - prob.lower, 1e-12)
    base_rhobeg = 0.02 * float(np.min(box_rng))

    remaining = int(budget)
    best = float("inf")
    tries = 0

    while remaining > 0 and tries < 12:
        tries += 1
        before = evalf.nfe

        slice_budget = int(min(max(10*dim, 50), remaining))
        x0 = prob.lower + rng.random(dim) * (prob.upper - prob.lower)

        rhobeg = base_rhobeg * (0.5 ** max(0, tries - 1))
        n = dim
        npt = min(max(2*n + 1, n + 2), (n + 1) * (n + 2) // 2)

        def f_obj(x):
            return float(evalf(np.asarray(x, float)))

        try:
            res = pybobyqa.solve(
                f_obj, x0=x0,
                bounds=(prob.lower, prob.upper),
                maxfun=int(slice_budget),
                npt=npt,
                rhobeg=rhobeg,
                scaling_within_bounds=True,
                seek_global_minimum=False,
                objfun_has_noise=False,
                print_progress=False,
            )
        except TypeError:
            res = pybobyqa.solve(
                f_obj, x0=x0,
                bounds=(prob.lower, prob.upper),
                maxfun=int(slice_budget),
            )
        except Exception:
            res = None

        spent = evalf.nfe - before

        if res is not None:
            for attr in ("f", "fun", "fx"):
                if hasattr(res, attr):
                    try:
                        best = min(best, float(getattr(res, attr)))
                        break
                    except Exception:
                        pass

        if spent <= 1 and remaining > 0:
            try:
                res2 = pybobyqa.solve(
                    f_obj,
                    x0=(prob.lower + prob.upper) / 2.0,
                    bounds=(prob.lower, prob.upper),
                    maxfun=int(min(max(10, slice_budget//2), remaining)),
                    rhobeg=0.005 * float(np.min(box_rng)),
                    scaling_within_bounds=True,
                    print_progress=False,
                )
                for attr in ("f", "fun", "fx"):
                    if hasattr(res2, attr):
                        try:
                            best = min(best, float(getattr(res2, attr)))
                            break
                        except Exception:
                            pass
            except Exception:
                pass

        spent = max(0, evalf.nfe - before)
        remaining -= spent
        if spent <= 1:
            seed += 13
            rng = np.random.default_rng(seed)

    if evalf.nfe == 0:
        _ = evalf((prob.lower + prob.upper) / 2.0)

    if not np.isfinite(best):
        best = float(evalf.best)

    return float(best), int(evalf.nfe)

# --- PyADE jSO / L-SHADE (version-tolerant; fallback path) ------------
def run_pyade(prob: Problem, budget: int, seed: int, which: str):
    if not HAVE_PYADE:
        raise RuntimeError("pyade not installed or import failed")
    import inspect

    evalf = EvalCounter(
        lambda z: prob.f(project_box_open(np.asarray(z, float), prob.lower, prob.upper)),
        prob.fopt, 1e-8
    )

    algo = pyade_jso if which.lower() == "jso" else pyade_lshade
    bounds_mat = np.stack([prob.lower, prob.upper], axis=1).astype(float)

    try:
        defaults = dict(algo.get_default_params(dim=prob.dim))
    except Exception:
        defaults = {}
    NP = int(max(defaults.get("NP", 0), 5 * prob.dim, 20))
    iters = max(1, int(budget) // NP)

    def f_handle(x, *args, **kwargs):
        return float(evalf(np.asarray(x, float)))

    sig = inspect.signature(algo.apply)
    accepted = set(sig.parameters.keys())

    base_kw = {}
    for n in ("func", "fobj", "function"):
        if n in accepted:
            base_kw[n] = f_handle; break
    if "bounds" in accepted:
        base_kw["bounds"] = bounds_mat
    else:
        if "lower" in accepted: base_kw["lower"] = prob.lower.astype(float)
        if "upper" in accepted: base_kw["upper"] = prob.upper.astype(float)
    for n in ("max_evals", "maxEvaluations", "max_evaluation", "maxFEs", "budget"):
        if n in accepted: base_kw[n] = int(budget); break
    if "seed" in accepted: base_kw["seed"] = int(seed)
    for n in ("iters", "n_iters", "max_iters"):
        if n in accepted: base_kw[n] = int(iters); break
    for n in ("NP", "pop_size", "population_size"):
        if n in accepted: base_kw[n] = int(NP); break

    try:
        sol, fit = algo.apply(**base_kw)
        if evalf.nfe == 0:
            _ = f_handle(prob.lower + (prob.upper - prob.lower) * 0.5)
        return float(fit), int(evalf.nfe)
    except TypeError:
        pass

    # positional legacy path
    params = [p for p in sig.parameters.values()
              if p.kind in (inspect.Parameter.POSITIONAL_ONLY,
                            inspect.Parameter.POSITIONAL_OR_KEYWORD)]
    opts = defaults.copy(); opts.setdefault("NP", NP)
    mem_size = int(defaults.get("H", defaults.get("memory_size", 10)))
    def _cb(*_a, **_k): return None
    value_by_name = {
        "func": f_handle, "fobj": f_handle, "function": f_handle,
        "bounds": bounds_mat, "lower": prob.lower.astype(float), "upper": prob.upper.astype(float),
        "max_evals": int(budget), "maxEvaluations": int(budget),
        "max_evaluation": int(budget), "maxFEs": int(budget), "budget": int(budget),
        "seed": int(seed),
        "iters": int(iters), "n_iters": int(iters), "max_iters": int(iters),
        "NP": int(NP), "pop_size": int(NP), "population_size": int(NP),
        "individual_size": int(prob.dim),
        "opts": opts, "options": opts,
        "memory_size": mem_size,
        "callback": _cb,
    }
    pos_args=[]
    for p in params:
        name = p.name
        if name in value_by_name: pos_args.append(value_by_name[name])
        elif p.default is not inspect._empty: continue
        else:
            lname = name.lower()
            if lname.startswith("func"):  pos_args.append(f_handle); continue
            if lname.startswith("bound"): pos_args.append(bounds_mat); continue
            if "eval" in lname or "fe" in lname: pos_args.append(int(budget)); continue
            if name == "seed": pos_args.append(int(seed)); continue
            raise TypeError(f"PyADE apply(...) requires unknown positional arg '{name}'")

    sol, fit = algo.apply(*pos_args)
    if evalf.nfe == 0:
        _ = f_handle(prob.lower + (prob.upper - prob.lower) * 0.5)
    return float(fit), int(evalf.nfe)

# --- Minion backend (preferred for jSO/LSHADE & friends) --------------
MINION_ALLOWED = {
    "LSHADE","DE","JADE","jSO","NelderMead","LSRTDE","NLSHADE_RSP",
    "j2020","GWO_DE","ARRDE","ABC","DA","L_BFGS_B"
}
MINION_ALIASES = {
    "NL_SHADE_RSP": "NLSHADE_RSP",
    "NLSHADE-RSP":  "NLSHADE_RSP",
    "NLSHADE RSP":  "NLSHADE_RSP",
    "J-SO":         "jSO",
    "J_SO":         "jSO",
    "JADE2020":     "j2020",
    "J2020":        "j2020",
    # approximate mappings – not identical implementations:
    "iL-SHADE-RSP":   "NLSHADE_RSP",
    "LSHADE-cnEpSin": "LSHADE",
    "mLSHADE-RL":     "LSHADE",
    "mLSHADE-SPACMA": "LSHADE",
}
def normalize_minion_algo(name: str) -> str:
    key = str(name).strip()
    if key in MINION_ALLOWED:
        return key
    canon = key.replace("_","").replace("-","").lower()
    for raw, target in MINION_ALIASES.items():
        if canon == raw.replace("_","").replace("-","").lower():
            return target
    raise ValueError(
        f"Unknown algorithm '{name}' for this backend. "
        f"Allowed: {sorted(MINION_ALLOWED)}. "
        f"Known aliases: {sorted(MINION_ALIASES.keys())}."
    )

def _vectorized_func_from_evalf(evalf, prob):
    def fvec(X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[None, :]
        out = []
        for x in X:
            x = project_box_open(x, prob.lower, prob.upper)
            out.append(float(evalf(x)))
        return out
    return fvec

def _bounds_list(prob):
    return list(zip(prob.lower.tolist(), prob.upper.tolist()))

def run_minion(prob: Problem, budget: int, seed: int, algo_name: str):
    if not HAVE_MINION:
        raise RuntimeError("minionpy not installed")
    algo_id = normalize_minion_algo(algo_name)
    evalf = EvalCounter(lambda z: prob.f(project_box_open(np.asarray(z, float), prob.lower, prob.upper)),
                        prob.fopt, 1e-8)
    fvec = _vectorized_func_from_evalf(evalf, prob)
    opt = mpy.Minimizer(
        func=fvec,
        x0=None,
        bounds=_bounds_list(prob),
        algo=algo_id,
        relTol=0.0,
        maxevals=int(budget),
        seed=int(seed),
        options=None
    )
    res = opt.optimize()
    return float(res.fun), int(evalf.nfe)

def run_minion_jso(prob: Problem, budget: int, seed: int):
    return run_minion(prob, budget, seed, "jSO")

def run_minion_lshade(prob: Problem, budget: int, seed: int):
    return run_minion(prob, budget, seed, "LSHADE")

def run_jso_external(prob: Problem, budget: int, seed: int):
    try:
        return run_minion_jso(prob, budget, seed)
    except Exception:
        return run_pyade(prob, budget, seed, "jso")

def run_lshade_external(prob: Problem, budget: int, seed: int):
    try:
        return run_minion_lshade(prob, budget, seed)
    except Exception:
        return run_pyade(prob, budget, seed, "lshade")

# --- Hybrid: IPOP-CMA-ES → Py-BOBYQA polish ---------------------------
def _pybobyqa_polish(prob: Problem, evalf: EvalCounter, x_start: np.ndarray, budget: int, seed: int):
    if not HAVE_PYBOBYQA or budget <= 0:
        return float(evalf.best)
    rng = np.random.default_rng(seed)
    dim = int(prob.dim)
    box_rng = np.maximum(prob.upper - prob.lower, 1e-12)
    rhobeg = 0.02 * float(np.min(box_rng))
    npt = min(max(2*dim + 1, dim + 2), (dim + 1) * (dim + 2) // 2)
    def f_obj(x):
        return float(evalf(project_box_open(np.asarray(x, float), prob.lower, prob.upper)))
    try:
        res = pybobyqa.solve(
            f_obj, x0=np.asarray(x_start, float),
            bounds=(prob.lower, prob.upper),
            maxfun=int(budget), npt=npt, rhobeg=rhobeg,
            scaling_within_bounds=True, seek_global_minimum=False,
            objfun_has_noise=False, print_progress=False,
        )
    except TypeError:
        res = pybobyqa.solve(f_obj, x0=np.asarray(x_start, float),
                             bounds=(prob.lower, prob.upper), maxfun=int(budget))
    f_best = None
    for attr in ("f", "fun", "fx"):
        if hasattr(res, attr):
            try: f_best = float(getattr(res, attr)); break
            except Exception: pass
    if (getattr(res, "nf", getattr(res, "nfev", 0)) <= 1) or (f_best is None) or (not np.isfinite(f_best)):
        try:
            res2 = pybobyqa.solve(
                f_obj, x0=(prob.lower + prob.upper)/2.0,
                bounds=(prob.lower, prob.upper),
                maxfun=int(max(10, budget//2)),
                rhobeg=0.005 * float(np.min(box_rng)),
                scaling_within_bounds=True, print_progress=False,
            )
            for attr in ("f", "fun", "fx"):
                if hasattr(res2, attr):
                    try: f_best = float(getattr(res2, attr)); break
                    except Exception: pass
        except Exception:
            pass
    return float(evalf.best if (f_best is None or not np.isfinite(f_best)) else f_best)

def run_ipop_cmaes_bobyqa(prob: Problem, budget: int, seed: int):
    if not HAVE_CMA:
        raise RuntimeError("cma not installed")
    evalf = EvalCounter(lambda z: prob.f(project_box_open(np.asarray(z, float), prob.lower, prob.upper)),
                        prob.fopt, 1e-8)
    x0 = (prob.lower + prob.upper) / 2.0
    sigma0 = 0.3 * float(np.max(prob.upper - prob.lower))
    base_pop = int(4 + np.floor(3 * np.log(prob.dim)))
    popsize = max(4, base_pop)
    budget_stage1 = int(0.8 * budget)
    f_best = float("inf")
    x_best = np.asarray(x0, float)
    restart_id = 0
    while evalf.nfe < budget_stage1:
        remaining = int(budget_stage1 - evalf.nfe)
        if remaining <= 0: break
        opts = dict(bounds=[prob.lower, prob.upper],
                    maxfevals=remaining, seed=int(seed + restart_id),
                    verbose=-9, CMA_active=True, popsize=int(popsize))
        x_restart, f_restart, _, _nfe, *_ = cma.fmin(
            lambda x: float(evalf(np.asarray(x, float))), x0, sigma0, options=opts
        )
        if float(f_restart) < f_best:
            f_best = float(f_restart)
            x_best = np.asarray(x_restart, float)
        restart_id += 1
        popsize *= 2
        x0 = project_box_open(x0 + 0.05 * (np.random.rand(prob.dim) - 0.5), prob.lower, prob.upper)
    remaining = int(budget - evalf.nfe)
    if remaining > 0:
        f_best = min(f_best, _pybobyqa_polish(prob, evalf, x_best, remaining, seed + 777))
    if evalf.nfe == 0:
        _ = evalf(x0)
    return float(f_best if np.isfinite(f_best) else evalf.best), int(evalf.nfe)

# --- Optional: Mealpy adapters (PSO/GWO/DE/GA/SHADE/SSA) --------------
MEALPY_IMPORTS = {
    "pso":   ("mealpy.swarm_based.PSO",  ["OriginalPSO", "ImprovedPSO", "BasePSO"]),
    "gwo":   ("mealpy.swarm_based.GWO",  ["OriginalGWO", "BaseGWO"]),
    "de":    ("mealpy.evolutionary_based.DE",    ["OriginalDE", "BaseDE"]),
    "ga":    ("mealpy.evolutionary_based.GA",    ["OriginalGA", "BaseGA"]),
    "shade": ("mealpy.evolutionary_based.SHADE", ["OriginalSHADE", "BaseSHADE"]),
    "woa":   ("mealpy.swarm_based.WOA",  ["OriginalWOA", "BaseWOA"]),
    "ssa":   ("mealpy.swarm_based.SSA",  ["OriginalSSA", "BaseSSA"]),
}
def _get_mealpy_cls(which: str):
    which = which.lower()
    mod_path, class_names = MEALPY_IMPORTS[which]
    mod = importlib.import_module(mod_path)
    for cname in class_names:
        if hasattr(mod, cname):
            return getattr(mod, cname)
    raise ImportError(f"No suitable class found in {mod_path} among {class_names}")

def run_mealpy(prob: Problem, budget: int, seed: int, which: str, **kw):
    try:
        import random
        from importlib import import_module as _imp
        from mealpy import Problem as MP, FloatVar as FV
        HAVE_MEALPY = True
    except Exception:
        try:
            from mealpy.utils.problem import Problem as MP
            from mealpy.utils.space import FloatVar as FV
            HAVE_MEALPY = True
        except Exception:
            HAVE_MEALPY = False
    if not HAVE_MEALPY:
        raise RuntimeError("mealpy not installed")

    np.random.seed(int(seed))
    try:
        import random as _r; _r.seed(int(seed))
    except Exception:
        pass

    evalf = EvalCounter(
        lambda z: prob.f(project_box_open(np.asarray(z, float), prob.lower, prob.upper)),
        prob.fopt, 1e-8
    )

    pop = int(kw.pop("pop_size", max(20, 5 * int(prob.dim))))
    epoch = max(1, int(budget) // pop)

    algo_cls = _get_mealpy_cls(which)

    ctor_kw = dict(epoch=epoch, pop_size=pop, verbose=False)
    if "population_size" in getattr(algo_cls.__init__, "__code__", type("", (), {"co_varnames": ()})).co_varnames:
        ctor_kw["population_size"] = pop
    ctor_kw.update(kw)
    algo = algo_cls(**ctor_kw)

    class _MP(MP):
        def __init__(self):
            super().__init__(bounds=FV(prob.lower, prob.upper), minmax="min")
        def obj_func(self, sol):
            return float(evalf(np.asarray(sol, float)))

    best_f = float("inf")
    tried = False
    try:
        tried = True
        ret = algo.solve(_MP(), seed=int(seed))
        if isinstance(ret, tuple) and len(ret) >= 2:
            best_f = float(ret[1])
        elif hasattr(ret, "loss_best"):
            best_f = float(ret.loss_best)
        elif hasattr(ret, "best"):
            best_f = float(ret.best)
        elif np.isscalar(ret):
            best_f = float(ret)
    except TypeError:
        ret = algo.solve(_MP())
        if isinstance(ret, tuple) and len(ret) >= 2:
            best_f = float(ret[1])
    except Exception:
        if not tried:
            raise
    if evalf.nfe == 0:
        x0 = prob.lower + np.random.default_rng(seed).random(prob.dim) * (prob.upper - prob.lower)
        _ = evalf(x0)
    if not np.isfinite(best_f):
        best_f = evalf.best
    return float(best_f), int(evalf.nfe)

# =====================================================================
#  Mapping of CLI names to runner functions
# =====================================================================
ALG_MAP: Dict[str, Callable[[Problem, int, int], Tuple[float, int]]] = {
    "Spiral-LSHADE": run_Spiral-LSHADE,
    "CMAES":      run_cmaes,
    "IPOP-CMAES": run_ipop_cmaes,
    "BIPOP-CMAES": run_bipop_cmaes,
    "SciPyDE":    run_scipy_de,
    "LBFGSB":     run_lbfgsb,
    "Py-BOBYQA":  run_pybobyqa,

    # Preferred external implementations
    "jSO":        run_jso_external,
    "L_SHADE":    run_lshade_external,
    "L-SHADE":    run_lshade_external,

    # Minion-native (must be installed)
    "NLSHADE_RSP": lambda p,b,s: run_minion(p,b,s,"NLSHADE_RSP"),
    "JADE":        lambda p,b,s: run_minion(p,b,s,"JADE"),
    "LSHADE":      lambda p,b,s: run_minion(p,b,s,"LSHADE"),
    "DE":          lambda p,b,s: run_minion(p,b,s,"DE"),
    "j2020":       lambda p,b,s: run_minion(p,b,s,"j2020"),
    "LSRTDE":      lambda p,b,s: run_minion(p,b,s,"LSRTDE"),
    "GWO_DE":      lambda p,b,s: run_minion(p,b,s,"GWO_DE"),
    "ARRDE":       lambda p,b,s: run_minion(p,b,s,"ARRDE"),
    "ABC":         lambda p,b,s: run_minion(p,b,s,"ABC"),
    "DA":          lambda p,b,s: run_minion(p,b,s,"DA"),
    "NelderMead":  lambda p,b,s: run_minion(p,b,s,"NelderMead"),
    "L_BFGS_B":    lambda p,b,s: run_minion(p,b,s,"L_BFGS_B"),

    # Hybrid
    "IPOP-CMAES+BOBYQA": run_ipop_cmaes_bobyqa,

    # Mealpy Swarm/Evo (optional)
    "PSO":  lambda p,b,s: run_mealpy(p, b, s, "pso"),
    "GWO":  lambda p,b,s: run_mealpy(p, b, s, "gwo"),
    "MealpyDE": lambda p,b,s: run_mealpy(p, b, s, "de"),
    "MealpyGA": lambda p,b,s: run_mealpy(p, b, s, "ga"),
    "SHADE": lambda p,b,s: run_mealpy(p, b, s, "shade"),
    "SSA":   lambda p,b,s: run_mealpy(p, b, s, "ssa"),
}

# convenience keys that route through Minion normalizer
ALG_MAP.update({
    "NL_SHADE_RSP": lambda p,b,s: run_minion(p,b,s,"NL_SHADE_RSP"),
    "NLSHADE-RSP":  lambda p,b,s: run_minion(p,b,s,"NLSHADE-RSP"),
    "iL-SHADE-RSP": lambda p,b,s: run_minion(p,b,s,"iL-SHADE-RSP"),
    "LSHADE-cnEpSin": lambda p,b,s: run_minion(p,b,s,"LSHADE-cnEpSin"),
    "mLSHADE-RL":     lambda p,b,s: run_minion(p,b,s,"mLSHADE-RL"),
    "mLSHADE-SPACMA": lambda p,b,s: run_minion(p,b,s,"mLSHADE-SPACMA"),
})

# =====================================================================
#  ERT summary helpers
# =====================================================================
@dataclass
class RunResult:
    alg: str; suite: str; fid: int; dim: int; run: int
    fbest: float; err: float; nfe: int; hit: int; time_sec: float; budget: int

def summarize_ert(rows: List[RunResult], budget_mult: int) -> List[Dict[str, float]]:
    from collections import defaultdict
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r.alg, r.suite, r.fid, r.dim)].append(r)
    summary=[]
    for (alg, suite, fid, dim), lst in grouped.items():
        successes = sum(r.hit for r in lst)
        total_evals = sum(r.nfe if r.hit else dim * budget_mult for r in lst)
        ert = total_evals / successes if successes else float("nan")
        best_err = min(r.err for r in lst)
        mean_err = float(np.mean([r.err for r in lst]))
        summary.append(dict(alg=alg, suite=suite, fid=fid, dim=dim,
                            ERT=ert, succ=successes, runs=len(lst),
                            best_err=best_err, mean_err=mean_err))
    return summary

# =====================================================================
#  Main
# =====================================================================
def parse_ids(spec: str) -> List[int]:
    res=[]
    for part in str(spec).split(","):
        s = part.strip()
        if not s:
            continue
        if "-" in s:
            a, b = map(int, s.split("-"))
            lo, hi = (a, b) if a <= b else (b, a)
            res.extend(range(lo, hi+1))
        else:
            res.append(int(s))
    return sorted(set(res))

def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--suite", choices=["cec2014", "cec2022"], default="cec2022")
    ap.add_argument("--dims",  type=int, nargs="+", default=[10, 20])
    ap.add_argument("--fids",  type=str, default=None, help="e.g. '1-12'")
    ap.add_argument("--runs",  type=int, default=10)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--budget-mult", type=int, default=20000,
                    help="budget = budget_mult × dim")
    ap.add_argument("--target-tol", type=float, default=1e-8)
    ap.add_argument("--algs",  nargs="+", default=["Spiral-LSHADE", "CMAES", "SciPyDE"])
    ap.add_argument("--outdir", type=str, default="results_cec")
    args = ap.parse_args()

    if args.fids is None:
        args.fids = "1-12" if args.suite == "cec2022" else "1-30"
    fids = parse_ids(args.fids)

    os.makedirs(args.outdir, exist_ok=True)
    results_csv = os.path.join(args.outdir, f"{args.suite}_results.csv")
    with open(results_csv, "w", newline="") as fh:
        csv.writer(fh).writerow(
            ["alg","suite","fid","dim","run","fbest","err","nfe","hit","time_sec"])

    all_rows: List[RunResult] = []
    for dim in args.dims:
        budget = args.budget_mult * dim
        for fid in fids:
            try:
                prob = build_problem(args.suite, fid, dim)
            except Exception as e:
                print(f"[SKIP] F{fid:02d} D{dim}: {e}", file=sys.stderr)
                continue
            for run_idx in range(args.runs):
                seed = args.seed0 + run_idx
                for alg in args.algs:
                    if alg not in ALG_MAP:
                        print(f"[SKIP] unknown alg {alg}", file=sys.stderr)
                        continue
                    runner = ALG_MAP[alg]
                    t0 = time.time()
                    try:
                        fbest, nfe = runner(prob, budget, seed)
                    except Exception as e:
                        print(f"[ERR] {alg} F{fid:02d}D{dim}: {e}", file=sys.stderr)
                        fbest, nfe = float("inf"), 0
                    t1 = time.time()
                    err = abs(fbest - prob.fopt)
                    hit = int(err <= args.target_tol)
                    row = RunResult(alg=alg, suite=args.suite, fid=fid, dim=dim,
                                    run=seed, fbest=fbest, err=err, nfe=nfe, hit=hit,
                                    time_sec=t1 - t0, budget=budget)
                    all_rows.append(row)
                    with open(results_csv, "a", newline="") as fh:
                        csv.writer(fh).writerow(
                            [row.alg,row.suite,row.fid,row.dim,row.run,
                             row.fbest,row.err,row.nfe,row.hit,row.time_sec])
                    print(f"{alg:14s}|{args.suite} F{fid:02d} D{dim:02d} run{run_idx:02d} "
                          f"| f_best={fbest:.3e} err={err:.1e} nfe={nfe:7d} {'HIT' if hit else ''}")

    # ------------- ERT summary -------------
    ert_csv = os.path.join(args.outdir, f"{args.suite}_ERT_summary.csv")
    ert_rows = summarize_ert(all_rows, budget_mult=args.budget_mult)
    with open(ert_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["alg","suite","fid","dim","ERT","succ","runs","best_err","mean_err"])
        for d in sorted(ert_rows, key=lambda z: (z["alg"], z["dim"], z["fid"])):
            w.writerow([d["alg"], d["suite"], d["fid"], d["dim"],
                        d["ERT"], d["succ"], d["runs"], d["best_err"], d["mean_err"]])
    print("Saved:", results_csv)
    print("Saved:", ert_csv)

# ---------------------------------------------------------------------
if __name__ == "__main__":
    main()
