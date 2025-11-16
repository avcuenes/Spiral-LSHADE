# -*- coding: utf-8 -*-
"""
slo_bench.bbob_run
------------------
Clean BBOB portfolio runner with target-hit logging.

CLI  (example):
    python3 -m slo_bench.bbob_run                \
        --dims 2 3 5 10                         \
        --functions 1-24                        \
        --instances 1-15                        \
        --algs Spiral-LSHADE CMAES SciPyDE LBFGSB     \
        --budget-mult 4000                      \
        --seed 42                               \
        --outdir exdata
"""
from __future__ import annotations
import argparse, csv, os, sys, time, importlib
from typing import Callable, Dict, List, Tuple, Any
import numpy as np
import cocoex

# optional libs
try: import cocopp; HAVE_COCOPP = True
except Exception: HAVE_COCOPP = False
try: import cma;   HAVE_CMA    = True
except Exception: HAVE_CMA    = False
try:
    from scipy.optimize import minimize, differential_evolution
    HAVE_SCIPY = True
except Exception: HAVE_SCIPY = False
try:
    from mealpy.swarm_based import GWO, PSO, WOA, SSA
    from mealpy.evolutionary_based import SHADE, GA
    from mealpy import Problem as MealpyProblem, FloatVar
    HAVE_MEALPY = True
except Exception: HAVE_MEALPY = False

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataclasses import fields
# minionpy (preferred external differential evolution family)
try:
    import minionpy as mpy
    HAVE_MINION = True
except Exception:
    HAVE_MINION = False


def _bounds_list(lb, ub):
    return list(zip(lb.tolist(), ub.tolist()))

def _vectorized_counter_func(f):
    """Return (fvec, counter_dict) so we can report n_evals accurately."""
    calls = {"n": 0}
    def fvec(X):
        X = np.asarray(X, float)
        if X.ndim == 1:
            X = X[None, :]
        out = []
        for x in X:
            calls["n"] += 1
            out.append(float(f(x)))
        return out
    return fvec, calls

def run_minion_algo(algo_name: str):
    """Factory: returns a runner(f, lb, ub, budget, seed, kw) using minionpy."""
    if not HAVE_MINION:
        raise RuntimeError("minionpy not installed")
    def _runner(f, lb, ub, budget, seed, kw):
        fvec, calls = _vectorized_counter_func(f)
        opt = mpy.Minimizer(
            func=fvec,
            x0=None,
            bounds=_bounds_list(lb, ub),
            algo=algo_name,                 # 'jSO', 'LSHADE', 'JADE', 'NLSHADE_RSP'
            relTol=0.0,
            maxevals=int(budget),
            seed=int(seed),
            options=None
        )
        res = opt.optimize()
        return float(res.fun), int(calls["n"])
    return _runner

def build_port(selected: List[str]):
    port = []
    for n in selected:
        u = n.upper()

        # --- Your hybrid & classical baselines ---
        if u in ("Spiral-LSHADE", "SLO_HYBRID"):
            port.append(("Spiral-LSHADE", run_spiral_lshade_hybrid))
        elif u == "CMAES":
            port.append(("CMAES", run_cmaes))
        elif u in ("SCIPYDE", "DE"):
            port.append(("SciPyDE", run_scipy_de))
        elif u == "LBFGSB":
            port.append(("LBFGSB", run_lbfgsb))

        # --- Minion family (preferred) + graceful fallbacks ---
        elif u in ("JSO",):
            if HAVE_MINION:
                port.append(("jSO", run_minion_algo("jSO")))
            elif HAVE_MEALPY:
                # approximate fallback with SHADE
                from mealpy.evolutionary_based import SHADE
                port.append(("jSO", lambda *a, **k: run_mealpy(SHADE.OriginalSHADE, *a, **k)))
            else:
                print(f"[SKIP] {n} unavailable (need minionpy or mealpy)", file=sys.stderr)

        elif u in ("LSHADE", "L_SHADE", "L-SHADE"):
            if HAVE_MINION:
                port.append(("LSHADE", run_minion_algo("LSHADE")))
            elif HAVE_MEALPY:
                from mealpy.evolutionary_based import SHADE
                port.append(("LSHADE", lambda *a, **k: run_mealpy(SHADE.OriginalSHADE, *a, **k)))
            else:
                print(f"[SKIP] {n} unavailable (need minionpy or mealpy)", file=sys.stderr)

        elif u in ("NLSHADE-RSP", "NLSHADE_RSP", "NL_SHADE_RSP"):
            if HAVE_MINION:
                port.append(("NLSHADE_RSP", run_minion_algo("NLSHADE_RSP")))
            elif HAVE_MEALPY:
                # use SHADE as closest available fallback
                from mealpy.evolutionary_based import SHADE
                port.append(("NLSHADE_RSP", lambda *a, **k: run_mealpy(SHADE.OriginalSHADE, *a, **k)))
            else:
                print(f"[SKIP] {n} unavailable (need minionpy or mealpy)", file=sys.stderr)

        elif u == "JADE":
            if HAVE_MINION:
                port.append(("JADE", run_minion_algo("JADE")))
            elif HAVE_MEALPY:
                # fall back to DE (not JADE, but similar family)
                from mealpy.evolutionary_based import DE
                port.append(("JADE", lambda *a, **k: run_mealpy(DE.OriginalDE, *a, **k)))
            else:
                print(f"[SKIP] {n} unavailable (need minionpy or mealpy)", file=sys.stderr)

        # --- Mealpy swarm/evo (when available) ---
        elif u == "GWO" and HAVE_MEALPY:
            from mealpy.swarm_based import GWO
            port.append(("GWO", lambda *fargs, **k: run_mealpy(GWO.OriginalGWO, *fargs, **k)))
        elif u == "PSO" and HAVE_MEALPY:
            from mealpy.swarm_based import PSO
            port.append(("PSO", lambda *fargs, **k: run_mealpy(PSO.OriginalPSO, *fargs, **k)))
        elif u in ("GA", "MEALPYGA") and HAVE_MEALPY:
            from mealpy.evolutionary_based import GA
            # GA in mealpy often exposes BaseGA; use it if OriginalGA doesn't exist
            try:
                port.append(("MealpyGA", lambda *fargs, **k: run_mealpy(GA.OriginalGA, *fargs, **k)))
            except AttributeError:
                port.append(("MealpyGA", lambda *fargs, **k: run_mealpy(GA.BaseGA, *fargs, **k)))
        elif u == "SSA" and HAVE_MEALPY:
            from mealpy.swarm_based import SSA
            port.append(("SSA", lambda *fargs, **k: run_mealpy(SSA.OriginalSSA, *fargs, **k)))

        else:
            print(f"[SKIP] {n} unavailable", file=sys.stderr)

    return port

# -----------------------------------------------------------------------------
#  helpers
# -----------------------------------------------------------------------------
TARGETS = [1e-1, 1e-3, 1e-5, 1e-8]

def project_open_box(x: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64); lb = lb.astype(np.float64); ub = ub.astype(np.float64)
    rng = ub - lb
    eps = np.maximum(1e-8 * np.maximum(rng, 1.0), 1e-12)
    eps = np.minimum(eps, 0.45 * np.maximum(rng, 1e-30))
    return np.minimum(ub - eps, np.maximum(lb + eps, x))

def rand_open_box(d: int, lb: np.ndarray, ub: np.ndarray, seed: int) -> np.ndarray:
    """Uniform random point strictly inside (lb, ub)."""
    lb = np.asarray(lb, dtype=float)
    ub = np.asarray(ub, dtype=float)
    rng = np.random.default_rng(seed)
    x = lb + (ub - lb) * rng.random(d)
    # keep strictly in the open box (avoid hitting bounds exactly)
    eps = 1e-12
    return np.minimum(np.maximum(x, lb + eps * (ub - lb)), ub - eps * (ub - lb))

def parse_ids(spec: str) -> List[int]:
    res = []
    for part in spec.split(','):
        part = part.strip()
        if not part: continue
        if '-' in part:
            a,b = map(int, part.split('-'))
            if a<=b: res.extend(range(a,b+1))
            else:    res.extend(range(a,b-1,-1))
        else: res.append(int(part))
    return sorted(set(res))


# -----------------------------------------------------------------------------
#  Algorithm runners
# -----------------------------------------------------------------------------

from spiral_lshade import _slo_lshade_core, SpiralLSHADEParams

def run_spiral_lshade_hybrid(
    f: Callable[[np.ndarray], float],
    lb: np.ndarray,
    ub: np.ndarray,
    budget: int,
    seed: int,
    kw: Optional[Dict] = None,
) -> Tuple[float, int]:
    """Adapter matching (f, lb, ub, budget, seed, kw) → (f_best, n_evals)."""
    # normalise kw handling
    if isinstance(kw, SpiralLSHADEParams):
        kw = {"params": kw}
    kw = dict(kw) if kw else {}
    # pull params & rng seed
    p = kw.get("params", SpiralLSHADEParams())
    rng_seed_val = kw.pop("rng_seed", seed)
    if isinstance(rng_seed_val, np.random.Generator):
        rng = rng_seed_val
    else:
        rng = np.random.default_rng(int(rng_seed_val))

    lb = np.asarray(lb, float); ub = np.asarray(ub, float)
    best_f, evals = _slo_lshade_core(f, lb, ub, budget, rng, p)
    return float(best_f), int(evals)


def run_cmaes(f, lb, ub, budget, seed, kw):
    if not HAVE_CMA: raise RuntimeError("cma not installed")
    x0 = (lb+ub)/2
    sigma0 = kw.get("sigma0", 0.3*float(np.max(ub-lb)))
    opts = dict(bounds=[lb,ub], maxfevals=int(budget), seed=int(seed), verbose=-9, CMA_active=True)
    opts.update({k:v for k,v in kw.items() if k not in ("sigma0",)})
    _, fb, _, ne, *_ = cma.fmin(f, x0, sigma0, options=opts)
    return float(fb), int(ne)

def run_scipy_de(f, lb, ub, budget, seed, kw):
    if not HAVE_SCIPY: raise RuntimeError("scipy not installed")
    dim, pop = lb.size, int(kw.get("popsize",15))
    res = differential_evolution(f, list(zip(lb,ub)),
                                 popsize=pop, maxiter=max(1,budget//(pop*dim)),
                                 polish=False, updating='deferred', seed=seed)
    return float(res.fun), int(res.nfev)

def run_lbfgsb(f, lb, ub, budget, seed, kw):
    if not HAVE_SCIPY: raise RuntimeError("scipy not installed")
    x0 = rand_open_box(lb.size, lb, ub, seed)
    res = minimize(f, x0, method="L-BFGS-B", bounds=list(zip(lb,ub)),
                   options={'maxfun':int(budget),'disp':False})
    return float(res.fun), int(getattr(res,'nfev',0))

def run_mealpy(mealpy_cls, f, lb, ub, budget, seed, kw):
    if not HAVE_MEALPY:
        raise RuntimeError("mealpy not installed")
    pop = int(kw.get("pop_size", 50))
    epoch = max(1, budget // pop)

    class P(MealpyProblem):  # wrap
        def __init__(self):
            super().__init__(bounds=FloatVar(lb, ub), minmax="min")
        def obj_func(self, sol): 
            return float(f(np.asarray(sol, float)))

    ctor_kwargs = {k: v for k, v in kw.items() if k not in ('pop_size',)}
    try:
        algo = mealpy_cls(epoch=epoch, pop_size=pop, verbose=False, **ctor_kwargs)
    except TypeError:
        # some versions don't accept verbose
        algo = mealpy_cls(epoch=epoch, pop_size=pop, **ctor_kwargs)

    # some versions return (pos, fit), others an object; extract a float
    ret = algo.solve(P())
    if isinstance(ret, tuple) and len(ret) >= 2:
        best_fit = float(ret[1])
    elif hasattr(ret, "loss_best"):
        best_fit = float(ret.loss_best)
    elif hasattr(ret, "best"):
        best_fit = float(ret.best)
    else:
        best_fit = float(ret)

    return float(best_fit), int(epoch * pop)





# def build_port(selected: List[str]):
#     port=[]
#     for n in selected:
#         u=n.upper()
#         if u in ("Spiral-LSHADE","Spiral-LSHADE"): port.append(("Spiral-LSHADE",run_spiral_lshade_hybrid))
#         elif u=="CMAES":                port.append(("CMAES",run_cmaes))
#         elif u in ("SCIPYDE","DE"):     port.append(("SciPyDE",run_scipy_de))
#         elif u=="LBFGSB":               port.append(("LBFGSB",run_lbfgsb))
#         elif u=="GWO"   and HAVE_MEALPY:port.append(("GWO", lambda*fargs,**k: run_mealpy(GWO.OriginalGWO,*fargs,**k)))
#         elif u=="PSO"   and HAVE_MEALPY:port.append(("PSO", lambda*fargs,**k: run_mealpy(PSO.OriginalPSO,*fargs,**k)))
#         elif u=="WOA"   and HAVE_MEALPY:port.append(("WOA", lambda*fargs,**k: run_mealpy(WOA.OriginalWOA,*fargs,**k)))
#         elif u=="SSA"   and HAVE_MEALPY:port.append(("SSA", lambda*fargs,**k: run_mealpy(SSA.OriginalSSA,*fargs,**k)))
#         elif u=="SHADE" and HAVE_MEALPY:port.append(("SHADE",lambda*fargs,**k: run_mealpy(SHADE.OriginalSHADE,*fargs,**k)))
#         elif u=="GA"    and HAVE_MEALPY:port.append(("GA",   lambda*fargs,**k: run_mealpy(GA.BaseGA,*fargs,**k)))
#         else: print(f"[SKIP] {n} unavailable", file=sys.stderr)
#     return port

# -----------------------------------------------------------------------------
#  Suite runner
# -----------------------------------------------------------------------------
def run_suite(alg_name, runner, dims, fids, insts, outdir, budget_mult, seed, quiet):
    req_folder = os.path.join(outdir, alg_name)
    obs = cocoex.Observer("bbob", f"result_folder: {req_folder} algorithm_name: {alg_name}")
    # suite = cocoex.Suite("bbob","year:2009",
    #      f"dimensions:{','.join(map(str,dims))} "
    #      f"function_indices:{','.join(map(str,fids))} "
    #      f"instance_indices:{','.join(map(str,insts))}")
    suite = cocoex.Suite("bbob", "instances: 1-15",
        f"dimensions:{','.join(map(str,dims))} "
        f"function_indices:{','.join(map(str,fids))}")

    csv_path=os.path.join(obs.result_folder,"results_bbob_summary.csv")
    with open(csv_path,"w",newline="") as fh:
        csv.writer(fh).writerow(
            ["alg","fid","inst","dim","best_f","evals_total",*[f"t_hit_{t:g}" for t in TARGETS],"duration_sec"])

    for pr in suite:
        pr.observe_with(obs)
        dim = pr.dimension
        fid = int(getattr(pr,"id_function",0)); inst=int(getattr(pr,"id_instance",0))
        lb,ub = np.asarray(pr.lower_bounds), np.asarray(pr.upper_bounds)
        hits={t:None for t in TARGETS}; nfe=0; best=np.inf
        def f(x,pr=pr,lb=lb,ub=ub):
            nonlocal nfe,best,hits
            nfe+=1; y=float(pr(project_open_box(x,lb,ub)))
            if y<best:
                best=y
                for t in TARGETS:
                    if hits[t] is None and best<=t: hits[t]=nfe
            return y
        bud=budget_mult*dim; t0=time.time()
        try: best_f, tot=runner(f,lb,ub,bud,seed,{})
        except Exception as e:
            print(f"[{alg_name}] error f{fid}i{inst}d{dim}: {e}",file=sys.stderr)
            best_f, tot=float("inf"), nfe
        t1=time.time()
        with open(csv_path,"a",newline="") as fh:
            csv.writer(fh).writerow([alg_name,fid,inst,dim,best_f,tot,*[(hits[t] or -1) for t in TARGETS],t1-t0])
        if not quiet:
            hstr=", ".join([f"t{t}:{hits[t] or '-'}" for t in TARGETS])
            print(f"{alg_name:10s}|f{fid:02d}i{inst:02d}d{dim:02d}|best={best_f:.1e} evals={tot:5d}|{hstr}")
    print(f"[{alg_name}] data in {obs.result_folder}")
    return obs.result_folder

# -----------------------------------------------------------------------------
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--dims",nargs="+",type=int,default=[2,3,5,10])
    ap.add_argument("--functions",default="1-24")
    ap.add_argument("--instances",default="1-15")
    ap.add_argument("--algs",nargs="+",default=["CMAES","SciPyDE","LBFGSB"])
    ap.add_argument("--budget-mult",type=int,default=1000)
    ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--outdir",default="exdata")
    ap.add_argument("--quiet",action="store_true")
    ap.add_argument("--no-cocopp",action="store_true")
    args=ap.parse_args()

    os.makedirs(args.outdir,exist_ok=True)
    port=build_port(args.algs)
    if not port: sys.exit("No valid algs.")

    dims=args.dims; fids=parse_ids(args.functions); insts=parse_ids(args.instances)
    folders=[]
    for name,run in port:
        folders.append(run_suite(name,run,dims,fids,insts,args.outdir,args.budget_mult,args.seed,args.quiet))

    print("\nFolders:",*folders,sep="\n  - ")
    if HAVE_COCOPP and not args.no_cocopp:
        print("Running cocopp …")
        import cocopp; cocopp.main(folders)
        print("Done ➜ ppdata/index.html")

if __name__=="__main__": main()
