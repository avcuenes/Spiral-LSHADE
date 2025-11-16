# -*- coding: utf-8 -*-
"""
slo_bench.eng_run  (formerly cec_run2)
--------------------------------------
Benchmark runner with multiple optimizers.

Supports:
- Your engineering problems suite: --suite eng
- (Optional) CEC-2014 / CEC-2022 via opfunu if installed

CSV per-run log  →  <outdir>/<suite>_results.csv
CSV ERT summary →  <outdir>/<suite>_ERT_summary.csv
ENG stats       →  <outdir>/eng_mean_std.csv
ENG plot        →  <outdir>/eng_err_mean_bar.png

Examples
--------
# Run your engineering problems (IDs 1..16)
python3 -m slo_bench.cec_run2 \
  --suite eng --fids 1-16 --runs 5 \
  --algs Spiral-LSHADE CMAES SciPyDE LBFGSB \
  --budget-mult 20000 --target-tol 1e-8 \
  --outdir results_eng
"""
from __future__ import annotations
import argparse, csv, os, sys, time, importlib, warnings, math
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
#  Optional: Your Spiral-LSHADE hybrid (Spiral-LSHADE)
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

try:    import pybobyqa;                                          HAVE_PYBOBYQA = True
except Exception:                                                 HAVE_PYBOBYQA = False

try:
    import pyade.jso   as pyade_jso
    import pyade.lshade as pyade_lshade
    HAVE_PYADE = True
except Exception:
    HAVE_PYADE = False

try:    import minionpy as mpy;                                   HAVE_MINION = True
except Exception:                                                 HAVE_MINION = False

# OPFUNU – CEC suites (optional)
try:    from opfunu.cec_based import cec2014 as ofu2014;          HAVE_2014   = True
except Exception:                                                 HAVE_2014   = False
try:    from opfunu.cec_based import cec2022 as ofu2022;          HAVE_2022   = True
except Exception:                                                 HAVE_2022   = False


# =====================================================================
#  Penalty utilities and your engineering problem classes
# =====================================================================
ALPHA = 1e7
BETA  = 2

def _viol(x): return np.maximum(x, 0.0)

class _Problem:
    name:  str
    dim:   int
    lower: np.ndarray
    upper: np.ndarray
    fopt:  float | None
    def f_raw(self, x): ...
    def g(self, x): ...
    def f(self, x):
        # robust, positive penalty objective
        try:
            val = float(self.f_raw(x))
        except Exception:
            return 1e300
        if not np.isfinite(val):
            return 1e300
        try:
            v = np.sum(_viol(self.g(x))**BETA)
        except Exception:
            return 1e300
        tot = val + ALPHA*v
        return tot if np.isfinite(tot) else 1e300

# --- 1.1 Pressure vessel (continuous) ----------------------------------
class PressureVessel(_Problem):
    name  = "pressure_vessel"
    lower = np.array([0.0, 0.0, 10.0, 10.0])
    upper = np.array([99. , 99. , 200. , 240. ])  # align with L-constraint
    dim   = 4
    fopt  = 6059.714
    def f_raw(self,x):
        Ts,Th,R,L = x
        return (0.6224*Ts*R*L + 1.7781*Th*R**2 +
                3.1661*Ts**2*L + 19.84*Ts**2*R)
    def g(self,x):
        Ts,Th,R,L = x
        g1 = -Ts + 0.0193*R
        g2 = -Th + 0.00954*R
        g3 = -math.pi*R**2*L - (4/3)*math.pi*R**3 + 750*1728
        g4 =  L - 240
        return np.array([g1,g2,g3,g4])

# --- 1.2 Spring --------------------------------------------------------
class Spring(_Problem):
    name="tension_spring"
    lower=np.array([0.05,0.25,11.0]); upper=np.array([2.0,1.3,12.0]); dim=3
    fopt=2.6254
    def f_raw(self,x): return (x[2]+2)*x[1]*x[0]**2
    def g(self,x):
        d,D,N = x
        g1 = 1 - (D**3*N)/(71785*d**4)
        g2 = (4*D**2 - d*D)/(12566*(D*d**3 - d**4)) + 1/(5108*d**2) - 1
        g3 = 1 - 140.45*d/(D**2*N)
        g4 = (D+d)/1.5 - 1
        return np.array([g1,g2,g3,g4])

# --- 1.3 Welded-beam (corrected σ, δ) ---------------------------------
P,L,E,G = 6000.,14.,30e6,12e6
tau_max,sig_max,delta_max = 13600.,30000.,0.25
class WeldedBeam(_Problem):
    name="welded_beam"; lower=np.array([0.1,0.1,0.1,0.1])
    upper=np.array([2.0,3.5,10.0,2.0]); dim=4; fopt=1.724852
    # x = [h, l, b, t]
    def f_raw(self,x): h,l,b,t = x; return 1.10471*h**2*l + 0.04811*b*t*(14+l)
    def _tau(self,x):
        h,l,b,t = x
        tau1 = P/(math.sqrt(2)*h*l)
        M    = P*(L + l/2)
        R    = math.sqrt(l**2/4 + ((h + b)/2)**2)
        J    = 2*((h*l*math.sqrt(2))*(l**2/12 + ((h + b)/2)**2))
        tau2 = M*R/J
        return math.sqrt(tau1**2 + tau1*tau2*l/R + tau2**2)
    def _delta(self,x): _,_,b,t = x; return 4*P*L**3/(E*b*t**3)
    def _sigma(self,x): _,_,b,t = x; return 6*P*L/(b*t**2)
    def _Pc(self,x):
        _,_,b,t = x
        return (4.013*E*math.sqrt(b**2*t**6/36)/L**2) * (1 - b/(2*L)*math.sqrt(E/(4*G)))
    def g(self,x):
        g = np.zeros(7)
        g[0]= self._tau(x)-tau_max
        g[1]= self._sigma(x)-sig_max
        g[2]= x[0]-x[3]
        g[3]= 0.10471*x[0]**2 + 0.04811*x[2]*x[3]*(14+x[1]) - 5
        g[4]= 0.125 - x[0]
        g[5]= self._delta(x)-delta_max
        g[6]= P - self._Pc(x)
        return g

# --- 1.4 Gear-box (custom surrogate) ----------------------------------
i,rho,n,sigma_y = 4,8,6,294.3
y_,b2_,Kv,Kw,N1,Pwr = 0.102,0.193,0.389,0.8,1500,7.5
class GearBox(_Problem):
    name="gear_box"; lower=np.array([20,10,30,18,2.75])
    upper=np.array([32,30,40,25,4]); dim=5; fopt=None
    def f_raw(self,x):
        b,d1,d2,Z1,m=x
        Dr=m*(i*Z1-2.5); lw=2.5*m; Di=Dr-2*lw; bw=3.5*m
        d0=d2+25; dp=0.25*(Di-d0)
        return (math.pi/4)*(rho/1000)*(b*m**2*Z1**2*(i**2+1)
              -(Di**2-d0**2)*(1-bw) - n*dp**2*bw - (d1+d2)*b)
    def g(self,x):
        b,d1,d2,Z1,m=x
        Dr=m*(i*Z1-2.5); lw=2.5*m; Di=Dr-2*lw; d0=d2+25
        D1=m*Z1; D2=i*m*Z1; Z2=Z1*D2/D1
        v=math.pi*D1*N1/60000; b1=102*Pwr/v; b3=4.97e6*Pwr/(N1*2)
        Fs=math.pi*Kv*Kw*sigma_y*b*m*y_; Fp=2*Kv*Kw*D1*b*Z2/(Z1+Z2)
        g1=-Fs+b1; g2=-(Fs/Fp)+b2_; g3=-(d1**3)+b3
        return np.array([g1,g2,g3])

# --- 1.5 Speed-reducer (corrected) ------------------------------------
class SpeedReducer(_Problem):
    name  = "speed_reducer"
    lower = np.array([2.6 ,0.7 ,17. ,7.3 ,7.3 ,2.9 ,5.0])
    upper = np.array([3.6 ,0.8 ,28. ,8.3 ,8.3 ,3.9 ,5.5])
    dim   = 7
    fopt  = 2994.0
    def f_raw(self, x):
        x1,x2,x3,x4,x5,x6,x7 = x
        return (0.7854*x1*x2**2*(3.3333333333*x3**2 + 14.933*x3 - 43.0934)
              - 1.508*x1*(x6**2 + x7**2)
              + 7.477*(x6**3 + x7**3)
              + 0.7854*(x4*x6**2 + x5*x7**2))  # fixed
    def g(self, x):
        x1,x2,x3,x4,x5,x6,x7 = x
        g = np.zeros(11)
        g[0]  = 27/(x1*x2**2*x3) - 1
        g[1]  = 397.5/(x1*x2**2*x3**2) - 1
        g[2]  = 1.93*x4**3/(x2*x3*x6**4) - 1
        g[3]  = 1.93*x5**3/(x2*x3*x7**4) - 1
        g[4]  = math.sqrt((745*x4/(x2*x3))**2 + 1.69e6)/(110*x6**3) - 1  # fixed
        g[5]  = math.sqrt((745*x5/(x2*x3))**2 + 1.69e6)/(110*x7**3) - 1  # fixed
        g[6]  = x2*x3 - 40
        g[7]  = x1/x2 - 12
        g[8]  = 5 - x1/x2
        g[9]  = 1.9 - x4 + 1.5*x6
        g[10] = 1.9 - x5 + 1.1*x7
        return g

# --- 1.6 Car side-impact (minor index fix) ----------------------------
class CarSideImpact(_Problem):
    name  = "car_side_impact"
    lower = np.array([0.5 ,0.45 ,0.5 ,0.5 ,0.875 ,0.4 ,0.4 ,0.5 ,0.5 ,0.875 ,0.4])
    upper = np.array([1.5 ,1.35 ,1.5 ,1.5 ,2.625,1.2 ,1.2 ,1.5 ,1.5 ,2.625 ,1.2])
    dim   = 11
    fopt  = None
    def f_raw(self, x):
        return (1.98 + 4.9*x[0] + 6.67*x[1] + 6.98*x[2]
              + 4.01*x[3] + 1.78*x[4] + 2.73*x[5])
    def g(self, x):
        g = np.zeros(10)
        g[0] = 1.16 - 0.3717*x[1]*x[3] - 0.0092928*x[2]
        g[1] = 0.261 - 0.0159*x[1]*x[2] - 0.06486*x[0]
        g[2] = 0.214 + 0.00817*x[4] - 0.045195*x[1] - 0.0135168*x[2]  # fixed index
        g[3] = 0.74  - 0.698*x[4] - 0.173*x[5]
        g[4] = 0.68  - 0.2*x[2] - 0.051*x[3] - 0.102*x[0]
        g[5] = 1.0   - 0.1353*x[0] + 0.007283*x[3] + 0.009498*x[2]
        g[6] = (math.pi/4)*(x[0]**2 - x[6]**2) - 40
        g[7] = (math.pi/4)*(x[5]**2 - x[7]**2) - 20
        g[8] = (math.pi/4)*(x[4]**2*x[8] - x[5]**2*x[8])
        g[9] = x[9] - x[10]
        return g

# --- 1.7 Hydrostatic thrust bearing ----------------------------------
class HydrostaticBearing(_Problem):
    name  = "hydrostatic_thrust_bearing"
    lower = np.array([0.5 ,0.35,17.0,7.0])
    upper = np.array([1.1 ,0.95,28.0,1.9])
    dim   = 4
    fopt  = None
    def f_raw(self, x):
        x1,x2,x3,x4 = x
        return 4.9e-5*x1**(-0.5)*x2**2*math.sqrt(x3) + 1.62e-3*x4/x3
    def g(self, x):
        x1,x2,x3,x4 = x
        return np.array([
            -x1 + 0.5,   x1 - 1.1,
            -x2 + 0.35,  x2 - 0.95,
            -x3 + 17.0,  x3 - 28.0,
            x4 - 1.9])

# --- 1.8 Four-bar truss ----------------------------------------------
class FourBarTruss(_Problem):
    name="four_bar_truss"
    lower=np.array([1.,1.,1.]); upper=np.array([5.,5.,5.]); dim=3
    fopt =1.0
    def f_raw(self,x): return x[0]*x[2]*(1+x[1])
    def g(self,x):
        g1=(x[0]+2*x[1])*x[2]-5
        g2=x[0]*x[1]-25
        g3=x[0]-2*x[1]
        return np.array([g1,g2,g3])

# --- 1.9 Ten-bar planar truss (surrogate; no canonical fopt) ---------
LEN  = np.array([1,1,math.sqrt(2),math.sqrt(2),1,1,math.sqrt(2),math.sqrt(2),math.sqrt(2),math.sqrt(2)])
RHO  = 0.1
class TenBarTruss(_Problem):
    name  = "ten_bar_truss"
    lower = np.full(10,0.1); upper = np.full(10,35.0); dim=10
    fopt  = None  # surrogate scale ≠ canonical (505)
    def f_raw(self,x): return RHO*np.dot(LEN,x)
    def g(self,x):
        σ = 5.0/x
        return σ/25.0 - 1

# --- 1.10 Cantilever beam (continuous) --------------------------------
class CantileverCont(_Problem):
    name="cantilever_cont"
    lower=np.array([0.01,0.01]); upper=np.array([0.2,0.2]); dim=2; fopt=None
    _P,_L,_E = 500,200,2e5
    def f_raw(self,x):
        b,h=x; return self._P*self._L**3/(3*self._E*b*h**3)
    def g(self,x):
        b,h=x; σ=6*self._P*self._L/(b*h**2)
        return np.array([σ-140])

# --- 1.11 Cantilever beam (discrete-section) (fixed g) ----------------
_SECT=np.array([1,2,3,4,5,6])/10
class CantileverDisc(_Problem):
    name="cantilever_disc"
    lower=np.zeros(5); upper=np.full(5,5); dim=5; fopt=None
    def _round_idx(self,idx): return np.clip(np.round(idx).astype(int),0,5)
    def f_raw(self,idx):
        A=_SECT[self._round_idx(idx)]
        return 100*np.sum(A)
    def g(self,idx):
        A=_SECT[self._round_idx(idx)]
        return np.array([(1000.0/A).max() - 140.0])  # fixed

# --- 1.12 Stepped-column buckling (surrogate) -------------------------
class SteppedColumn(_Problem):
    name="stepped_column"
    lower=np.full(10,0.1); upper=np.full(10,10.0); dim=10; fopt=None
    def f_raw(self,A): return 0.1*np.sum(A)
    def g(self,A):
        Pcr=(math.pi**2)*2.1e5*np.sum(A)/(100**2)
        return np.array([1e5/Pcr-1])

# --- 1.13 Machining parameters ----------------------------------------
class MachiningCost(_Problem):
    name="machining_cost"
    lower=np.array([0.5,0.1,50,0.01]); upper=np.array([2.0,0.5,400,0.05])
    dim=4; fopt=2.385
    def f_raw(self,x):
        d,f,v,t=x
        return 0.8*d+0.01*f+0.0005*v+0.2*t
    def g(self,x):
        d,f,v,_=x
        g1=1-0.002*v*f
        g2=0.1-d+0.5*f
        return np.array([g1,g2])

# --- 1.14 Heat-exchanger (variant; fopt unknown here) -----------------
class HeatExchanger(_Problem):
    name="heat_exchanger"
    lower=np.array([0.3,5,100]); upper=np.array([1.5,9,600]); dim=3; fopt=None
    def f_raw(self,x):
        D,L,u=x
        return 0.6224*D*L*u+131*D+146.3
    def g(self,x):
        D,L,u=x
        Q=0.8*D*L*u; return np.array([60-Q])

# --- 1.15 Thick-wall pressure vessel (discrete Ts,Th) -----------------
_mult=0.0625
class ThickPressureVessel(_Problem):
    name="pressure_vessel_thick"
    lower=np.array([0.0625,0.0625,10,10])
    upper=np.array([2,2,100,240]); dim=4; fopt=5885.3
    def _snap(self,x): return np.round(x/_mult)*_mult
    def f_raw(self,x):
        Ts,Th,R,L=self._snap(x[0]),self._snap(x[1]),x[2],x[3]
        return (0.6224*Ts*R*L+1.7781*Th*R**2+3.1661*Ts**2*L+19.84*Ts**2*R)
    def g(self,x):
        Ts,Th,R,L=self._snap(x[0]),self._snap(x[1]),x[2],x[3]
        return np.array([
            -Ts+0.0625, -Th+0.0625, -R+10, R-100, -L+10, L-240,
            math.pi*R**2*L+4/3*math.pi*R**3-1.0e5])

# --- 1.16 Gear-train integer ------------------------------------------
_target=1/6.931
class GearTrain(_Problem):
    name="gear_train"
    lower=np.full(4,12); upper=np.full(4,60); dim=4; fopt=2.7e-4
    def _round(self,z): return np.clip(np.round(z).astype(int),12,60)
    def f_raw(self,z):
        z=self._round(z)
        ratio=(z[0]*z[1])/(z[2]*z[3])
        return abs(ratio-_target)
    def g(self,z): return np.zeros(1)

# Registry for engineering suite
ENGINEERING_PROBLEMS = {
     1: ("pressure_vessel",            PressureVessel),
     2: ("tension_spring",             Spring),
     3: ("welded_beam",                WeldedBeam),
     4: ("gear_box",                   GearBox),
     5: ("speed_reducer",              SpeedReducer),
     6: ("car_side_impact",            CarSideImpact),
     7: ("hydrostatic_thrust_bearing", HydrostaticBearing),
     8: ("four_bar_truss",             FourBarTruss),
     9: ("ten_bar_truss",              TenBarTruss),
    10: ("cantilever_cont",            CantileverCont),
    11: ("cantilever_disc",            CantileverDisc),
    12: ("stepped_column",             SteppedColumn),
    13: ("machining_cost",             MachiningCost),
    14: ("heat_exchanger",             HeatExchanger),
    15: ("pressure_vessel_thick",      ThickPressureVessel),
    16: ("gear_train",                 GearTrain),
}


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

    # ----- Your engineering suite -----
    if suite == "eng":
        if fid not in ENGINEERING_PROBLEMS:
            raise ValueError(f"fid {fid} not in suite 'eng' (valid 1..{len(ENGINEERING_PROBLEMS)})")
        pname, pcls = ENGINEERING_PROBLEMS[fid]
        try:
            pobj = pcls()
        except NameError as e:
            raise RuntimeError(f"{pcls.__name__} must be defined: {e}")

        lower = np.asarray(pcls.lower, dtype=float).copy()
        upper = np.asarray(pcls.upper, dtype=float).copy()
        pdim  = int(pcls.dim)
        fopt_attr = getattr(pcls, "fopt", None)
        fopt  = float("nan") if (fopt_attr is None) else float(fopt_attr)

        def f(x: np.ndarray) -> float:
            return float(pobj.f(np.asarray(x, float)))

        return Problem(fid=fid, name=pname, dim=pdim, lower=lower, upper=upper, fopt=fopt, f=f)

    # ----- Original CEC suites (optional) -----
    if suite == "cec2014":
        if not HAVE_2014:
            raise RuntimeError("opfunu cec2014 not available")
        fmap = {i: getattr(ofu2014, f"F{i}2014") for i in range(1, 31)}
    elif suite == "cec2022":
        if not HAVE_2022:
            raise RuntimeError("opfunu cec2022 not available")
        fmap = {i: getattr(ofu2022, f"F{i}2022") for i in range(1, 13)}
    else:
        raise ValueError("suite must be 'eng', 'cec2014', or 'cec2022'")

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
#  Evaluation counter with early-hit (robust to unknown fopt)
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
        self._check_hit = np.isfinite(self.fopt)  # False if NaN/inf

    def __call__(self, x: np.ndarray) -> float:
        y = float(self.f(x)); self.nfe += 1
        if y < self.best:
            self.best = y
            if self._check_hit and abs(self.best - self.fopt) <= self.target_tol:
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
        # ERT: if no successes, NaN
        total_evals = sum(r.nfe if r.hit else dim * budget_mult for r in lst)
        ert = total_evals / successes if successes else float("nan")
        errs = np.array([r.err for r in lst], dtype=float)
        best_err = float(np.nanmin(errs)) if np.any(np.isfinite(errs)) else float("nan")
        mean_err = float(np.nanmean(errs)) if np.any(np.isfinite(errs)) else float("nan")
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

def _write_eng_stats_and_plot(all_rows: List[RunResult], outdir: str):
    """Create eng_mean_std.csv and eng_err_mean_bar.png (Seaborn preferred)."""
    try:
        import pandas as pd
    except Exception as e:
        print(f"[WARN] pandas not available; skipping ENG stats: {e}", file=sys.stderr)
        return

    # Build DataFrame
    df = pd.DataFrame([r.__dict__ for r in all_rows if r.suite == "eng"])
    if df.empty:
        print("[WARN] No ENG rows to summarize.", file=sys.stderr)
        return

    # Map fid → problem name
    fid2name = {fid: name for fid, (name, _) in ENGINEERING_PROBLEMS.items()}
    df["problem"] = df["fid"].map(fid2name)

    # Mean / std per alg × problem × dim
    stats = (df.groupby(["alg","fid","problem","dim"], dropna=False)
               .agg(fbest_mean=("fbest","mean"),
                    fbest_std =("fbest","std"),
                    err_mean  =("err","mean"),
                    err_std   =("err","std"),
                    nfe_mean  =("nfe","mean"),
                    succ_rate =("hit","mean"))
               .reset_index())

    stats_path = os.path.join(outdir, "eng_mean_std.csv")
    stats.to_csv(stats_path, index=False)
    print("Saved:", stats_path)

    # Plot: Mean absolute error per problem (log scale), colored by alg
    stats_plot = stats[stats["err_mean"].apply(np.isfinite)]
    if stats_plot.empty:
        print("[WARN] No finite err_mean values to plot.", file=sys.stderr)
        return

    png_path = os.path.join(outdir, "eng_err_mean_bar.png")
    try:
        import seaborn as sns
        import matplotlib.pyplot as plt
        plt.figure(figsize=(max(10, 0.6*stats_plot["problem"].nunique()*len(stats_plot["alg"].unique())), 6))
        sns.barplot(data=stats_plot, x="problem", y="err_mean", hue="alg", errorbar=None)
        plt.yscale("log")
        plt.ylabel("Mean absolute error (log scale)")
        plt.xlabel("Problem")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(png_path, dpi=200)
        plt.close()
        print("Saved:", png_path)
    except Exception as e:
        # Fallback to matplotlib-only grouped bars
        try:
            import matplotlib.pyplot as plt
            # Prepare grouped bar data
            probs = list(stats_plot["problem"].unique())
            algs  = list(stats_plot["alg"].unique())
            x = np.arange(len(probs))
            width = 0.8 / max(1, len(algs))
            fig = plt.figure(figsize=(max(10, 0.6*len(probs)*len(algs)), 6))
            for i, alg in enumerate(algs):
                y = [float(stats_plot[(stats_plot["problem"] == p) & (stats_plot["alg"] == alg)]["err_mean"].values[0])
                     if not stats_plot[(stats_plot["problem"] == p) & (stats_plot["alg"] == alg)].empty else np.nan
                     for p in probs]
                plt.bar(x + i*width - (len(algs)-1)*width/2, y, width=width, label=alg)
            plt.yscale("log")
            plt.ylabel("Mean absolute error (log scale)")
            plt.xlabel("Problem")
            plt.xticks(x, probs, rotation=45, ha="right")
            plt.legend()
            plt.tight_layout()
            plt.savefig(png_path, dpi=200)
            plt.close()
            print("Saved (matplotlib fallback):", png_path)
        except Exception as e2:
            print(f"[WARN] Could not generate plot: {e} / fallback: {e2}", file=sys.stderr)

def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--suite", choices=["cec2014", "cec2022", "eng"], default="eng")
    ap.add_argument("--dims",  type=int, nargs="+", default=[10, 20],
                    help="CEC only; ignored for --suite eng")
    ap.add_argument("--fids",  type=str, default=None, help="e.g. '1-12' (or '1-16' for eng)")
    ap.add_argument("--runs",  type=int, default=10)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--budget-mult", type=int, default=20000,
                    help="budget = budget_mult × dim (uses each problem's dim for eng)")
    ap.add_argument("--target-tol", type=float, default=1e-8)
    ap.add_argument("--algs",  nargs="+", default=["Spiral-LSHADE", "CMAES", "SciPyDE"])
    ap.add_argument("--outdir", type=str, default="results_eng")
    args = ap.parse_args()

    if args.fids is None:
        if args.suite == "cec2022":
            args.fids = "1-12"
        elif args.suite == "cec2014":
            args.fids = "1-30"
        else:  # eng
            args.fids = f"1-{len(ENGINEERING_PROBLEMS)}"
    fids = parse_ids(args.fids)

    os.makedirs(args.outdir, exist_ok=True)
    results_csv = os.path.join(args.outdir, f"{args.suite}_results.csv")
    with open(results_csv, "w", newline="") as fh:
        csv.writer(fh).writerow(
            ["alg","suite","fid","dim","run","fbest","err","nfe","hit","time_sec"])

    all_rows: List[RunResult] = []

    # For eng, ignore --dims to avoid duplicating runs; do a single pass.
    dims_iter = args.dims if args.suite != "eng" else [0]

    for dim_spec in dims_iter:
        for fid in fids:
            try:
                prob = build_problem(args.suite, fid, dim_spec)
            except Exception as e:
                print(f"[SKIP] fid={fid} dim={dim_spec}: {e}", file=sys.stderr)
                continue

            # Use actual problem dimension for budget (both eng & CEC)
            budget = int(args.budget_mult) * int(prob.dim)

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
                        print(f"[ERR] {alg} {prob.name} D{prob.dim}: {e}", file=sys.stderr)
                        fbest, nfe = float("inf"), 0
                    t1 = time.time()
                    err = abs(fbest - prob.fopt)  # may be NaN if fopt unknown
                    hit = int(np.isfinite(prob.fopt) and (err <= args.target_tol))

                    row = RunResult(alg=alg, suite=args.suite, fid=fid, dim=prob.dim,
                                    run=seed, fbest=fbest, err=err, nfe=nfe, hit=hit,
                                    time_sec=t1 - t0, budget=budget)
                    all_rows.append(row)
                    with open(results_csv, "a", newline="") as fh:
                        csv.writer(fh).writerow(
                            [row.alg,row.suite,row.fid,row.dim,row.run,
                             row.fbest,row.err,row.nfe,row.hit,row.time_sec])
                    print(f"{alg:14s}|{args.suite} {prob.name:>24s} D{prob.dim:02d} run{run_idx:02d} "
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
    print("Saved:", ert_csv)

    # ------------- ENG mean/std + Seaborn plot -------------
    if any(r.suite == "eng" for r in all_rows):
        _write_eng_stats_and_plot(all_rows, args.outdir)

    print("Saved:", results_csv)

# ---------------------------------------------------------------------
if __name__ == "__main__":
    main()
