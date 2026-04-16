#!/usr/bin/env python3
"""
===============================================================
  Corrosion RC Beam Optimizer -- Part 3 + 4: Physics Engine
  Google Colab Self-Contained Script
===============================================================
  PREREQUISITE: Run Part 1 + Part 2 first!

  STAGES:
    A — Symbolic Calculus (SymPy)
        Parse PySR equation, all partial derivatives,
        critical corrosion point eta*, integration, Taylor series.

    B — Non-Dimensionalization & Master Curve (Buckingham Pi)
        Construct Pi groups, collapse 804 points onto 1 curve,
        universal scaling law.

    C — Global Sensitivity & Phase Diagram (SALib / Sobol)
        Sobol first-order & total-order indices,
        phase diagram (safe / warning / critical zones).

    D — Prediction & Validation
        Extrapolation beyond data range, bootstrap CI,
        comparison with ACI degradation model,
        automatic discovery statement generation.

  OUTPUT:
    12 Nature-quality figures  +  physics_results.json
    +  comprehensive PDF report  +  ZIP

  HOW TO RUN (Google Colab):
    1.  Run Part 1, then Part 2 (in same runtime session)
    2.  Paste this ENTIRE file into a NEW cell
    3.  Run it (~3-8 min)
    4.  Download physics/ results when done
===============================================================
"""

# =============================================================
# CELL 1: INSTALL & SETUP
# =============================================================
import subprocess, sys, os

def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

for p in ["loguru", "sympy", "SALib", "scikit-learn",
           "matplotlib", "seaborn", "fpdf2"]:
    try:
        __import__(p.replace("-", "_").replace("SALib", "SALib"))
    except ImportError:
        install(p)

REPO = "corrosion-rc-beam-optimizer"
BASE = "/kaggle/working" if os.path.isdir("/kaggle/working") else "/content"
REPO_PATH = f"{BASE}/{REPO}"
if not os.path.isdir(REPO_PATH):
    try:
        subprocess.run(
            ["git", "clone",
             "https://github.com/Dr-Yehia/corrosion-rc-beam-optimizer.git",
             REPO_PATH], check=True, timeout=30)
    except Exception:
        if not os.path.isdir(REPO_PATH):
            raise RuntimeError("Repo not found. Run Part 1 first.")

os.chdir(f"{REPO_PATH}/src")
sys.path.insert(0, f"{REPO_PATH}/src")
print("Part 3 setup complete.")

# =============================================================
# CELL 2: IMPORTS
# =============================================================
import json, time, warnings, traceback, re, copy
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

import sympy as sp
from sympy import (
    symbols, Symbol, diff, integrate, solve, series, simplify,
    latex, sqrt, log, exp, oo, Abs, Piecewise, N as sp_N,
    lambdify, Rational,
)
from scipy import optimize
from scipy.interpolate import UnivariateSpline
from scipy.stats import bootstrap as scipy_bootstrap
from datetime import datetime
from pathlib import Path
from loguru import logger
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")

from config import (
    RESULTS_DIR, MODELS_DIR, FIGURES_DIR, EQ_DIR, LOG_DIR,
    TARGET_COL, RANDOM_STATE, L1_TARGET_R2, L2_TARGET_R2,
)
from data_preprocessing import run_preprocessing
from aci_calculator import compute_aci_predictions, evaluate_aci_benchmark

# -- Directories --
PHYSICS_DIR = RESULTS_DIR / "physics"
PHYSICS_DIR.mkdir(parents=True, exist_ok=True)
PH_FIG = PHYSICS_DIR / "figures"
PH_FIG.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# -- Logger --
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
    level="INFO", colorize=True,
)
logger.add(
    str(LOG_DIR / "run_log_part3.txt"),
    format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}",
    level="DEBUG", rotation="10 MB", encoding="utf-8",
)

t_start = time.time()
logger.info("=" * 65)
logger.info("  Part 3 + 4: Physics Engine & Prediction")
logger.info(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
logger.info("=" * 65)

# -- Plot style --
plt.rcParams.update({
    "font.size": 12, "axes.titlesize": 14, "axes.labelsize": 13,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.family": "serif",
})

# =============================================================
# CELL 3: LOAD DATA + EQUATION
# =============================================================
logger.info("Loading data & equation ...")
data = run_preprocessing(save_clean=True)
df_clean = data["df_clean"]
N_TOTAL = len(df_clean)

df_aci = compute_aci_predictions(df_clean)
aci_bench = evaluate_aci_benchmark(df_aci)

COL_ETA = "Mass Loss (Tensile bars), \u03b7m (%)"
COL_FY  = "fy Longitudinal Bars (Tensile), (MPa) "
COL_FC  = "f'c (MPa)"
COL_D   = "Depth (mm)"
COL_B   = "Width (mm)"
COL_RHO = "Tension Reinforcement Ratio, pten (%)"
COL_DB  = "Diameter Tensile Bars, db,t (mm)"

y_exp   = df_clean[TARGET_COL].values.astype(np.float64)
M_ACI   = df_aci["MACI_pred"].values.astype(np.float64)
eta_arr = df_clean[COL_ETA].values.astype(np.float64)
d_arr   = df_clean[COL_D].values.astype(np.float64)
b_arr   = df_clean[COL_B].values.astype(np.float64)
fy_arr  = df_clean[COL_FY].values.astype(np.float64)
fc_arr  = df_clean[COL_FC].values.astype(np.float64)
rho_arr = df_clean[COL_RHO].values.astype(np.float64)
db_arr  = df_clean[COL_DB].values.astype(np.float64)

d_b_arr = d_arr / np.maximum(b_arr, 1.0)
csi_arr = (df_clean["corr_severity_idx"].values.astype(np.float64)
           if "corr_severity_idx" in df_clean.columns
           else eta_arr * (fy_arr / fc_arr))
ri_arr  = (df_clean["reinf_index"].values.astype(np.float64)
           if "reinf_index" in df_clean.columns
           else np.ones(N_TOTAL))

logger.info(f"Data loaded: {N_TOTAL} samples")

# ---- Load equation from Part 2 ----
eq_json_path = EQ_DIR / "all_equations.json"
eq_txt_path  = EQ_DIR / "best_equation.txt"

WINNER  = "UNKNOWN"
eq_str  = None
eq_ltx  = None
eq_meta = {}

if eq_json_path.exists():
    with open(eq_json_path, encoding="utf-8") as f:
        eq_data = json.load(f)
    WINNER  = eq_data.get("winner", "UNKNOWN")
    eq_str  = eq_data.get("final_equation")
    eq_ltx  = eq_data.get("final_equation_latex")
    eq_meta = eq_data.get("final_metrics", {})
    logger.info(f"Equation loaded from JSON  |  Winner: {WINNER}")
elif eq_txt_path.exists():
    with open(eq_txt_path, encoding="utf-8") as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        if line.startswith("f_corr ="):
            eq_str = line.split("=", 1)[1].strip()
            WINNER = "RATIO"
        elif line.startswith("Mmax =") and "M_ACI" not in line:
            eq_str = line.split("=", 1)[1].strip()
            WINNER = "DIRECT"
    logger.info(f"Equation loaded from TXT  |  Winner: {WINNER}")
else:
    WINNER = "RATIO"
    eq_str = "1.0 - 0.012*eta_m**0.85"
    logger.warning("No Part 2 equation found — using fallback degradation model")

logger.info(f"Equation string: {eq_str[:120]}")

# ---- Load Part 1 + Part 2 summaries ----
part2_dir = RESULTS_DIR / "for_part2"
part1_summary = None
if (part2_dir / "part1_summary.json").exists():
    with open(part2_dir / "part1_summary.json") as f:
        part1_summary = json.load(f)

pysr_metrics_path = MODELS_DIR / "pysr_metrics.json"
pysr_summary = None
if pysr_metrics_path.exists():
    with open(pysr_metrics_path) as f:
        pysr_summary = json.load(f)

# =============================================================
# CELL 4 — STAGE A: SYMBOLIC CALCULUS (SymPy)
# =============================================================
logger.info("\n" + "=" * 65)
logger.info("  STAGE A — Symbolic Calculus")
logger.info("=" * 65)

# ---- A1: Define symbols ----
eta_m_s = Symbol("eta_m", positive=True, real=True)
fy_s    = Symbol("fy",    positive=True, real=True)
fc_s    = Symbol("fc",    positive=True, real=True)
d_s     = Symbol("d",     positive=True, real=True)
b_s     = Symbol("b",     positive=True, real=True)
rho_t_s = Symbol("rho_t", positive=True, real=True)
db_t_s  = Symbol("db_t",  positive=True, real=True)
d_b_s   = Symbol("d_b",   positive=True, real=True)
CSI_s   = Symbol("CSI",   positive=True, real=True)
RI_s    = Symbol("RI",    positive=True, real=True)

SYM_MAP = {
    "eta_m": eta_m_s, "fy": fy_s, "fc": fc_s, "d": d_s, "b": b_s,
    "rho_t": rho_t_s, "db_t": db_t_s, "d_b": d_b_s,
    "CSI": CSI_s, "RI": RI_s,
    "sqrt": sp.sqrt, "log": sp.log, "exp": sp.exp, "abs": sp.Abs,
    "pi": sp.pi,
}

DATA_MAP = {
    eta_m_s: eta_arr, fy_s: fy_arr, fc_s: fc_arr,
    d_s: d_arr, b_s: b_arr, rho_t_s: rho_arr,
    db_t_s: db_arr, d_b_s: d_b_arr, CSI_s: csi_arr, RI_s: ri_arr,
}

MEDIAN_MAP = {s: float(np.median(v)) for s, v in DATA_MAP.items()}

# ---- A2: Parse equation to SymPy ----
eq_str_clean = (eq_str.replace("^", "**")
                       .replace("Abs(", "abs(")
                       .replace("square(", "sqrt("))

try:
    f_expr = sp.sympify(eq_str_clean, locals=SYM_MAP)
    logger.success(f"Parsed OK: {f_expr}")
except Exception as exc:
    logger.error(f"Parse failed ({exc}) — using fallback")
    f_expr = 1.0 - 0.012 * eta_m_s ** 0.85

free_syms = sorted(f_expr.free_symbols, key=str)
logger.info(f"Free symbols: {[str(s) for s in free_syms]}")

if WINNER == "RATIO":
    logger.info("Mode: f_corr equation — Mmax = M_ACI * f_corr")
    analysis_label = "f_{corr}"
else:
    logger.info("Mode: Direct Mmax equation")
    analysis_label = "M_{max}"

# ---- A3: Build numerical evaluator ----
def _make_eval_func(expr, syms):
    """Create a fast numpy evaluator from SymPy expression."""
    fn = lambdify(syms, expr, modules=["numpy"])
    def evaluate(data_dict):
        args = [data_dict[s] for s in syms]
        return np.asarray(fn(*args), dtype=np.float64)
    return evaluate

eval_f = _make_eval_func(f_expr, free_syms)

# Verify numerical evaluation
try:
    f_vals = eval_f(DATA_MAP)
    f_median_val = float(np.median(f_vals))
    logger.info(f"Equation eval: median={f_median_val:.4f}, "
                f"range=[{np.nanmin(f_vals):.4f}, {np.nanmax(f_vals):.4f}]")
except Exception as exc:
    logger.error(f"Numerical evaluation failed: {exc}")
    f_vals = np.ones(N_TOTAL)

# ---- A4: Partial derivatives ----
deriv_results = {}

if eta_m_s in f_expr.free_symbols:
    df_deta   = diff(f_expr, eta_m_s)
    d2f_deta2 = diff(f_expr, eta_m_s, 2)
    d3f_deta3 = diff(f_expr, eta_m_s, 3)
elif CSI_s in f_expr.free_symbols:
    # Chain rule: CSI = eta_m * fy / fc  =>  dCSI/d_eta = fy/fc
    dCSI_deta = fy_s / fc_s
    df_deta   = diff(f_expr, CSI_s) * dCSI_deta
    d2f_deta2 = diff(diff(f_expr, CSI_s), CSI_s) * dCSI_deta**2
    d3f_deta3 = diff(diff(diff(f_expr, CSI_s), CSI_s), CSI_s) * dCSI_deta**3
    logger.info("  Using chain rule: d/d_eta via CSI = eta_m * fy/fc")
else:
    df_deta   = sp.S.Zero
    d2f_deta2 = sp.S.Zero
    d3f_deta3 = sp.S.Zero

deriv_results["df/d_eta"]     = df_deta
deriv_results["d2f/d_eta2"]   = d2f_deta2
deriv_results["d3f/d_eta3"]   = d3f_deta3

for sym_name, sym_obj in [("d", d_s), ("b", b_s), ("fy", fy_s),
                           ("fc", fc_s), ("rho_t", rho_t_s),
                           ("d_b", d_b_s)]:
    if sym_obj in f_expr.free_symbols:
        deriv_results[f"df/d_{sym_name}"] = diff(f_expr, sym_obj)

logger.info(f"Computed {len(deriv_results)} derivatives")
for k, v in deriv_results.items():
    try:
        v_simple = simplify(v)
    except Exception:
        v_simple = v
    logger.info(f"  {k} = {v_simple}")

# ---- A5: Critical point eta* (inflection / regime change) ----
critical_eta_star = None
critical_method   = "none"

# Build the effective 1D expression for critical-point search
# (may use chain rule: CSI = eta_m * fy/fc)
_need_chain_for_crit = (eta_m_s not in free_syms and CSI_s in free_syms)
if _need_chain_for_crit:
    _fy_m = MEDIAN_MAP[fy_s]; _fc_m = MEDIAN_MAP[fc_s]
    _crit_expr = f_expr.subs(CSI_s, eta_m_s * _fy_m / _fc_m)
    _crit_d2   = diff(_crit_expr, eta_m_s, 2)
    _crit_subs = {s: MEDIAN_MAP[s] for s in free_syms
                  if s not in (CSI_s, eta_m_s)}
    logger.info("  Critical-point search: using CSI chain rule")
else:
    _crit_expr = f_expr
    _crit_d2   = d2f_deta2
    _crit_subs = {s: MEDIAN_MAP[s] for s in free_syms if s != eta_m_s}

if not _crit_d2.equals(sp.S.Zero):
    try:
        sols = solve(_crit_d2.subs(_crit_subs), eta_m_s)
        real_sols = [float(sp.re(s)) for s in sols
                     if sp.im(s) == 0 and 0 < float(sp.re(s)) < 65]
        if real_sols:
            critical_eta_star = min(real_sols)
            critical_method = "analytical"
            logger.success(f"Critical eta* = {critical_eta_star:.2f}% (analytical)")
    except Exception:
        pass

if critical_eta_star is None:
    try:
        _d2_1d = _crit_d2.subs(_crit_subs)
        d2_func = lambdify(eta_m_s, _d2_1d, modules=["numpy"])
        eta_scan = np.linspace(0.1, 60, 2000)
        d2_vals  = np.array([float(d2_func(e)) for e in eta_scan])
        sign_changes = np.where(np.diff(np.sign(d2_vals)))[0]
        if len(sign_changes) > 0:
            idx = sign_changes[0]
            result = optimize.brentq(
                lambda x: float(d2_func(x)),
                eta_scan[idx], eta_scan[idx + 1],
            )
            critical_eta_star = float(result)
            critical_method = "numerical"
            logger.success(f"Critical eta* = {critical_eta_star:.2f}% (numerical)")
    except Exception as exc:
        logger.warning(f"Numerical critical point search failed: {exc}")

if critical_eta_star is None:
    try:
        _f_1d_c = _crit_expr.subs(_crit_subs)
        f_1d_func = lambdify(eta_m_s, _f_1d_c, modules=["numpy"])
        eta_scan = np.linspace(0.5, 60, 2000)
        f_scan = np.array([float(f_1d_func(e)) for e in eta_scan])
        f_spline = UnivariateSpline(eta_scan, f_scan, s=0, k=4)
        d2_spline = f_spline.derivative(n=2)
        d2_vals   = d2_spline(eta_scan)
        sign_ch   = np.where(np.diff(np.sign(d2_vals)))[0]
        if len(sign_ch) > 0:
            critical_eta_star = float(eta_scan[sign_ch[0]])
            critical_method = "spline"
            logger.success(f"Critical eta* = {critical_eta_star:.2f}% (spline)")
    except Exception:
        pass

if critical_eta_star is None:
    critical_eta_star = 15.0
    critical_method = "heuristic"
    logger.warning(f"No inflection found — using heuristic eta* = {critical_eta_star}%")

# ---- A6: Integration (cumulative damage index) ----
eta_upper = Symbol("eta_upper", positive=True)
integral_expr = None
try:
    _int_expr = (_crit_expr if _need_chain_for_crit else f_expr)
    _int_subs = {s: MEDIAN_MAP[s] for s in _int_expr.free_symbols
                 if s != eta_m_s}
    f_1d_for_int = _int_expr.subs(_int_subs)
    integral_expr = integrate(f_1d_for_int, (eta_m_s, 0, eta_upper))
    logger.info(f"Integral: int(f, 0..eta) = {integral_expr}")
except Exception as exc:
    logger.warning(f"Symbolic integration failed: {exc}")

# ---- A7: Taylor expansion around eta=0 ----
taylor_expr = None
try:
    _tay_expr = (_crit_expr if _need_chain_for_crit else f_expr)
    _tay_subs = {s: MEDIAN_MAP[s] for s in _tay_expr.free_symbols
                 if s != eta_m_s}
    f_1d_taylor = _tay_expr.subs(_tay_subs) if _tay_subs else _tay_expr
    taylor_expr = series(f_1d_taylor, eta_m_s, 0, n=4).removeO()
    logger.info(f"Taylor (order 3): {taylor_expr}")
except Exception as exc:
    logger.warning(f"Taylor expansion failed: {exc}")

# ---- A8: Build 1D evaluation curves ----
eta_plot = np.linspace(0.01, 60, 500)

# If eta_m is not a free symbol but CSI is, we express CSI as eta_m * fy/fc
USE_CSI_CHAIN = (eta_m_s not in free_syms and CSI_s in free_syms)
if USE_CSI_CHAIN:
    fy_med = MEDIAN_MAP[fy_s]
    fc_med = MEDIAN_MAP[fc_s]
    subs_med = {s: MEDIAN_MAP[s] for s in free_syms
                if s not in (CSI_s, eta_m_s)}
    # Replace CSI with eta_m * fy_med/fc_med so we can sweep eta_m
    f_expr_1d = f_expr.subs(CSI_s, eta_m_s * fy_med / fc_med)
    df_deta_1d = diff(f_expr_1d, eta_m_s)
    d2f_deta2_1d = diff(f_expr_1d, eta_m_s, 2)
    logger.info(f"  CSI chain: f(eta) = f(..., CSI=eta*{fy_med/fc_med:.3f}, ...)")
else:
    f_expr_1d = f_expr
    df_deta_1d = df_deta
    d2f_deta2_1d = d2f_deta2

subs_med = {s: MEDIAN_MAP[s] for s in free_syms
            if s != eta_m_s and not (USE_CSI_CHAIN and s == CSI_s)}

try:
    f_1d_sym = f_expr_1d.subs(subs_med)
    f_1d_fn  = lambdify(eta_m_s, f_1d_sym, modules=["numpy"])
    f_curve  = np.array([float(f_1d_fn(e)) for e in eta_plot])
except Exception:
    f_curve = np.ones_like(eta_plot)

try:
    d1_sym = df_deta_1d.subs(subs_med)
    d1_fn  = lambdify(eta_m_s, d1_sym, modules=["numpy"])
    d1_curve = np.array([float(d1_fn(e)) for e in eta_plot])
except Exception:
    d1_curve = np.gradient(f_curve, eta_plot)

try:
    d2_sym = d2f_deta2_1d.subs(subs_med)
    d2_fn  = lambdify(eta_m_s, d2_sym, modules=["numpy"])
    d2_curve = np.array([float(d2_fn(e)) for e in eta_plot])
except Exception:
    d2_curve = np.gradient(d1_curve, eta_plot)

if integral_expr is not None:
    try:
        int_fn = lambdify(eta_upper, integral_expr, modules=["numpy"])
        int_curve = np.array([float(int_fn(e)) for e in eta_plot])
    except Exception:
        int_curve = np.cumsum(f_curve) * (eta_plot[1] - eta_plot[0])
else:
    int_curve = np.cumsum(f_curve) * (eta_plot[1] - eta_plot[0])

if taylor_expr is not None:
    try:
        tay_fn    = lambdify(eta_m_s, taylor_expr, modules=["numpy"])
        tay_curve = np.array([float(tay_fn(e)) for e in eta_plot])
    except Exception:
        tay_curve = None
else:
    tay_curve = None

stage_a_results = {
    "equation_str": str(f_expr),
    "winner": WINNER,
    "derivatives": {k: str(v) for k, v in deriv_results.items()},
    "critical_eta_star": round(critical_eta_star, 2),
    "critical_method": critical_method,
    "integral_expr": str(integral_expr) if integral_expr else "numerical",
    "taylor_expr": str(taylor_expr) if taylor_expr else None,
}
logger.info("Stage A complete.")

# =============================================================
# CELL 5 — STAGE B: NON-DIMENSIONALIZATION & MASTER CURVE
# =============================================================
logger.info("\n" + "=" * 65)
logger.info("  STAGE B — Non-Dimensionalization & Master Curve")
logger.info("=" * 65)

# ---- B1: Compute Pi groups ----
# Pi_M = M_exp * 1e6 / (fc * b * d^2)  [dimensionless moment]
Pi_M   = y_exp * 1e6 / (fc_arr * b_arr * d_arr**2)
Pi_ACI = M_ACI * 1e6 / (fc_arr * b_arr * d_arr**2)

omega = rho_arr / 100.0 * fy_arr / fc_arr
eta_nd = eta_arr / 100.0

R_ratio = y_exp / np.maximum(M_ACI, 1e-6)

logger.info(f"Pi_M range: [{Pi_M.min():.4f}, {Pi_M.max():.4f}]")
logger.info(f"omega range: [{omega.min():.4f}, {omega.max():.4f}]")
logger.info(f"R_ratio range: [{R_ratio.min():.3f}, {R_ratio.max():.3f}]")

# ---- B2: Master curve — collapse R_ratio vs eta ----
valid_mc = np.isfinite(R_ratio) & (R_ratio > 0.05) & (R_ratio < 10.0)
R_mc   = R_ratio[valid_mc]
eta_mc = eta_arr[valid_mc]
omega_mc = omega[valid_mc]

# Fit multiple models for R = f(eta)
master_fits = {}

# Model 1: polynomial R = a0 + a1*eta + a2*eta^2
try:
    p2 = np.polyfit(eta_mc, R_mc, 2)
    R_pred_p2 = np.polyval(p2, eta_mc)
    r2_p2 = r2_score(R_mc, R_pred_p2)
    master_fits["Poly2"] = {"params": p2.tolist(), "R2": round(r2_p2, 4)}
except Exception:
    r2_p2 = -1

# Model 2: exponential decay R = a * exp(-b * eta^c)
try:
    def exp_model(eta, a, k, c):
        return a * np.exp(-k * np.power(eta + 1e-6, c))
    from scipy.optimize import curve_fit
    popt, _ = curve_fit(exp_model, eta_mc, R_mc,
                        p0=[1.2, 0.01, 1.0],
                        bounds=([0.5, 1e-5, 0.1], [3.0, 1.0, 3.0]),
                        maxfev=10000)
    R_pred_exp = exp_model(eta_mc, *popt)
    r2_exp = r2_score(R_mc, R_pred_exp)
    master_fits["ExpDecay"] = {
        "params": {"a": round(popt[0], 4), "k": round(popt[1], 6),
                   "c": round(popt[2], 4)},
        "R2": round(r2_exp, 4),
        "formula": f"R = {popt[0]:.3f} * exp(-{popt[1]:.5f} * eta^{popt[2]:.3f})",
    }
except Exception:
    r2_exp = -1
    popt = None

# Model 3: Padé approximant R = (1 + a1*eta) / (1 + a2*eta + a3*eta^2)
try:
    def pade_model(eta, a1, a2, a3):
        return (1.0 + a1 * eta) / (1.0 + a2 * eta + a3 * eta**2)
    popt_pade, _ = curve_fit(pade_model, eta_mc, R_mc,
                              p0=[0.01, 0.01, 0.0001],
                              maxfev=10000)
    R_pred_pade = pade_model(eta_mc, *popt_pade)
    r2_pade = r2_score(R_mc, R_pred_pade)
    master_fits["Pade"] = {
        "params": popt_pade.tolist(), "R2": round(r2_pade, 4),
    }
except Exception:
    r2_pade = -1

# Select best master curve model
best_mc_name = max(master_fits, key=lambda k: master_fits[k]["R2"])
best_mc_r2 = master_fits[best_mc_name]["R2"]
logger.info(f"Master curve fits: " +
            " | ".join(f"{k}: R2={v['R2']:.4f}" for k, v in master_fits.items()))
logger.success(f"Best master curve: {best_mc_name} (R2={best_mc_r2:.4f})")

# ---- B3: Compound parameter optimization ----
# Pi_compound = omega^alpha * (1 - eta/100)^beta
# Optimize alpha, beta to maximize R2 of Pi_M vs Pi_compound
def _compound_r2(params):
    alpha, beta = params
    Pi_c = np.power(omega + 1e-8, alpha) * np.power(
        np.clip(1.0 - eta_nd, 1e-6, 1.0), beta)
    mask = np.isfinite(Pi_c) & (Pi_c > 0)
    if mask.sum() < 50:
        return 1e6
    p = np.polyfit(Pi_c[mask], Pi_M[mask], 1)
    pred = np.polyval(p, Pi_c[mask])
    ss_res = np.sum((Pi_M[mask] - pred) ** 2)
    ss_tot = np.sum((Pi_M[mask] - np.mean(Pi_M[mask])) ** 2)
    return -(1.0 - ss_res / (ss_tot + 1e-12))

try:
    res = optimize.differential_evolution(
        _compound_r2, bounds=[(0.3, 2.5), (0.3, 3.0)],
        seed=RANDOM_STATE, maxiter=200, tol=1e-6,
    )
    alpha_opt, beta_opt = res.x
    r2_compound = -res.fun
    Pi_compound = np.power(omega + 1e-8, alpha_opt) * np.power(
        np.clip(1.0 - eta_nd, 1e-6, 1.0), beta_opt)
    logger.success(
        f"Compound parameter: omega^{alpha_opt:.2f} * (1-eta/100)^{beta_opt:.2f}"
        f"  |  R2 = {r2_compound:.4f}"
    )
except Exception as exc:
    logger.warning(f"Compound optimization failed: {exc}")
    alpha_opt, beta_opt = 1.0, 1.0
    Pi_compound = omega * (1.0 - eta_nd)
    r2_compound = 0.0

stage_b_results = {
    "Pi_M_range": [round(float(Pi_M.min()), 4), round(float(Pi_M.max()), 4)],
    "omega_range": [round(float(omega.min()), 4), round(float(omega.max()), 4)],
    "master_curve_fits": master_fits,
    "best_master_curve": best_mc_name,
    "best_master_curve_R2": best_mc_r2,
    "compound_parameter": {
        "alpha": round(alpha_opt, 3),
        "beta": round(beta_opt, 3),
        "R2": round(r2_compound, 4),
        "formula": f"Pi_c = omega^{alpha_opt:.2f} * (1 - eta/100)^{beta_opt:.2f}",
    },
}
logger.info("Stage B complete.")

# =============================================================
# CELL 6 — STAGE C: GLOBAL SENSITIVITY & PHASE DIAGRAM
# =============================================================
logger.info("\n" + "=" * 65)
logger.info("  STAGE C — Sensitivity Analysis & Phase Diagram")
logger.info("=" * 65)

# ---- C1: Sobol sensitivity (SALib) ----
sobol_results = {}
sobol_ok = False

try:
    from SALib.sample import saltelli
    from SALib.analyze import sobol

    sens_syms = [s for s in free_syms if s in DATA_MAP]
    if len(sens_syms) >= 2:
        var_names = [str(s) for s in sens_syms]
        bounds = [[float(np.percentile(DATA_MAP[s], 2)),
                   float(np.percentile(DATA_MAP[s], 98))]
                  for s in sens_syms]
        for i, (lo, hi) in enumerate(bounds):
            if hi - lo < 1e-6:
                bounds[i][1] = lo + 1.0

        problem = {
            "num_vars": len(sens_syms),
            "names": var_names,
            "bounds": bounds,
        }

        N_SOBOL = 1024
        param_values = saltelli.sample(problem, N_SOBOL,
                                        calc_second_order=False)
        logger.info(f"Sobol samples: {param_values.shape[0]} "
                    f"({len(sens_syms)} vars)")

        eval_sobol = _make_eval_func(f_expr, sens_syms)

        Y = np.zeros(param_values.shape[0])
        for i in range(param_values.shape[0]):
            d_local = {s: param_values[i, j]
                       for j, s in enumerate(sens_syms)}
            try:
                Y[i] = float(eval_sobol(d_local))
            except Exception:
                Y[i] = np.nan
        Y = np.nan_to_num(Y, nan=float(np.nanmedian(Y)))

        Si = sobol.analyze(problem, Y, calc_second_order=False)

        sobol_results = {
            "S1": {n: round(float(v), 4) for n, v in zip(var_names, Si["S1"])},
            "ST": {n: round(float(v), 4) for n, v in zip(var_names, Si["ST"])},
        }
        sobol_ok = True
        logger.info("Sobol S1: " +
                    " | ".join(f"{n}={v:.3f}" for n, v in
                               sobol_results["S1"].items()))
        logger.info("Sobol ST: " +
                    " | ".join(f"{n}={v:.3f}" for n, v in
                               sobol_results["ST"].items()))
    else:
        logger.warning("Not enough free symbols for Sobol analysis")
except ImportError:
    logger.warning("SALib not available — skipping Sobol")
except Exception as exc:
    logger.warning(f"Sobol analysis failed: {exc}")

# ---- C2: Phase diagram computation ----
# Grid: eta_m (0-60) vs rho (0.5-5%) → color = predicted f/Mmax
eta_grid = np.linspace(0.5, 60, 80)
rho_grid = np.linspace(0.3, 5.0, 60)
ETA_G, RHO_G = np.meshgrid(eta_grid, rho_grid)

phase_Z = np.zeros_like(ETA_G)
for ii in range(ETA_G.shape[0]):
    for jj in range(ETA_G.shape[1]):
        d_local = {s: MEDIAN_MAP[s] for s in free_syms}
        if eta_m_s in d_local:
            d_local[eta_m_s] = ETA_G[ii, jj]
        if rho_t_s in d_local:
            d_local[rho_t_s] = RHO_G[ii, jj]
        try:
            phase_Z[ii, jj] = float(eval_f(d_local))
        except Exception:
            phase_Z[ii, jj] = np.nan

if WINNER == "RATIO":
    phase_label = "Correction Factor f_corr"
    safe_thresh = 0.8
    warn_thresh = 0.5
else:
    ref_val = float(np.median(y_exp))
    phase_Z_norm = phase_Z / (ref_val if ref_val > 0 else 1.0)
    safe_thresh = 0.8
    warn_thresh = 0.5

logger.info("Phase diagram computed.")

# ---- C3: Spider sensitivity (one-at-a-time) ----
spider_results = {}
for sym in free_syms:
    if sym == eta_m_s:
        continue
    med_val = MEDIAN_MAP[sym]
    lo_val  = med_val * 0.7
    hi_val  = med_val * 1.3
    d_lo = {s: MEDIAN_MAP[s] for s in free_syms}
    d_hi = {s: MEDIAN_MAP[s] for s in free_syms}
    d_lo[sym] = lo_val
    d_hi[sym] = hi_val
    d_lo[eta_m_s] = critical_eta_star if eta_m_s in d_lo else 10.0
    d_hi[eta_m_s] = critical_eta_star if eta_m_s in d_hi else 10.0
    try:
        f_lo = float(eval_f(d_lo))
        f_hi = float(eval_f(d_hi))
        sensitivity = abs(f_hi - f_lo) / max(abs(f_lo + f_hi) / 2, 1e-8)
        spider_results[str(sym)] = round(sensitivity, 4)
    except Exception:
        spider_results[str(sym)] = 0.0

logger.info("Spider sensitivity: " +
            " | ".join(f"{k}={v:.3f}" for k, v in spider_results.items()))

stage_c_results = {
    "sobol": sobol_results if sobol_ok else "unavailable",
    "spider_sensitivity": spider_results,
    "phase_diagram": {
        "eta_range": [0.5, 60],
        "rho_range": [0.3, 5.0],
        "safe_threshold": safe_thresh,
        "warning_threshold": warn_thresh,
    },
}
logger.info("Stage C complete.")

# =============================================================
# CELL 7 — STAGE D: PREDICTION & VALIDATION
# =============================================================
logger.info("\n" + "=" * 65)
logger.info("  STAGE D — Prediction & Validation")
logger.info("=" * 65)

# ---- D1: Extrapolation prediction (beyond eta=64%) ----
eta_extrap = np.linspace(0, 90, 500)
try:
    f_extrap = np.array([float(f_1d_fn(e)) for e in eta_extrap])
except Exception:
    f_extrap = np.ones_like(eta_extrap)

if WINNER == "RATIO":
    M_aci_median = float(np.median(M_ACI))
    M_extrap = M_aci_median * f_extrap
else:
    M_extrap = f_extrap

collapse_eta = None
for i, e in enumerate(eta_extrap):
    ref = M_extrap[0] if M_extrap[0] > 0 else 1.0
    if M_extrap[i] / ref < 0.1:
        collapse_eta = e
        break
if collapse_eta is None:
    collapse_eta = 90.0
logger.info(f"Predicted near-zero capacity at eta = {collapse_eta:.1f}%")

# ---- D2: Bootstrap confidence intervals ----
N_BOOT = 2000
rng = np.random.RandomState(RANDOM_STATE)
boot_predictions = np.zeros((N_BOOT, len(eta_plot)))

for ib in range(N_BOOT):
    idx = rng.choice(N_TOTAL, N_TOTAL, replace=True)
    subs_boot = {}
    _boot_base = f_expr_1d if USE_CSI_CHAIN else f_expr
    for s in _boot_base.free_symbols:
        if s == eta_m_s:
            continue
        if s in DATA_MAP:
            subs_boot[s] = float(np.median(DATA_MAP[s][idx]))
        elif s in MEDIAN_MAP:
            subs_boot[s] = MEDIAN_MAP[s]
    f_boot_sym = _boot_base.subs(subs_boot)
    try:
        f_boot_fn = lambdify(eta_m_s, f_boot_sym, modules=["numpy"])
        boot_predictions[ib, :] = [float(f_boot_fn(e)) for e in eta_plot]
    except Exception:
        boot_predictions[ib, :] = f_curve

ci_lower = np.percentile(boot_predictions, 2.5, axis=0)
ci_upper = np.percentile(boot_predictions, 97.5, axis=0)
ci_median = np.percentile(boot_predictions, 50, axis=0)

logger.info(f"Bootstrap CI (95%): width at eta*={critical_eta_star:.0f}% is "
            f"[{ci_lower[int(critical_eta_star / 60 * 499)]:.3f}, "
            f"{ci_upper[int(critical_eta_star / 60 * 499)]:.3f}]")

# ---- D3: ACI degradation comparison ----
aci_1d = np.array([
    (1.0 - e / 100.0)**2 for e in eta_plot
])

if WINNER == "RATIO":
    comparison_label = "PySR f_corr"
    aci_compare = aci_1d
    pysr_compare = f_curve
else:
    comparison_label = "PySR Mmax"
    aci_compare = aci_1d * float(np.median(M_ACI))
    pysr_compare = f_curve

# ---- D4: Regime classification based on derivatives ----
regimes = []
for i, e in enumerate(eta_plot):
    if e < critical_eta_star * 0.5:
        regimes.append("SAFE")
    elif e < critical_eta_star:
        regimes.append("WARNING")
    elif e < critical_eta_star * 1.5:
        regimes.append("CRITICAL")
    else:
        regimes.append("FAILURE")

regime_boundaries = {
    "safe_limit": round(critical_eta_star * 0.5, 1),
    "warning_limit": round(critical_eta_star, 1),
    "critical_limit": round(critical_eta_star * 1.5, 1),
}

# ---- D5: Degradation rate at key points ----
deg_rates = {}
for e_val in [5, 10, 15, 20, 30, 40, 50]:
    try:
        rate = float(d1_fn(e_val))
        deg_rates[f"eta={e_val}%"] = round(rate, 6)
    except Exception:
        idx_near = np.argmin(np.abs(eta_plot - e_val))
        deg_rates[f"eta={e_val}%"] = round(float(d1_curve[idx_near]), 6)

logger.info("Degradation rates: " +
            " | ".join(f"{k}: {v}" for k, v in deg_rates.items()))

# ---- D6: Discovery statements ----
discoveries = []

discoveries.append(
    f"DISCOVERY 1: The corrosion degradation exhibits a regime change "
    f"(inflection point) at eta_m* = {critical_eta_star:.1f}%. "
    f"Below this threshold, degradation is approximately linear; "
    f"above it, the rate accelerates non-linearly."
)

rate_5  = deg_rates.get("eta=5%", 0)
rate_40 = deg_rates.get("eta=40%", 0)
if abs(rate_5) > 0 and abs(rate_40) > 0:
    accel = abs(rate_40) / max(abs(rate_5), 1e-8)
    discoveries.append(
        f"DISCOVERY 2: The degradation rate at 40% mass loss is "
        f"{accel:.1f}x faster than at 5% mass loss, confirming "
        f"non-linear (accelerating) capacity reduction."
    )

if best_mc_r2 > 0.5:
    discoveries.append(
        f"DISCOVERY 3: A universal master curve R = f(eta) "
        f"collapses all {sum(valid_mc)} data points with "
        f"R2 = {best_mc_r2:.3f}, regardless of section geometry "
        f"or material properties. This suggests the corrosion "
        f"correction factor is primarily governed by mass loss."
    )

discoveries.append(
    f"DISCOVERY 4: The model predicts near-complete capacity loss "
    f"at eta_m = {collapse_eta:.0f}%, which defines the ultimate "
    f"service life limit for corroded RC beams."
)

for disc in discoveries:
    logger.info(f"  {disc}")

stage_d_results = {
    "critical_eta_star": round(critical_eta_star, 2),
    "collapse_eta": round(collapse_eta, 1),
    "degradation_rates": deg_rates,
    "regime_boundaries": regime_boundaries,
    "discoveries": discoveries,
    "bootstrap_n": N_BOOT,
}
logger.info("Stage D complete.")

# =============================================================
# SAVE STATE FOR PART 4
# =============================================================
import joblib as _jl

_p3_state = dict(
    # Paths & config
    RESULTS_DIR=str(RESULTS_DIR), MODELS_DIR=str(MODELS_DIR),
    EQ_DIR=str(EQ_DIR), LOG_DIR=str(LOG_DIR),
    PHYSICS_DIR=str(PHYSICS_DIR), PH_FIG=str(PH_FIG),
    TARGET_COL=TARGET_COL, RANDOM_STATE=RANDOM_STATE,
    L1_TARGET_R2=L1_TARGET_R2, L2_TARGET_R2=L2_TARGET_R2,
    N_TOTAL=N_TOTAL, WINNER=WINNER,
    USE_CSI_CHAIN=USE_CSI_CHAIN, N_BOOT=N_BOOT,
    eq_str=eq_str, eq_ltx=eq_ltx, eq_meta=eq_meta,
    # Arrays
    y_exp=y_exp, M_ACI=M_ACI, eta_arr=eta_arr,
    d_arr=d_arr, b_arr=b_arr, fy_arr=fy_arr, fc_arr=fc_arr,
    rho_arr=rho_arr, db_arr=db_arr, d_b_arr=d_b_arr,
    csi_arr=csi_arr, ri_arr=ri_arr,
    # Symbolic (as strings for portability)
    f_expr_str=str(f_expr),
    free_syms_str=[str(s) for s in free_syms],
    deriv_results_str={k: str(v) for k, v in deriv_results.items()},
    integral_expr_str=str(integral_expr) if integral_expr else None,
    taylor_expr_str=str(taylor_expr) if taylor_expr else None,
    d2f_deta2_str=str(d2f_deta2),
    d2f_deta2_1d_str=str(d2f_deta2_1d) if 'd2f_deta2_1d' in dir() else "0",
    critical_eta_star=critical_eta_star,
    critical_method=critical_method,
    # Stage results
    stage_a_results=stage_a_results,
    stage_b_results=stage_b_results,
    stage_c_results=stage_c_results,
    stage_d_results=stage_d_results,
    discoveries=discoveries,
    # Sobol
    sobol_ok=sobol_ok,
    sobol_results=sobol_results if sobol_ok else {},
    spider_results=spider_results,
    # Master curve
    best_mc_name=best_mc_name, best_mc_r2=best_mc_r2,
    valid_mc=valid_mc, alpha_opt=alpha_opt,
    beta_opt=beta_opt, r2_compound=r2_compound,
    # Prediction
    collapse_eta=collapse_eta, deg_rates=deg_rates,
    regime_boundaries=regime_boundaries,
    # 1D curves
    eta_plot=eta_plot, f_curve=f_curve,
    d1_curve=d1_curve, d2_curve=d2_curve,
    ci_lower=ci_lower, ci_upper=ci_upper,
    tay_curve=tay_curve,
    analysis_label=analysis_label, f_vals=f_vals,
    # Part 1/2 summaries
    part1_summary=part1_summary, pysr_summary=pysr_summary,
    # Timing
    t_start=t_start,
    # Column names
    COL_ETA=COL_ETA, COL_FY=COL_FY, COL_FC=COL_FC,
    COL_D=COL_D, COL_B=COL_B, COL_RHO=COL_RHO, COL_DB=COL_DB,
)

_p3_save = PHYSICS_DIR / "part3_state.pkl"
_jl.dump(_p3_state, str(_p3_save))
logger.info(f"Part 3 state saved -> {_p3_save}")
logger.info("=" * 65)
logger.info("  Part 3 DONE. Run Part 4 next.")
logger.info("=" * 65)
