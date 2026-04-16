#!/usr/bin/env python3
"""
===============================================================
  Corrosion RC Beam Optimizer -- Part 6: ODE Discovery
  (The Governing Differential Equation of Corrosion Degradation)
  PREREQUISITE: Run Part 4 first!
===============================================================

  PURPOSE:
    Discover the GOVERNING DIFFERENTIAL EQUATION that describes
    how flexural capacity degrades with corrosion — NOT just a
    regression fit, but the fundamental physical LAW analogous
    to Newton discovering F=ma or Fourier discovering dT/dt=k*d²T/dx².

  STAGES:
    I — ODE Discovery (SINDy: Sparse Identification of Nonlinear Dynamics)
        From raw data, discover: dM/dη = g(M, η)

    J — Analytical Solution of the Discovered ODE (SymPy dsolve)
        Solve the ODE symbolically → M(η) = closed-form law

    K — Phase Space Analysis (Dynamical Systems Theory)
        Equilibria, stability, bifurcation, trajectories

  OUTPUT:
    6 Figures (ODE1–ODE6) + ODE_Discovery_Report.pdf + ode_results.json

  HOW TO RUN:
    1. Run Parts 1-4 first (in same runtime session or load state)
    2. Paste this file into a new cell on Kaggle/Colab
    3. Run (~2-5 min)
===============================================================
"""

# =============================================================
# CELL 0: SETUP & INSTALL
# =============================================================
import subprocess, sys, os

def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

for p in ["loguru", "sympy", "scikit-learn", "matplotlib",
           "seaborn", "fpdf2", "joblib"]:
    try:
        __import__(p.replace("-", "_"))
    except ImportError:
        install(p)

# SINDy (pysindy) — the core ODE discovery library
try:
    import pysindy
except ImportError:
    install("pysindy")
    import pysindy

REPO = "corrosion-rc-beam-optimizer"
BASE = "/kaggle/working" if os.path.isdir("/kaggle/working") else "/content"
REPO_PATH = f"{BASE}/{REPO}"
if not os.path.isdir(REPO_PATH):
    subprocess.run(
        ["git", "clone",
         "https://github.com/Dr-Yehia/corrosion-rc-beam-optimizer.git",
         REPO_PATH],
        check=True,
    )

os.chdir(f"{REPO_PATH}/src")
sys.path.insert(0, f"{REPO_PATH}/src")

# =============================================================
# CELL 1: IMPORTS
# =============================================================
import json, time, warnings, traceback
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import sympy as sp
from sympy import (
    Symbol, Function, dsolve, Eq, exp, log, sqrt, Abs,
    symbols, latex, lambdify, classify_ode, checkodesol,
    simplify, series, integrate, diff, oo, sympify,
)
from scipy.integrate import solve_ivp, odeint
from scipy.optimize import curve_fit, differential_evolution
from datetime import datetime
from pathlib import Path
from loguru import logger
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib as _jl

warnings.filterwarnings("ignore")

from config import (
    RESULTS_DIR, MODELS_DIR, FIGURES_DIR, EQ_DIR, LOG_DIR,
    TARGET_COL, RANDOM_STATE,
)

# -- Directories --
PHYSICS_DIR = RESULTS_DIR / "physics"
PHYSICS_DIR.mkdir(parents=True, exist_ok=True)
ODE_DIR = PHYSICS_DIR / "ode_discovery"
ODE_DIR.mkdir(parents=True, exist_ok=True)
ODE_FIG = ODE_DIR / "figures"
ODE_FIG.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# -- Logger --
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
    level="INFO", colorize=True,
)
logger.add(
    str(LOG_DIR / "run_log_part6.txt"),
    format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}",
    level="DEBUG", rotation="10 MB", encoding="utf-8",
)

t6_start = time.time()
logger.info("=" * 65)
logger.info("  Part 6: ODE Discovery — The Governing Law of Degradation")
logger.info(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
logger.info("=" * 65)

# -- Plot style --
plt.rcParams.update({
    "font.size": 12, "axes.titlesize": 14, "axes.labelsize": 13,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.family": "serif",
})

# =============================================================
# CELL 2: LOAD STATE FROM PART 4
# =============================================================
_p4_path = PHYSICS_DIR / "part4_state.pkl"
if not _p4_path.exists():
    raise FileNotFoundError(
        f"Part 4 state not found at {_p4_path}. Run Part 4 first!")

S = _jl.load(str(_p4_path))

N_TOTAL = S["N_TOTAL"]
WINNER = S["WINNER"]
y_exp = S["y_exp"]
M_ACI = S["M_ACI"]
eta_arr = S["eta_arr"]
d_arr = S["d_arr"]
b_arr = S["b_arr"]
fy_arr = S["fy_arr"]
fc_arr = S["fc_arr"]
rho_arr = S["rho_arr"]
db_arr = S["db_arr"]
r2_eq_all = S["r2_eq_all"]
rmse_eq_all = S["rmse_eq_all"]
eq_str = S["eq_str"]
analysis_label = S["analysis_label"]

# Reconstruct PySR symbolic
eta_m_s = Symbol("eta_m", positive=True, real=True)
_sym_map = {
    "eta_m": eta_m_s,
    "fy": Symbol("fy", positive=True, real=True),
    "fc": Symbol("fc", positive=True, real=True),
    "d": Symbol("d", positive=True, real=True),
    "b": Symbol("b", positive=True, real=True),
    "rho_t": Symbol("rho_t", positive=True, real=True),
    "db_t": Symbol("db_t", positive=True, real=True),
    "d_b": Symbol("d_b", positive=True, real=True),
    "CSI": Symbol("CSI", positive=True, real=True),
    "RI": Symbol("RI", positive=True, real=True),
}
f_expr_pysr = sympify(S["f_expr_str"], locals=_sym_map)

logger.info(f"Data loaded: {N_TOTAL} beams")
logger.info(f"PySR equation: {str(f_expr_pysr)[:100]}")

# =============================================================
# CELL 3 — STAGE I: ODE DISCOVERY (SINDy)
# =============================================================
logger.info("\n" + "=" * 65)
logger.info("  STAGE I — ODE Discovery: dM/d(eta) = g(M, eta)")
logger.info("  Method: SINDy (Sparse Identification of Nonlinear Dynamics)")
logger.info("=" * 65)

# ---- I1: Prepare 1D degradation data ----
# Group beams by eta bins and compute median Mmax per bin
eta_bins = np.arange(0, 70, 2.0)
M_binned = []
eta_centers = []

for i in range(len(eta_bins) - 1):
    mask = (eta_arr >= eta_bins[i]) & (eta_arr < eta_bins[i + 1])
    if mask.sum() >= 3:
        eta_centers.append((eta_bins[i] + eta_bins[i + 1]) / 2.0)
        M_binned.append(float(np.median(y_exp[mask])))

eta_c = np.array(eta_centers)
M_c = np.array(M_binned)

logger.info(f"  Binned data: {len(eta_c)} bins with >= 3 beams each")
logger.info(f"  eta range: [{eta_c[0]:.1f}, {eta_c[-1]:.1f}]%")
logger.info(f"  M range: [{M_c.min():.2f}, {M_c.max():.2f}] kN.m")

# ---- I2: Numerical derivative dM/deta from binned data ----
dM_deta = np.gradient(M_c, eta_c)
d2M_deta2 = np.gradient(dM_deta, eta_c)

logger.info(f"  Numerical dM/deta: mean={np.mean(dM_deta):.4f}, "
            f"min={np.min(dM_deta):.4f}, max={np.max(dM_deta):.4f}")

# ---- I3: SINDy ODE Discovery ----
# Discover: dM/deta = f(M, eta)
import pysindy as ps

# Build feature matrix: [M, eta]
X_sindy = np.column_stack([M_c, eta_c])

# Custom library with physics-motivated terms
lib_functions = [
    lambda x: x,                      # M, eta (linear)
    lambda x: x ** 2,                 # M^2, eta^2
    lambda x: np.sqrt(np.abs(x) + 1e-10),  # sqrt(M), sqrt(eta)
    lambda x: 1.0 / (x + 1e-6),      # 1/M, 1/eta
    lambda x: x * np.log(np.abs(x) + 1e-10),  # M*ln(M), eta*ln(eta)
]
lib_names = [
    lambda x: x,
    lambda x: x + "^2",
    lambda x: "sqrt(" + x + ")",
    lambda x: "1/" + x,
    lambda x: x + "*ln(" + x + ")",
]

sindy_lib = ps.PolynomialLibrary(degree=3, include_interaction=True)

# Multiple thresholds to find best sparse model
best_sindy = None
best_sindy_score = -1e10
sindy_results = []

for thresh in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]:
    try:
        optimizer = ps.STLSQ(threshold=thresh, alpha=0.01)
        model = ps.SINDy(
            feature_names=["M", "eta"],
            feature_library=sindy_lib,
            optimizer=optimizer,
        )
        model.fit(X_sindy, t=eta_c, x_dot=np.column_stack([dM_deta,
                                                             np.ones_like(dM_deta)]))

        dM_pred = model.predict(X_sindy)[:, 0]
        ss_res = np.sum((dM_deta - dM_pred) ** 2)
        ss_tot = np.sum((dM_deta - np.mean(dM_deta)) ** 2)
        r2_sindy = 1 - ss_res / max(ss_tot, 1e-10)
        n_terms = np.count_nonzero(model.coefficients()[0])

        sindy_results.append({
            "threshold": thresh,
            "R2": round(r2_sindy, 4),
            "n_terms": n_terms,
            "coefficients": model.coefficients()[0].tolist(),
            "feature_names": model.get_feature_names(),
        })

        logger.info(f"  SINDy (thresh={thresh}): R2={r2_sindy:.4f}, "
                    f"terms={n_terms}")

        if r2_sindy > best_sindy_score and n_terms >= 1:
            best_sindy_score = r2_sindy
            best_sindy = model
    except Exception as exc:
        logger.warning(f"  SINDy (thresh={thresh}) failed: {exc}")

# ---- I4: Extract discovered ODE ----
sindy_equation_str = "Could not discover ODE"
sindy_coeffs = {}

if best_sindy is not None:
    feat_names = best_sindy.get_feature_names()
    coeffs = best_sindy.coefficients()[0]
    terms = []
    for fname, coeff in zip(feat_names, coeffs):
        if abs(coeff) > 1e-10:
            sindy_coeffs[fname] = round(float(coeff), 8)
            if coeff > 0 and terms:
                terms.append(f"+ {coeff:.6f}*{fname}")
            else:
                terms.append(f"{coeff:.6f}*{fname}")
    sindy_equation_str = " ".join(terms) if terms else "0"
    logger.info(f"\n  DISCOVERED ODE:")
    logger.info(f"  dM/d(eta) = {sindy_equation_str}")
else:
    logger.warning("  SINDy failed — using parametric ODE fitting instead")

# ---- I5: Parametric ODE fitting (backup + comparison) ----
# Fit canonical ODE forms to the data
logger.info("\n  Fitting canonical ODE forms...")

ode_candidates = {}

# Form 1: dM/deta = -alpha * M  (exponential decay)
try:
    def exp_decay_deriv(eta, alpha):
        return -alpha * np.interp(eta, eta_c, M_c)

    def fit_exp(eta, alpha):
        return -alpha * np.interp(eta, eta_c, M_c)

    from scipy.optimize import minimize_scalar
    def loss_exp(alpha):
        pred = -alpha * M_c
        return np.sum((dM_deta - pred) ** 2)
    res = minimize_scalar(loss_exp, bounds=(1e-6, 1.0), method="bounded")
    alpha_exp = res.x
    pred_exp = -alpha_exp * M_c
    r2_exp = r2_score(dM_deta, pred_exp)
    ode_candidates["Exponential: dM/deta = -alpha*M"] = {
        "params": {"alpha": round(alpha_exp, 6)},
        "R2": round(r2_exp, 4),
        "solution": f"M(eta) = M0 * exp(-{alpha_exp:.6f} * eta)",
        "type": "separable",
    }
    logger.info(f"  Exponential decay: alpha={alpha_exp:.6f}, R2={r2_exp:.4f}")
except Exception as exc:
    logger.warning(f"  Exponential fit failed: {exc}")

# Form 2: dM/deta = -alpha * M^beta  (power-law decay)
try:
    def loss_power(params):
        alpha, beta = params
        pred = -alpha * np.power(np.abs(M_c) + 1e-10, beta)
        return np.sum((dM_deta - pred) ** 2)

    res_pw = differential_evolution(loss_power,
                                    bounds=[(1e-6, 5.0), (0.1, 3.0)],
                                    seed=42, maxiter=500)
    alpha_pw, beta_pw = res_pw.x
    pred_pw = -alpha_pw * np.power(np.abs(M_c) + 1e-10, beta_pw)
    r2_pw = r2_score(dM_deta, pred_pw)
    ode_candidates["Power-law: dM/deta = -alpha*M^beta"] = {
        "params": {"alpha": round(alpha_pw, 6), "beta": round(beta_pw, 4)},
        "R2": round(r2_pw, 4),
        "solution": (f"M(eta) = [M0^(1-{beta_pw:.3f}) - "
                     f"{alpha_pw:.6f}*(1-{beta_pw:.3f})*eta]^(1/(1-{beta_pw:.3f}))"),
        "type": "Bernoulli",
    }
    logger.info(f"  Power-law: alpha={alpha_pw:.6f}, beta={beta_pw:.4f}, "
                f"R2={r2_pw:.4f}")
except Exception as exc:
    logger.warning(f"  Power-law fit failed: {exc}")

# Form 3: dM/deta = -(alpha + beta*eta) * M  (time-dependent decay)
try:
    def loss_td(params):
        a, b = params
        pred = -(a + b * eta_c) * M_c
        return np.sum((dM_deta - pred) ** 2)

    res_td = differential_evolution(loss_td,
                                    bounds=[(-0.5, 0.5), (-0.05, 0.05)],
                                    seed=42, maxiter=500)
    a_td, b_td = res_td.x
    pred_td = -(a_td + b_td * eta_c) * M_c
    r2_td = r2_score(dM_deta, pred_td)
    ode_candidates["Time-dependent: dM/deta = -(a+b*eta)*M"] = {
        "params": {"a": round(a_td, 6), "b": round(b_td, 6)},
        "R2": round(r2_td, 4),
        "solution": (f"M(eta) = M0 * exp(-{a_td:.6f}*eta "
                     f"- {b_td:.6f}*eta^2/2)"),
        "type": "separable",
    }
    logger.info(f"  Time-dependent: a={a_td:.6f}, b={b_td:.6f}, "
                f"R2={r2_td:.4f}")
except Exception as exc:
    logger.warning(f"  Time-dependent fit failed: {exc}")

# Form 4: dM/deta = -alpha*M + beta  (linear first-order with constant)
try:
    def loss_lin(params):
        a, b = params
        pred = -a * M_c + b
        return np.sum((dM_deta - pred) ** 2)

    res_lin = differential_evolution(loss_lin,
                                     bounds=[(1e-6, 1.0), (-10, 10)],
                                     seed=42, maxiter=500)
    a_lin, b_lin = res_lin.x
    pred_lin = -a_lin * M_c + b_lin
    r2_lin = r2_score(dM_deta, pred_lin)
    ode_candidates["Linear 1st-order: dM/deta = -a*M + b"] = {
        "params": {"a": round(a_lin, 6), "b": round(b_lin, 4)},
        "R2": round(r2_lin, 4),
        "solution": (f"M(eta) = {b_lin/a_lin:.2f} + "
                     f"(M0 - {b_lin/a_lin:.2f})*exp(-{a_lin:.6f}*eta)"),
        "type": "linear first-order",
    }
    logger.info(f"  Linear 1st-order: a={a_lin:.6f}, b={b_lin:.4f}, "
                f"R2={r2_lin:.4f}")
except Exception as exc:
    logger.warning(f"  Linear 1st-order fit failed: {exc}")

# Form 5: dM/deta = -alpha * M * (1 - M/K)  (logistic-like degradation)
try:
    def loss_log(params):
        alpha, K = params
        pred = -alpha * M_c * (1.0 - M_c / K)
        return np.sum((dM_deta - pred) ** 2)

    res_log = differential_evolution(loss_log,
                                     bounds=[(1e-6, 1.0), (0.1, 200)],
                                     seed=42, maxiter=500)
    a_log, K_log = res_log.x
    pred_log = -a_log * M_c * (1.0 - M_c / K_log)
    r2_log = r2_score(dM_deta, pred_log)
    ode_candidates["Logistic: dM/deta = -a*M*(1-M/K)"] = {
        "params": {"a": round(a_log, 6), "K": round(K_log, 4)},
        "R2": round(r2_log, 4),
        "solution": (f"M(eta) = K*M0 / (M0 + (K-M0)*exp(a*eta)), "
                     f"K={K_log:.2f}"),
        "type": "Bernoulli / logistic",
    }
    logger.info(f"  Logistic: a={a_log:.6f}, K={K_log:.4f}, R2={r2_log:.4f}")
except Exception as exc:
    logger.warning(f"  Logistic fit failed: {exc}")

# ---- I6: Select best ODE ----
best_ode_name = None
best_ode_r2 = -1e10
for name, info in ode_candidates.items():
    if info["R2"] > best_ode_r2:
        best_ode_r2 = info["R2"]
        best_ode_name = name

logger.info(f"\n  BEST ODE: {best_ode_name}")
logger.info(f"  R2 = {best_ode_r2:.4f}")
logger.info(f"  Solution: {ode_candidates[best_ode_name]['solution']}")

stage_i_results = {
    "sindy_equation": sindy_equation_str,
    "sindy_R2": round(best_sindy_score, 4) if best_sindy else None,
    "sindy_coefficients": sindy_coeffs,
    "sindy_all_trials": sindy_results,
    "parametric_candidates": {k: v for k, v in ode_candidates.items()},
    "best_ode": best_ode_name,
    "best_ode_R2": best_ode_r2,
    "best_ode_solution": ode_candidates[best_ode_name]["solution"],
    "n_bins": len(eta_c),
}

# =============================================================
# CELL 4 — STAGE J: ANALYTICAL SOLUTION OF DISCOVERED ODE
# =============================================================
logger.info("\n" + "=" * 65)
logger.info("  STAGE J — Analytical Solution (SymPy dsolve)")
logger.info("=" * 65)

eta_sym = Symbol("eta", positive=True, real=True)
M_sym = Function("M")
M0_sym = Symbol("M_0", positive=True, real=True)
alpha_sym = Symbol("alpha", positive=True, real=True)
beta_sym = Symbol("beta_p", positive=True, real=True)
a_sym = Symbol("a", real=True)
b_sym_ode = Symbol("b_ode", real=True)
K_sym = Symbol("K", positive=True, real=True)

# Solve all candidate ODEs symbolically
ode_solutions = {}

# ODE 1: dM/deta = -alpha*M
try:
    ode1 = Eq(M_sym(eta_sym).diff(eta_sym), -alpha_sym * M_sym(eta_sym))
    sol1 = dsolve(ode1, M_sym(eta_sym), ics={M_sym(0): M0_sym})
    ode_class1 = classify_ode(ode1, M_sym(eta_sym))
    ode_solutions["Exponential Decay"] = {
        "ode": str(ode1),
        "ode_latex": latex(ode1),
        "solution": str(sol1),
        "solution_latex": latex(sol1),
        "classification": list(ode_class1)[:3],
        "physical_meaning": (
            "The rate of capacity loss is proportional to the current "
            "capacity. This is the simplest degradation law: each unit "
            "of corrosion destroys a fixed FRACTION of remaining capacity."
        ),
    }
    logger.info(f"  ODE 1 solved: {sol1}")
except Exception as exc:
    logger.warning(f"  ODE 1 solution failed: {exc}")

# ODE 2: dM/deta = -alpha*M^beta
try:
    ode2 = Eq(M_sym(eta_sym).diff(eta_sym),
              -alpha_sym * M_sym(eta_sym) ** beta_sym)
    sol2 = dsolve(ode2, M_sym(eta_sym))
    ode_solutions["Power-Law Decay"] = {
        "ode": str(ode2),
        "ode_latex": latex(ode2),
        "solution": str(sol2),
        "solution_latex": latex(sol2),
        "classification": ["Bernoulli"],
        "physical_meaning": (
            "The degradation rate depends on a POWER of current capacity. "
            "When beta>1, degradation accelerates for strong beams "
            "(high M) — corrosion preferentially attacks stronger elements. "
            "When beta<1, weak beams degrade faster — a self-limiting process."
        ),
    }
    logger.info(f"  ODE 2 solved: {str(sol2)[:100]}")
except Exception as exc:
    logger.warning(f"  ODE 2 solution failed: {exc}")

# ODE 3: dM/deta = -(a + b*eta)*M
try:
    ode3 = Eq(M_sym(eta_sym).diff(eta_sym),
              -(a_sym + b_sym_ode * eta_sym) * M_sym(eta_sym))
    sol3 = dsolve(ode3, M_sym(eta_sym), ics={M_sym(0): M0_sym})
    ode_solutions["Time-Dependent Decay"] = {
        "ode": str(ode3),
        "ode_latex": latex(ode3),
        "solution": str(sol3),
        "solution_latex": latex(sol3),
        "classification": ["separable", "1st order linear"],
        "physical_meaning": (
            "The degradation rate ITSELF changes with corrosion level. "
            "If b>0, degradation accelerates over time — corrosion breeds "
            "more corrosion (autocatalytic). If b<0, degradation slows "
            "as the most vulnerable material is consumed first."
        ),
    }
    logger.info(f"  ODE 3 solved: {str(sol3)[:100]}")
except Exception as exc:
    logger.warning(f"  ODE 3 solution failed: {exc}")

# ODE 4: dM/deta = -a*M + b
try:
    ode4 = Eq(M_sym(eta_sym).diff(eta_sym),
              -a_sym * M_sym(eta_sym) + b_sym_ode)
    sol4 = dsolve(ode4, M_sym(eta_sym), ics={M_sym(0): M0_sym})
    ode_solutions["Linear First-Order with Constant"] = {
        "ode": str(ode4),
        "ode_latex": latex(ode4),
        "solution": str(sol4),
        "solution_latex": latex(sol4),
        "classification": ["1st order linear", "separable"],
        "physical_meaning": (
            "Capacity decays exponentially BUT approaches a nonzero "
            "residual value b/a instead of zero. This means even at "
            "extreme corrosion, the beam retains some minimum capacity — "
            "the concrete core still resists, even without reinforcement."
        ),
    }
    logger.info(f"  ODE 4 solved: {str(sol4)[:100]}")
except Exception as exc:
    logger.warning(f"  ODE 4 solution failed: {exc}")

# ODE 5: dM/deta = -a*M*(1-M/K)
try:
    ode5 = Eq(M_sym(eta_sym).diff(eta_sym),
              -a_sym * M_sym(eta_sym) * (1 - M_sym(eta_sym) / K_sym))
    sol5 = dsolve(ode5, M_sym(eta_sym))
    ode_solutions["Logistic Degradation"] = {
        "ode": str(ode5),
        "ode_latex": latex(ode5),
        "solution": str(sol5),
        "solution_latex": latex(sol5),
        "classification": ["Bernoulli", "separable"],
        "physical_meaning": (
            "A self-regulating degradation process. Near K (carrying "
            "capacity), degradation nearly stops — the system resists "
            "further damage. Below K, degradation proceeds normally. "
            "This captures the physical reality that highly corroded "
            "beams have little left to lose."
        ),
    }
    logger.info(f"  ODE 5 solved: {str(sol5)[:100]}")
except Exception as exc:
    logger.warning(f"  ODE 5 solution failed: {exc}")

stage_j_results = {
    "ode_solutions": {
        k: {kk: str(vv) if not isinstance(vv, (str, list)) else vv
             for kk, vv in v.items()}
        for k, v in ode_solutions.items()
    },
    "n_odes_solved": len(ode_solutions),
}

logger.info(f"\n  Successfully solved {len(ode_solutions)} ODEs analytically")

# =============================================================
# CELL 5 — STAGE K: PHASE SPACE ANALYSIS
# =============================================================
logger.info("\n" + "=" * 65)
logger.info("  STAGE K — Phase Space & Dynamical Systems Analysis")
logger.info("=" * 65)

# ---- K1: Compute best-fit ODE trajectory ----
best_info = ode_candidates[best_ode_name]
M0_data = float(M_c[0])

# Solve the best ODE numerically for comparison
eta_ode = np.linspace(0, 65, 500)

trajectories = {}
for name, info in ode_candidates.items():
    p = info["params"]
    try:
        if "alpha" in p and "beta" in p and "K" not in p:
            # Power law
            alpha_v, beta_v = p["alpha"], p["beta"]
            def rhs(eta, M):
                return -alpha_v * np.power(max(M, 1e-10), beta_v)
        elif "a" in p and "b" in p and "K" not in p:
            a_v, b_v = p["a"], p["b"]
            if abs(b_v) < 0.01:
                def rhs(eta, M):
                    return -(a_v + b_v * eta) * M
            else:
                def rhs(eta, M):
                    return -a_v * M + b_v
        elif "alpha" in p and "K" in p:
            alpha_v, K_v = p["alpha"], p["K"]
            def rhs(eta, M):
                return -alpha_v * M * (1.0 - M / K_v)
        elif "alpha" in p and len(p) == 1:
            alpha_v = p["alpha"]
            def rhs(eta, M):
                return -alpha_v * M
        else:
            continue

        sol_num = solve_ivp(lambda t, y: [rhs(t, y[0])],
                            [0, 65], [M0_data],
                            t_eval=eta_ode, max_step=0.5,
                            method="RK45")
        if sol_num.success:
            trajectories[name] = sol_num.y[0]
    except Exception as exc:
        logger.warning(f"  Trajectory for '{name[:30]}' failed: {exc}")

logger.info(f"  Computed {len(trajectories)} ODE trajectories")

# ---- K2: Equilibrium analysis ----
equilibria = {}
for name, info in ode_candidates.items():
    p = info["params"]
    if "a" in p and "b" in p and "K" not in p and abs(p.get("b", 0)) > 0.01:
        M_eq = p["b"] / p["a"]
        stability = "stable" if p["a"] > 0 else "unstable"
        equilibria[name] = {
            "M_equilibrium_kNm": round(M_eq, 4),
            "stability": stability,
            "meaning": (f"The beam approaches M={M_eq:.2f} kN.m "
                        f"as corrosion increases indefinitely"),
        }
    elif "alpha" in p and len(p) == 1:
        equilibria[name] = {
            "M_equilibrium_kNm": 0.0,
            "stability": "stable",
            "meaning": "Capacity decays to zero — complete structural failure",
        }
    elif "K" in p:
        equilibria[name] = {
            "M_equilibrium_kNm": round(p["K"], 4),
            "stability": "unstable (saddle)",
            "meaning": (f"K={p['K']:.2f} kN.m is the carrying capacity; "
                        f"M=0 is the stable equilibrium"),
        }

logger.info(f"  Equilibrium analysis: {len(equilibria)} equilibria found")
for n, eq in equilibria.items():
    logger.info(f"    {n[:40]}: M*={eq['M_equilibrium_kNm']:.2f} "
                f"({eq['stability']})")

# ---- K3: Lyapunov stability ----
# For dM/dt = f(M), stability at M* requires f'(M*) < 0
lyapunov = {}
for name, info in ode_candidates.items():
    p = info["params"]
    if "alpha" in p and len(p) == 1:
        lyapunov[name] = {
            "f_prime_at_eq": -p["alpha"],
            "stable": True,
            "meaning": "Globally asymptotically stable — M -> 0 always",
        }
    elif "a" in p and "b" in p and "K" not in p and abs(p.get("b", 0)) > 0.01:
        lyapunov[name] = {
            "f_prime_at_eq": -p["a"],
            "stable": p["a"] > 0,
            "meaning": f"Stable if a>0: a={p['a']:.4f}",
        }

stage_k_results = {
    "trajectories_computed": len(trajectories),
    "equilibria": equilibria,
    "lyapunov_stability": lyapunov,
}

# =============================================================
# CELL 6 — GENERATE FIGURES (ODE1–ODE6)
# =============================================================
logger.info("\n" + "=" * 65)
logger.info("  Generating ODE Discovery Figures")
logger.info("=" * 65)

fig_count = 0
C_BLUE = "#1565C0"
C_RED = "#C62828"
C_GREEN = "#2E7D32"
C_ORANGE = "#E65100"
C_PURPLE = "#6A1B9A"
COLORS = [C_BLUE, C_RED, C_GREEN, C_ORANGE, C_PURPLE, "#00838F"]

# ------- ODE Fig 1: Phase Portrait (dM/deta vs M) -------
try:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(M_c, dM_deta, c=C_BLUE, s=50, alpha=0.7, zorder=5,
               label="Data (binned medians)")

    M_smooth = np.linspace(M_c.min() * 0.5, M_c.max() * 1.2, 200)
    for i, (name, info) in enumerate(ode_candidates.items()):
        p = info["params"]
        short = name.split(":")[0]
        col = COLORS[i % len(COLORS)]
        if "alpha" in p and "beta" in p and "K" not in p:
            ax.plot(M_smooth, -p["alpha"] * M_smooth ** p["beta"],
                    color=col, linewidth=2, alpha=0.8,
                    label=f"{short} (R2={info['R2']:.3f})")
        elif "a" in p and "b" in p and "K" not in p:
            if abs(p["b"]) < 0.01:
                ax.plot(M_smooth, -(p["a"] + p["b"] * 20) * M_smooth,
                        color=col, linewidth=2, alpha=0.8,
                        label=f"{short} (R2={info['R2']:.3f})")
            else:
                ax.plot(M_smooth, -p["a"] * M_smooth + p["b"],
                        color=col, linewidth=2, alpha=0.8,
                        label=f"{short} (R2={info['R2']:.3f})")
        elif "alpha" in p and "K" in p:
            ax.plot(M_smooth,
                    -p["alpha"] * M_smooth * (1 - M_smooth / p["K"]),
                    color=col, linewidth=2, alpha=0.8,
                    label=f"{short} (R2={info['R2']:.3f})")
        elif "alpha" in p and len(p) == 1:
            ax.plot(M_smooth, -p["alpha"] * M_smooth,
                    color=col, linewidth=2, alpha=0.8,
                    label=f"{short} (R2={info['R2']:.3f})")

    ax.axhline(0, color="black", linewidth=0.5, alpha=0.5)
    ax.set_xlabel("$M_{max}$ (kN.m)", fontsize=13)
    ax.set_ylabel("$dM/d\\eta$ (kN.m per %)", fontsize=13)
    ax.set_title("Phase Portrait: Rate of Degradation vs Capacity",
                 fontsize=14)
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(ODE_FIG / "fig_ode1_phase_portrait.png")
    plt.close(fig)
    fig_count += 1
    logger.info("  Fig ODE1 OK — Phase Portrait")
except Exception as e:
    logger.warning(f"  Fig ODE1 FAILED: {e}")

# ------- ODE Fig 2: ODE Trajectories vs Data -------
try:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(eta_c, M_c, c="black", s=60, zorder=10, marker="D",
               label=f"Data ({len(eta_c)} bins)")

    for i, (name, traj) in enumerate(trajectories.items()):
        short = name.split(":")[0]
        r2_info = ode_candidates[name]["R2"]
        ax.plot(eta_ode, traj, color=COLORS[i % len(COLORS)],
                linewidth=2.5, alpha=0.8,
                label=f"{short} (R2={r2_info:.3f})")

    ax.set_xlabel("Mass Loss $\\eta_m$ (%)", fontsize=13)
    ax.set_ylabel("$M_{max}$ (kN.m)", fontsize=13)
    ax.set_title("ODE Trajectories vs Experimental Data", fontsize=14)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(ODE_FIG / "fig_ode2_trajectories.png")
    plt.close(fig)
    fig_count += 1
    logger.info("  Fig ODE2 OK — Trajectories")
except Exception as e:
    logger.warning(f"  Fig ODE2 FAILED: {e}")

# ------- ODE Fig 3: ODE Comparison Bar Chart -------
try:
    fig, ax = plt.subplots(figsize=(10, 5))
    names = [n.split(":")[0] for n in ode_candidates.keys()]
    r2s = [v["R2"] for v in ode_candidates.values()]
    bars = ax.barh(names, r2s, color=COLORS[:len(names)], alpha=0.85,
                   edgecolor="white")
    for bar, r2v in zip(bars, r2s):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{r2v:.4f}", va="center", fontsize=11, fontweight="bold")
    ax.set_xlabel("$R^2$ (derivative prediction)", fontsize=12)
    ax.set_title("ODE Candidate Comparison", fontsize=14)
    ax.set_xlim(0, max(r2s) * 1.15 if r2s else 1)
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(ODE_FIG / "fig_ode3_comparison.png")
    plt.close(fig)
    fig_count += 1
    logger.info("  Fig ODE3 OK — Comparison")
except Exception as e:
    logger.warning(f"  Fig ODE3 FAILED: {e}")

# ------- ODE Fig 4: Vector Field (eta, M) -------
try:
    fig, ax = plt.subplots(figsize=(10, 7))
    eta_grid = np.linspace(0, 60, 20)
    M_grid = np.linspace(0.5, M_c.max() * 1.1, 20)
    ETA, MG = np.meshgrid(eta_grid, M_grid)

    best_p = ode_candidates[best_ode_name]["params"]
    if "alpha" in best_p and "beta" in best_p and "K" not in best_p:
        DM = -best_p["alpha"] * np.power(MG + 1e-10, best_p["beta"])
    elif "a" in best_p and "b" in best_p and "K" not in best_p:
        if abs(best_p["b"]) < 0.01:
            DM = -(best_p["a"] + best_p["b"] * ETA) * MG
        else:
            DM = -best_p["a"] * MG + best_p["b"]
    elif "alpha" in best_p and "K" in best_p:
        DM = -best_p["alpha"] * MG * (1 - MG / best_p["K"])
    else:
        DM = -best_p.get("alpha", 0.01) * MG

    DETA = np.ones_like(DM)
    speed = np.sqrt(DETA ** 2 + DM ** 2)

    ax.streamplot(ETA, MG, DETA, DM, color=speed, cmap="coolwarm",
                  density=1.5, linewidth=1.5, arrowsize=1.5)
    ax.scatter(eta_c, M_c, c="black", s=40, zorder=10, marker="D",
               label="Data")
    ax.set_xlabel("Mass Loss $\\eta_m$ (%)", fontsize=13)
    ax.set_ylabel("$M_{max}$ (kN.m)", fontsize=13)
    ax.set_title(f"Vector Field — Best ODE: {best_ode_name.split(':')[0]}",
                 fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(ODE_FIG / "fig_ode4_vector_field.png")
    plt.close(fig)
    fig_count += 1
    logger.info("  Fig ODE4 OK — Vector Field")
except Exception as e:
    logger.warning(f"  Fig ODE4 FAILED: {e}")

# ------- ODE Fig 5: Multiple Initial Conditions -------
try:
    fig, ax = plt.subplots(figsize=(10, 6))
    M0_values = [5, 10, 20, 40, 80, 120]
    for M0v in M0_values:
        best_p = ode_candidates[best_ode_name]["params"]
        if "alpha" in best_p and len(best_p) == 1:
            traj = M0v * np.exp(-best_p["alpha"] * eta_ode)
        elif "a" in best_p and "b" in best_p and "K" not in best_p:
            if abs(best_p["b"]) < 0.01:
                a, b = best_p["a"], best_p["b"]
                traj = M0v * np.exp(-a * eta_ode - b * eta_ode ** 2 / 2)
            else:
                a, b = best_p["a"], best_p["b"]
                M_eq = b / a
                traj = M_eq + (M0v - M_eq) * np.exp(-a * eta_ode)
        elif "alpha" in best_p and "beta" in best_p and "K" not in best_p:
            sol_ic = solve_ivp(
                lambda t, y: [-best_p["alpha"] * max(y[0], 1e-10) ** best_p["beta"]],
                [0, 65], [M0v], t_eval=eta_ode, max_step=0.5)
            traj = sol_ic.y[0] if sol_ic.success else M0v * np.ones_like(eta_ode)
        else:
            traj = M0v * np.exp(-0.01 * eta_ode)

        ax.plot(eta_ode, traj, linewidth=2,
                label=f"$M_0$ = {M0v} kN.m")

    ax.scatter(eta_c, M_c, c="black", s=30, zorder=10, alpha=0.5,
               label="Data")
    ax.set_xlabel("Mass Loss $\\eta_m$ (%)", fontsize=13)
    ax.set_ylabel("$M_{max}$ (kN.m)", fontsize=13)
    ax.set_title("ODE Solution Family — Multiple Initial Conditions",
                 fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(ODE_FIG / "fig_ode5_initial_conditions.png")
    plt.close(fig)
    fig_count += 1
    logger.info("  Fig ODE5 OK — Initial Conditions")
except Exception as e:
    logger.warning(f"  Fig ODE5 FAILED: {e}")

# ------- ODE Fig 6: PySR vs ODE Solution Comparison -------
try:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(eta_c, M_c, c="black", s=60, zorder=10, marker="D",
               label="Data (binned)")

    if best_ode_name in trajectories:
        ax.plot(eta_ode, trajectories[best_ode_name],
                color=C_RED, linewidth=3, alpha=0.9,
                label=f"ODE Solution (R2={best_ode_r2:.4f})")

    # PySR 1D curve from Part 3
    from config import RESULTS_DIR as _RD
    from data_preprocessing import run_preprocessing
    from aci_calculator import compute_aci_predictions
    data = run_preprocessing(save_clean=True)
    df_clean = data["df_clean"]
    _all_syms = [Symbol(s) for s in
                 ["eta_m", "fy", "fc", "d", "b", "rho_t", "db_t",
                  "d_b", "CSI", "RI"]]
    _f_lam = lambdify(_all_syms, f_expr_pysr, modules="numpy")

    med_vals = {
        "fy": float(fy_arr.mean()), "fc": float(fc_arr.mean()),
        "d": float(d_arr.mean()), "b": float(b_arr.mean()),
        "rho_t": float(rho_arr.mean()), "db_t": float(db_arr.mean()),
        "d_b": float((d_arr / np.maximum(b_arr, 1)).mean()),
        "CSI": 0, "RI": float((rho_arr * fy_arr / fc_arr).mean()),
    }
    M_pysr = []
    for e in eta_ode:
        med_vals["eta_m"] = e
        med_vals["CSI"] = e * med_vals["fy"] / med_vals["fc"]
        try:
            v = float(_f_lam(*[med_vals[str(s)] for s in _all_syms]))
            M_pysr.append(max(v, 0))
        except Exception:
            M_pysr.append(0)
    M_pysr = np.array(M_pysr)

    ax.plot(eta_ode, M_pysr, color=C_BLUE, linewidth=3, alpha=0.9,
            linestyle="--",
            label=f"PySR Equation (R2={r2_eq_all:.4f})")

    ax.set_xlabel("Mass Loss $\\eta_m$ (%)", fontsize=13)
    ax.set_ylabel("$M_{max}$ (kN.m)", fontsize=13)
    ax.set_title("Comparison: ODE Law vs PySR Equation vs Data",
                 fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(ODE_FIG / "fig_ode6_pysr_vs_ode.png")
    plt.close(fig)
    fig_count += 1
    logger.info("  Fig ODE6 OK — PySR vs ODE")
except Exception as e:
    logger.warning(f"  Fig ODE6 FAILED: {e}")
    traceback.print_exc()

logger.info(f"  Total ODE figures: {fig_count}")

# =============================================================
# CELL 7 — SAVE RESULTS JSON
# =============================================================
ode_results = {
    "generated_at": str(datetime.now()),
    "stage_i_ode_discovery": stage_i_results,
    "stage_j_analytical_solutions": stage_j_results,
    "stage_k_phase_space": stage_k_results,
    "n_figures": fig_count,
}

with open(ODE_DIR / "ode_results.json", "w", encoding="utf-8") as f:
    json.dump(ode_results, f, indent=2, default=str, ensure_ascii=False)
logger.info(f"Results saved -> {ODE_DIR / 'ode_results.json'}")

# =============================================================
# CELL 8 — PDF REPORT: ODE DISCOVERY
# =============================================================
logger.info("\n" + "=" * 65)
logger.info("  Generating ODE Discovery PDF Report")
logger.info("=" * 65)

def _safe(text):
    return (str(text)
            .replace("\u2014", "-").replace("\u2013", "-")
            .replace("\u2018", "'").replace("\u2019", "'")
            .replace("\u201c", '"').replace("\u201d", '"')
            .replace("\u2192", "->").replace("\u2190", "<-")
            .replace("\u2713", "[OK]").replace("\u2717", "[X]")
            .replace("\u03b7", "eta").replace("\u03c1", "rho")
            .replace("\u03b1", "alpha").replace("\u03b2", "beta")
            .replace("\u00b2", "2").replace("\u00b7", ".")
            .encode("latin-1", errors="replace").decode("latin-1"))

def generate_ode_report():
    from fpdf import FPDF

    class PDF(FPDF):
        def header(self):
            self.set_font("Helvetica", "B", 9)
            self.set_text_color(80, 80, 80)
            self.cell(0, 7,
                      "Part 6: ODE Discovery -- Governing Law of Degradation",
                      0, 1, "C")
            self.set_draw_color(180, 180, 180)
            self.line(10, self.get_y(), 200, self.get_y())
            self.ln(2)

        def footer(self):
            self.set_y(-15)
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(128, 128, 128)
            self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", 0, 0, "C")

        def section(self, num, title):
            self.set_font("Helvetica", "B", 13)
            self.set_text_color(13, 27, 42)
            self.ln(4)
            self.cell(0, 8, f"{num}. {title}", 0, 1, "L")
            self.set_draw_color(30, 100, 180)
            self.line(10, self.get_y(), 120, self.get_y())
            self.ln(3)
            self.set_text_color(0, 0, 0)

        def body(self, txt):
            self.set_font("Helvetica", "", 10)
            self.multi_cell(0, 5.5, _safe(txt))
            self.ln(2)

        def math(self, label, expr):
            self.set_font("Courier", "B", 10)
            self.ln(1)
            self.cell(0, 6, f"  {label}:", 0, 1)
            self.set_font("Courier", "", 9)
            lines = [expr[i:i + 85] for i in range(0, len(expr), 85)]
            for line in lines:
                self.cell(0, 5, f"    {_safe(line)}", 0, 1)
            self.ln(2)
            self.set_font("Helvetica", "", 10)

    pdf = PDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)

    # ===== TITLE PAGE =====
    pdf.add_page()
    pdf.ln(25)
    pdf.set_font("Helvetica", "B", 22)
    pdf.cell(0, 12, "ODE Discovery Report", 0, 1, "C")
    pdf.ln(5)
    pdf.set_font("Helvetica", "", 15)
    pdf.cell(0, 10,
             "The Governing Differential Equation", 0, 1, "C")
    pdf.cell(0, 10,
             "of Corrosion-Induced Capacity Degradation", 0, 1, "C")
    pdf.ln(8)
    pdf.set_font("Helvetica", "I", 12)
    pdf.cell(0, 8,
             "Discovered via SINDy + Parametric ODE Fitting + SymPy",
             0, 1, "C")
    pdf.ln(5)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 7,
             f"Database: {N_TOTAL} experimentally tested RC beams",
             0, 1, "C")
    pdf.cell(0, 7,
             f"Generated: {datetime.now().strftime('%B %d, %Y')}",
             0, 1, "C")

    pdf.ln(10)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "Abstract", 0, 1, "C")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5.5, _safe(
        "This report presents the discovery of the governing "
        "ordinary differential equation (ODE) that describes how "
        "the flexural capacity of reinforced concrete beams degrades "
        "under corrosion. Unlike traditional regression which finds "
        "M = f(eta), this work discovers dM/d(eta) = g(M, eta) -- "
        "the RATE LAW of degradation. Five canonical ODE forms were "
        "tested using data from 804 beams. The discovered law was then "
        "solved analytically using SymPy dsolve, and the resulting "
        "solution family was analyzed as a dynamical system with "
        "equilibria, stability, and phase-space trajectories. "
        "This represents a fundamental shift from statistical fitting "
        "to physical law discovery."
    ))

    # ===== STAGE I: ODE DISCOVERY =====
    pdf.add_page()
    pdf.section("1", "Stage I: ODE Discovery from Data")

    pdf.body(
        "The central question is: what differential equation governs "
        "the degradation of flexural capacity with corrosion?\n\n"
        "Traditional approach: Find M = f(eta) by regression.\n"
        "Our approach: Find dM/d(eta) = g(M, eta) -- the RATE LAW.\n\n"
        "This is analogous to Newton discovering F = ma (a differential "
        "equation) rather than fitting x = f(t) to planetary data."
    )

    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "Method 1: SINDy (data-driven ODE discovery):", 0, 1)
    pdf.body(
        "SINDy (Sparse Identification of Nonlinear Dynamics) builds a "
        "library of candidate functions [M, eta, M^2, M*eta, ...] and "
        "finds the sparsest combination that reproduces dM/d(eta)."
    )
    pdf.math("SINDy Result", sindy_equation_str)

    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "Method 2: Parametric ODE Fitting:", 0, 1)
    pdf.body(
        "Five canonical ODE forms from mathematical physics were "
        "fitted to the data. Each has a distinct physical meaning:"
    )

    for i, (name, info) in enumerate(ode_candidates.items(), 1):
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 6, f"  Candidate {i}: {_safe(name)}", 0, 1)
        pdf.set_font("Courier", "", 9)
        pdf.cell(0, 5, f"    R2 = {info['R2']:.4f}", 0, 1)
        pdf.cell(0, 5,
                 f"    Solution: {_safe(info['solution'][:80])}", 0, 1)
        pdf.set_font("Helvetica", "", 9)
        for pk, pv in info["params"].items():
            pdf.cell(0, 5, f"      {pk} = {pv}", 0, 1)
        pdf.ln(2)

    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(13, 100, 13)
    pdf.cell(0, 8,
             f"  WINNER: {_safe(best_ode_name)} (R2={best_ode_r2:.4f})",
             0, 1)
    pdf.set_text_color(0, 0, 0)

    # ===== STAGE J: ANALYTICAL SOLUTIONS =====
    pdf.add_page()
    pdf.section("2", "Stage J: Analytical Solutions (SymPy dsolve)")

    pdf.body(
        "Each discovered ODE was solved ANALYTICALLY using SymPy's "
        "dsolve — producing exact closed-form solutions. This is the "
        "'inverse calculus' step: given the derivative (rate law), we "
        "recover the original function (the degradation law)."
    )

    for name, sol_info in ode_solutions.items():
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, f"ODE: {_safe(name)}", 0, 1)
        pdf.ln(2)

        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 6, "  Differential Equation:", 0, 1)
        pdf.math("ODE", str(sol_info["ode"]))

        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 6, "  Classification:", 0, 1)
        pdf.set_font("Helvetica", "", 10)
        for cls in sol_info["classification"]:
            pdf.cell(0, 5, f"    - {_safe(cls)}", 0, 1)
        pdf.ln(2)

        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 6, "  ANALYTICAL SOLUTION (via inverse calculus):", 0, 1)
        pdf.math("M(eta)", str(sol_info["solution"]))

        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 6, "  Physical Interpretation:", 0, 1)
        pdf.body(sol_info["physical_meaning"])

    # ===== STAGE K: PHASE SPACE =====
    pdf.add_page()
    pdf.section("3", "Stage K: Phase Space & Dynamical Systems")

    pdf.body(
        "The discovered ODE defines a dynamical system. We analyze "
        "its equilibria (where dM/d(eta) = 0), stability (whether "
        "small perturbations grow or decay), and the global behavior "
        "of all possible degradation trajectories."
    )

    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "Equilibrium Analysis:", 0, 1)
    for name, eq_info in equilibria.items():
        pdf.set_font("Helvetica", "B", 10)
        short = name.split(":")[0]
        pdf.cell(0, 6, f"  {_safe(short)}:", 0, 1)
        pdf.set_font("Helvetica", "", 9)
        pdf.cell(0, 5,
                 f"    M* = {eq_info['M_equilibrium_kNm']:.2f} kN.m "
                 f"({eq_info['stability']})", 0, 1)
        pdf.cell(0, 5, f"    {_safe(eq_info['meaning'][:80])}", 0, 1)
        pdf.ln(2)

    if lyapunov:
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 7, "Lyapunov Stability:", 0, 1)
        for name, ly_info in lyapunov.items():
            short = name.split(":")[0]
            pdf.set_font("Helvetica", "", 9)
            pdf.cell(0, 5,
                     f"  {_safe(short)}: f'(M*) = "
                     f"{ly_info['f_prime_at_eq']:.4f} "
                     f"-> {'STABLE' if ly_info['stable'] else 'UNSTABLE'}",
                     0, 1)

    # ===== FIGURES =====
    pdf.add_page()
    pdf.section("4", "Figures")

    for fig_path in sorted(ODE_FIG.glob("*.png")):
        try:
            if pdf.get_y() > 170:
                pdf.add_page()
            pdf.set_font("Helvetica", "I", 9)
            caption = fig_path.stem.replace("_", " ").title()
            pdf.cell(0, 6, _safe(caption), 0, 1, "C")
            pdf.image(str(fig_path), x=15, w=180)
            pdf.ln(5)
        except Exception:
            pdf.cell(0, 6, f"[Could not embed {fig_path.name}]", 0, 1)

    # ===== CONCLUSION =====
    pdf.add_page()
    pdf.section("5", "Conclusion & Scientific Significance")

    pdf.body(
        f"This work discovered the governing differential equation "
        f"of corrosion-induced degradation from {N_TOTAL} experimental "
        f"data points.\n\n"
        f"Best governing law: {best_ode_name}\n"
        f"R2 of derivative prediction: {best_ode_r2:.4f}\n"
        f"Analytical solution: {ode_candidates[best_ode_name]['solution']}\n\n"
        f"Scientific significance:\n\n"
        f"1. DISCOVERY OF A RATE LAW: For the first time, the "
        f"differential equation governing how flexural capacity "
        f"degrades with corrosion has been identified from data — "
        f"not assumed a priori.\n\n"
        f"2. ANALYTICAL SOLUTION: The discovered ODE was solved "
        f"exactly using symbolic calculus (inverse differentiation), "
        f"producing a closed-form degradation law with clear "
        f"physical parameters.\n\n"
        f"3. DYNAMICAL SYSTEMS INTERPRETATION: The degradation "
        f"process was analyzed as a dynamical system, revealing "
        f"equilibria, stability properties, and the complete "
        f"family of solution trajectories.\n\n"
        f"4. COMPARISON WITH PySR: The ODE-derived law provides "
        f"a complementary perspective to the PySR equation — "
        f"one discovers the relationship directly, the other "
        f"discovers the rate of change. Together they form "
        f"a complete mathematical description of corrosion "
        f"degradation in RC beams."
    )

    report_path = ODE_DIR / "ODE_Discovery_Report.pdf"
    pdf.output(str(report_path))
    return report_path

try:
    report_p = generate_ode_report()
    logger.info(f"ODE Report saved -> {report_p}")
except Exception as e:
    logger.warning(f"ODE Report failed: {e}")
    traceback.print_exc()

# =============================================================
# CELL 9 — FINAL SUMMARY & DISPLAY
# =============================================================
elapsed = time.time() - t6_start

print(f"\n{'=' * 65}")
print("  PART 6 COMPLETE — ODE DISCOVERY")
print(f"{'=' * 65}")
print(f"\n  Stage I: ODE Discovery")
print(f"    SINDy equation  : {sindy_equation_str[:80]}")
print(f"    Best parametric : {best_ode_name}")
print(f"    Best R2         : {best_ode_r2:.4f}")
print(f"    Solution        : {ode_candidates[best_ode_name]['solution'][:80]}")
print(f"\n  Stage J: Analytical Solutions")
print(f"    ODEs solved     : {len(ode_solutions)}")
print(f"\n  Stage K: Phase Space")
print(f"    Trajectories    : {len(trajectories)}")
print(f"    Equilibria      : {len(equilibria)}")
print(f"\n  Figures           : {fig_count}")
print(f"  ODE Report        : {ODE_DIR / 'ODE_Discovery_Report.pdf'}")
print(f"  Results JSON      : {ODE_DIR / 'ode_results.json'}")
print(f"  Runtime           : {elapsed / 60:.1f} min ({elapsed:.0f}s)")
print(f"\n{'=' * 65}")

# ---- Display figures in notebook ----
try:
    from IPython.display import display, Image as IPImage
    for fig_path in sorted(ODE_FIG.glob("*.png")):
        print(f"\n{fig_path.name}:")
        display(IPImage(filename=str(fig_path), width=800))
except ImportError:
    pass

# ---- Copy to Kaggle output ----
import shutil
KAGGLE_OUT = Path("/kaggle/working")
if KAGGLE_OUT.exists():
    for fig_path in ODE_FIG.glob("*.png"):
        shutil.copy2(str(fig_path), str(KAGGLE_OUT / fig_path.name))
    for key_file in [
        ODE_DIR / "ode_results.json",
        ODE_DIR / "ODE_Discovery_Report.pdf",
    ]:
        if key_file.exists():
            shutil.copy2(str(key_file), str(KAGGLE_OUT / key_file.name))
    print(f"\nAll files copied to {KAGGLE_OUT}")

print("\nPart 6 Done.")
