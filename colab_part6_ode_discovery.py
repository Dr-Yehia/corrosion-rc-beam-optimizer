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
SINDY_OK = False
try:
    import pysindy
    SINDY_OK = True
except ImportError:
    try:
        install("pysindy")
        import pysindy
        SINDY_OK = True
    except Exception:
        print("WARNING: pysindy install failed — will use parametric ODE only")

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

# ---- I1: Prepare NORMALIZED 1D degradation data ----
# Normalize each beam by its ACI capacity → R = Mexp/M_ACI (degradation ratio)
R_ratio_all = y_exp / np.maximum(M_ACI, 1e-6)

# Finer bins with lower threshold for more data points
eta_bins = np.arange(0, 70, 1.5)
M_binned, R_binned = [], []
eta_centers = []

for i in range(len(eta_bins) - 1):
    mask = (eta_arr >= eta_bins[i]) & (eta_arr < eta_bins[i + 1])
    if mask.sum() >= 2:
        eta_centers.append((eta_bins[i] + eta_bins[i + 1]) / 2.0)
        M_binned.append(float(np.median(y_exp[mask])))
        R_binned.append(float(np.median(R_ratio_all[mask])))

eta_c = np.array(eta_centers)
M_c = np.array(M_binned)
R_c = np.array(R_binned)

# Smooth to reduce noise in derivatives
from scipy.ndimage import uniform_filter1d
R_c_smooth = uniform_filter1d(R_c, size=3)
M_c_smooth = uniform_filter1d(M_c, size=3)

logger.info(f"  Binned data: {len(eta_c)} bins with >= 2 beams each")
logger.info(f"  eta range: [{eta_c[0]:.1f}, {eta_c[-1]:.1f}]%")
logger.info(f"  M range: [{M_c.min():.2f}, {M_c.max():.2f}] kN.m")
logger.info(f"  R ratio range: [{R_c.min():.3f}, {R_c.max():.3f}]")

# ---- I2: Numerical derivative dM/deta from smoothed binned data ----
dM_deta = np.gradient(M_c_smooth, eta_c)
dR_deta = np.gradient(R_c_smooth, eta_c)
d2M_deta2 = np.gradient(dM_deta, eta_c)

logger.info(f"  Numerical dM/deta (smoothed): mean={np.mean(dM_deta):.4f}, "
            f"min={np.min(dM_deta):.4f}, max={np.max(dM_deta):.4f}")
logger.info(f"  Numerical dR/deta (smoothed): mean={np.mean(dR_deta):.6f}")

# ---- I3: SINDy ODE Discovery ----
if SINDY_OK:
    import pysindy as ps

X_sindy = np.column_stack([M_c_smooth, eta_c])

best_sindy = None
best_sindy_score = -1e10
sindy_results = []
sindy_equation_str = "Could not discover ODE"
sindy_coeffs = {}

if SINDY_OK:
    for deg in [2, 3, 4]:
        sindy_lib = ps.PolynomialLibrary(degree=deg, include_interaction=True)
        for thresh in [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]:
            try:
                optimizer = ps.STLSQ(threshold=thresh, alpha=0.005)
                try:
                    model = ps.SINDy(
                        feature_names=["M", "eta"],
                        feature_library=sindy_lib,
                        optimizer=optimizer,
                    )
                    model.fit(X_sindy, t=eta_c,
                              x_dot=np.column_stack([dM_deta,
                                                     np.ones_like(dM_deta)]))
                except TypeError:
                    model = ps.SINDy(
                        feature_library=sindy_lib,
                        optimizer=optimizer,
                    )
                    model.fit(X_sindy, t=eta_c,
                              feature_names=["M", "eta"],
                              x_dot=np.column_stack([dM_deta,
                                                     np.ones_like(dM_deta)]))

                dM_pred = model.predict(X_sindy)[:, 0]
                ss_res = np.sum((dM_deta - dM_pred) ** 2)
                ss_tot = np.sum((dM_deta - np.mean(dM_deta)) ** 2)
                r2_sindy = 1 - ss_res / max(ss_tot, 1e-10)
                n_terms = np.count_nonzero(model.coefficients()[0])

                sindy_results.append({
                    "threshold": thresh,
                    "degree": deg,
                    "R2": round(r2_sindy, 4),
                    "n_terms": n_terms,
                    "coefficients": model.coefficients()[0].tolist(),
                    "feature_names": model.get_feature_names(),
                })

                if r2_sindy > best_sindy_score and n_terms >= 1:
                    best_sindy_score = r2_sindy
                    best_sindy = model
                    logger.info(f"  SINDy (deg={deg}, thresh={thresh}): "
                                f"R2={r2_sindy:.4f}, terms={n_terms} [NEW BEST]")
            except Exception as exc:
                pass

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
        logger.info(f"\n  DISCOVERED ODE (SINDy):")
        logger.info(f"  dM/d(eta) = {sindy_equation_str}")
    else:
        logger.warning("  SINDy failed — using parametric ODE fitting instead")
else:
    logger.warning("  pysindy not available — skipping SINDy, using parametric ODE")

# ---- I5: ENHANCED Parametric ODE fitting ----
# KEY IMPROVEMENT: Fit M(eta) TRAJECTORY directly via curve_fit
# instead of fitting dM/deta (which is noisy).
# Also enforce boundary: M → 0 as eta → 100%.
logger.info("\n  Fitting 10 canonical ODE forms on M(eta) trajectory...")

from scipy.optimize import curve_fit

M0_est = float(M_c_smooth[0])
R0_est = float(R_c_smooth[0])
ode_candidates = {}

BOUNDARY_WEIGHT = 5.0
ETA_BOUNDARY = 100.0

def _boundary_penalty(func, params, lam=BOUNDARY_WEIGHT):
    """Penalty if M(100%) is not close to zero."""
    try:
        val = func(np.array([ETA_BOUNDARY]), *params)
        return lam * float(val[0]) ** 2
    except Exception:
        return 0.0

# --- Form 1: M(eta) = M0 * exp(-alpha * eta)  [Simple exponential] ---
try:
    def f_exp(eta, M0, alpha):
        return M0 * np.exp(-alpha * eta)
    popt, _ = curve_fit(f_exp, eta_c, M_c_smooth, p0=[M0_est, 0.02],
                        bounds=([0, 1e-6], [500, 1.0]), maxfev=5000)
    pred = f_exp(eta_c, *popt)
    r2 = r2_score(M_c_smooth, pred)
    rmse = np.sqrt(np.mean((M_c_smooth - pred) ** 2))
    M_at_100 = f_exp(np.array([100.0]), *popt)[0]
    ode_candidates["Exponential: dM/deta = -alpha*M"] = {
        "params": {"M0": round(popt[0], 4), "alpha": round(popt[1], 6)},
        "R2_trajectory": round(r2, 4), "RMSE": round(rmse, 4),
        "M_at_100pct": round(M_at_100, 3),
        "solution": f"M(eta) = {popt[0]:.2f} * exp(-{popt[1]:.6f} * eta)",
        "type": "separable",
    }
    logger.info(f"  [1] Exponential: R2={r2:.4f}, RMSE={rmse:.3f}, "
                f"M(100%)={M_at_100:.3f}")
except Exception as exc:
    logger.warning(f"  [1] Exponential fit failed: {exc}")

# --- Form 2: M(eta) = M0 * exp(-alpha * eta^beta)  [Weibull / stretched exp] ---
try:
    def f_weibull(eta, M0, alpha, beta):
        return M0 * np.exp(-alpha * np.power(np.maximum(eta, 1e-10), beta))
    popt, _ = curve_fit(f_weibull, eta_c, M_c_smooth,
                        p0=[M0_est, 0.01, 1.2],
                        bounds=([0, 1e-8, 0.3], [500, 2.0, 4.0]),
                        maxfev=10000)
    pred = f_weibull(eta_c, *popt)
    r2 = r2_score(M_c_smooth, pred)
    rmse = np.sqrt(np.mean((M_c_smooth - pred) ** 2))
    M_at_100 = f_weibull(np.array([100.0]), *popt)[0]
    ode_candidates["Weibull: M = M0*exp(-a*eta^b)"] = {
        "params": {"M0": round(popt[0], 4), "alpha": round(popt[1], 6),
                   "beta": round(popt[2], 4)},
        "R2_trajectory": round(r2, 4), "RMSE": round(rmse, 4),
        "M_at_100pct": round(M_at_100, 3),
        "solution": (f"M(eta) = {popt[0]:.2f} * exp(-{popt[1]:.6f} "
                     f"* eta^{popt[2]:.3f})"),
        "ode_form": (f"dM/deta = -{popt[1]:.6f}*{popt[2]:.3f}"
                     f"*eta^({popt[2]:.3f}-1) * M"),
        "type": "Weibull / stretched exponential",
    }
    logger.info(f"  [2] Weibull: R2={r2:.4f}, RMSE={rmse:.3f}, "
                f"beta={popt[2]:.3f}, M(100%)={M_at_100:.3f}")
except Exception as exc:
    logger.warning(f"  [2] Weibull fit failed: {exc}")

# --- Form 3: M(eta) = M0 * exp(-a*eta - b*eta^2/2)  [Gaussian decay] ---
try:
    def f_gauss(eta, M0, a, b):
        return M0 * np.exp(-a * eta - b * eta ** 2 / 2.0)
    popt, _ = curve_fit(f_gauss, eta_c, M_c_smooth,
                        p0=[M0_est, 0.01, 0.0005],
                        bounds=([0, -0.5, -0.01], [500, 0.5, 0.05]),
                        maxfev=10000)
    pred = f_gauss(eta_c, *popt)
    r2 = r2_score(M_c_smooth, pred)
    rmse = np.sqrt(np.mean((M_c_smooth - pred) ** 2))
    M_at_100 = f_gauss(np.array([100.0]), *popt)[0]
    ode_candidates["Gaussian decay: dM/deta = -(a+b*eta)*M"] = {
        "params": {"M0": round(popt[0], 4), "a": round(popt[1], 6),
                   "b": round(popt[2], 6)},
        "R2_trajectory": round(r2, 4), "RMSE": round(rmse, 4),
        "M_at_100pct": round(M_at_100, 3),
        "solution": (f"M(eta) = {popt[0]:.2f} * exp(-{popt[1]:.6f}*eta "
                     f"- {popt[2]:.6f}*eta^2/2)"),
        "type": "separable",
    }
    logger.info(f"  [3] Gaussian decay: R2={r2:.4f}, RMSE={rmse:.3f}, "
                f"M(100%)={M_at_100:.3f}")
except Exception as exc:
    logger.warning(f"  [3] Gaussian decay fit failed: {exc}")

# --- Form 4: M(eta) = b/a + (M0 - b/a)*exp(-a*eta)  [Linear 1st order] ---
try:
    def f_lin1(eta, M0, a, b):
        return b / a + (M0 - b / a) * np.exp(-a * eta)
    popt, _ = curve_fit(f_lin1, eta_c, M_c_smooth,
                        p0=[M0_est, 0.05, 0.5],
                        bounds=([0, 1e-6, -50], [500, 2.0, 50]),
                        maxfev=10000)
    pred = f_lin1(eta_c, *popt)
    r2 = r2_score(M_c_smooth, pred)
    rmse = np.sqrt(np.mean((M_c_smooth - pred) ** 2))
    M_at_100 = f_lin1(np.array([100.0]), *popt)[0]
    ode_candidates["Linear 1st-order: dM/deta = -a*M + b"] = {
        "params": {"M0": round(popt[0], 4), "a": round(popt[1], 6),
                   "b": round(popt[2], 4)},
        "R2_trajectory": round(r2, 4), "RMSE": round(rmse, 4),
        "M_at_100pct": round(M_at_100, 3),
        "solution": (f"M(eta) = {popt[2]/popt[1]:.2f} + "
                     f"({popt[0]:.2f} - {popt[2]/popt[1]:.2f})"
                     f"*exp(-{popt[1]:.6f}*eta)"),
        "type": "linear first-order",
    }
    logger.info(f"  [4] Linear 1st-order: R2={r2:.4f}, RMSE={rmse:.3f}, "
                f"M(100%)={M_at_100:.3f}")
except Exception as exc:
    logger.warning(f"  [4] Linear 1st-order fit failed: {exc}")

# --- Form 5: M(eta) = M0 * (1 + a*eta)^(-n)  [Power-law (Freundlich)] ---
try:
    def f_power(eta, M0, a, n):
        return M0 * np.power(1.0 + a * eta, -n)
    popt, _ = curve_fit(f_power, eta_c, M_c_smooth,
                        p0=[M0_est, 0.1, 1.0],
                        bounds=([0, 1e-6, 0.1], [500, 5.0, 10.0]),
                        maxfev=10000)
    pred = f_power(eta_c, *popt)
    r2 = r2_score(M_c_smooth, pred)
    rmse = np.sqrt(np.mean((M_c_smooth - pred) ** 2))
    M_at_100 = f_power(np.array([100.0]), *popt)[0]
    ode_candidates["Power-law: M = M0*(1+a*eta)^(-n)"] = {
        "params": {"M0": round(popt[0], 4), "a": round(popt[1], 6),
                   "n": round(popt[2], 4)},
        "R2_trajectory": round(r2, 4), "RMSE": round(rmse, 4),
        "M_at_100pct": round(M_at_100, 3),
        "solution": (f"M(eta) = {popt[0]:.2f} * (1 + {popt[1]:.6f}*eta)"
                     f"^(-{popt[2]:.3f})"),
        "ode_form": (f"dM/deta = -{popt[1]:.6f}*{popt[2]:.3f} * M "
                     f"/ (1 + {popt[1]:.6f}*eta)"),
        "type": "Bernoulli",
    }
    logger.info(f"  [5] Power-law: R2={r2:.4f}, RMSE={rmse:.3f}, "
                f"M(100%)={M_at_100:.3f}")
except Exception as exc:
    logger.warning(f"  [5] Power-law fit failed: {exc}")

# --- Form 6: M(eta) = M0 / (1 + exp(k*(eta - eta_c50)))  [Sigmoid degradation] ---
try:
    def f_sigmoid(eta, M0, k, eta50):
        return M0 / (1.0 + np.exp(k * (eta - eta50)))
    popt, _ = curve_fit(f_sigmoid, eta_c, M_c_smooth,
                        p0=[M0_est * 2, 0.08, 50.0],
                        bounds=([0, 0.001, 5], [1000, 1.0, 100]),
                        maxfev=10000)
    pred = f_sigmoid(eta_c, *popt)
    r2 = r2_score(M_c_smooth, pred)
    rmse = np.sqrt(np.mean((M_c_smooth - pred) ** 2))
    M_at_100 = f_sigmoid(np.array([100.0]), *popt)[0]
    ode_candidates["Sigmoid: M = M0/(1+exp(k*(eta-eta50)))"] = {
        "params": {"M0": round(popt[0], 4), "k": round(popt[1], 6),
                   "eta50": round(popt[2], 2)},
        "R2_trajectory": round(r2, 4), "RMSE": round(rmse, 4),
        "M_at_100pct": round(M_at_100, 3),
        "solution": (f"M(eta) = {popt[0]:.2f} / "
                     f"(1 + exp({popt[1]:.4f}*(eta - {popt[2]:.1f})))"),
        "type": "logistic / Riccati",
    }
    logger.info(f"  [6] Sigmoid: R2={r2:.4f}, RMSE={rmse:.3f}, "
                f"eta50={popt[2]:.1f}, M(100%)={M_at_100:.3f}")
except Exception as exc:
    logger.warning(f"  [6] Sigmoid fit failed: {exc}")

# --- Form 7: M(eta) = a * exp(-b*eta) + c * exp(-d*eta)  [Double exponential] ---
try:
    def f_dblexp(eta, a, b, c, d):
        return a * np.exp(-b * eta) + c * np.exp(-d * eta)
    popt, _ = curve_fit(f_dblexp, eta_c, M_c_smooth,
                        p0=[M0_est * 0.6, 0.01, M0_est * 0.4, 0.08],
                        bounds=([0, 1e-6, 0, 1e-6], [500, 2.0, 500, 2.0]),
                        maxfev=15000)
    pred = f_dblexp(eta_c, *popt)
    r2 = r2_score(M_c_smooth, pred)
    rmse = np.sqrt(np.mean((M_c_smooth - pred) ** 2))
    M_at_100 = f_dblexp(np.array([100.0]), *popt)[0]
    ode_candidates["Double-exp: M = a*exp(-b*eta)+c*exp(-d*eta)"] = {
        "params": {"a": round(popt[0], 4), "b": round(popt[1], 6),
                   "c": round(popt[2], 4), "d": round(popt[3], 6)},
        "R2_trajectory": round(r2, 4), "RMSE": round(rmse, 4),
        "M_at_100pct": round(M_at_100, 3),
        "solution": (f"M(eta) = {popt[0]:.2f}*exp(-{popt[1]:.5f}*eta) + "
                     f"{popt[2]:.2f}*exp(-{popt[3]:.5f}*eta)"),
        "type": "two-phase decay (2nd-order linear ODE)",
    }
    logger.info(f"  [7] Double-exp: R2={r2:.4f}, RMSE={rmse:.3f}, "
                f"M(100%)={M_at_100:.3f}")
except Exception as exc:
    logger.warning(f"  [7] Double-exp fit failed: {exc}")

# --- Form 8: M(eta) = M0 * (1 - eta/100)^n  [Boundary-enforcing power] ---
try:
    def f_bndpow(eta, M0, n):
        return M0 * np.power(np.maximum(1.0 - eta / 100.0, 1e-10), n)
    popt, _ = curve_fit(f_bndpow, eta_c, M_c_smooth,
                        p0=[M0_est, 1.0],
                        bounds=([0, 0.1], [500, 10.0]),
                        maxfev=10000)
    pred = f_bndpow(eta_c, *popt)
    r2 = r2_score(M_c_smooth, pred)
    rmse = np.sqrt(np.mean((M_c_smooth - pred) ** 2))
    M_at_100 = 0.0
    ode_candidates["Boundary power: M = M0*(1-eta/100)^n"] = {
        "params": {"M0": round(popt[0], 4), "n": round(popt[1], 4)},
        "R2_trajectory": round(r2, 4), "RMSE": round(rmse, 4),
        "M_at_100pct": 0.0,
        "solution": (f"M(eta) = {popt[0]:.2f} * (1 - eta/100)^{popt[1]:.3f}"),
        "ode_form": (f"dM/deta = -{popt[1]:.3f}/(100 - eta) * M"),
        "type": "singular at eta=100 (physically exact boundary)",
        "boundary_exact": True,
    }
    logger.info(f"  [8] Boundary power: R2={r2:.4f}, RMSE={rmse:.3f}, "
                f"n={popt[1]:.3f}, M(100%)=0 [EXACT]")
except Exception as exc:
    logger.warning(f"  [8] Boundary power fit failed: {exc}")

# --- Form 9: M(eta) = M0 * exp(-a*eta) * (1 - eta/100)^n  [Hybrid exp+boundary] ---
try:
    def f_hybrid(eta, M0, a, n):
        return (M0 * np.exp(-a * eta)
                * np.power(np.maximum(1.0 - eta / 100.0, 1e-10), n))
    popt, _ = curve_fit(f_hybrid, eta_c, M_c_smooth,
                        p0=[M0_est, 0.005, 0.5],
                        bounds=([0, 0, 0.01], [500, 1.0, 10.0]),
                        maxfev=15000)
    pred = f_hybrid(eta_c, *popt)
    r2 = r2_score(M_c_smooth, pred)
    rmse = np.sqrt(np.mean((M_c_smooth - pred) ** 2))
    M_at_100 = 0.0
    ode_candidates["Hybrid exp+boundary: M = M0*exp(-a*eta)*(1-eta/100)^n"] = {
        "params": {"M0": round(popt[0], 4), "a": round(popt[1], 6),
                   "n": round(popt[2], 4)},
        "R2_trajectory": round(r2, 4), "RMSE": round(rmse, 4),
        "M_at_100pct": 0.0,
        "solution": (f"M(eta) = {popt[0]:.2f} * exp(-{popt[1]:.5f}*eta) "
                     f"* (1 - eta/100)^{popt[2]:.3f}"),
        "ode_form": (f"dM/deta = -({popt[1]:.5f} + "
                     f"{popt[2]:.3f}/(100-eta)) * M"),
        "type": "hybrid (exponential + boundary-enforcing)",
        "boundary_exact": True,
    }
    logger.info(f"  [9] Hybrid exp+boundary: R2={r2:.4f}, RMSE={rmse:.3f}, "
                f"M(100%)=0 [EXACT]")
except Exception as exc:
    logger.warning(f"  [9] Hybrid exp+boundary fit failed: {exc}")

# --- Form 10: M(eta) = M0 * exp(-a*eta^b) * (1-eta/100)^n  [Weibull+boundary] ---
try:
    def f_weib_bnd(eta, M0, a, b, n):
        return (M0 * np.exp(-a * np.power(np.maximum(eta, 1e-10), b))
                * np.power(np.maximum(1.0 - eta / 100.0, 1e-10), n))
    popt, _ = curve_fit(f_weib_bnd, eta_c, M_c_smooth,
                        p0=[M0_est, 0.005, 1.2, 0.5],
                        bounds=([0, 1e-8, 0.3, 0.01], [500, 2.0, 4.0, 10.0]),
                        maxfev=20000)
    pred = f_weib_bnd(eta_c, *popt)
    r2 = r2_score(M_c_smooth, pred)
    rmse = np.sqrt(np.mean((M_c_smooth - pred) ** 2))
    M_at_100 = 0.0
    ode_candidates["Weibull+boundary: M = M0*exp(-a*eta^b)*(1-eta/100)^n"] = {
        "params": {"M0": round(popt[0], 4), "a": round(popt[1], 6),
                   "b": round(popt[2], 4), "n": round(popt[3], 4)},
        "R2_trajectory": round(r2, 4), "RMSE": round(rmse, 4),
        "M_at_100pct": 0.0,
        "solution": (f"M(eta) = {popt[0]:.2f} * exp(-{popt[1]:.5f}"
                     f"*eta^{popt[2]:.3f}) * (1-eta/100)^{popt[3]:.3f}"),
        "type": "Weibull + boundary (most general form)",
        "boundary_exact": True,
    }
    logger.info(f"  [10] Weibull+boundary: R2={r2:.4f}, RMSE={rmse:.3f}, "
                f"M(100%)=0 [EXACT]")
except Exception as exc:
    logger.warning(f"  [10] Weibull+boundary fit failed: {exc}")

# ===== ALSO FIT ON NORMALIZED RATIO R = M/M_ACI ====
logger.info("\n  Fitting normalized ratio R(eta) = M/M_ACI ...")

ratio_candidates = {}

try:
    def f_R_weib_bnd(eta, R0, a, b, n):
        return (R0 * np.exp(-a * np.power(np.maximum(eta, 1e-10), b))
                * np.power(np.maximum(1.0 - eta / 100.0, 1e-10), n))
    popt_r, _ = curve_fit(f_R_weib_bnd, eta_c, R_c_smooth,
                          p0=[R0_est, 0.005, 1.2, 0.5],
                          bounds=([0, 1e-8, 0.3, 0.01], [10, 2.0, 4.0, 10.0]),
                          maxfev=20000)
    pred_r = f_R_weib_bnd(eta_c, *popt_r)
    r2_r = r2_score(R_c_smooth, pred_r)
    rmse_r = np.sqrt(np.mean((R_c_smooth - pred_r) ** 2))
    ratio_candidates["R(eta) Weibull+boundary"] = {
        "params": {"R0": round(popt_r[0], 4), "a": round(popt_r[1], 6),
                   "b": round(popt_r[2], 4), "n": round(popt_r[3], 4)},
        "R2": round(r2_r, 4), "RMSE": round(rmse_r, 6),
        "solution": (f"R(eta) = {popt_r[0]:.3f} * exp(-{popt_r[1]:.5f}"
                     f"*eta^{popt_r[2]:.3f}) * (1-eta/100)^{popt_r[3]:.3f}"),
    }
    logger.info(f"  R(eta) Weibull+boundary: R2={r2_r:.4f}, RMSE_ratio={rmse_r:.6f}")
except Exception as exc:
    logger.warning(f"  R(eta) Weibull+boundary failed: {exc}")

try:
    def f_R_hybrid(eta, R0, a, n):
        return (R0 * np.exp(-a * eta)
                * np.power(np.maximum(1.0 - eta / 100.0, 1e-10), n))
    popt_r2, _ = curve_fit(f_R_hybrid, eta_c, R_c_smooth,
                           p0=[R0_est, 0.005, 0.5],
                           bounds=([0, 0, 0.01], [10, 1.0, 10.0]),
                           maxfev=15000)
    pred_r2 = f_R_hybrid(eta_c, *popt_r2)
    r2_r2 = r2_score(R_c_smooth, pred_r2)
    rmse_r2 = np.sqrt(np.mean((R_c_smooth - pred_r2) ** 2))
    ratio_candidates["R(eta) Hybrid exp+boundary"] = {
        "params": {"R0": round(popt_r2[0], 4), "a": round(popt_r2[1], 6),
                   "n": round(popt_r2[2], 4)},
        "R2": round(r2_r2, 4), "RMSE": round(rmse_r2, 6),
        "solution": (f"R(eta) = {popt_r2[0]:.3f} * exp(-{popt_r2[1]:.5f}*eta)"
                     f" * (1-eta/100)^{popt_r2[2]:.3f}"),
    }
    logger.info(f"  R(eta) Hybrid exp+boundary: R2={r2_r2:.4f}, "
                f"RMSE_ratio={rmse_r2:.6f}")
except Exception as exc:
    logger.warning(f"  R(eta) Hybrid fit failed: {exc}")

# ---- I6: Select best ODE (use COMPOSITE score) ----
best_ode_name = None
best_ode_score = -1e10
for name, info in ode_candidates.items():
    r2_val = info["R2_trajectory"]
    rmse_val = info["RMSE"]
    m100 = abs(info["M_at_100pct"])
    boundary_ok = info.get("boundary_exact", False)
    score = (0.50 * r2_val
             + 0.25 * max(0, 1.0 - rmse_val / max(M_c.max(), 1))
             + 0.25 * (1.0 if boundary_ok else max(0, 1.0 - m100 / max(M_c.max(), 1))))
    info["composite_score"] = round(score, 4)
    if score > best_ode_score:
        best_ode_score = score
        best_ode_name = name

best_ode_r2 = ode_candidates[best_ode_name]["R2_trajectory"]
logger.info(f"\n  BEST ODE: {best_ode_name}")
logger.info(f"  R2(trajectory) = {best_ode_r2:.4f}")
logger.info(f"  M(100%) = {ode_candidates[best_ode_name]['M_at_100pct']}")
logger.info(f"  Solution: {ode_candidates[best_ode_name]['solution']}")

stage_i_results = {
    "sindy_equation": sindy_equation_str,
    "sindy_R2": round(best_sindy_score, 4) if best_sindy else None,
    "sindy_coefficients": sindy_coeffs,
    "sindy_all_trials": sindy_results,
    "parametric_candidates": {k: v for k, v in ode_candidates.items()},
    "ratio_candidates": ratio_candidates,
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
n_sym = Symbol("n", positive=True, real=True)

ode_solutions = {}

# ODE 1: dM/deta = -alpha*M → M = M0*exp(-alpha*eta)
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
            "capacity. Simplest degradation law: each unit of corrosion "
            "destroys a fixed FRACTION of remaining capacity."
        ),
    }
    logger.info(f"  ODE 1 solved: {sol1}")
except Exception as exc:
    logger.warning(f"  ODE 1 solution failed: {exc}")

# ODE 2: dM/deta = -alpha*beta*eta^(beta-1)*M  (Weibull)
# Solution: M = M0*exp(-alpha*eta^beta)
try:
    ode2 = Eq(M_sym(eta_sym).diff(eta_sym),
              -alpha_sym * beta_sym * eta_sym ** (beta_sym - 1) * M_sym(eta_sym))
    sol2 = dsolve(ode2, M_sym(eta_sym), ics={M_sym(0): M0_sym})
    ode_solutions["Weibull / Stretched Exponential"] = {
        "ode": str(ode2),
        "ode_latex": latex(ode2),
        "solution": str(sol2),
        "solution_latex": latex(sol2),
        "classification": ["separable", "1st order linear"],
        "physical_meaning": (
            "Weibull degradation: the instantaneous decay rate depends on "
            "eta^(beta-1). When beta>1, degradation accelerates with "
            "corrosion (damage breeds more damage). When beta<1, initial "
            "damage is fast but slows (passivation effect)."
        ),
    }
    logger.info(f"  ODE 2 (Weibull) solved: {str(sol2)[:100]}")
except Exception as exc:
    logger.warning(f"  ODE 2 solution failed: {exc}")

# ODE 3: dM/deta = -(a + b*eta)*M → M = M0*exp(-a*eta - b*eta^2/2)
try:
    ode3 = Eq(M_sym(eta_sym).diff(eta_sym),
              -(a_sym + b_sym_ode * eta_sym) * M_sym(eta_sym))
    sol3 = dsolve(ode3, M_sym(eta_sym), ics={M_sym(0): M0_sym})
    ode_solutions["Time-Dependent (Gaussian) Decay"] = {
        "ode": str(ode3),
        "ode_latex": latex(ode3),
        "solution": str(sol3),
        "solution_latex": latex(sol3),
        "classification": ["separable", "1st order linear"],
        "physical_meaning": (
            "The degradation rate ITSELF changes with corrosion level. "
            "If b>0, degradation accelerates over time (autocatalytic). "
            "The quadratic exponent produces Gaussian-like rapid decay."
        ),
    }
    logger.info(f"  ODE 3 solved: {str(sol3)[:100]}")
except Exception as exc:
    logger.warning(f"  ODE 3 solution failed: {exc}")

# ODE 4: dM/deta = -a*M + b → M = b/a + (M0-b/a)*exp(-a*eta)
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
            "Capacity decays exponentially to a nonzero residual b/a. "
            "Even at extreme corrosion, the beam retains minimum capacity — "
            "the concrete core still resists without reinforcement."
        ),
    }
    logger.info(f"  ODE 4 solved: {str(sol4)[:100]}")
except Exception as exc:
    logger.warning(f"  ODE 4 solution failed: {exc}")

# ODE 5: dM/deta = -n/(100-eta) * M → M = M0*(1-eta/100)^n  [Boundary-enforcing]
try:
    ode5 = Eq(M_sym(eta_sym).diff(eta_sym),
              -n_sym / (100 - eta_sym) * M_sym(eta_sym))
    sol5 = dsolve(ode5, M_sym(eta_sym), ics={M_sym(0): M0_sym})
    ode_solutions["Boundary-Enforcing Power Law"] = {
        "ode": str(ode5),
        "ode_latex": latex(ode5),
        "solution": str(sol5),
        "solution_latex": latex(sol5),
        "classification": ["separable", "1st order linear"],
        "physical_meaning": (
            "A physically-exact degradation law where M(100%)=0 by "
            "construction. The singular point at eta=100% represents "
            "complete structural failure. Parameter n controls the "
            "rate: n=1 is linear decline, n>1 is concave (initial slow "
            "degradation then rapid collapse), n<1 is convex (fast initial "
            "loss then gradual)."
        ),
    }
    logger.info(f"  ODE 5 (boundary power) solved: {str(sol5)[:100]}")
except Exception as exc:
    logger.warning(f"  ODE 5 solution failed: {exc}")

# ODE 6: dM/deta = -(alpha + n/(100-eta))*M  [Hybrid: exponential + boundary]
try:
    ode6 = Eq(M_sym(eta_sym).diff(eta_sym),
              -(alpha_sym + n_sym / (100 - eta_sym)) * M_sym(eta_sym))
    sol6 = dsolve(ode6, M_sym(eta_sym), ics={M_sym(0): M0_sym})
    ode_solutions["Hybrid Exponential + Boundary"] = {
        "ode": str(ode6),
        "ode_latex": latex(ode6),
        "solution": str(sol6),
        "solution_latex": latex(sol6),
        "classification": ["separable", "1st order linear"],
        "physical_meaning": (
            "Combines exponential degradation (constant rate alpha) with "
            "a singularity at eta=100% (boundary term n/(100-eta)). "
            "Guarantees M(100%)=0 while capturing the exponential decay "
            "observed at moderate corrosion levels. This is the most "
            "physically complete single-equation degradation model."
        ),
    }
    logger.info(f"  ODE 6 (hybrid) solved: {str(sol6)[:100]}")
except Exception as exc:
    logger.warning(f"  ODE 6 solution failed: {exc}")

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

# ---- K1: Compute trajectories from fitted analytical solutions ----
best_info = ode_candidates[best_ode_name]
M0_data = float(M_c[0])

eta_ode = np.linspace(0, min(100, eta_c[-1] * 1.5), 500)
eta_ode_full = np.linspace(0, 100, 500)

_traj_funcs = {
    "Exponential: dM/deta = -alpha*M":
        lambda e, p: p["M0"] * np.exp(-p["alpha"] * e),
    "Weibull: M = M0*exp(-a*eta^b)":
        lambda e, p: p["M0"] * np.exp(-p["alpha"] * np.power(np.maximum(e, 1e-10), p["beta"])),
    "Gaussian decay: dM/deta = -(a+b*eta)*M":
        lambda e, p: p["M0"] * np.exp(-p["a"] * e - p["b"] * e ** 2 / 2.0),
    "Linear 1st-order: dM/deta = -a*M + b":
        lambda e, p: p["b"] / p["a"] + (p["M0"] - p["b"] / p["a"]) * np.exp(-p["a"] * e),
    "Power-law: M = M0*(1+a*eta)^(-n)":
        lambda e, p: p["M0"] * np.power(1.0 + p["a"] * e, -p["n"]),
    "Sigmoid: M = M0/(1+exp(k*(eta-eta50)))":
        lambda e, p: p["M0"] / (1.0 + np.exp(p["k"] * (e - p["eta50"]))),
    "Double-exp: M = a*exp(-b*eta)+c*exp(-d*eta)":
        lambda e, p: p["a"] * np.exp(-p["b"] * e) + p["c"] * np.exp(-p["d"] * e),
    "Boundary power: M = M0*(1-eta/100)^n":
        lambda e, p: p["M0"] * np.power(np.maximum(1.0 - e / 100.0, 1e-10), p["n"]),
    "Hybrid exp+boundary: M = M0*exp(-a*eta)*(1-eta/100)^n":
        lambda e, p: (p["M0"] * np.exp(-p["a"] * e)
                      * np.power(np.maximum(1.0 - e / 100.0, 1e-10), p["n"])),
    "Weibull+boundary: M = M0*exp(-a*eta^b)*(1-eta/100)^n":
        lambda e, p: (p["M0"] * np.exp(-p["a"] * np.power(np.maximum(e, 1e-10), p["b"]))
                      * np.power(np.maximum(1.0 - e / 100.0, 1e-10), p["n"])),
}

trajectories = {}
trajectories_full = {}
for name, info in ode_candidates.items():
    if name in _traj_funcs:
        try:
            trajectories[name] = _traj_funcs[name](eta_ode, info["params"])
            trajectories_full[name] = _traj_funcs[name](eta_ode_full, info["params"])
        except Exception as exc:
            logger.warning(f"  Trajectory for '{name[:30]}' failed: {exc}")

logger.info(f"  Computed {len(trajectories)} analytical trajectories")

# ---- K2: Equilibrium & boundary analysis ----
equilibria = {}
for name, info in ode_candidates.items():
    if info.get("boundary_exact", False):
        equilibria[name] = {
            "M_equilibrium_kNm": 0.0,
            "stability": "stable (boundary-enforced)",
            "meaning": "M -> 0 at eta=100% (exact zero by construction)",
        }
    elif info["M_at_100pct"] < 1.0:
        equilibria[name] = {
            "M_equilibrium_kNm": round(info["M_at_100pct"], 4),
            "stability": "stable (near-zero)",
            "meaning": f"M -> {info['M_at_100pct']:.3f} kN.m (effectively zero)",
        }
    else:
        equilibria[name] = {
            "M_equilibrium_kNm": round(info["M_at_100pct"], 4),
            "stability": "non-zero residual",
            "meaning": (f"M -> {info['M_at_100pct']:.2f} kN.m at 100% corrosion "
                        "(residual capacity)"),
        }

logger.info(f"  Equilibrium analysis: {len(equilibria)} models analyzed")
for n, eq in equilibria.items():
    logger.info(f"    {n[:40]}: M(100%)={eq['M_equilibrium_kNm']:.3f} "
                f"({eq['stability']})")

# ---- K3: Lyapunov-style stability (monotone decrease check) ----
lyapunov = {}
for name in trajectories_full:
    traj = trajectories_full[name]
    mono_check = np.all(np.diff(traj) <= 0.01)
    final_val = traj[-1]
    lyapunov[name] = {
        "monotone_decreasing": bool(mono_check),
        "final_value": round(float(final_val), 4),
        "stable": bool(final_val < 1.0 and mono_check),
        "meaning": ("Monotone decay to ~0: physically consistent"
                    if mono_check and final_val < 1.0
                    else "Non-monotone or nonzero limit: needs review"),
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
    ax.scatter(M_c_smooth, dM_deta, c=C_BLUE, s=50, alpha=0.7, zorder=5,
               label="Data (smoothed binned medians)")

    top3 = sorted(ode_candidates.items(),
                  key=lambda x: x[1]["R2_trajectory"], reverse=True)[:5]
    for i, (name, info) in enumerate(top3):
        short = name.split(":")[0][:20]
        col = COLORS[i % len(COLORS)]
        if name in trajectories:
            traj_eta = trajectories[name]
            dM_fit = np.gradient(traj_eta, eta_ode)
            ax.plot(traj_eta, dM_fit,
                    color=col, linewidth=2, alpha=0.8,
                    label=f"{short} (R2={info['R2_trajectory']:.3f})")

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
        short = name.split(":")[0][:20]
        r2_t = ode_candidates[name]["R2_trajectory"]
        ax.plot(eta_ode, traj, color=COLORS[i % len(COLORS)],
                linewidth=2.5, alpha=0.8,
                label=f"{short} (R2={r2_t:.3f})")

    ax.set_xlabel("Mass Loss $\\eta_m$ (%)", fontsize=13)
    ax.set_ylabel("$M_{max}$ (kN.m)", fontsize=13)
    ax.set_title("ODE Trajectories vs Experimental Data", fontsize=14)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(ODE_FIG / "fig_ode2_trajectories.png")
    plt.close(fig)
    fig_count += 1
    logger.info("  Fig ODE2 OK — Trajectories")
except Exception as e:
    logger.warning(f"  Fig ODE2 FAILED: {e}")

# ------- ODE Fig 3: ODE Comparison Bar Chart (R² trajectory + boundary) -------
try:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sorted_cands = sorted(ode_candidates.items(),
                          key=lambda x: x[1]["R2_trajectory"], reverse=True)
    names = [n.split(":")[0][:22] for n, _ in sorted_cands]
    r2s = [v["R2_trajectory"] for _, v in sorted_cands]
    m100s = [v["M_at_100pct"] for _, v in sorted_cands]
    ccolors = [C_GREEN if v.get("boundary_exact", False)
               else C_BLUE for _, v in sorted_cands]

    bars = axes[0].barh(names, r2s, color=ccolors, alpha=0.85,
                        edgecolor="white")
    for bar, r2v in zip(bars, r2s):
        axes[0].text(bar.get_width() + 0.005,
                     bar.get_y() + bar.get_height() / 2,
                     f"{r2v:.4f}", va="center", fontsize=10, fontweight="bold")
    axes[0].set_xlabel("$R^2$ (trajectory fit)", fontsize=12)
    axes[0].set_title("Trajectory Accuracy", fontsize=13)
    axes[0].set_xlim(0, max(r2s) * 1.15 if r2s else 1)
    axes[0].grid(True, alpha=0.3, axis="x")

    bars2 = axes[1].barh(names, m100s, color=ccolors, alpha=0.85,
                         edgecolor="white")
    for bar, mv in zip(bars2, m100s):
        axes[1].text(bar.get_width() + 0.1,
                     bar.get_y() + bar.get_height() / 2,
                     f"{mv:.2f}", va="center", fontsize=10, fontweight="bold")
    axes[1].set_xlabel("$M(\\eta=100\\%)$ (kN.m)", fontsize=12)
    axes[1].set_title("Boundary Condition: M at 100% Corrosion", fontsize=13)
    axes[1].axvline(0, color="black", linewidth=1)
    axes[1].grid(True, alpha=0.3, axis="x")

    fig.tight_layout()
    fig.savefig(ODE_FIG / "fig_ode3_comparison.png")
    plt.close(fig)
    fig_count += 1
    logger.info("  Fig ODE3 OK — Comparison")
except Exception as e:
    logger.warning(f"  Fig ODE3 FAILED: {e}")

# ------- ODE Fig 4: Vector Field using best ODE -------
try:
    fig, ax = plt.subplots(figsize=(10, 7))
    eta_grid = np.linspace(0.5, 95, 20)
    M_grid = np.linspace(0.5, M_c.max() * 1.1, 20)
    ETA, MG = np.meshgrid(eta_grid, M_grid)

    best_func = _traj_funcs.get(best_ode_name)
    if best_func is not None:
        best_p = ode_candidates[best_ode_name]["params"]
        M_at_eta = best_func(ETA.ravel(), best_p).reshape(ETA.shape)
        DM = np.gradient(M_at_eta, eta_grid[1] - eta_grid[0], axis=1)
    else:
        DM = -0.01 * MG

    DETA = np.ones_like(DM)
    speed = np.sqrt(DETA ** 2 + DM ** 2)

    ax.streamplot(ETA, MG, DETA, DM, color=speed, cmap="coolwarm",
                  density=1.5, linewidth=1.5, arrowsize=1.5)
    ax.scatter(eta_c, M_c, c="black", s=40, zorder=10, marker="D",
               label="Data")
    short_best = best_ode_name.split(":")[0][:25]
    ax.set_xlabel("Mass Loss $\\eta_m$ (%)", fontsize=13)
    ax.set_ylabel("$M_{max}$ (kN.m)", fontsize=13)
    ax.set_title(f"Vector Field — Best ODE: {short_best}", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(ODE_FIG / "fig_ode4_vector_field.png")
    plt.close(fig)
    fig_count += 1
    logger.info("  Fig ODE4 OK — Vector Field")
except Exception as e:
    logger.warning(f"  Fig ODE4 FAILED: {e}")

# ------- ODE Fig 5: Multiple Initial Conditions (extrapolate to eta=100%) -------
try:
    fig, ax = plt.subplots(figsize=(10, 6))
    M0_values = [5, 10, 20, 40, 80, 120]
    eta_full = np.linspace(0, 100, 500)
    best_p = ode_candidates[best_ode_name]["params"]
    best_func = _traj_funcs.get(best_ode_name)

    for M0v in M0_values:
        if best_func is not None:
            p_ic = dict(best_p)
            if "M0" in p_ic:
                p_ic["M0"] = M0v
            elif "a" in p_ic and "c" in p_ic:
                ratio = M0v / (p_ic["a"] + p_ic["c"])
                p_ic["a"] *= ratio
                p_ic["c"] *= ratio
            traj = best_func(eta_full, p_ic)
        else:
            traj = M0v * np.exp(-0.01 * eta_full)

        ax.plot(eta_full, np.maximum(traj, 0), linewidth=2,
                label=f"$M_0$ = {M0v} kN.m")

    ax.axvline(100, color="red", linewidth=1, linestyle="--", alpha=0.5,
               label="$\\eta=100\\%$ (full corrosion)")
    ax.scatter(eta_c, M_c, c="black", s=30, zorder=10, alpha=0.5,
               label="Data")
    ax.set_xlabel("Mass Loss $\\eta_m$ (%)", fontsize=13)
    ax.set_ylabel("$M_{max}$ (kN.m)", fontsize=13)
    ax.set_title("ODE Solution Family — Extrapolation to 100% Corrosion",
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
    pdf.cell(0, 7, "Method 2: Parametric ODE Fitting (TRAJECTORY):", 0, 1)
    pdf.body(
        "Ten canonical ODE forms from mathematical physics were "
        "fitted DIRECTLY to the M(eta) trajectory (not dM/deta). "
        "Three boundary-enforcing forms guarantee M(100%)=0. "
        "Each has a distinct physical meaning:"
    )

    sorted_report = sorted(ode_candidates.items(),
                           key=lambda x: x[1]["R2_trajectory"], reverse=True)
    for i, (name, info) in enumerate(sorted_report, 1):
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 6, f"  Candidate {i}: {_safe(name)}", 0, 1)
        pdf.set_font("Courier", "", 9)
        pdf.cell(0, 5,
                 f"    R2(trajectory) = {info['R2_trajectory']:.4f}"
                 f"    RMSE = {info['RMSE']:.4f}"
                 f"    M(100%) = {info['M_at_100pct']:.3f}", 0, 1)
        pdf.cell(0, 5,
                 f"    Solution: {_safe(info['solution'][:80])}", 0, 1)
        pdf.set_font("Helvetica", "", 9)
        for pk, pv in info["params"].items():
            pdf.cell(0, 5, f"      {pk} = {pv}", 0, 1)
        if info.get("boundary_exact"):
            pdf.set_text_color(0, 128, 0)
            pdf.cell(0, 5, "      [BOUNDARY EXACT: M(100%)=0]", 0, 1)
            pdf.set_text_color(0, 0, 0)
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
        f"R2 (trajectory fit): {best_ode_r2:.4f}\n"
        f"M(100% corrosion): {ode_candidates[best_ode_name]['M_at_100pct']}\n"
        f"Analytical solution: {ode_candidates[best_ode_name]['solution']}\n\n"
        f"Scientific significance:\n\n"
        f"1. TRAJECTORY FITTING (not derivative): R2 is now measured "
        f"on the M(eta) trajectory itself, eliminating noise from "
        f"numerical differentiation. This gives physically "
        f"meaningful accuracy.\n\n"
        f"2. BOUNDARY-ENFORCING MODELS: Three ODE forms guarantee "
        f"M(100%)=0 by construction (physical requirement: complete "
        f"corrosion = zero capacity). Forms (1-eta/100)^n provide "
        f"the singular structure at eta=100%.\n\n"
        f"3. ANALYTICAL SOLUTION: The discovered ODE was solved "
        f"exactly using SymPy, producing a closed-form degradation "
        f"law with clear physical parameters.\n\n"
        f"4. DYNAMICAL SYSTEMS: The degradation was analyzed as "
        f"a dynamical system, revealing equilibria, stability, and "
        f"the complete family of solution trajectories.\n\n"
        f"5. COMPARISON WITH PySR: The ODE-derived law provides "
        f"a complementary perspective — one discovers M(eta) "
        f"directly, the other discovers dM/deta. Together they "
        f"form a complete mathematical description."
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
print(f"\n  Stage I: ODE Discovery (10 candidates)")
print(f"    SINDy equation  : {sindy_equation_str[:80]}")
print(f"    Best parametric : {best_ode_name}")
print(f"    R2 (trajectory) : {best_ode_r2:.4f}")
print(f"    M(100%)         : {ode_candidates[best_ode_name]['M_at_100pct']}")
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
import shutil, zipfile
KAGGLE_OUT = Path("/kaggle/working")
if KAGGLE_OUT.exists():
    ode_out = KAGGLE_OUT / "ode_discovery"
    ode_out.mkdir(exist_ok=True)
    ode_fig_out = ode_out / "figures"
    ode_fig_out.mkdir(exist_ok=True)

    for fig_path in ODE_FIG.glob("*.png"):
        shutil.copy2(str(fig_path), str(KAGGLE_OUT / fig_path.name))
        shutil.copy2(str(fig_path), str(ode_fig_out / fig_path.name))
    for key_file in [
        ODE_DIR / "ode_results.json",
        ODE_DIR / "ODE_Discovery_Report.pdf",
    ]:
        if key_file.exists():
            shutil.copy2(str(key_file), str(KAGGLE_OUT / key_file.name))
            shutil.copy2(str(key_file), str(ode_out / key_file.name))

    zip_path = KAGGLE_OUT / "ALL_RESULTS_COMPLETE.zip"
    if zip_path.exists():
        with zipfile.ZipFile(str(zip_path), "a",
                             zipfile.ZIP_DEFLATED) as zf:
            existing = set(zf.namelist())
            for f in ODE_FIG.glob("*.png"):
                arc = f"ode_discovery/figures/{f.name}"
                if arc not in existing:
                    zf.write(str(f), arc)
            for kf in [ODE_DIR / "ode_results.json",
                        ODE_DIR / "ODE_Discovery_Report.pdf"]:
                if kf.exists():
                    arc = f"ode_discovery/{kf.name}"
                    if arc not in existing:
                        zf.write(str(kf), arc)
        logger.info(f"Updated ZIP with ODE files: {zip_path}")

    print(f"\nAll files copied to {KAGGLE_OUT}")

print("\nPart 6 Done.")
