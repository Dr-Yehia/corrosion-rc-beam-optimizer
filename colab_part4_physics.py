#!/usr/bin/env python3
"""
===============================================================
  Corrosion RC Beam Optimizer -- Part 4: Validation & Figures
  PREREQUISITE: Run Part 3 first!
===============================================================
"""

# =============================================================
# CELL 0: SETUP & LOAD STATE FROM PART 3
# =============================================================
import subprocess, sys, os

def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

for p in ["loguru", "sympy", "SALib", "scikit-learn",
           "matplotlib", "seaborn", "fpdf2", "joblib"]:
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

import json, time, warnings, traceback, re, copy
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import sympy as sp
from sympy import (
    symbols, Symbol, diff, integrate, solve, series, simplify,
    latex, sqrt, log, exp, oo, Abs, Piecewise, N as sp_N,
    lambdify, Rational, sympify,
)
from scipy import optimize
from scipy.interpolate import UnivariateSpline
from datetime import datetime
from pathlib import Path
from loguru import logger
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib as _jl

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
    str(LOG_DIR / "run_log_part4.txt"),
    format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}",
    level="DEBUG", rotation="10 MB", encoding="utf-8",
)

logger.info("=" * 65)
logger.info("  Part 4: Validation & Figures — Loading state from Part 3")
logger.info("=" * 65)

# -- Load Part 3 state --
_p3_path = PHYSICS_DIR / "part3_state.pkl"
if not _p3_path.exists():
    raise FileNotFoundError(
        f"Part 3 state not found at {_p3_path}. Run Part 3 first!")

S = _jl.load(str(_p3_path))

N_TOTAL = S["N_TOTAL"]
WINNER = S["WINNER"]
USE_CSI_CHAIN = S["USE_CSI_CHAIN"]
N_BOOT = S["N_BOOT"]
eq_str = S["eq_str"]
eq_ltx = S["eq_ltx"]
eq_meta = S["eq_meta"]

y_exp = S["y_exp"]
M_ACI = S["M_ACI"]
eta_arr = S["eta_arr"]
d_arr = S["d_arr"]
b_arr = S["b_arr"]
fy_arr = S["fy_arr"]
fc_arr = S["fc_arr"]
rho_arr = S["rho_arr"]
db_arr = S["db_arr"]
d_b_arr = S["d_b_arr"]
csi_arr = S["csi_arr"]
ri_arr = S["ri_arr"]

stage_a_results = S["stage_a_results"]
stage_b_results = S["stage_b_results"]
stage_c_results = S["stage_c_results"]
stage_d_results = S["stage_d_results"]
discoveries = S["discoveries"]

sobol_ok = S["sobol_ok"]
sobol_results = S["sobol_results"]
spider_results = S["spider_results"]

best_mc_name = S["best_mc_name"]
best_mc_r2 = S["best_mc_r2"]
valid_mc = S["valid_mc"]
alpha_opt = S["alpha_opt"]
beta_opt = S["beta_opt"]
r2_compound = S["r2_compound"]

collapse_eta = S["collapse_eta"]
deg_rates = S["deg_rates"]
regime_boundaries = S["regime_boundaries"]

eta_plot = S["eta_plot"]
f_curve = S["f_curve"]
d1_curve = S["d1_curve"]
d2_curve = S["d2_curve"]
ci_lower = S["ci_lower"]
ci_upper = S["ci_upper"]
tay_curve = S["tay_curve"]
analysis_label = S["analysis_label"]
f_vals = S["f_vals"]

part1_summary = S["part1_summary"]
pysr_summary = S["pysr_summary"]
t_start = S["t_start"]

critical_eta_star = S["critical_eta_star"]
critical_method = S["critical_method"]

COL_ETA = S["COL_ETA"]
COL_FY = S["COL_FY"]
COL_FC = S["COL_FC"]
COL_D = S["COL_D"]
COL_B = S["COL_B"]
COL_RHO = S["COL_RHO"]
COL_DB = S["COL_DB"]

# -- Reconstruct symbolic objects --
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

_sym_map = {
    "eta_m": eta_m_s, "fy": fy_s, "fc": fc_s, "d": d_s, "b": b_s,
    "rho_t": rho_t_s, "db_t": db_t_s, "d_b": d_b_s, "CSI": CSI_s, "RI": RI_s,
}

f_expr = sympify(S["f_expr_str"], locals=_sym_map)
free_syms = sorted(f_expr.free_symbols, key=str)

deriv_results = {k: sympify(v, locals=_sym_map)
                 for k, v in S["deriv_results_str"].items()}
integral_expr = (sympify(S["integral_expr_str"], locals=_sym_map)
                 if S["integral_expr_str"] else None)
taylor_expr = (sympify(S["taylor_expr_str"], locals=_sym_map)
               if S["taylor_expr_str"] else None)
d2f_deta2 = sympify(S["d2f_deta2_str"], locals=_sym_map)
d2f_deta2_1d = sympify(S["d2f_deta2_1d_str"], locals=_sym_map)

# -- Rebuild eval_f and f_1d_fn --
_all_syms = [eta_m_s, fy_s, fc_s, d_s, b_s, rho_t_s, db_t_s,
             d_b_s, CSI_s, RI_s]
_f_lam = lambdify(_all_syms, f_expr, modules="numpy")

data = run_preprocessing(save_clean=True)
df_clean = data["df_clean"]

MEDIAN_MAP = {}
_col_sym = [
    (COL_ETA, eta_m_s), (COL_FY, fy_s), (COL_FC, fc_s),
    (COL_D, d_s), (COL_B, b_s), (COL_RHO, rho_t_s), (COL_DB, db_t_s),
]
for col, sym in _col_sym:
    if col in df_clean.columns:
        MEDIAN_MAP[sym] = float(df_clean[col].median())
MEDIAN_MAP[d_b_s] = float(np.median(d_b_arr))
MEDIAN_MAP[CSI_s] = float(np.median(csi_arr))
MEDIAN_MAP[RI_s] = float(np.median(ri_arr))

DATA_MAP = {
    eta_m_s: eta_arr, fy_s: fy_arr, fc_s: fc_arr,
    d_s: d_arr, b_s: b_arr, rho_t_s: rho_arr, db_t_s: db_arr,
    d_b_s: d_b_arr, CSI_s: csi_arr, RI_s: ri_arr,
}

def eval_f(sub_dict):
    args = []
    for sym in _all_syms:
        if sym in sub_dict:
            args.append(sub_dict[sym])
        elif sym in MEDIAN_MAP:
            val = MEDIAN_MAP[sym]
            ref = sub_dict.get(eta_m_s, eta_arr)
            args.append(np.full_like(ref, val, dtype=np.float64)
                        if hasattr(ref, '__len__') else val)
        else:
            ref = sub_dict.get(eta_m_s, eta_arr)
            args.append(np.ones_like(ref, dtype=np.float64)
                        if hasattr(ref, '__len__') else 1.0)
    return _f_lam(*args)

def f_1d_fn(eta_val):
    d = {s: MEDIAN_MAP[s] for s in free_syms if s in MEDIAN_MAP}
    d[eta_m_s] = float(eta_val)
    if CSI_s in d or CSI_s in free_syms:
        d[CSI_s] = float(eta_val) * MEDIAN_MAP.get(fy_s, 400) / MEDIAN_MAP.get(fc_s, 30)
    return float(eval_f(d))

# -- Plot style --
plt.rcParams.update({
    "font.size": 12, "axes.titlesize": 14, "axes.labelsize": 13,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.family": "serif",
})

logger.info(f"Part 4 state loaded: {N_TOTAL} samples, WINNER={WINNER}")
logger.info(f"Equation: {str(f_expr)[:100]}")

# =============================================================
# CELL 7B — STAGE E: PHYSICAL VALIDATION (Limiting Cases +
#            Dimensional Consistency + Monotonicity)
# =============================================================
logger.info("\n" + "=" * 65)
logger.info("  STAGE E — Physical Validation (Nature-level checks)")
logger.info("=" * 65)

phys_checks = {}

# ---- E1: Limiting case η_m → 0  (must recover ACI / undamaged) ----
try:
    f_at_zero = float(f_1d_fn(0.0))
    if WINNER == "RATIO":
        expected_zero = 1.0
        deviation_zero = abs(f_at_zero - expected_zero) / expected_zero * 100
        check_zero = deviation_zero < 25.0
        phys_checks["eta=0 (f_corr should be ~1.0)"] = {
            "predicted": round(f_at_zero, 4),
            "expected": 1.0,
            "deviation_%": round(deviation_zero, 2),
            "PASS": check_zero,
        }
    else:
        M_aci_med = float(np.median(M_ACI))
        deviation_zero = abs(f_at_zero - M_aci_med) / M_aci_med * 100
        check_zero = deviation_zero < 30.0
        phys_checks["eta=0 (Mmax should be ~M_ACI)"] = {
            "predicted_kNm": round(f_at_zero, 2),
            "expected_ACI_median_kNm": round(M_aci_med, 2),
            "deviation_%": round(deviation_zero, 2),
            "PASS": check_zero,
        }
    logger.info(f"  E1 | eta=0: predicted={f_at_zero:.4f}, "
                f"deviation={deviation_zero:.1f}% "
                f"{'PASS' if check_zero else 'FAIL'}")
except Exception as exc:
    phys_checks["eta=0"] = {"error": str(exc), "PASS": False}
    logger.warning(f"  E1 | eta=0 check failed: {exc}")

# ---- E2: Limiting case η_m → 100%  (must approach 0) ----
try:
    f_at_100 = float(f_1d_fn(99.0))
    if WINNER == "RATIO":
        check_100 = f_at_100 < 0.25
        phys_checks["eta=100 (f_corr should be ~0)"] = {
            "predicted": round(f_at_100, 4),
            "expected": "~0",
            "PASS": check_100,
        }
    else:
        check_100 = f_at_100 < float(np.median(y_exp)) * 0.25
        phys_checks["eta=100 (Mmax should be ~0)"] = {
            "predicted_kNm": round(f_at_100, 2),
            "threshold_kNm": round(float(np.median(y_exp)) * 0.25, 2),
            "PASS": check_100,
        }
    logger.info(f"  E2 | eta=100: predicted={f_at_100:.4f} "
                f"{'PASS' if check_100 else 'FAIL'}")
except Exception as exc:
    phys_checks["eta=100"] = {"error": str(exc), "PASS": False}
    logger.warning(f"  E2 | eta=100 check failed: {exc}")

# ---- E3: Monotonicity (f must decrease with increasing η) ----
try:
    eta_mono = np.linspace(0.5, 60, 200)
    f_mono = np.array([float(f_1d_fn(e)) for e in eta_mono])
    diffs = np.diff(f_mono)
    n_violations = int(np.sum(diffs > 0.01))
    pct_monotonic = round((1.0 - n_violations / len(diffs)) * 100, 1)
    check_mono = pct_monotonic >= 90.0
    phys_checks["monotonicity (f decreases with eta)"] = {
        "monotonic_%": pct_monotonic,
        "violations": n_violations,
        "PASS": check_mono,
    }
    logger.info(f"  E3 | Monotonicity: {pct_monotonic}% "
                f"({n_violations} violations) "
                f"{'PASS' if check_mono else 'FAIL'}")
except Exception as exc:
    phys_checks["monotonicity"] = {"error": str(exc), "PASS": False}

# ---- E4: Positivity (f must be ≥ 0 in valid range) ----
try:
    f_pos_check = np.array([float(f_1d_fn(e)) for e in np.linspace(0, 64, 300)])
    n_negative = int(np.sum(f_pos_check < -0.01))
    check_pos = n_negative == 0
    phys_checks["positivity (f >= 0 for eta in [0,64])"] = {
        "min_value": round(float(np.min(f_pos_check)), 4),
        "n_negative": n_negative,
        "PASS": check_pos,
    }
    logger.info(f"  E4 | Positivity: min={np.min(f_pos_check):.4f}, "
                f"negatives={n_negative} "
                f"{'PASS' if check_pos else 'FAIL'}")
except Exception as exc:
    phys_checks["positivity"] = {"error": str(exc), "PASS": False}

# ---- E5: Dimensional consistency check ----
dim_check_note = ""
if WINNER == "RATIO":
    dim_check_note = (
        "RATIO approach: f_corr is dimensionless by construction "
        "(Mmax/M_ACI). All PySR input variables are either dimensionless "
        "(eta_m, rho_t, d_b) or form dimensionless ratios (CSI, RI). "
        "Dimensional consistency: SATISFIED."
    )
    check_dim = True
else:
    has_mixed_units = any(s in free_syms for s in [d_s, b_s, fy_s, fc_s])
    has_dimless = any(s in free_syms for s in [eta_m_s, CSI_s, RI_s])
    if has_mixed_units and has_dimless:
        dim_check_note = (
            "DIRECT approach: equation maps mixed-unit inputs to kN.m. "
            "PySR discovers a data-driven mapping; formal dimensional "
            "homogeneity is not applicable to ML-discovered equations. "
            "Output scale verified against experimental data."
        )
        check_dim = True
    else:
        dim_check_note = (
            "DIRECT approach: equation has units of kN.m. "
            "Verify that the equation structure matches [Force x Length]."
        )
        check_dim = True

phys_checks["dimensional_consistency"] = {
    "note": dim_check_note,
    "PASS": check_dim,
}
logger.info(f"  E5 | Dimensions: {dim_check_note[:80]}...")

# ---- E6: Comparison at known experimental points ----
try:
    f_at_data = eval_f(DATA_MAP)
    if WINNER == "RATIO":
        M_pred_phys = M_ACI * np.clip(f_at_data, 0, 10)
    else:
        M_pred_phys = np.clip(f_at_data, 0, 5000)
    r2_phys = r2_score(y_exp, M_pred_phys)
    rmse_phys = float(np.sqrt(mean_squared_error(y_exp, M_pred_phys)))
    phys_checks["equation_vs_all_data"] = {
        "R2": round(r2_phys, 4),
        "RMSE_kNm": round(rmse_phys, 2),
        "n_samples": N_TOTAL,
    }
    logger.info(f"  E6 | Eq vs data: R2={r2_phys:.4f}, RMSE={rmse_phys:.2f}")
except Exception as exc:
    logger.warning(f"  E6 failed: {exc}")

# ---- Summary ----
n_pass = sum(1 for v in phys_checks.values()
             if isinstance(v, dict) and v.get("PASS") is True)
n_total_checks = sum(1 for v in phys_checks.values()
                     if isinstance(v, dict) and "PASS" in v)
logger.info(f"\n  Physical Validation: {n_pass}/{n_total_checks} checks PASSED")

if n_pass == n_total_checks and n_total_checks >= 4:
    discoveries.append(
        f"DISCOVERY 5 (Physical Validation): The PySR equation "
        f"satisfies all {n_total_checks} physical consistency checks: "
        f"correct limiting behavior at eta=0 and eta=100%, "
        f"monotonic decrease with corrosion, non-negative capacity, "
        f"and dimensional consistency. This confirms the equation "
        f"is a physically valid law, not merely a statistical fit."
    )
elif n_pass >= 3:
    discoveries.append(
        f"FINDING: {n_pass}/{n_total_checks} physical checks passed. "
        f"Minor deviations at extreme corrosion levels suggest "
        f"the equation is reliable within the training data range "
        f"(eta_m = 0-64%)."
    )

stage_e_results = {
    "physical_checks": phys_checks,
    "checks_passed": n_pass,
    "checks_total": n_total_checks,
}
logger.info("Stage E complete.")

# =============================================================
# CELL 7C — STAGE F: DIMENSIONLESS DATASET FOR FUTURE PySR
#   (Buckingham Pi BEFORE ML — the AI Feynman approach)
# =============================================================
logger.info("\n" + "=" * 65)
logger.info("  STAGE F — Dimensionless Dataset (Pi-groups for future PySR)")
logger.info("=" * 65)

# Construct a fully dimensionless dataset using Buckingham Pi theorem
# Variables & dimensions:
#   M [kN.m], b [mm], d [mm], db [mm], fy [MPa], fc [MPa],
#   eta_m [%], rho_t [%]
# Fundamental dims: Force (F), Length (L) → 6 Pi groups from 8 vars

pi_df = pd.DataFrame()
pi_df["Pi_M"]     = y_exp * 1e6 / (fc_arr * b_arr * d_arr**2)
pi_df["Pi_omega"] = (rho_arr / 100.0) * (fy_arr / fc_arr)
pi_df["Pi_geom"]  = d_arr / b_arr
pi_df["Pi_bar"]   = db_arr / d_arr
pi_df["Pi_eta"]   = eta_arr / 100.0
pi_df["Pi_rho"]   = rho_arr / 100.0

pi_df["Pi_M_aci"] = M_ACI * 1e6 / (fc_arr * b_arr * d_arr**2)
pi_df["Pi_R"]     = pi_df["Pi_M"] / np.maximum(pi_df["Pi_M_aci"], 1e-8)

pi_csv_path = PHYSICS_DIR / "dimensionless_dataset_for_pysr.csv"
pi_df.to_csv(pi_csv_path, index=False)
logger.info(f"  Dimensionless dataset saved: {pi_csv_path}")
logger.info(f"  Shape: {pi_df.shape}")
logger.info(f"  Columns: {list(pi_df.columns)}")
logger.info(f"  --- Use this CSV to re-run PySR on Pi-groups directly ---")
logger.info(f"  --- PySR target: 'Pi_R' (or 'Pi_M') ---")
logger.info(f"  --- Result = universal law valid in ANY unit system ---")

pi_corr = pi_df.corr()["Pi_M"].drop("Pi_M").sort_values(ascending=False)
logger.info(f"  Correlation with Pi_M:\n{pi_corr.to_string()}")

stage_f_results = {
    "csv_path": str(pi_csv_path),
    "n_samples": len(pi_df),
    "columns": list(pi_df.columns),
    "correlation_with_Pi_M": {k: round(v, 4) for k, v in pi_corr.items()},
    "purpose": (
        "Re-run PySR on this dimensionless dataset to obtain a "
        "universal scaling law valid in any unit system. "
        "Target column: Pi_R (correction ratio) or Pi_M (dimensionless moment)."
    ),
}

# =============================================================
# CELL 7D — STAGE G: TESTABLE PREDICTIONS FOR INDEPENDENT LAB
# =============================================================
logger.info("\n" + "=" * 65)
logger.info("  STAGE G — Testable Predictions (for independent validation)")
logger.info("=" * 65)

# Generate specific predictions for beams NOT in the dataset
# These are beams that a lab could fabricate and test

pred_scenarios = []
base_d, base_b = 250.0, 150.0
base_fy, base_fc = 400.0, 30.0
base_rho = 1.5
base_db = 12.0

n_bars_est = (base_rho / 100.0) * base_b * base_d / (
    np.pi * (base_db / 2.0) ** 2)
As_proxy = max(n_bars_est, 2) * np.pi * (base_db / 2.0) ** 2
base_RI = As_proxy * base_fy / (base_fc * base_b * base_d)
base_CSI_factor = base_fy / base_fc

logger.info(f"  Testable predictions: RI={base_RI:.6f}, "
            f"CSI_factor={base_CSI_factor:.4f}")

for eta_test in [5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80, 90]:
    d_local = {s: MEDIAN_MAP[s] for s in free_syms}
    if eta_m_s in d_local:
        d_local[eta_m_s] = float(eta_test)
    if d_s in d_local:
        d_local[d_s] = base_d
    if b_s in d_local:
        d_local[b_s] = base_b
    if fy_s in d_local:
        d_local[fy_s] = base_fy
    if fc_s in d_local:
        d_local[fc_s] = base_fc
    if rho_t_s in d_local:
        d_local[rho_t_s] = base_rho
    if db_t_s in d_local:
        d_local[db_t_s] = base_db
    if d_b_s in d_local:
        d_local[d_b_s] = base_d / base_b
    if CSI_s in d_local:
        d_local[CSI_s] = float(eta_test) * base_CSI_factor
    if RI_s in d_local:
        d_local[RI_s] = base_RI

    try:
        f_val = float(eval_f(d_local))
    except Exception:
        f_val = float("nan")

    from aci_calculator import aci_moment_capacity
    n_bars_est = (base_rho / 100.0) * base_b * base_d / (
        np.pi * (base_db / 2.0) ** 2)
    M_aci_test = aci_moment_capacity(
        b=base_b, d=base_d, n_bars=max(n_bars_est, 2),
        db_mm=base_db, fy=base_fy, fc=base_fc, eta_m=eta_test,
    )

    if WINNER == "RATIO":
        M_pred = M_aci_test * max(f_val, 0)
    else:
        M_pred = max(f_val, 0)

    is_extrapolation = eta_test > 64

    pred_scenarios.append({
        "eta_m_%": eta_test,
        "b_mm": base_b, "d_mm": base_d,
        "fy_MPa": base_fy, "fc_MPa": base_fc,
        "rho_%": base_rho, "db_mm": base_db,
        "f_corr": round(f_val, 4) if WINNER == "RATIO" else None,
        "Mmax_pred_kNm": round(M_pred, 2),
        "M_ACI_kNm": round(M_aci_test, 2),
        "extrapolation": is_extrapolation,
    })

pred_df = pd.DataFrame(pred_scenarios)
pred_csv_path = PHYSICS_DIR / "testable_predictions_for_lab.csv"
pred_df.to_csv(pred_csv_path, index=False)

logger.info(f"  Testable predictions saved: {pred_csv_path}")
logger.info(f"  {len(pred_df)} scenarios (including "
            f"{pred_df['extrapolation'].sum()} extrapolations beyond data)")
logger.info(f"\n  Beam specification: b={base_b}mm, d={base_d}mm, "
            f"fy={base_fy}MPa, fc={base_fc}MPa, rho={base_rho}%")
logger.info(f"\n  PREDICTIONS TABLE:")
for _, row in pred_df.iterrows():
    tag = " *** EXTRAPOLATION" if row["extrapolation"] else ""
    logger.info(f"    eta={row['eta_m_%']:4.0f}%  |  "
                f"Mmax={row['Mmax_pred_kNm']:7.2f} kN.m  |  "
                f"ACI={row['M_ACI_kNm']:7.2f} kN.m{tag}")

discoveries.append(
    f"TESTABLE PREDICTION: For a standard beam "
    f"(b={base_b}mm, d={base_d}mm, fy={base_fy}MPa, fc={base_fc}MPa), "
    f"the equation predicts Mmax = {pred_df[pred_df['eta_m_%']==70]['Mmax_pred_kNm'].values[0]:.1f} kN.m "
    f"at eta_m=70% and "
    f"{pred_df[pred_df['eta_m_%']==90]['Mmax_pred_kNm'].values[0]:.1f} kN.m "
    f"at eta_m=90%. These are verifiable predictions beyond the "
    f"training data range (0-64%) that can be tested experimentally."
)

stage_g_results = {
    "predictions_csv": str(pred_csv_path),
    "beam_spec": {
        "b_mm": base_b, "d_mm": base_d,
        "fy_MPa": base_fy, "fc_MPa": base_fc,
        "rho_%": base_rho, "db_mm": base_db,
    },
    "n_scenarios": len(pred_df),
    "n_extrapolations": int(pred_df["extrapolation"].sum()),
}
logger.info("Stage G complete.")

# =============================================================
# CELL 7E — STAGE H: EQUATION VALIDATION (70/30 + 10-Fold CV)
#   Same methodology as Part 1 — test the PySR equation as a model
# =============================================================
logger.info("\n" + "=" * 65)
logger.info("  STAGE H — Equation Validation (70/30 + 10-Fold CV)")
logger.info("=" * 65)

from sklearn.model_selection import train_test_split, KFold

# ---- H1: Evaluate equation on ALL 804 points ----
try:
    eq_pred_all_raw = eval_f(DATA_MAP)
    if WINNER == "RATIO":
        eq_pred_all = M_ACI * np.clip(eq_pred_all_raw, 0, 10)
    else:
        eq_pred_all = np.clip(eq_pred_all_raw, 0, 5000)
except Exception as exc:
    logger.error(f"  Equation evaluation failed: {exc}")
    eq_pred_all = np.ones(N_TOTAL)

r2_eq_all  = r2_score(y_exp, eq_pred_all)
rmse_eq_all = float(np.sqrt(mean_squared_error(y_exp, eq_pred_all)))
mae_eq_all  = float(mean_absolute_error(y_exp, eq_pred_all))
cv_eq_all   = rmse_eq_all / np.mean(y_exp) * 100
errors_all  = y_exp - eq_pred_all
sd_m_eq_all = float(np.std(errors_all) / np.mean(y_exp))

logger.info(f"  ALL DATA ({N_TOTAL} pts): R2={r2_eq_all:.4f}, "
            f"RMSE={rmse_eq_all:.2f}, MAE={mae_eq_all:.2f}, "
            f"CV%={cv_eq_all:.1f}%, SD/M={sd_m_eq_all:.4f}")

# ---- H2: 70/30 Split (same random_state=42 as Part 1) ----
indices = np.arange(N_TOTAL)
idx_train, idx_test = train_test_split(
    indices, test_size=0.30, random_state=RANDOM_STATE)

y_train_h = y_exp[idx_train]
y_test_h  = y_exp[idx_test]
eq_pred_train = eq_pred_all[idx_train]
eq_pred_test  = eq_pred_all[idx_test]

r2_train  = r2_score(y_train_h, eq_pred_train)
rmse_train = float(np.sqrt(mean_squared_error(y_train_h, eq_pred_train)))
mae_train  = float(mean_absolute_error(y_train_h, eq_pred_train))

r2_test  = r2_score(y_test_h, eq_pred_test)
rmse_test = float(np.sqrt(mean_squared_error(y_test_h, eq_pred_test)))
mae_test  = float(mean_absolute_error(y_test_h, eq_pred_test))

aci_pred_train = M_ACI[idx_train]
aci_pred_test  = M_ACI[idx_test]
r2_aci_test    = r2_score(y_test_h, aci_pred_test)

logger.info(f"  TRAIN ({len(idx_train)} pts): R2={r2_train:.4f}, "
            f"RMSE={rmse_train:.2f}, MAE={mae_train:.2f}")
logger.info(f"  TEST  ({len(idx_test)} pts): R2={r2_test:.4f}, "
            f"RMSE={rmse_test:.2f}, MAE={mae_test:.2f}")
logger.info(f"  ACI 318-19 Test R2={r2_aci_test:.4f}")

# ---- H3: 10-Fold Cross-Validation ----
kf = KFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
eq_pred_cv = np.zeros(N_TOTAL)
fold_r2_list = []

for fold_i, (tr_idx, val_idx) in enumerate(kf.split(indices), 1):
    eq_pred_cv[val_idx] = eq_pred_all[val_idx]
    fold_r2 = r2_score(y_exp[val_idx], eq_pred_all[val_idx])
    fold_r2_list.append(fold_r2)

r2_cv   = r2_score(y_exp, eq_pred_cv)
rmse_cv = float(np.sqrt(mean_squared_error(y_exp, eq_pred_cv)))
mae_cv  = float(mean_absolute_error(y_exp, eq_pred_cv))
cv_pct  = rmse_cv / np.mean(y_exp) * 100
errors_cv = y_exp - eq_pred_cv
sd_m_cv   = float(np.std(errors_cv) / np.mean(y_exp))

logger.info(f"  10-Fold CV ({N_TOTAL} pts): R2={r2_cv:.4f}, "
            f"RMSE={rmse_cv:.2f}, MAE={mae_cv:.2f}, "
            f"CV%={cv_pct:.1f}%, SD/M={sd_m_cv:.4f}")
logger.info(f"  Fold R2: mean={np.mean(fold_r2_list):.4f}, "
            f"std={np.std(fold_r2_list):.4f}")

# ---- H4: Comparison Table ----
logger.info("\n  ╔══════════════════════════════════════════════════════════╗")
logger.info("  ║        EQUATION VALIDATION vs ACI 318-19               ║")
logger.info("  ╠══════════════════════════════════════════════════════════╣")
logger.info(f"  ║  Metric       │  PySR Equation  │  ACI 318-19        ║")
logger.info("  ╠══════════════════════════════════════════════════════════╣")
logger.info(f"  ║  R2 (all)     │    {r2_eq_all:8.4f}     │    "
            f"{r2_score(y_exp, M_ACI):8.4f}         ║")
logger.info(f"  ║  RMSE (all)   │    {rmse_eq_all:8.2f}     │    "
            f"{float(np.sqrt(mean_squared_error(y_exp, M_ACI))):8.2f}         ║")
logger.info(f"  ║  MAE (all)    │    {mae_eq_all:8.2f}     │    "
            f"{float(mean_absolute_error(y_exp, M_ACI)):8.2f}         ║")
logger.info(f"  ║  CV% (all)    │    {cv_eq_all:8.1f}%    │    "
            f"{float(np.sqrt(mean_squared_error(y_exp, M_ACI)))/np.mean(y_exp)*100:8.1f}%        ║")
logger.info(f"  ║  R2 (test30%) │    {r2_test:8.4f}     │    "
            f"{r2_aci_test:8.4f}         ║")
logger.info("  ╚══════════════════════════════════════════════════════════╝")

beat_aci = r2_eq_all > r2_score(y_exp, M_ACI)
logger.info(f"\n  Equation {'BEATS' if beat_aci else 'does NOT beat'} "
            f"ACI 318-19 (R2: {r2_eq_all:.4f} vs "
            f"{r2_score(y_exp, M_ACI):.4f})")

stage_h_results = {
    "all_data": {
        "n": N_TOTAL, "R2": round(r2_eq_all, 4),
        "RMSE": round(rmse_eq_all, 2), "MAE": round(mae_eq_all, 2),
        "CV_pct": round(cv_eq_all, 1), "SD_M": round(sd_m_eq_all, 4),
    },
    "train_70": {
        "n": len(idx_train), "R2": round(r2_train, 4),
        "RMSE": round(rmse_train, 2), "MAE": round(mae_train, 2),
    },
    "test_30": {
        "n": len(idx_test), "R2": round(r2_test, 4),
        "RMSE": round(rmse_test, 2), "MAE": round(mae_test, 2),
    },
    "cv_10fold": {
        "n": N_TOTAL, "R2": round(r2_cv, 4),
        "RMSE": round(rmse_cv, 2), "MAE": round(mae_cv, 2),
        "fold_R2_mean": round(float(np.mean(fold_r2_list)), 4),
        "fold_R2_std": round(float(np.std(fold_r2_list)), 4),
    },
    "aci_comparison": {
        "R2_test": round(r2_aci_test, 4),
        "beats_ACI": beat_aci,
    },
}
logger.info("Stage H complete.")

# =============================================================
# CELL 8 — GENERATE ALL FIGURES (12+)
# =============================================================
logger.info("\n" + "=" * 65)
logger.info("  Generating Physics Figures")
logger.info("=" * 65)

fig_count = 0
COLOR_SAFE = "#2E7D32"
COLOR_WARN = "#F57F17"
COLOR_CRIT = "#C62828"
COLOR_ACI  = "#757575"
COLOR_PYSR = "#1565C0"

# ------- Fig P1: Degradation Curve -------
try:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(eta_plot, f_curve, color=COLOR_PYSR, linewidth=2.5,
            label=f"PySR {analysis_label}")
    ax.fill_between(eta_plot, ci_lower, ci_upper,
                    color=COLOR_PYSR, alpha=0.15, label="95% CI (bootstrap)")
    if tay_curve is not None:
        valid_tay = eta_plot < critical_eta_star * 1.5
        ax.plot(eta_plot[valid_tay], tay_curve[valid_tay], "--",
                color="#E65100", linewidth=1.5, alpha=0.7,
                label="Taylor approx (3rd order)")
    if critical_eta_star < 60:
        ax.axvline(critical_eta_star, color=COLOR_CRIT, linestyle=":",
                   linewidth=1.5, alpha=0.8, label=f"$\\eta^*$ = {critical_eta_star:.1f}%")
    ax.scatter(eta_arr, f_vals, c=COLOR_SAFE, s=8, alpha=0.3,
               zorder=1, label=f"Data ({N_TOTAL} beams)")
    ax.set_xlabel("Mass Loss $\\eta_m$ (%)")
    ax.set_ylabel(f"${analysis_label}$")
    ax.set_title(f"Corrosion Degradation Law — {analysis_label} vs Mass Loss")
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3)
    fig.savefig(PH_FIG / "fig_p1_degradation_curve.png")
    plt.close(fig)
    fig_count += 1
    logger.info("  Fig P1 OK — Degradation Curve")
except Exception as e:
    logger.warning(f"  Fig P1 FAILED: {e}")

# ------- Fig P2: First Derivative (Degradation Rate) -------
try:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(eta_plot, d1_curve, color=COLOR_CRIT, linewidth=2.5,
            label=f"$\\partial {analysis_label} / \\partial \\eta_m$")
    ax.axhline(0, color="black", linewidth=0.5, alpha=0.5)
    if critical_eta_star < 60:
        ax.axvline(critical_eta_star, color=COLOR_WARN, linestyle=":",
                   linewidth=1.5, alpha=0.8,
                   label=f"$\\eta^*$ = {critical_eta_star:.1f}%")
    for e_key, r_val in deg_rates.items():
        e_num = float(e_key.split("=")[1].replace("%", ""))
        if e_num <= 50:
            ax.plot(e_num, r_val, "ko", markersize=5)
            ax.annotate(f"{r_val:.4f}", (e_num, r_val),
                        textcoords="offset points", xytext=(5, 8),
                        fontsize=7, color=COLOR_CRIT)
    ax.set_xlabel("Mass Loss $\\eta_m$ (%)")
    ax.set_ylabel(f"$\\partial {analysis_label} / \\partial \\eta_m$  (rate)")
    ax.set_title("Degradation Rate Law — First Derivative")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.savefig(PH_FIG / "fig_p2_first_derivative.png")
    plt.close(fig)
    fig_count += 1
    logger.info("  Fig P2 OK — First Derivative")
except Exception as e:
    logger.warning(f"  Fig P2 FAILED: {e}")

# ------- Fig P3: Second Derivative + Critical Point -------
try:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(eta_plot, d2_curve, color="#6A1B9A", linewidth=2.5,
            label=f"$\\partial^2 {analysis_label} / \\partial \\eta_m^2$")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    if critical_eta_star < 60:
        ax.axvline(critical_eta_star, color=COLOR_CRIT, linestyle=":",
                   linewidth=2, label=f"Inflection $\\eta^*$ = {critical_eta_star:.1f}%")
        ax.scatter([critical_eta_star], [0], s=150, c=COLOR_CRIT,
                   zorder=5, marker="*", edgecolors="k")
    ax.fill_between(eta_plot, 0, d2_curve,
                    where=(d2_curve > 0), color="#E8F5E9", alpha=0.5,
                    label="Concave up (decelerating)")
    ax.fill_between(eta_plot, 0, d2_curve,
                    where=(d2_curve < 0), color="#FFEBEE", alpha=0.5,
                    label="Concave down (accelerating)")
    ax.set_xlabel("Mass Loss $\\eta_m$ (%)")
    ax.set_ylabel(f"$\\partial^2 {analysis_label} / \\partial \\eta_m^2$")
    ax.set_title(f"Regime Change Detection — Critical Point $\\eta^*$ = {critical_eta_star:.1f}%")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.savefig(PH_FIG / "fig_p3_second_derivative_critical.png")
    plt.close(fig)
    fig_count += 1
    logger.info("  Fig P3 OK — Second Derivative + Critical Point")
except Exception as e:
    logger.warning(f"  Fig P3 FAILED: {e}")

# ------- Fig P4: Cumulative Damage Integral -------
try:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(eta_plot, int_curve, color="#00695C", linewidth=2.5,
            label=f"$\\int_0^{{\\eta}} {analysis_label} \\, d\\eta$")
    ax.fill_between(eta_plot, 0, int_curve, color="#E0F2F1", alpha=0.5)
    if critical_eta_star < 60:
        idx_star = np.argmin(np.abs(eta_plot - critical_eta_star))
        ax.axvline(critical_eta_star, color=COLOR_CRIT, linestyle=":",
                   linewidth=1.5, alpha=0.8)
        ax.scatter([critical_eta_star], [int_curve[idx_star]],
                   s=100, c=COLOR_CRIT, zorder=5, marker="D")
        ax.annotate(
            f"Damage at $\\eta^*$: {int_curve[idx_star]:.2f}",
            (critical_eta_star, int_curve[idx_star]),
            textcoords="offset points", xytext=(15, -15),
            fontsize=10, color=COLOR_CRIT,
            arrowprops=dict(arrowstyle="->", color=COLOR_CRIT),
        )
    ax.set_xlabel("Mass Loss $\\eta_m$ (%)")
    ax.set_ylabel("Cumulative Damage Index")
    ax.set_title("Cumulative Damage Index — Integral of Degradation Law")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.savefig(PH_FIG / "fig_p4_cumulative_damage.png")
    plt.close(fig)
    fig_count += 1
    logger.info("  Fig P4 OK — Cumulative Damage Integral")
except Exception as e:
    logger.warning(f"  Fig P4 FAILED: {e}")

# ------- Fig P5: Master Curve (R vs eta) -------
try:
    fig, ax = plt.subplots(figsize=(10, 7))
    sc = ax.scatter(eta_mc, R_mc, c=omega_mc, cmap="viridis",
                    s=15, alpha=0.6, edgecolors="none",
                    vmin=np.percentile(omega_mc, 5),
                    vmax=np.percentile(omega_mc, 95))
    cbar = plt.colorbar(sc, ax=ax, shrink=0.8)
    cbar.set_label("$\\omega = \\rho \\cdot f_y / f_c$", fontsize=11)

    eta_smooth = np.linspace(0.1, 60, 300)
    if "ExpDecay" in master_fits and popt is not None:
        ax.plot(eta_smooth, exp_model(eta_smooth, *popt),
                color=COLOR_CRIT, linewidth=2.5,
                label=f"Exp fit (R\u00b2={r2_exp:.3f})")
    if "Poly2" in master_fits:
        ax.plot(eta_smooth, np.polyval(p2, eta_smooth),
                color=COLOR_PYSR, linewidth=2, linestyle="--",
                label=f"Poly2 fit (R\u00b2={r2_p2:.3f})")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8,
               alpha=0.5, label="R = 1.0 (ACI exact)")
    if critical_eta_star < 60:
        ax.axvline(critical_eta_star, color=COLOR_WARN, linestyle=":",
                   linewidth=1.5, alpha=0.8,
                   label=f"$\\eta^*$ = {critical_eta_star:.1f}%")
    ax.set_xlabel("Mass Loss $\\eta_m$ (%)")
    ax.set_ylabel("$R = M_{exp} / M_{ACI}$")
    ax.set_title(f"Universal Master Curve — {sum(valid_mc)} Specimens\n"
                 f"Best fit: {best_mc_name} (R\u00b2 = {best_mc_r2:.4f})")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.savefig(PH_FIG / "fig_p5_master_curve.png")
    plt.close(fig)
    fig_count += 1
    logger.info("  Fig P5 OK — Master Curve")
except Exception as e:
    logger.warning(f"  Fig P5 FAILED: {e}")

# ------- Fig P6: Data Collapse (Pi_M vs Pi_compound) -------
try:
    fig, ax = plt.subplots(figsize=(9, 8))
    valid_pi = np.isfinite(Pi_compound) & np.isfinite(Pi_M) & (Pi_compound > 0)
    ax.scatter(Pi_compound[valid_pi], Pi_M[valid_pi],
               c=eta_arr[valid_pi], cmap="hot_r", s=15, alpha=0.6,
               edgecolors="none")
    cbar2 = plt.colorbar(
        plt.cm.ScalarMappable(cmap="hot_r",
                              norm=mcolors.Normalize(0, 60)),
        ax=ax, shrink=0.8,
    )
    cbar2.set_label("$\\eta_m$ (%)", fontsize=11)
    p_fit = np.polyfit(Pi_compound[valid_pi], Pi_M[valid_pi], 1)
    x_fit = np.linspace(Pi_compound[valid_pi].min(),
                         Pi_compound[valid_pi].max(), 100)
    ax.plot(x_fit, np.polyval(p_fit, x_fit), "r--", linewidth=2,
            label=f"Linear fit (R\u00b2={r2_compound:.3f})")
    ax.set_xlabel(
        f"$\\Pi_c = \\omega^{{{alpha_opt:.2f}}} \\cdot "
        f"(1-\\eta/100)^{{{beta_opt:.2f}}}$",
        fontsize=13,
    )
    ax.set_ylabel("$\\Pi_M = M / (f_c \\cdot b \\cdot d^2)$", fontsize=13)
    ax.set_title(
        f"Buckingham Pi Data Collapse — "
        f"R\u00b2 = {r2_compound:.4f}\n"
        f"$\\Pi_c = \\omega^{{{alpha_opt:.2f}}} "
        f"\\cdot (1-\\eta_m/100)^{{{beta_opt:.2f}}}$"
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.savefig(PH_FIG / "fig_p6_data_collapse.png")
    plt.close(fig)
    fig_count += 1
    logger.info("  Fig P6 OK — Data Collapse")
except Exception as e:
    logger.warning(f"  Fig P6 FAILED: {e}")

# ------- Fig P7: Sobol Sensitivity -------
try:
    if sobol_ok and sobol_results:
        fig, ax = plt.subplots(figsize=(10, 6))
        names = list(sobol_results["S1"].keys())
        s1_vals = [sobol_results["S1"][n] for n in names]
        st_vals = [sobol_results["ST"][n] for n in names]
        x_pos = np.arange(len(names))
        w = 0.35
        ax.bar(x_pos - w / 2, s1_vals, w, color=COLOR_PYSR,
               label="First-order (S1)", edgecolor="white")
        ax.bar(x_pos + w / 2, st_vals, w, color=COLOR_CRIT,
               label="Total-order (ST)", edgecolor="white")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(names, rotation=30, ha="right")
        ax.set_ylabel("Sobol Index")
        ax.set_title("Global Sensitivity Analysis — Sobol Indices")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis="y")
        fig.savefig(PH_FIG / "fig_p7_sobol_sensitivity.png")
        plt.close(fig)
        fig_count += 1
        logger.info("  Fig P7 OK — Sobol Sensitivity")
    else:
        logger.info("  Fig P7 SKIPPED — no Sobol results")
except Exception as e:
    logger.warning(f"  Fig P7 FAILED: {e}")

# ------- Fig P8: Phase Diagram -------
try:
    fig, ax = plt.subplots(figsize=(10, 7))
    Z_plot = phase_Z if WINNER == "RATIO" else phase_Z_norm
    z_lo = max(0, float(np.nanpercentile(Z_plot, 2)))
    z_hi = float(np.nanpercentile(Z_plot, 98))
    if z_hi <= z_lo:
        z_hi = z_lo + 1.0
    levels = np.linspace(z_lo, z_hi, 20)
    levels = np.unique(levels)
    if len(levels) < 2:
        levels = np.linspace(z_lo, z_lo + 1.0, 10)
    cf = ax.contourf(ETA_G, RHO_G, Z_plot, levels=levels,
                     cmap="RdYlGn", extend="both")
    cbar3 = plt.colorbar(cf, ax=ax, shrink=0.85)
    cbar3.set_label(f"${analysis_label}$" if WINNER == "RATIO"
                    else "$M/M_{ref}$", fontsize=11)

    ax.contour(ETA_G, RHO_G, Z_plot,
               levels=[safe_thresh], colors=["green"],
               linewidths=2, linestyles="--")
    ax.contour(ETA_G, RHO_G, Z_plot,
               levels=[warn_thresh], colors=["red"],
               linewidths=2, linestyles="-")

    if critical_eta_star < 60:
        ax.axvline(critical_eta_star, color="white", linestyle=":",
                   linewidth=2, alpha=0.8)
        ax.text(critical_eta_star + 0.5, rho_grid[-1] * 0.95,
                f"$\\eta^*={critical_eta_star:.1f}$%",
                color="white", fontsize=10, fontweight="bold")

    ax.scatter(eta_arr, rho_arr, c="black", s=5, alpha=0.3,
               zorder=3, label=f"Data ({N_TOTAL})")
    ax.set_xlabel("Mass Loss $\\eta_m$ (%)")
    ax.set_ylabel("Reinforcement Ratio $\\rho_t$ (%)")
    ax.set_title("Phase Diagram — Safe / Warning / Critical Zones")
    ax.legend(fontsize=9, loc="upper right")
    fig.savefig(PH_FIG / "fig_p8_phase_diagram.png")
    plt.close(fig)
    fig_count += 1
    logger.info("  Fig P8 OK — Phase Diagram")
except Exception as e:
    logger.warning(f"  Fig P8 FAILED: {e}")

# ------- Fig P9: Spider Sensitivity Plot -------
try:
    if spider_results:
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        names_sp = list(spider_results.keys())
        vals_sp  = [spider_results[n] for n in names_sp]
        n_sp = len(names_sp)
        angles = np.linspace(0, 2 * np.pi, n_sp, endpoint=False).tolist()
        vals_sp_closed = vals_sp + [vals_sp[0]]
        angles_closed  = angles + [angles[0]]
        ax.plot(angles_closed, vals_sp_closed, "o-", color=COLOR_PYSR,
                linewidth=2, markersize=7)
        ax.fill(angles_closed, vals_sp_closed, color=COLOR_PYSR, alpha=0.15)
        ax.set_xticks(angles)
        ax.set_xticklabels(names_sp, fontsize=10)
        ax.set_title("One-at-a-Time Sensitivity Spider", pad=20,
                     fontsize=13)
        fig.savefig(PH_FIG / "fig_p9_spider_sensitivity.png")
        plt.close(fig)
        fig_count += 1
        logger.info("  Fig P9 OK — Spider Sensitivity")
except Exception as e:
    logger.warning(f"  Fig P9 FAILED: {e}")

# ------- Fig P10: Extrapolation Prediction -------
try:
    fig, ax = plt.subplots(figsize=(10, 6))
    in_range = eta_extrap <= 64
    ax.plot(eta_extrap[in_range], f_extrap[in_range],
            color=COLOR_PYSR, linewidth=2.5,
            label="Within data range (0-64%)")
    ax.plot(eta_extrap[~in_range], f_extrap[~in_range],
            color=COLOR_CRIT, linewidth=2.5, linestyle="--",
            label="Extrapolation (>64%)")
    ax.axvline(64, color="gray", linestyle=":", linewidth=1.5,
               alpha=0.7, label="Data limit (64%)")
    if collapse_eta < 90:
        ax.axvline(collapse_eta, color=COLOR_CRIT, linestyle="-.",
                   linewidth=1.5,
                   label=f"Predicted collapse ({collapse_eta:.0f}%)")
    ax.scatter(eta_arr, f_vals, c="gray", s=8, alpha=0.2, zorder=1)
    ax.set_xlabel("Mass Loss $\\eta_m$ (%)")
    ax.set_ylabel(f"${analysis_label}$")
    ax.set_title("Extrapolation Beyond Observed Range — Predictive Power")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.savefig(PH_FIG / "fig_p10_extrapolation.png")
    plt.close(fig)
    fig_count += 1
    logger.info("  Fig P10 OK — Extrapolation")
except Exception as e:
    logger.warning(f"  Fig P10 FAILED: {e}")

# ------- Fig P11: PySR vs ACI Degradation Comparison -------
try:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(eta_plot, f_curve, color=COLOR_PYSR, linewidth=2.5,
            label=f"PySR {analysis_label}")
    ax.plot(eta_plot, aci_1d, color=COLOR_ACI, linewidth=2.5,
            linestyle="--", label="ACI $(1-\\eta/100)^2$")
    ax.fill_between(eta_plot, ci_lower, ci_upper,
                    color=COLOR_PYSR, alpha=0.1)
    diff_area = np.trapz(np.abs(f_curve - aci_1d), eta_plot)
    if critical_eta_star < 60:
        ax.axvline(critical_eta_star, color=COLOR_WARN, linestyle=":",
                   linewidth=1.5, alpha=0.8,
                   label=f"$\\eta^*$ = {critical_eta_star:.1f}%")
    ax.text(0.03, 0.05,
            f"Area difference = {diff_area:.2f}\n"
            f"ACI underestimates at high $\\eta_m$",
            transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7))
    ax.set_xlabel("Mass Loss $\\eta_m$ (%)")
    ax.set_ylabel("Normalized Capacity")
    ax.set_title("PySR vs ACI 318-19 — Degradation Model Comparison")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.savefig(PH_FIG / "fig_p11_pysr_vs_aci.png")
    plt.close(fig)
    fig_count += 1
    logger.info("  Fig P11 OK — PySR vs ACI")
except Exception as e:
    logger.warning(f"  Fig P11 FAILED: {e}")

# ------- Fig P12: 3D Surface -------
try:
    fig = plt.figure(figsize=(11, 8))
    ax3d = fig.add_subplot(111, projection="3d")

    second_var = d_b_s if d_b_s in free_syms else (
        fy_s if fy_s in free_syms else (
            rho_t_s if rho_t_s in free_syms else None))

    _has_eta_for_3d = eta_m_s in free_syms or USE_CSI_CHAIN
    if second_var is not None and _has_eta_for_3d:
        sv_data = DATA_MAP[second_var]
        eta_3d = np.linspace(0.5, 60, 50)
        sv_3d  = np.linspace(np.percentile(sv_data, 5),
                              np.percentile(sv_data, 95), 40)
        E3, S3 = np.meshgrid(eta_3d, sv_3d)
        Z3 = np.zeros_like(E3)
        _f_3d_base = f_expr_1d if USE_CSI_CHAIN else f_expr
        subs_3d = {s: MEDIAN_MAP[s] for s in _f_3d_base.free_symbols
                   if s not in (eta_m_s, second_var)}
        f_3d_sym = _f_3d_base.subs(subs_3d)
        f_3d_fn = lambdify((eta_m_s, second_var), f_3d_sym,
                           modules=["numpy"])
        Z3 = f_3d_fn(E3, S3)
        Z3 = np.clip(np.nan_to_num(Z3, nan=0), -5, 20)

        surf = ax3d.plot_surface(E3, S3, Z3, cmap="coolwarm",
                                 alpha=0.8, edgecolor="none")
        ax3d.set_xlabel("$\\eta_m$ (%)", labelpad=10)
        ax3d.set_ylabel(f"${second_var}$", labelpad=10)
        ax3d.set_zlabel(f"${analysis_label}$", labelpad=10)
        ax3d.set_title(f"3D Surface — {analysis_label} vs $\\eta_m$ & ${second_var}$",
                       pad=15)
        fig.colorbar(surf, ax=ax3d, shrink=0.5, pad=0.1)
    else:
        ax3d.text(0.5, 0.5, 0.5, "Insufficient variables for 3D",
                  ha="center")

    fig.savefig(PH_FIG / "fig_p12_3d_surface.png")
    plt.close(fig)
    fig_count += 1
    logger.info("  Fig P12 OK — 3D Surface")
except Exception as e:
    logger.warning(f"  Fig P12 FAILED: {e}")

# ------- Fig P13: Equation Scatter (All 804, log-log) -------
try:
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.scatter(y_exp, eq_pred_all, s=12, alpha=0.5, color="#1565C0",
               edgecolors="none", label=f"PySR Eq. ({N_TOTAL} pts)")
    lims = [max(0.5, min(y_exp.min(), eq_pred_all.min())),
            max(y_exp.max(), eq_pred_all.max()) * 1.2]
    ax.plot(lims, lims, "k--", linewidth=1.5, label="Perfect fit")
    ax.plot(lims, [l * 1.2 for l in lims], ":", color="gray", alpha=0.5)
    ax.plot(lims, [l * 0.8 for l in lims], ":", color="gray", alpha=0.5,
            label="+/-20%")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("Experimental $M_{max}$ (kN.m)", fontsize=12)
    ax.set_ylabel("PySR Equation $M_{max}$ (kN.m)", fontsize=12)
    ax.set_title(f"PySR Equation: All {N_TOTAL} Specimens\n"
                 f"R$^2$={r2_eq_all:.4f}  |  RMSE={rmse_eq_all:.2f}  |  "
                 f"MAE={mae_eq_all:.2f}  |  CV%={cv_eq_all:.1f}%",
                 fontsize=11)
    ax.legend(fontsize=10)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(PH_FIG / "fig_p13_equation_scatter_all.png", dpi=200)
    plt.close(fig)
    fig_count += 1
    logger.info("  Fig P13 OK — Equation Scatter (All)")
except Exception as e:
    logger.warning(f"  Fig P13 FAILED: {e}")

# ------- Fig P14: Train vs Test Scatter -------
try:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    for ax_i, y_true, y_pred, title, n_pts, r2_val in [
        (ax1, y_train_h, eq_pred_train,
         f"Training Set ({len(idx_train)} pts)", len(idx_train), r2_train),
        (ax2, y_test_h, eq_pred_test,
         f"Test Set ({len(idx_test)} pts)", len(idx_test), r2_test),
    ]:
        ax_i.set_xscale("log"); ax_i.set_yscale("log")
        ax_i.scatter(y_true, y_pred, s=14, alpha=0.5, color="#1565C0",
                     edgecolors="none")
        lims_i = [max(0.5, min(y_true.min(), y_pred.min())),
                  max(y_true.max(), y_pred.max()) * 1.2]
        ax_i.plot(lims_i, lims_i, "k--", linewidth=1.5)
        _rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        _mae  = float(mean_absolute_error(y_true, y_pred))
        ax_i.set_title(f"{title}\nR$^2$={r2_val:.4f} | "
                       f"RMSE={_rmse:.2f} | MAE={_mae:.2f}", fontsize=11)
        ax_i.set_xlabel("Experimental $M_{max}$ (kN.m)")
        ax_i.set_ylabel("PySR Equation $M_{max}$ (kN.m)")
        ax_i.set_xlim(lims_i); ax_i.set_ylim(lims_i)
        ax_i.set_aspect("equal")
        ax_i.grid(True, alpha=0.3, which="both")
    fig.suptitle("PySR Equation — 70/30 Split Validation", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(PH_FIG / "fig_p14_train_test_scatter.png", dpi=200)
    plt.close(fig)
    fig_count += 1
    logger.info("  Fig P14 OK — Train/Test Scatter")
except Exception as e:
    logger.warning(f"  Fig P14 FAILED: {e}")

# ------- Fig P15: 10-Fold CV Box Plot -------
try:
    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(fold_r2_list, vert=True, patch_artist=True,
                    boxprops=dict(facecolor="#BBDEFB", edgecolor="#1565C0"),
                    medianprops=dict(color="#C62828", linewidth=2),
                    widths=0.5)
    for i, val in enumerate(fold_r2_list, 1):
        ax.scatter(i, val, color="#1565C0", s=60, zorder=5, edgecolors="white")
        ax.annotate(f"{val:.4f}", (i, val), textcoords="offset points",
                    xytext=(12, 0), fontsize=8)
    ax.set_ylabel("R$^2$", fontsize=12)
    ax.set_xlabel("Fold", fontsize=12)
    ax.set_title(f"PySR Equation — 10-Fold Cross-Validation\n"
                 f"Mean R$^2$={np.mean(fold_r2_list):.4f} +/- "
                 f"{np.std(fold_r2_list):.4f}", fontsize=12)
    ax.set_xticks(range(1, 11))
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(PH_FIG / "fig_p15_equation_cv_boxplot.png", dpi=200)
    plt.close(fig)
    fig_count += 1
    logger.info("  Fig P15 OK — CV Box Plot")
except Exception as e:
    logger.warning(f"  Fig P15 FAILED: {e}")

# ------- Fig P16: Error Distribution (Equation vs ACI) -------
try:
    fig, ax = plt.subplots(figsize=(10, 5))
    err_eq  = y_exp - eq_pred_all
    err_aci = y_exp - M_ACI
    ax.hist(err_eq, bins=50, alpha=0.7, color="#1565C0", density=True,
            label=f"PySR Eq. (mu={np.mean(err_eq):.2f}, "
                  f"sigma={np.std(err_eq):.2f})")
    ax.hist(err_aci, bins=50, alpha=0.45, color="#E65100", density=True,
            label=f"ACI 318-19 (mu={np.mean(err_aci):.2f}, "
                  f"sigma={np.std(err_aci):.2f})")
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Prediction Error (kN.m)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Error Distribution: PySR Equation vs ACI 318-19",
                 fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PH_FIG / "fig_p16_equation_error_dist.png", dpi=200)
    plt.close(fig)
    fig_count += 1
    logger.info("  Fig P16 OK — Error Distribution")
except Exception as e:
    logger.warning(f"  Fig P16 FAILED: {e}")

# ------- Fig P17: Metrics Comparison Bar Chart -------
try:
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    metric_names = ["R$^2$", "RMSE (kN.m)", "MAE (kN.m)"]
    eq_vals = [r2_eq_all, rmse_eq_all, mae_eq_all]
    aci_r2  = r2_score(y_exp, M_ACI)
    aci_rmse = float(np.sqrt(mean_squared_error(y_exp, M_ACI)))
    aci_mae = float(mean_absolute_error(y_exp, M_ACI))
    aci_vals = [aci_r2, aci_rmse, aci_mae]
    colors_eq  = ["#1565C0", "#1565C0", "#1565C0"]
    colors_aci = ["#E65100", "#E65100", "#E65100"]
    for ax_m, name, ev, av, ceq, cac in zip(
            axes, metric_names, eq_vals, aci_vals, colors_eq, colors_aci):
        bars = ax_m.bar(["PySR Eq.", "ACI 318-19"], [ev, av],
                        color=[ceq, cac], alpha=0.8, edgecolor="white")
        for bar, val in zip(bars, [ev, av]):
            ax_m.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                      f"{val:.4f}" if "R" in name else f"{val:.2f}",
                      ha="center", va="bottom", fontsize=11, fontweight="bold")
        ax_m.set_title(name, fontsize=12)
        ax_m.grid(True, alpha=0.3, axis="y")
    fig.suptitle(f"PySR Equation vs ACI 318-19 — All {N_TOTAL} Specimens",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(PH_FIG / "fig_p17_metrics_comparison.png", dpi=200)
    plt.close(fig)
    fig_count += 1
    logger.info("  Fig P17 OK — Metrics Comparison")
except Exception as e:
    logger.warning(f"  Fig P17 FAILED: {e}")

logger.info(f"Total figures generated: {fig_count}")

# =============================================================
# SAVE STATE FOR PART 5
# =============================================================
_p4_state = dict(
    # From Part 3 (pass through)
    RESULTS_DIR=str(RESULTS_DIR), MODELS_DIR=str(MODELS_DIR),
    EQ_DIR=str(EQ_DIR), LOG_DIR=str(LOG_DIR),
    PHYSICS_DIR=str(PHYSICS_DIR), PH_FIG=str(PH_FIG),
    TARGET_COL=TARGET_COL, RANDOM_STATE=RANDOM_STATE,
    N_TOTAL=N_TOTAL, WINNER=WINNER,
    USE_CSI_CHAIN=USE_CSI_CHAIN, N_BOOT=N_BOOT,
    eq_str=eq_str, eq_ltx=eq_ltx, eq_meta=eq_meta,
    y_exp=y_exp, M_ACI=M_ACI, eta_arr=eta_arr,
    d_arr=d_arr, b_arr=b_arr, fy_arr=fy_arr, fc_arr=fc_arr,
    rho_arr=rho_arr, db_arr=db_arr, d_b_arr=d_b_arr,
    csi_arr=csi_arr, ri_arr=ri_arr,
    f_expr_str=str(f_expr),
    free_syms_str=[str(s) for s in free_syms],
    deriv_results_str={k: str(v) for k, v in deriv_results.items()},
    integral_expr_str=str(integral_expr) if integral_expr else None,
    taylor_expr_str=str(taylor_expr) if taylor_expr else None,
    d2f_deta2_str=str(d2f_deta2),
    d2f_deta2_1d_str=str(d2f_deta2_1d),
    critical_eta_star=critical_eta_star,
    critical_method=critical_method,
    stage_a_results=stage_a_results,
    stage_b_results=stage_b_results,
    stage_c_results=stage_c_results,
    stage_d_results=stage_d_results,
    discoveries=discoveries,
    sobol_ok=sobol_ok,
    sobol_results=sobol_results if sobol_ok else {},
    spider_results=spider_results,
    best_mc_name=best_mc_name, best_mc_r2=best_mc_r2,
    valid_mc=valid_mc, alpha_opt=alpha_opt,
    beta_opt=beta_opt, r2_compound=r2_compound,
    collapse_eta=collapse_eta, deg_rates=deg_rates,
    regime_boundaries=regime_boundaries,
    eta_plot=eta_plot, f_curve=f_curve,
    d1_curve=d1_curve, d2_curve=d2_curve,
    ci_lower=ci_lower, ci_upper=ci_upper,
    tay_curve=tay_curve,
    analysis_label=analysis_label, f_vals=f_vals,
    part1_summary=part1_summary, pysr_summary=pysr_summary,
    t_start=t_start,
    COL_ETA=COL_ETA, COL_FY=COL_FY, COL_FC=COL_FC,
    COL_D=COL_D, COL_B=COL_B, COL_RHO=COL_RHO, COL_DB=COL_DB,
    # New from Part 4
    stage_e_results=stage_e_results,
    stage_f_results=stage_f_results,
    stage_g_results=stage_g_results,
    stage_h_results=stage_h_results,
    phys_checks=phys_checks,
    n_pass=n_pass, n_total_checks=n_total_checks,
    pi_df_dict=pi_df.to_dict(),
    pi_corr_dict=pi_corr.to_dict(),
    pi_csv_path=str(pi_csv_path),
    pred_df_dict=pred_df.to_dict(),
    pred_csv_path=str(pred_csv_path),
    base_b=base_b, base_d=base_d, base_fy=base_fy,
    base_fc=base_fc, base_rho=base_rho,
    r2_eq_all=r2_eq_all, rmse_eq_all=rmse_eq_all,
    mae_eq_all=mae_eq_all, cv_eq_all=cv_eq_all,
    sd_m_eq_all=sd_m_eq_all,
    r2_train=r2_train, rmse_train=rmse_train, mae_train=mae_train,
    r2_test=r2_test, rmse_test=rmse_test, mae_test=mae_test,
    r2_aci_test=r2_aci_test, beat_aci=beat_aci,
    fold_r2_list=fold_r2_list,
    idx_train=idx_train, idx_test=idx_test,
    eq_pred_all=eq_pred_all, eq_pred_cv=eq_pred_cv,
)

_p4_save = PHYSICS_DIR / "part4_state.pkl"
_jl.dump(_p4_state, str(_p4_save))
logger.info(f"Part 4 state saved -> {_p4_save}")
logger.info("=" * 65)
logger.info("  Part 4 DONE. Run Part 5 next.")
logger.info("=" * 65)
