#!/usr/bin/env python3
"""
===============================================================
  Corrosion RC Beam Optimizer -- Part 2: PySR Equation Discovery
  Google Colab Self-Contained Script
===============================================================
  PREREQUISITE: Run Part 1 (colab_part1_training.py) first!

  PIPELINE:
    1. Re-run preprocessing + ACI (fast, <1 min)
    2. PySR Ratio approach (Mmax_exp / M_ACI)
    3. PySR Direct Mmax approach
    4. Compare & select winner
    5. Generate equation figures
    6. PDF Report (combines Part 1 + Part 2 results)
    7. Save final equations + ZIP

  HOW TO RUN (Google Colab):
    1. Run Part 1 first (in same runtime session)
    2. Paste this ENTIRE file into a NEW cell
    3. Run it (takes ~2-4 hours for PySR)
    4. Download final_results/ when done
===============================================================
"""

# =============================================================
# CELL 1: INSTALL & CLONE
# =============================================================
import subprocess, sys, os

def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

for p in ["loguru", "pysr", "scikit-learn", "matplotlib", "seaborn", "fpdf2"]:
    try:
        __import__(p.replace("-", "_"))
    except ImportError:
        install(p)

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

# Patch 70/30 if not already patched
config_path = f"{REPO_PATH}/src/config.py"
with open(config_path, "r") as f:
    cfg_txt = f.read()
if "TEST_SIZE    = 0.20" in cfg_txt:
    cfg_txt = cfg_txt.replace("TEST_SIZE    = 0.20", "TEST_SIZE    = 0.30")
    with open(config_path, "w") as f:
        f.write(cfg_txt)
    print("CONFIG PATCHED: TEST_SIZE = 0.30")

os.chdir(f"{REPO_PATH}/src")
sys.path.insert(0, f"{REPO_PATH}/src")
print("Setup complete.")

# =============================================================
# CELL 2: IMPORTS
# =============================================================
import json
import time
import warnings
import traceback
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
from datetime import datetime
from pathlib import Path
from loguru import logger
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")

from config import (
    RESULTS_DIR, MODELS_DIR, FIGURES_DIR, EQ_DIR, LOG_DIR,
    TARGET_COL, RANDOM_STATE,
    L1_TARGET_R2, L2_TARGET_R2,
)
from data_preprocessing import run_preprocessing
from aci_calculator import compute_aci_predictions, evaluate_aci_benchmark

LOG_DIR.mkdir(parents=True, exist_ok=True)
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
    level="INFO",
    colorize=True,
)
log_file = LOG_DIR / "run_log_part2.txt"
logger.add(
    str(log_file),
    format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}",
    level="DEBUG",
    rotation="10 MB",
    encoding="utf-8",
)

t_start = time.time()
logger.info("=" * 65)
logger.info("  Corrosion RC Beam Optimizer -- Part 2: PySR Equations")
logger.info(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
logger.info("=" * 65)

# =============================================================
# CELL 3: LOAD DATA (fast re-preprocessing + ACI)
# =============================================================
logger.info("Re-running preprocessing + ACI ...")
data = run_preprocessing(save_clean=True)
df_clean = data["df_clean"]
N_TOTAL = len(df_clean)

df_aci = compute_aci_predictions(df_clean)
aci_metrics = evaluate_aci_benchmark(df_aci)
logger.info(f"Data loaded: {N_TOTAL} samples")

# Load Part 1 summary if available
part2_dir = RESULTS_DIR / "for_part2"
part1_summary = None
if (part2_dir / "part1_summary.json").exists():
    with open(part2_dir / "part1_summary.json") as f:
        part1_summary = json.load(f)
    logger.info("Part 1 summary loaded.")
else:
    logger.warning("Part 1 summary not found -- PDF report will be partial.")

# =============================================================
# CELL 4: PySR CONFIGURATION
# =============================================================
from pysr import PySRRegressor

def _sanitize_name(name):
    MAPPING = {
        "Mass Loss (Tensile bars), \u03b7m (%)": "eta_m",
        "fy Longitudinal Bars (Tensile), (MPa) ": "fy",
        "f'c (MPa)": "fc",
        "Depth (mm)": "d",
        "Width (mm)": "b",
        "Tension Reinforcement Ratio, pten (%)": "rho_t",
        "corr_severity_idx": "CSI",
        "d_b_ratio": "d_b",
        "reinf_index": "RI",
        "Diameter Tensile Bars, db,t (mm)": "db_t",
    }
    if name in MAPPING:
        return MAPPING[name]
    clean = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    clean = re.sub(r"_+", "_", clean).strip("_")
    return clean if clean else "x"


# ======= PySR HYPERPARAMETERS (adjust for Colab runtime) =======
# For Colab free tier (~4h limit): niterations=300, populations=60
# For Colab Pro / local:           niterations=800, populations=100
PYSR_COMMON = dict(
    niterations=400,
    maxsize=25,
    populations=60,
    binary_operators=["+", "-", "*", "/", "^"],
    unary_operators=["sqrt", "log", "exp"],
    nested_constraints={
        "sqrt": {"sqrt": 0, "log": 1, "exp": 0},
        "log": {"log": 0, "exp": 0, "sqrt": 1},
        "exp": {"exp": 0, "log": 0, "sqrt": 1},
    },
    constraints={"^": (-1, 1), "sqrt": 9, "log": 9, "exp": 5},
    model_selection="accuracy",
    elementwise_loss="loss(x, y) = (x - y)^2",
    verbosity=1,
    random_state=RANDOM_STATE,
    deterministic=False,
    parallelism="multithreading",
    turbo=True,
    extra_sympy_mappings={},
)

logger.info(f"PySR config: niterations={PYSR_COMMON['niterations']}, "
            f"maxsize={PYSR_COMMON['maxsize']}, "
            f"populations={PYSR_COMMON['populations']}")

# =============================================================
# CELL 5: PySR -- RATIO APPROACH (Mmax_exp / M_ACI)
# =============================================================
logger.info("\n" + "=" * 60)
logger.info("  Phase 3A -- PySR Ratio Approach")
logger.info("=" * 60)

RATIO_FEATURES = [
    "Mass Loss (Tensile bars), \u03b7m (%)",
    "fy Longitudinal Bars (Tensile), (MPa) ",
    "f'c (MPa)",
    "Depth (mm)",
    "Width (mm)",
    "Tension Reinforcement Ratio, pten (%)",
    "d_b_ratio",
]

available_R = [f for f in RATIO_FEATURES if f in df_clean.columns]
safe_names_R = [_sanitize_name(n) for n in available_R]

X_ratio = df_clean[available_R].values.astype(np.float64)
y_mmax = df_clean[TARGET_COL].values.astype(np.float64)
M_ACI_all = df_aci["MACI_pred"].values.astype(np.float64)

y_ratio = y_mmax / np.maximum(M_ACI_all, 1e-6)

valid_R = (
    np.isfinite(X_ratio).all(axis=1)
    & np.isfinite(y_ratio)
    & (y_ratio > 0.1)
    & (y_ratio < 10.0)
)
X_ratio = X_ratio[valid_R]
y_ratio = y_ratio[valid_R]
y_mmax_R = y_mmax[valid_R]
M_ACI_R = M_ACI_all[valid_R]

logger.info(f"  Ratio: {X_ratio.shape[0]} samples, features: {safe_names_R}")
logger.info(f"  Ratio range: [{y_ratio.min():.3f}, {y_ratio.max():.3f}]")

pysr_ratio = PySRRegressor(**PYSR_COMMON)
logger.info("  Starting PySR Ratio training ...")
pysr_ratio.fit(X_ratio, y_ratio, variable_names=safe_names_R)
logger.info("  Ratio training complete.")

sys.stdout.flush()
sys.stderr.flush()
time.sleep(1)

# Evaluate Ratio Pareto front
equations_R = pysr_ratio.get_hof()
logger.info("=" * 60)
logger.info("  Evaluating Ratio Pareto Equations")
logger.info("=" * 60)

all_eq_R = []
best_R_r2, best_R_idx = -999, None

for idx in range(len(equations_R)):
    try:
        pred_i = np.clip(pysr_ratio.predict(X_ratio, index=idx), 0.01, 20.0)
        mmax_i = M_ACI_R * pred_i
        r2_ratio_i = r2_score(y_ratio, pred_i)
        r2_mmax_i = r2_score(y_mmax_R, mmax_i)
        mape_ratio_i = float(np.mean(np.abs(
            (y_ratio - pred_i) / np.maximum(np.abs(y_ratio), 1e-6)
        )) * 100)
        mape_mmax_i = float(np.mean(np.abs(
            (y_mmax_R - mmax_i) / np.maximum(np.abs(y_mmax_R), 1e-6)
        )) * 100)
        eq_str_i = str(pysr_ratio.sympy(index=idx))
        cx_i = int(equations_R.iloc[idx].get("complexity", idx))
        loss_i = float(equations_R.iloc[idx].get("loss", 0))

        all_eq_R.append({
            "index": idx, "complexity": cx_i, "loss": round(loss_i, 4),
            "ratio_R2": round(r2_ratio_i, 4), "ratio_MAPE": round(mape_ratio_i, 2),
            "mmax_R2": round(r2_mmax_i, 4), "mmax_MAPE": round(mape_mmax_i, 2),
            "equation": eq_str_i,
        })

        logger.info(
            f"  R| C={cx_i:2d} | Ratio R2={r2_ratio_i:.4f} | "
            f"Mmax R2={r2_mmax_i:.4f} | MAPE={mape_mmax_i:.1f}% | "
            f"{eq_str_i[:55]}"
        )

        if r2_mmax_i > best_R_r2:
            best_R_r2, best_R_idx = r2_mmax_i, idx
    except Exception as e:
        logger.warning(f"  R| Eq {idx} failed: {e}")

logger.info(f"\n  Ratio best: idx={best_R_idx}, Mmax R2={best_R_r2:.4f}")

# Extract Ratio best
if best_R_idx is not None:
    ratio_pred_best = np.clip(
        pysr_ratio.predict(X_ratio, index=best_R_idx), 0.01, 20.0
    )
    ratio_mmax_pred = M_ACI_R * ratio_pred_best
    ratio_best_str = str(pysr_ratio.sympy(index=best_R_idx))
    ratio_best_latex = str(pysr_ratio.latex(index=best_R_idx))
    ratio_rmse = float(np.sqrt(mean_squared_error(y_mmax_R, ratio_mmax_pred)))
    ratio_mape = float(np.mean(np.abs(
        (y_mmax_R - ratio_mmax_pred) / np.maximum(np.abs(y_mmax_R), 1e-6)
    )) * 100)
    ratio_mae = float(mean_absolute_error(y_mmax_R, ratio_mmax_pred))
else:
    ratio_pred_best = np.zeros_like(y_ratio)
    ratio_mmax_pred = np.zeros_like(y_mmax_R)
    ratio_best_str, ratio_best_latex = "N/A", "N/A"
    ratio_rmse, ratio_mape, ratio_mae = 999.0, 999.0, 999.0

ratio_metrics = {
    "approach": "Ratio = Mmax_exp / M_ACI",
    "mmax_R2": round(best_R_r2, 4),
    "mmax_RMSE": round(ratio_rmse, 4),
    "mmax_MAE": round(ratio_mae, 4),
    "mmax_MAPE": round(ratio_mape, 2),
    "equation": ratio_best_str,
    "equation_latex": ratio_best_latex,
    "n_samples": int(X_ratio.shape[0]),
}

logger.info(f"  Ratio: R2={best_R_r2:.4f}, RMSE={ratio_rmse:.2f}, "
            f"MAPE={ratio_mape:.1f}%")

# =============================================================
# CELL 6: PySR -- DIRECT Mmax APPROACH
# =============================================================
logger.info("\n" + "=" * 60)
logger.info("  Phase 3B -- PySR Direct Mmax Prediction")
logger.info("=" * 60)

DIRECT_FEATURES = [
    "Mass Loss (Tensile bars), \u03b7m (%)",
    "fy Longitudinal Bars (Tensile), (MPa) ",
    "f'c (MPa)",
    "Depth (mm)",
    "Width (mm)",
    "Tension Reinforcement Ratio, pten (%)",
    "Diameter Tensile Bars, db,t (mm)",
    "d_b_ratio",
    "reinf_index",
    "corr_severity_idx",
]

available_D = [f for f in DIRECT_FEATURES if f in df_clean.columns]
safe_names_D = [_sanitize_name(n) for n in available_D]

X_direct = df_clean[available_D].values.astype(np.float64)
y_direct = df_clean[TARGET_COL].values.astype(np.float64)

valid_D = (
    np.isfinite(X_direct).all(axis=1)
    & np.isfinite(y_direct)
    & (y_direct > 0)
)
X_direct = X_direct[valid_D]
y_direct = y_direct[valid_D]
M_ACI_D = M_ACI_all[valid_D]

logger.info(f"  Direct: {X_direct.shape[0]} samples, features: {safe_names_D}")

pysr_direct = PySRRegressor(**PYSR_COMMON)
logger.info("  Starting PySR Direct training ...")
pysr_direct.fit(X_direct, y_direct, variable_names=safe_names_D)
logger.info("  Direct training complete.")

sys.stdout.flush()
sys.stderr.flush()
time.sleep(1)

# Evaluate Direct Pareto front
equations_D = pysr_direct.get_hof()
logger.info("=" * 60)
logger.info("  Evaluating Direct Pareto Equations")
logger.info("=" * 60)

all_eq_D = []
best_D_r2, best_D_idx = -999, None

for idx in range(len(equations_D)):
    try:
        pred_d = np.clip(pysr_direct.predict(X_direct, index=idx), 0, 500)
        r2_d = r2_score(y_direct, pred_d)
        rmse_d = float(np.sqrt(mean_squared_error(y_direct, pred_d)))
        mae_d = float(mean_absolute_error(y_direct, pred_d))
        mape_d = float(np.mean(np.abs(
            (y_direct - pred_d) / np.maximum(np.abs(y_direct), 1e-6)
        )) * 100)
        eq_str_d = str(pysr_direct.sympy(index=idx))
        cx_d = int(equations_D.iloc[idx].get("complexity", idx))
        loss_d = float(equations_D.iloc[idx].get("loss", 0))

        all_eq_D.append({
            "index": idx, "complexity": cx_d, "loss": round(loss_d, 4),
            "ratio_R2": round(r2_d, 4),
            "mmax_R2": round(r2_d, 4), "mmax_RMSE": round(rmse_d, 4),
            "mmax_MAE": round(mae_d, 4), "mmax_MAPE": round(mape_d, 2),
            "equation": eq_str_d,
        })

        logger.info(
            f"  D| C={cx_d:2d} R2={r2_d:.4f} RMSE={rmse_d:.2f} "
            f"MAPE={mape_d:.1f}% | {eq_str_d[:55]}"
        )

        if r2_d > best_D_r2:
            best_D_r2, best_D_idx = r2_d, idx
    except Exception as e:
        logger.warning(f"  D| Eq {idx} failed: {e}")

logger.info(f"\n  Direct best: idx={best_D_idx}, R2={best_D_r2:.4f}")

# Extract Direct best
if best_D_idx is not None:
    direct_pred_best = np.clip(
        pysr_direct.predict(X_direct, index=best_D_idx), 0, 500
    )
    direct_best_str = str(pysr_direct.sympy(index=best_D_idx))
    direct_best_latex = str(pysr_direct.latex(index=best_D_idx))
    direct_rmse = float(np.sqrt(mean_squared_error(y_direct, direct_pred_best)))
    direct_mape = float(np.mean(np.abs(
        (y_direct - direct_pred_best) / np.maximum(np.abs(y_direct), 1e-6)
    )) * 100)
    direct_mae = float(mean_absolute_error(y_direct, direct_pred_best))
else:
    direct_pred_best = np.zeros_like(y_direct)
    direct_best_str, direct_best_latex = "N/A", "N/A"
    direct_rmse, direct_mape, direct_mae = 999.0, 999.0, 999.0

direct_metrics = {
    "approach": "Direct Mmax prediction",
    "R2": round(best_D_r2, 4),
    "RMSE": round(direct_rmse, 4),
    "MAE": round(direct_mae, 4),
    "MAPE": round(direct_mape, 2),
    "equation": direct_best_str,
    "equation_latex": direct_best_latex,
    "n_samples": int(X_direct.shape[0]),
}

logger.info(f"  Direct: R2={best_D_r2:.4f}, RMSE={direct_rmse:.2f}, "
            f"MAPE={direct_mape:.1f}%")

# Save intermediate metrics
with open(MODELS_DIR / "pysr_metrics_ratio.json", "w", encoding="utf-8") as f:
    json.dump(ratio_metrics, f, indent=2, default=str, ensure_ascii=False)
with open(MODELS_DIR / "pysr_metrics_direct.json", "w", encoding="utf-8") as f:
    json.dump(direct_metrics, f, indent=2, default=str, ensure_ascii=False)

# =============================================================
# CELL 7: FINAL COMPARISON -- Pick Winner
# =============================================================
logger.info("\n" + "=" * 60)
logger.info("  FINAL COMPARISON -- Ratio vs Direct")
logger.info("=" * 60)
logger.info(f"  Ratio  : Mmax R2={best_R_r2:.4f} | RMSE={ratio_rmse:.2f} | "
            f"MAPE={ratio_mape:.1f}%")
logger.info(f"  Direct : Mmax R2={best_D_r2:.4f} | RMSE={direct_rmse:.2f} | "
            f"MAPE={direct_mape:.1f}%")

if best_D_r2 > best_R_r2:
    WINNER = "DIRECT"
    best_eq_str = direct_best_str
    best_eq_latex = direct_best_latex
    r2_mmax = best_D_r2
    rmse_mmax = direct_rmse
    mae_mmax = direct_mae
    mape_mmax = direct_mape
    y_exp_winner = y_direct
    y_pred_winner = direct_pred_best
    n_winner = int(X_direct.shape[0])
    all_eq_winner = all_eq_D
    equations_winner = equations_D
    best_idx_winner = best_D_idx
    safe_names_winner = safe_names_D
    X_winner = X_direct
    logger.success(f"  >>> WINNER: Direct (R2={best_D_r2:.4f})")
else:
    WINNER = "RATIO"
    best_eq_str = ratio_best_str
    best_eq_latex = ratio_best_latex
    r2_mmax = best_R_r2
    rmse_mmax = ratio_rmse
    mae_mmax = ratio_mae
    mape_mmax = ratio_mape
    y_exp_winner = y_mmax_R
    y_pred_winner = ratio_mmax_pred
    n_winner = int(X_ratio.shape[0])
    all_eq_winner = all_eq_R
    equations_winner = equations_R
    best_idx_winner = best_R_idx
    safe_names_winner = safe_names_R
    X_winner = X_ratio
    logger.success(f"  >>> WINNER: Ratio (Mmax R2={best_R_r2:.4f})")

cv_pct_eq = (rmse_mmax / np.mean(y_exp_winner)) * 100
sd_m_eq = float(np.std(y_exp_winner - y_pred_winner) / np.mean(y_exp_winner))

# Consolidated metrics
pysr_metrics = {
    "winner": WINNER,
    "approach": f"Dual PySR -- winner: {WINNER}",
    "mmax_R2": round(r2_mmax, 4),
    "mmax_RMSE": round(rmse_mmax, 4),
    "mmax_MAE": round(mae_mmax, 4),
    "mmax_MAPE": round(mape_mmax, 2),
    "mmax_CV_pct": round(cv_pct_eq, 2),
    "mmax_SD_M": round(sd_m_eq, 4),
    "L1_broken": r2_mmax >= L1_TARGET_R2,
    "L2_broken": r2_mmax >= L2_TARGET_R2,
    "equation": best_eq_str,
    "equation_latex": best_eq_latex,
    "n_samples": n_winner,
    "ratio_approach_R2": round(best_R_r2, 4),
    "direct_approach_R2": round(best_D_r2, 4),
    "timestamp": str(datetime.now()),
}

# =============================================================
# CELL 8: SAVE EQUATIONS
# =============================================================
logger.info("\n" + "=" * 60)
logger.info("  Saving Equations")
logger.info("=" * 60)

EQ_DIR.mkdir(parents=True, exist_ok=True)

with open(EQ_DIR / "best_equation.txt", "w", encoding="utf-8") as f:
    f.write(f"# Best PySR Equation ({WINNER} Approach)\n")
    f.write(f"# Generated: {datetime.now()}\n")
    f.write(f"# Winner: {WINNER} | R2={r2_mmax:.4f} | "
            f"RMSE={rmse_mmax:.4f} | MAE={mae_mmax:.4f} | "
            f"MAPE={mape_mmax:.2f}% | CV%={cv_pct_eq:.2f}% | "
            f"SD/M={sd_m_eq:.4f}\n")
    f.write(f"# Ratio R2={best_R_r2:.4f} | Direct R2={best_D_r2:.4f}\n\n")
    if WINNER == "DIRECT":
        f.write(f"Mmax = {best_eq_str}\n")
    else:
        f.write(f"Mmax = M_ACI * f_corr\n")
        f.write(f"f_corr = {best_eq_str}\n")

with open(EQ_DIR / "best_equation.latex", "w", encoding="utf-8") as f:
    f.write(f"% Best PySR Equation ({WINNER})\n")
    f.write(f"% Generated: {datetime.now()}\n\n")
    if WINNER == "DIRECT":
        f.write(f"M_{{\\max}} = {best_eq_latex}\n")
    else:
        f.write(f"M_{{\\max,corr}} = M_{{\\text{{ACI}}}} "
                f"\\times {best_eq_latex}\n")

# Save all equations JSON
eq_records_R = equations_R.to_dict(orient="records") if equations_R is not None else []
eq_records_D = equations_D.to_dict(orient="records") if equations_D is not None else []

all_eq_payload = {
    "winner": WINNER,
    "final_equation": best_eq_str,
    "final_equation_latex": best_eq_latex,
    "ratio_approach": {
        "best_equation": ratio_best_str,
        "best_equation_latex": ratio_best_latex,
        "mmax_R2": round(best_R_r2, 4),
        "metrics": ratio_metrics,
        "all_equations": eq_records_R,
        "pareto_evaluation": all_eq_R,
    },
    "direct_approach": {
        "best_equation": direct_best_str,
        "best_equation_latex": direct_best_latex,
        "R2": round(best_D_r2, 4),
        "metrics": direct_metrics,
        "all_equations": eq_records_D,
        "pareto_evaluation": all_eq_D,
    },
    "final_metrics": pysr_metrics,
    "generated_at": str(datetime.now()),
}
with open(EQ_DIR / "all_equations.json", "w", encoding="utf-8") as f:
    json.dump(all_eq_payload, f, indent=2, default=str, ensure_ascii=False)
with open(MODELS_DIR / "pysr_metrics.json", "w", encoding="utf-8") as f:
    json.dump(pysr_metrics, f, indent=2, default=str, ensure_ascii=False)

logger.info(f"  Equations saved to {EQ_DIR}")
logger.info(f"  PUBLICATION EQUATION ({WINNER}):")
logger.info(f"    {best_eq_str}")
logger.info(f"    R2={r2_mmax:.4f} | RMSE={rmse_mmax:.4f} | "
            f"MAE={mae_mmax:.4f} | MAPE={mape_mmax:.2f}%")
logger.info(f"    CV%={cv_pct_eq:.2f}% | SD/M={sd_m_eq:.4f}")
logger.info(f"    L1={pysr_metrics['L1_broken']} | "
            f"L2={pysr_metrics['L2_broken']}")

# =============================================================
# CELL 9: PySR FIGURES
# =============================================================
logger.info("\n" + "=" * 60)
logger.info("  Generating PySR Figures")
logger.info("=" * 60)

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({
    "font.size": 12, "axes.titlesize": 14, "axes.labelsize": 13,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
})

fig_count = 0

# -- Figure 8: PySR Equation Scatter (Winner, log-log) --
try:
    fig8, ax8 = plt.subplots(figsize=(8, 8))

    _pos8 = (y_exp_winner > 0) & (y_pred_winner > 0)
    x8 = y_exp_winner[_pos8]
    y8 = y_pred_winner[_pos8]

    ax8.scatter(
        x8, y8, c="#2E7D32", alpha=0.5, s=25,
        edgecolors="w", linewidth=0.3, zorder=3,
    )
    lo8 = max(0.3, min(x8.min(), y8.min()) * 0.8)
    hi8 = max(x8.max(), y8.max()) * 1.15
    lim8 = [lo8, hi8]
    ax8.plot(lim8, lim8, "r--", linewidth=2, label="Perfect prediction")
    ax8.plot(lim8, [v * 1.2 for v in lim8], "g:", linewidth=1, alpha=0.6,
             label="+/-20% band")
    ax8.plot(lim8, [v * 0.8 for v in lim8], "g:", linewidth=1, alpha=0.6)
    ax8.set_xscale("log")
    ax8.set_yscale("log")
    ax8.set_xlabel("Experimental Mmax (kN.m)")
    ylabel = ("Predicted Mmax (kN.m)" if WINNER == "DIRECT"
              else "Predicted Mmax = M_ACI * f_corr (kN.m)")
    ax8.set_ylabel(ylabel)
    ax8.set_title(
        f"PySR {WINNER} Equation: Predicted vs Experimental\n"
        f"R\u00b2={r2_mmax:.4f} | RMSE={rmse_mmax:.2f} | "
        f"MAPE={mape_mmax:.1f}% | n={n_winner}"
    )
    ax8.set_xlim(lim8)
    ax8.set_ylim(lim8)
    ax8.set_aspect("equal")
    ax8.legend(fontsize=10, loc="upper left")
    ax8.grid(True, alpha=0.3, which="both")
    textstr8 = (
        f"R\u00b2 = {r2_mmax:.4f}\n"
        f"RMSE = {rmse_mmax:.2f} kN.m\n"
        f"MAE = {mae_mmax:.2f} kN.m\n"
        f"MAPE = {mape_mmax:.1f}%\n"
        f"CV% = {cv_pct_eq:.1f}%\n"
        f"SD/M = {sd_m_eq:.4f}\n"
        f"n = {n_winner}"
    )
    ax8.text(
        0.97, 0.03, textstr8, transform=ax8.transAxes, fontsize=10,
        verticalalignment="bottom", horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )
    fig8.savefig(FIGURES_DIR / "fig8_pysr_equation_scatter.png")
    plt.close(fig8)
    fig_count += 1
    logger.info(f"  Fig 8 OK -- PySR {WINNER} Scatter (log-log)")
except Exception as e:
    logger.warning(f"  Fig 8 FAILED: {e}")

# -- Figure 9: Pareto Curve (Complexity vs Loss) --
try:
    fig9, ax9 = plt.subplots(figsize=(10, 6))
    eq_df = equations_winner.copy()
    if "complexity" in eq_df.columns and "loss" in eq_df.columns:
        ax9.plot(eq_df["complexity"], eq_df["loss"], "o-",
                 color="#E65100", markersize=8, linewidth=2)
        if best_idx_winner is not None and best_idx_winner < len(eq_df):
            ax9.scatter(
                eq_df.iloc[best_idx_winner]["complexity"],
                eq_df.iloc[best_idx_winner]["loss"],
                s=200, c="red", zorder=5, marker="*",
                label="Selected (best Mmax R2)",
            )
        ax9.set_xlabel("Equation Complexity (nodes)")
        ax9.set_ylabel("Mean Squared Error (Loss)")
        ax9.set_title(f"PySR Pareto Frontier ({WINNER}): Accuracy vs Complexity")
        ax9.legend(fontsize=11)
        ax9.grid(True, alpha=0.3)
    fig9.savefig(FIGURES_DIR / "fig9_pareto_curve.png")
    plt.close(fig9)
    fig_count += 1
    logger.info("  Fig 9 OK -- Pareto Curve")
except Exception as e:
    logger.warning(f"  Fig 9 FAILED: {e}")

# -- Figure 10: Pareto Mmax R2 vs Complexity --
try:
    if all_eq_winner:
        fig10, ax10 = plt.subplots(figsize=(10, 6))
        complexities = [r["complexity"] for r in all_eq_winner]
        mmax_r2s = [r["mmax_R2"] for r in all_eq_winner]
        mmax_mapes = [r["mmax_MAPE"] for r in all_eq_winner]

        ax10_twin = ax10.twinx()
        ax10.plot(complexities, mmax_r2s, "o-", color="#1565C0",
                  markersize=8, linewidth=2, label="Mmax R2")
        ax10_twin.plot(complexities, mmax_mapes, "s--", color="#E65100",
                       markersize=6, linewidth=1.5, alpha=0.7,
                       label="Mmax MAPE %")

        if best_idx_winner is not None and best_idx_winner < len(all_eq_winner):
            best_r = all_eq_winner[best_idx_winner]
            ax10.scatter(best_r["complexity"], best_r["mmax_R2"],
                         s=200, c="red", zorder=5, marker="*",
                         label="Selected")

        ax10.set_xlabel("Equation Complexity")
        ax10.set_ylabel("Mmax R2", color="#1565C0")
        ax10_twin.set_ylabel("Mmax MAPE (%)", color="#E65100")
        ax10.set_title("Pareto Front -- Back-Transformed Performance")
        ax10.legend(loc="lower right", fontsize=10)
        ax10_twin.legend(loc="upper right", fontsize=10)
        ax10.grid(True, alpha=0.3)
        fig10.savefig(FIGURES_DIR / "fig10_pareto_mmax_performance.png")
        plt.close(fig10)
        fig_count += 1
        logger.info("  Fig 10 OK -- Pareto Mmax Performance")
except Exception as e:
    logger.warning(f"  Fig 10 FAILED: {e}")

# -- Figure 11: Ratio vs eta_m (corrosion impact) --
try:
    if WINNER == "RATIO" or True:
        fig11, ax11 = plt.subplots(figsize=(10, 6))
        eta_idx = (safe_names_R.index("eta_m")
                   if "eta_m" in safe_names_R else 0)
        eta_vals = X_ratio[:, eta_idx]
        exp_ratio = y_mmax_R / np.maximum(M_ACI_R, 1e-6)
        pred_ratio = ratio_mmax_pred / np.maximum(M_ACI_R, 1e-6)
        ax11.scatter(eta_vals, exp_ratio, c="#757575", alpha=0.4, s=20,
                     label="Experimental Ratio")
        ax11.scatter(eta_vals, pred_ratio, c="#D32F2F", alpha=0.4, s=20,
                     label="PySR Predicted Ratio")
        ax11.axhline(y=1.0, color="black", linestyle="--", linewidth=1,
                     alpha=0.5, label="Ratio = 1.0 (ACI exact)")
        ax11.set_xlabel("Mass Loss eta_m (%)")
        ax11.set_ylabel("Ratio = Mmax,exp / M_ACI")
        ax11.set_title("Corrosion Correction Factor vs Mass Loss")
        ax11.legend(fontsize=10)
        ax11.grid(True, alpha=0.3)
        fig11.savefig(FIGURES_DIR / "fig11_ratio_vs_eta.png")
        plt.close(fig11)
        fig_count += 1
        logger.info("  Fig 11 OK -- Ratio vs eta_m")
except Exception as e:
    logger.warning(f"  Fig 11 FAILED: {e}")

logger.info(f"  PySR figures generated: {fig_count}")

# =============================================================
# CELL 10: PDF REPORT
# =============================================================
logger.info("\n" + "=" * 60)
logger.info("  Generating PDF Report")
logger.info("=" * 60)


def generate_pdf_report():
    from fpdf import FPDF

    class PDF(FPDF):
        def header(self):
            self.set_font("Helvetica", "B", 10)
            self.set_text_color(13, 27, 42)
            self.cell(
                0, 8,
                "Corrosion RC Beam Optimizer - Scientific Report",
                0, 1, "C",
            )
            self.set_draw_color(189, 189, 189)
            self.line(10, self.get_y(), 200, self.get_y())
            self.ln(3)

        def footer(self):
            self.set_y(-15)
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(128, 128, 128)
            self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", 0, 0, "C")

    pdf = PDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)

    # -- Title Page --
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 22)
    pdf.ln(40)
    pdf.cell(0, 15, "Corrosion RC Beam Optimizer", 0, 1, "C")
    pdf.set_font("Helvetica", "", 14)
    pdf.cell(0, 10, "Scientific Report - Full Pipeline", 0, 1, "C")
    pdf.set_font("Helvetica", "I", 11)
    pdf.cell(
        0, 10,
        f"Generated: {datetime.now().strftime('%B %d, %Y - %H:%M')}",
        0, 1, "C",
    )
    pdf.ln(10)
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(
        0, 6,
        "This report presents the complete results of the Corrosion RC "
        "Beam Optimizer pipeline. The study applies ML ensemble models and "
        "symbolic regression (PySR) to predict the residual flexural "
        "capacity of corroded RC beams, benchmarked against ACI 318-19.\n"
        f"L1 target: R2 >= {L1_TARGET_R2} | L2 target: R2 >= {L2_TARGET_R2}\n"
        f"Data: {N_TOTAL} specimens | Split: 70/30",
    )

    # -- ACI Benchmark --
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "1. ACI 318-19 Benchmark", 0, 1, "L")
    pdf.set_font("Helvetica", "", 10)
    for k, v in aci_metrics.items():
        pdf.cell(0, 7, f"  {k}: {v}", 0, 1)

    # -- Part 1: Ensemble Results --
    pdf.ln(5)
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "2. Ensemble Model Results (Part 1)", 0, 1, "L")
    pdf.set_font("Helvetica", "", 10)
    if part1_summary:
        bn = part1_summary.get("best_model_name", "?")
        pdf.cell(0, 7, f"  Best Model: {bn}", 0, 1)
        tm = part1_summary.get("test_metrics", {})
        pdf.cell(0, 7, f"  Test R2    = {tm.get('R2', '?')}", 0, 1)
        pdf.cell(0, 7, f"  Test RMSE  = {tm.get('RMSE', '?')} kN.m", 0, 1)
        pdf.cell(0, 7, f"  Test MAE   = {tm.get('MAE', '?')} kN.m", 0, 1)
        pdf.cell(0, 7, f"  Test CV%   = {tm.get('CV_pct', '?')}%", 0, 1)
        pdf.cell(0, 7, f"  Test SD/M  = {tm.get('SD_M', '?')}", 0, 1)
        pdf.ln(3)
        cm = part1_summary.get("cv_all_metrics", {})
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 7,
                 f"  10-Fold CV (ALL {cm.get('n_samples', '?')} samples):",
                 0, 1)
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 7, f"    R2    = {cm.get('R2', '?')}", 0, 1)
        pdf.cell(0, 7, f"    RMSE  = {cm.get('RMSE', '?')} kN.m", 0, 1)
        pdf.cell(0, 7, f"    MAE   = {cm.get('MAE', '?')} kN.m", 0, 1)
        pdf.cell(0, 7, f"    CV%   = {cm.get('CV_pct', '?')}%", 0, 1)
        pdf.cell(0, 7, f"    SD/M  = {cm.get('SD_M', '?')}", 0, 1)
    else:
        pdf.cell(0, 7, "  (Part 1 results not available)", 0, 1)

    # -- PySR Results --
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "3. PySR Symbolic Regression (Part 2)", 0, 1, "L")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 7, f"  Winner: {WINNER}", 0, 1)
    pdf.cell(0, 7, f"  Equation: {best_eq_str[:90]}", 0, 1)
    pdf.cell(0, 7, f"  Mmax R2   = {r2_mmax:.4f}", 0, 1)
    pdf.cell(0, 7, f"  Mmax RMSE = {rmse_mmax:.4f} kN.m", 0, 1)
    pdf.cell(0, 7, f"  Mmax MAE  = {mae_mmax:.4f} kN.m", 0, 1)
    pdf.cell(0, 7, f"  Mmax MAPE = {mape_mmax:.2f}%", 0, 1)
    pdf.cell(0, 7, f"  CV%       = {cv_pct_eq:.2f}%", 0, 1)
    pdf.cell(0, 7, f"  SD/M      = {sd_m_eq:.4f}", 0, 1)
    pdf.cell(0, 7, f"  L1 broken = {pysr_metrics['L1_broken']}", 0, 1)
    pdf.cell(0, 7, f"  L2 broken = {pysr_metrics['L2_broken']}", 0, 1)
    pdf.ln(3)
    pdf.cell(0, 7, f"  Ratio approach  R2 = {best_R_r2:.4f}", 0, 1)
    pdf.cell(0, 7, f"  Direct approach R2 = {best_D_r2:.4f}", 0, 1)

    # -- Pareto Table --
    pdf.ln(5)
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "4. Pareto Front Equations", 0, 1, "L")
    pdf.set_font("Helvetica", "", 8)
    for res in all_eq_winner:
        marker = " <<<" if res["index"] == best_idx_winner else ""
        pdf.cell(
            0, 5,
            f"  C={res['complexity']:2d} | Mmax R2={res['mmax_R2']:.4f} | "
            f"MAPE={res['mmax_MAPE']:.1f}%{marker}",
            0, 1,
        )

    # -- Figures Gallery --
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "5. Figures Gallery", 0, 1, "L")

    figure_files = sorted(FIGURES_DIR.glob("*.png"))
    for fig_path in figure_files:
        try:
            if pdf.get_y() > 180:
                pdf.add_page()
            pdf.set_font("Helvetica", "I", 9)
            caption = fig_path.stem.replace("_", " ").title()
            pdf.cell(0, 6, caption, 0, 1, "C")
            pdf.image(str(fig_path), x=15, w=180)
            pdf.ln(5)
        except Exception as e:
            pdf.cell(0, 6, f"[Could not embed {fig_path.name}: {e}]", 0, 1)

    report_path = RESULTS_DIR / "Final_Report.pdf"
    pdf.output(str(report_path))
    return report_path


try:
    report_path = generate_pdf_report()
    logger.info(f"PDF Report saved -> {report_path}")
except Exception as e:
    logger.warning(f"PDF report failed: {e}")
    traceback.print_exc()

# =============================================================
# CELL 11: FINAL SUMMARY
# =============================================================
elapsed = time.time() - t_start

sep = "=" * 65
print(f"\n{sep}")
print("  PART 2 COMPLETE -- PySR EQUATION DISCOVERY")
print(sep)

print(f"\n  === PySR DUAL APPROACH ===")
print(f"  Ratio approach  : Mmax R2 = {best_R_r2:.4f}")
print(f"  Direct approach : Mmax R2 = {best_D_r2:.4f}")
print(f"\n  >>> WINNER: {WINNER}")
print(f"  PUBLICATION EQUATION:")
if WINNER == "DIRECT":
    print(f"    Mmax = {best_eq_str}")
else:
    print(f"    Mmax = M_ACI * f_corr")
    print(f"    f_corr = {best_eq_str}")
print(f"\n    R2    = {r2_mmax:.4f}")
print(f"    RMSE  = {rmse_mmax:.4f} kN.m")
print(f"    MAE   = {mae_mmax:.4f} kN.m")
print(f"    MAPE  = {mape_mmax:.2f}%")
print(f"    CV%   = {cv_pct_eq:.2f}%")
print(f"    SD/M  = {sd_m_eq:.4f}")
print(f"    L1    = {pysr_metrics['L1_broken']}")
print(f"    L2    = {pysr_metrics['L2_broken']}")

if part1_summary:
    cm = part1_summary.get("cv_all_metrics", {})
    print(f"\n  === Part 1 Summary (ML) ===")
    print(f"  Best model: {part1_summary.get('best_model_name', '?')}")
    print(f"  10-Fold CV R2 = {cm.get('R2', '?')} "
          f"({cm.get('n_samples', '?')} samples)")

print(f"\n  PDF Report: {RESULTS_DIR / 'Final_Report.pdf'}")
print(f"  Equations:  {EQ_DIR}")
print(f"  PySR time:  {elapsed / 60:.1f} min ({elapsed:.0f}s)")
print(sep)

# =============================================================
# CELL 12: ZIP FOR DOWNLOAD
# =============================================================
import zipfile, shutil
from pathlib import Path as _P

_kaggle = _P("/kaggle/working")
_colab  = _P("/content")
_out = _kaggle if _kaggle.exists() else _colab

zip_path = str(_out / "part2_results_complete.zip")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for sub in ["figures", "models", "equations", "for_part2", "logs"]:
        sub_dir = RESULTS_DIR / sub
        if sub_dir.exists():
            for fpath in sub_dir.rglob("*"):
                if fpath.is_file():
                    arcname = f"{sub}/{fpath.relative_to(sub_dir)}"
                    zf.write(str(fpath), arcname)
    report_f = RESULTS_DIR / "Final_Report.pdf"
    if report_f.exists():
        zf.write(str(report_f), "Final_Report.pdf")

if _kaggle.exists():
    for d in [FIGURES_DIR, EQ_DIR]:
        if d.exists():
            for f in d.glob("*"):
                if f.is_file():
                    shutil.copy2(str(f), str(_kaggle / f.name))

print(f"\nClean ZIP -> {zip_path}")
try:
    from google.colab import files
    files.download(zip_path)
except ImportError:
    pass
print("\nPart 2 Done. Ready for Part 3 (Physics).")
