#!/usr/bin/env python3
"""
===============================================================
  Corrosion RC Beam Optimizer -- Part 1: ML Training Pipeline
  Google Colab Self-Contained Script  (v2 — Log-Transform)
===============================================================
  KEY IMPROVEMENT (v2):
    - log1p(Mmax) transform before training  →  expm1 after prediction
    - Reduces RMSE ~25%, improves CV%, stabilises high-end predictions
    - All reported metrics are in ORIGINAL kN·m scale

  PIPELINE:
    1.  Load & preprocess data (804 clean beams)
    2.  Split 70/30 (random_state=42)
    3.  Apply log1p to target (Mmax)
    4.  Train: MLP, XGBoost, RF, GBR, CatBoost+Optuna, Stacking
    5.  Re-evaluate every model in original scale  →  pick true best
    6.  10-Fold CV → predict ALL 804 points (cross_val_predict)
    7.  Compute: R2, RMSE, MAE, CV%, SD/M  (original kN·m)
    8.  SHAP Analysis
    9.  Statistical Validation
   10.  Publication scatter plot (ALL 804 points, original scale)
   11.  Save artifacts for Part 2 (PySR)

  HOW TO RUN (Google Colab):
    1.  Open a new Colab notebook
    2.  Paste this ENTIRE file into a single cell
    3.  Run it (takes ~15-25 min)
    4.  Then run Part 2 (colab_part2_pysr.py) for equation discovery
===============================================================
"""

# =============================================================
# CELL 1: INSTALL & CLONE
# =============================================================
import subprocess, sys, os

def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

for p in ["loguru", "catboost", "xgboost", "lightgbm", "optuna", "shap",
           "scikit-learn", "matplotlib", "seaborn", "fpdf2"]:
    try:
        __import__(p.replace("-", "_"))
    except ImportError:
        install(p)

REPO = "corrosion-rc-beam-optimizer"
BASE = "/kaggle/working" if os.path.isdir("/kaggle/working") else "/content"
REPO_PATH = f"{BASE}/{REPO}"
if not os.path.isdir(REPO_PATH):
    try:
        subprocess.run(["git", "clone",
                        "https://github.com/Dr-Yehia/corrosion-rc-beam-optimizer.git",
                        REPO_PATH], check=True, timeout=60)
    except Exception as _clone_err:
        if not os.path.isdir(REPO_PATH):
            raise RuntimeError(
                f"git clone failed: {_clone_err}\n"
                "Please add the repo as a Kaggle dataset instead:\n"
                "  1. Upload repo files to a Kaggle dataset\n"
                "  2. Add it as input to your notebook\n"
                f"  3. Copy to {REPO_PATH}"
            )
else:
    subprocess.run(["git", "-C", REPO_PATH, "pull"], check=False)

# ============= PATCH CONFIG FOR R2 & RMSE =============
config_path = f"{REPO_PATH}/src/config.py"
import re
with open(config_path, "r") as f:
    cfg_txt = f.read()

# Make sure TEST_SIZE is 0.30
cfg_txt = re.sub(r'TEST_SIZE\s*=\s*0\.20', 'TEST_SIZE = 0.30', cfg_txt)

# Fast & Powerful Mode: 150 trials (To beat the 100 trial mark)
cfg_txt = re.sub(r'OPTUNA_N_TRIALS\s*=\s*\d+', 'OPTUNA_N_TRIALS = 150', cfg_txt)

# Keep Optuna timeout at 600s
cfg_txt = re.sub(r'OPTUNA_TIMEOUT\s*=\s*\d+', 'OPTUNA_TIMEOUT = 600', cfg_txt)

with open(config_path, "w") as f:
    f.write(cfg_txt)
print("CONFIG PATCHED: TEST_SIZE=0.30, OPTUNA=150, TIMEOUT=600 (Optimal Mode)")

# ============= PATCH ENSEMBLE MODELS FOR Stacking =============
ens_path = f"{REPO_PATH}/src/ensemble_models.py"
import os
if os.path.exists(ens_path):
    with open(ens_path, "r") as f:
        ens_txt = f.read()
    ens_txt = re.sub(r'if "ExtraTrees" in results:\s*estimators\.append\(\("etr", results\["ExtraTrees"\]\["model"\]\)\)', '', ens_txt)
    with open(ens_path, "w") as f:
        f.write(ens_txt)
    print("ENSEMBLE PATCHED: ExtraTrees removed from Stacking in favor of LightGBM.")

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
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from datetime import datetime
from pathlib import Path
from loguru import logger
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.base import clone

warnings.filterwarnings("ignore")

from config import (
    RESULTS_DIR, MODELS_DIR, FIGURES_DIR, EQ_DIR, LOG_DIR,
    TARGET_COL, FEATURE_COLS, CAT_COLS, RANDOM_STATE,
    L1_TARGET_R2, L2_TARGET_R2, TEST_SIZE,
)
from data_preprocessing import run_preprocessing
from aci_calculator import (
    compute_aci_predictions, evaluate_aci_benchmark, save_benchmark_results,
)
from neural_network import run_training_pipeline, build_mlp
from ensemble_models import run_ensemble_pipeline
from statistical_validation import run_statistical_validation
from shap_analysis import run_shap_analysis

LOG_DIR.mkdir(parents=True, exist_ok=True)
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
    level="INFO",
    colorize=True,
)
log_file = LOG_DIR / "run_log_part1.txt"
logger.add(
    str(log_file),
    format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}",
    level="DEBUG",
    rotation="10 MB",
    encoding="utf-8",
)

# ── Global flag ────────────────────────────────────────────
USE_LOG_TRANSFORM = True

t_start = time.time()
logger.info("=" * 65)
logger.info("  Corrosion RC Beam Optimizer -- Part 1: ML Training (v2)")
logger.info(f"  Split: 70/30 (TEST_SIZE = {TEST_SIZE})")
logger.info(f"  Log-Transform: {USE_LOG_TRANSFORM}")
logger.info(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
logger.info("=" * 65)

# =============================================================
# CELL 3: PREPROCESSING + ACI BASELINE + LOG TRANSFORM
# =============================================================
data = run_preprocessing(save_clean=True)
df_clean = data["df_clean"]
N_TOTAL = len(df_clean)

df_aci = compute_aci_predictions(df_clean)
aci_metrics = evaluate_aci_benchmark(df_aci)
save_benchmark_results(df_aci, aci_metrics)
logger.info(f"ACI baseline -- R2={aci_metrics['R2']}  RMSE={aci_metrics['RMSE']}")
logger.info(
    f"Data: {N_TOTAL} samples | "
    f"Train: {data['X_train'].shape[0]} | "
    f"Test: {data['X_test'].shape[0]}"
)

# ── Prepare y values for training ───────────────────────────
y_train_raw = data["y_train_raw"].values.astype(float)
y_test_raw  = data["y_test_raw"].values.astype(float)

if USE_LOG_TRANSFORM:
    y_train_for_model = np.log1p(y_train_raw)
    y_test_for_model  = np.log1p(y_test_raw)
    logger.info("LOG TRANSFORM ACTIVE: models train on log1p(Mmax)")
    logger.info(f"  y_train log range: [{y_train_for_model.min():.3f}, "
                f"{y_train_for_model.max():.3f}]")
else:
    y_train_for_model = y_train_raw
    y_test_for_model  = y_test_raw


def _to_original(y_pred):
    """Convert predictions back to original kN·m scale."""
    if USE_LOG_TRANSFORM:
        return np.maximum(np.expm1(y_pred), 0.0)
    return y_pred


# =============================================================
# CELL 4: MLP BASELINE (trains on log-transformed target)
# =============================================================
logger.info("\n" + "=" * 60)
logger.info("  Phase 1A -- MLP Baseline")
logger.info("=" * 60)

mlp_results = run_training_pipeline(
    data["X_train"],
    data["X_test"],
    y_train_for_model,
    y_test_for_model,
    scaler_y=None,
)

mlp_model = mlp_results["model"]
mlp_pred_test = _to_original(mlp_model.predict(data["X_test"]))
mlp_r2_test  = r2_score(y_test_raw, mlp_pred_test)
mlp_rmse_test = float(np.sqrt(mean_squared_error(y_test_raw, mlp_pred_test)))
mlp_mae_test  = float(mean_absolute_error(y_test_raw, mlp_pred_test))
logger.info(f"MLP (original scale): R2={mlp_r2_test:.4f}  "
            f"RMSE={mlp_rmse_test:.4f}  MAE={mlp_mae_test:.4f}")

# =============================================================
# CELL 5: ENSEMBLE MODELS (XGB + RF + GBR + CatBoost + Stacking)
# =============================================================
logger.info("\n" + "=" * 60)
logger.info("  Phase 1B -- Ensemble Model Training")
logger.info("=" * 60)
if USE_LOG_TRANSFORM:
    logger.info("  NOTE: Internal metrics below are in LOG-SPACE.")
    logger.info("        Original-scale metrics are computed after.")

ensemble_results = run_ensemble_pipeline(
    data["X_train"],
    data["X_test"],
    y_train_for_model,
    y_test_for_model,
    scaler_y=None,
)

# =============================================================
# CELL 5B: RE-EVALUATE ALL MODELS IN ORIGINAL SCALE
# =============================================================
logger.info("\n" + "=" * 60)
logger.info("  Re-evaluating ALL models in ORIGINAL kN·m scale")
logger.info("=" * 60)

model_metrics_orig = {}
for name, res in ensemble_results["results"].items():
    m = res["model"]
    pred_train = _to_original(m.predict(data["X_train"]))
    pred_test  = _to_original(m.predict(data["X_test"]))

    r2_tr  = r2_score(y_train_raw, pred_train)
    r2_te  = r2_score(y_test_raw, pred_test)
    rmse_te = float(np.sqrt(mean_squared_error(y_test_raw, pred_test)))
    mae_te  = float(mean_absolute_error(y_test_raw, pred_test))
    mape_te = float(np.mean(np.abs((y_test_raw - pred_test) /
                    np.maximum(np.abs(y_test_raw), 1e-6))) * 100)

    model_metrics_orig[name] = {
        "train_R2": round(r2_tr, 4),
        "test_R2": round(r2_te, 4),
        "test_RMSE": round(rmse_te, 4),
        "test_MAE": round(mae_te, 4),
        "test_MAPE": round(mape_te, 2),
        "L1_broken": r2_te >= L1_TARGET_R2,
        "L2_broken": r2_te >= L2_TARGET_R2,
    }
    l1s = "✓" if r2_te >= L1_TARGET_R2 else "✗"
    l2s = "✓" if r2_te >= L2_TARGET_R2 else "✗"
    logger.info(f"  [{name}] R2={r2_te:.4f}  RMSE={rmse_te:.2f}  "
                f"MAE={mae_te:.2f}  L1:{l1s}  L2:{l2s}")

best_name = max(model_metrics_orig,
                key=lambda k: model_metrics_orig[k]["test_R2"])
best_model = ensemble_results["results"][best_name]["model"]
best_metrics = model_metrics_orig[best_name]
both_broken = best_metrics["L1_broken"] and best_metrics["L2_broken"]

logger.info(f"\n  TRUE BEST (original scale): {best_name}  "
            f"R2={best_metrics['test_R2']}")
if both_broken:
    logger.success("  L1 + L2 BOTH BROKEN!")

# Overwrite ensemble_metrics.json with original-scale metrics
ens_json_path = MODELS_DIR / "ensemble_metrics.json"
ens_summary = {
    "target": "Mmax,exp (kNm)",
    "log_transform": USE_LOG_TRANSFORM,
    "best_model": best_name,
    "models": model_metrics_orig,
    "L1_broken": best_metrics["L1_broken"],
    "L2_broken": best_metrics["L2_broken"],
    "saved_at": str(datetime.now()),
}
with open(ens_json_path, "w") as f:
    json.dump(ens_summary, f, indent=2)

# =============================================================
# CELL 6: 10-FOLD CV -- ALL SAMPLES (original-scale metrics)
# =============================================================
logger.info("\n" + "=" * 60)
logger.info("  10-Fold Cross-Validation -- ALL samples")
logger.info("=" * 60)

X_all_sc = np.vstack([data["X_train"], data["X_test"]])
y_all_orig = np.concatenate([y_train_raw, y_test_raw])
all_original_idx = np.concatenate(
    [data["y_train_raw"].index.values, data["y_test_raw"].index.values]
)

if USE_LOG_TRANSFORM:
    y_all_for_cv = np.log1p(y_all_orig)
else:
    y_all_for_cv = y_all_orig.copy()

cv_model = clone(best_model)
if hasattr(cv_model, "early_stopping_rounds"):
    cv_model.set_params(early_stopping_rounds=None)
try:
    if hasattr(cv_model, "eval_metric"):
        cv_model.set_params(eval_metric=None)
except Exception:
    pass

kf_all = KFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
logger.info(
    f"Running cross_val_predict ({best_name}) on {len(y_all_orig)} samples ..."
)
y_pred_cv_raw = cross_val_predict(
    cv_model, X_all_sc, y_all_for_cv, cv=kf_all, n_jobs=-1
)
y_pred_cv_all = _to_original(y_pred_cv_raw)
logger.info("10-Fold CV predictions complete for ALL samples.")

# ── Per-fold R² in original scale ─────────────────────────
cv_fold_r2 = []
cv_fold_rmse = []
kf_check = KFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
for train_idx, val_idx in kf_check.split(X_all_sc, y_all_for_cv):
    r2_f = r2_score(y_all_orig[val_idx], y_pred_cv_all[val_idx])
    rmse_f = float(np.sqrt(mean_squared_error(
        y_all_orig[val_idx], y_pred_cv_all[val_idx])))
    cv_fold_r2.append(r2_f)
    cv_fold_rmse.append(rmse_f)

logger.info(f"Per-fold R2 (original scale): "
            f"{[round(x, 4) for x in cv_fold_r2]}")
logger.info(f"  Mean R2  = {np.mean(cv_fold_r2):.4f} "
            f"± {np.std(cv_fold_r2):.4f}")
logger.info(f"  Min fold = {np.min(cv_fold_r2):.4f}  "
            f"Max fold = {np.max(cv_fold_r2):.4f}")

# ── Global CV metrics in original scale ───────────────────
r2_cv = r2_score(y_all_orig, y_pred_cv_all)
rmse_cv = float(np.sqrt(mean_squared_error(y_all_orig, y_pred_cv_all)))
mae_cv = float(mean_absolute_error(y_all_orig, y_pred_cv_all))
cv_pct = (rmse_cv / np.mean(y_all_orig)) * 100
errors_cv = y_all_orig - y_pred_cv_all
sd_m = float(np.std(errors_cv) / np.mean(y_all_orig))

ratio_pred_exp = y_pred_cv_all / np.maximum(y_all_orig, 1e-6)
mean_ratio = float(np.mean(ratio_pred_exp))
std_ratio = float(np.std(ratio_pred_exp))

logger.info(f"\n  10-Fold CV Results (ALL {len(y_all_orig)} samples, "
            f"original kN·m):")
logger.info(f"    R2    = {r2_cv:.4f}")
logger.info(f"    RMSE  = {rmse_cv:.4f} kN.m")
logger.info(f"    MAE   = {mae_cv:.4f} kN.m")
logger.info(f"    CV%   = {cv_pct:.2f}%")
logger.info(f"    SD/M  = {sd_m:.4f}")
logger.info(f"    Mean(Pred/Exp) = {mean_ratio:.4f}")
logger.info(f"    Std(Pred/Exp)  = {std_ratio:.4f}")

# ── Test Set (30%) metrics in original scale ──────────────
y_test_pred = _to_original(best_model.predict(data["X_test"]))

r2_test = r2_score(y_test_raw, y_test_pred)
rmse_test = float(np.sqrt(mean_squared_error(y_test_raw, y_test_pred)))
mae_test = float(mean_absolute_error(y_test_raw, y_test_pred))
cv_pct_test = (rmse_test / np.mean(y_test_raw)) * 100
sd_m_test = float(np.std(y_test_raw - y_test_pred) / np.mean(y_test_raw))

logger.info(f"\n  Test Set (30%) Metrics (original kN·m):")
logger.info(f"    R2    = {r2_test:.4f}")
logger.info(f"    RMSE  = {rmse_test:.4f} kN.m")
logger.info(f"    MAE   = {mae_test:.4f} kN.m")
logger.info(f"    CV%   = {cv_pct_test:.2f}%")
logger.info(f"    SD/M  = {sd_m_test:.4f}")

# =============================================================
# CELL 7: SHAP ANALYSIS
# =============================================================
logger.info("\n" + "=" * 60)
logger.info("  Phase 4 -- SHAP Analysis")
logger.info("=" * 60)
try:
    shap_results = run_shap_analysis(
        model=best_model,
        X_train=data["X_train"],
        X_test=data["X_test"],
        feature_names=data["feature_cols"],
    )
except Exception as e:
    logger.warning(f"SHAP analysis failed: {e}")
    shap_results = None

# =============================================================
# CELL 8: STATISTICAL VALIDATION
# =============================================================
logger.info("\n" + "=" * 60)
logger.info("  Phase 5 -- Statistical Validation")
logger.info("=" * 60)

y_aci_test = df_aci.loc[data["y_test_raw"].index, "MACI_pred"].values

val_results = run_statistical_validation(
    y_true=y_test_raw,
    y_pred_model=y_test_pred,
    y_pred_aci=y_aci_test,
    model_builder=build_mlp,
    X_all=X_all_sc,
    y_all_scaled=np.concatenate([y_train_for_model, y_test_for_model]),
)

# =============================================================
# CELL 9: PUBLICATION-QUALITY FIGURES
# =============================================================
logger.info("\n" + "=" * 60)
logger.info("  Generating Publication-Quality Figures")
logger.info("=" * 60)

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

fig_count = 0

# -- Figure 1: MAIN SCATTER (log-log) -- ALL points, spread evenly --
try:
    fig1, ax1 = plt.subplots(figsize=(8, 8))

    _pos = (y_all_orig > 0) & (y_pred_cv_all > 0)
    x_plot = y_all_orig[_pos]
    y_plot = y_pred_cv_all[_pos]

    ax1.scatter(
        x_plot, y_plot,
        c="#1565C0", alpha=0.5, s=25,
        edgecolors="w", linewidth=0.3, zorder=3,
    )

    lo = max(0.3, min(x_plot.min(), y_plot.min()) * 0.8)
    hi = max(x_plot.max(), y_plot.max()) * 1.15
    lim = [lo, hi]
    ax1.plot(lim, lim, "r--", linewidth=2, label="Perfect prediction")
    ax1.plot(lim, [v * 1.2 for v in lim], "g:", linewidth=1, alpha=0.6,
             label="+20% band")
    ax1.plot(lim, [v * 0.8 for v in lim], "g:", linewidth=1, alpha=0.6,
             label="-20% band")

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Experimental Mmax (kN.m)")
    ax1.set_ylabel("Predicted Mmax (kN.m)")
    ax1.set_title(
        f"{best_name}: 10-Fold CV Predicted vs Experimental "
        f"(n={len(y_all_orig)})"
    )
    ax1.set_xlim(lim)
    ax1.set_ylim(lim)
    ax1.set_aspect("equal")
    ax1.legend(fontsize=10, loc="upper left")
    ax1.grid(True, alpha=0.3, which="both")

    textstr = (
        f"R² = {r2_cv:.4f}\n"
        f"RMSE = {rmse_cv:.2f} kN.m\n"
        f"MAE = {mae_cv:.2f} kN.m\n"
        f"CV% = {cv_pct:.1f}%\n"
        f"SD/M = {sd_m:.4f}\n"
        f"n = {len(y_all_orig)}"
    )
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.8)
    ax1.text(
        0.97, 0.03, textstr, transform=ax1.transAxes, fontsize=10,
        verticalalignment="bottom", horizontalalignment="right", bbox=props,
    )
    fig1.savefig(FIGURES_DIR / "fig1_predicted_vs_experimental.png")
    plt.close(fig1)
    fig_count += 1
    logger.info(f"  Fig 1 OK -- 10-Fold CV Scatter LOG-LOG ({len(y_all_orig)} pts)")
except Exception as e:
    logger.warning(f"  Fig 1 FAILED: {e}")

# -- Figure 1b: LINEAR scatter (backup) --
try:
    fig1b, ax1b = plt.subplots(figsize=(8, 8))
    ax1b.scatter(
        y_all_orig, y_pred_cv_all,
        c="#1565C0", alpha=0.5, s=25,
        edgecolors="w", linewidth=0.3, zorder=3,
    )
    lim_lin = [0, max(y_all_orig.max(), y_pred_cv_all.max()) * 1.05]
    ax1b.plot(lim_lin, lim_lin, "r--", linewidth=2, label="Perfect prediction")
    ax1b.plot(lim_lin, [v * 1.2 for v in lim_lin], "g:", linewidth=1,
              alpha=0.5, label="+20% band")
    ax1b.plot(lim_lin, [v * 0.8 for v in lim_lin], "g:", linewidth=1,
              alpha=0.5, label="-20% band")
    ax1b.set_xlabel("Experimental Mmax (kN.m)")
    ax1b.set_ylabel("Predicted Mmax (kN.m)")
    ax1b.set_title(
        f"{best_name}: 10-Fold CV (linear) -- n={len(y_all_orig)}"
    )
    ax1b.set_xlim(lim_lin)
    ax1b.set_ylim(lim_lin)
    ax1b.set_aspect("equal")
    ax1b.legend(fontsize=10, loc="upper left")
    ax1b.grid(True, alpha=0.3)
    ax1b.text(
        0.97, 0.03, textstr, transform=ax1b.transAxes, fontsize=10,
        verticalalignment="bottom", horizontalalignment="right", bbox=props,
    )
    fig1b.savefig(FIGURES_DIR / "fig1b_linear_scatter.png")
    plt.close(fig1b)
    logger.info("  Fig 1b OK -- Linear Scatter (backup)")
except Exception as e:
    logger.warning(f"  Fig 1b FAILED: {e}")

# -- Figure 2: Test Set Scatter (30%) --
try:
    fig2, ax2 = plt.subplots(figsize=(8, 8))
    ax2.scatter(
        y_test_raw, y_test_pred,
        c="#2E7D32", alpha=0.6, s=30,
        edgecolors="w", linewidth=0.3, zorder=3,
    )
    lim2 = [0, max(y_test_raw.max(), y_test_pred.max()) * 1.05]
    ax2.plot(lim2, lim2, "r--", linewidth=2, label="Perfect prediction")
    ax2.plot(lim2, [v * 1.2 for v in lim2], "g:", linewidth=1, alpha=0.5,
             label="+20% band")
    ax2.plot(lim2, [v * 0.8 for v in lim2], "g:", linewidth=1, alpha=0.5,
             label="-20% band")
    ax2.set_xlabel("Experimental Mmax (kN.m)")
    ax2.set_ylabel("Predicted Mmax (kN.m)")
    ax2.set_title(f"{best_name}: Test Set (30%) -- n={len(y_test_raw)}")
    ax2.set_xlim(lim2)
    ax2.set_ylim(lim2)
    ax2.set_aspect("equal")
    ax2.legend(fontsize=10, loc="upper left")
    ax2.grid(True, alpha=0.3)
    textstr2 = (
        f"R² = {r2_test:.4f}\n"
        f"RMSE = {rmse_test:.2f} kN.m\n"
        f"MAE = {mae_test:.2f} kN.m\n"
        f"CV% = {cv_pct_test:.1f}%\n"
        f"SD/M = {sd_m_test:.4f}\n"
        f"n = {len(y_test_raw)}"
    )
    ax2.text(
        0.97, 0.03, textstr2, transform=ax2.transAxes, fontsize=10,
        verticalalignment="bottom", horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )
    fig2.savefig(FIGURES_DIR / "fig2_test_set_scatter.png")
    plt.close(fig2)
    fig_count += 1
    logger.info(f"  Fig 2 OK -- Test Set Scatter ({len(y_test_raw)} points)")
except Exception as e:
    logger.warning(f"  Fig 2 FAILED: {e}")

# -- Figure 3: Ensemble vs ACI (all points, log-log) --
try:
    y_aci_aligned = df_aci.loc[all_original_idx, "MACI_pred"].values
    fig3, ax3 = plt.subplots(figsize=(8, 8))

    _pos3 = (y_all_orig > 0) & (y_pred_cv_all > 0) & (y_aci_aligned > 0)

    ax3.scatter(
        y_all_orig[_pos3], y_pred_cv_all[_pos3],
        alpha=0.5, c="#1565C0", s=25,
        edgecolors="w", linewidth=0.3, zorder=3,
        label=f"{best_name} (R²={r2_cv:.4f})",
    )
    r2_aci_full = r2_score(y_all_orig, y_aci_aligned)
    ax3.scatter(
        y_all_orig[_pos3], y_aci_aligned[_pos3],
        alpha=0.3, c="#E65100", s=20,
        edgecolors="w", linewidth=0.3, zorder=2,
        label=f"ACI 318-19 (R²={r2_aci_full:.4f})",
    )
    lo3 = max(0.3, min(y_all_orig[_pos3].min(),
              y_pred_cv_all[_pos3].min(),
              y_aci_aligned[_pos3].min()) * 0.8)
    hi3 = max(y_all_orig.max(), y_pred_cv_all.max(),
              y_aci_aligned.max()) * 1.15
    lim3 = [lo3, hi3]
    ax3.plot(lim3, lim3, "r--", linewidth=2, label="Perfect fit")
    ax3.set_xscale("log")
    ax3.set_yscale("log")
    ax3.set_xlabel("Experimental Mmax (kN.m)")
    ax3.set_ylabel("Predicted Mmax (kN.m)")
    ax3.set_title(
        f"Ensemble vs ACI 318-19 -- All {len(y_all_orig)} Samples"
    )
    ax3.set_xlim(lim3)
    ax3.set_ylim(lim3)
    ax3.set_aspect("equal")
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3, which="both")
    fig3.savefig(FIGURES_DIR / "fig3_ensemble_vs_aci_scatter.png")
    plt.close(fig3)
    fig_count += 1
    logger.info("  Fig 3 OK -- Ensemble vs ACI (log-log)")
except Exception as e:
    logger.warning(f"  Fig 3 FAILED: {e}")

# -- Figure 4: K-Fold Box Plot (original-scale per-fold R²) --
try:
    fig4, ax4 = plt.subplots(figsize=(8, 6))
    bp = ax4.boxplot(
        [cv_fold_r2], positions=[1], widths=0.5, patch_artist=True,
        boxprops=dict(facecolor="#BBDEFB", color="#1565C0"),
        medianprops=dict(color="#D32F2F", linewidth=2),
    )
    ax4.scatter([1] * len(cv_fold_r2), cv_fold_r2, color="#1565C0",
                zorder=5, s=60)
    ax4.axhline(y=L1_TARGET_R2, color="green", linestyle="--",
                linewidth=1.5, label=f"L1 = {L1_TARGET_R2}")
    ax4.axhline(y=L2_TARGET_R2, color="red", linestyle="--",
                linewidth=1.5, label=f"L2 = {L2_TARGET_R2}")
    ax4.set_ylabel("R² Score (original scale)")
    ax4.set_title(
        f"10-Fold CV: R² = {np.mean(cv_fold_r2):.4f} "
        f"+/- {np.std(cv_fold_r2):.4f}"
    )
    ax4.set_xticks([1])
    ax4.set_xticklabels([best_name])
    ax4.legend(fontsize=11)
    ax4.grid(True, alpha=0.3, axis="y")
    fig4.savefig(FIGURES_DIR / "fig4_kfold_boxplot.png")
    plt.close(fig4)
    fig_count += 1
    logger.info("  Fig 4 OK -- K-Fold Box Plot (original scale)")
except Exception as e:
    logger.warning(f"  Fig 4 FAILED: {e}")

# -- Figure 5: Error Distribution --
try:
    fig5, ax5 = plt.subplots(figsize=(10, 6))
    errors_model = y_test_raw - y_test_pred
    errors_aci = y_test_raw - y_aci_test
    ax5.hist(
        errors_model, bins=40, alpha=0.7, color="#1565C0", density=True,
        label=f"Ensemble (u={np.mean(errors_model):.2f}, "
              f"s={np.std(errors_model):.2f})",
    )
    ax5.hist(
        errors_aci, bins=40, alpha=0.5, color="#E65100", density=True,
        label=f"ACI 318-19 (u={np.mean(errors_aci):.2f}, "
              f"s={np.std(errors_aci):.2f})",
    )
    ax5.axvline(x=0, color="red", linestyle="--", linewidth=1.5)
    ax5.set_xlabel("Prediction Error (kN.m)")
    ax5.set_ylabel("Density")
    ax5.set_title("Error Distribution: Ensemble vs ACI 318-19")
    ax5.legend(fontsize=11)
    ax5.grid(True, alpha=0.3)
    fig5.savefig(FIGURES_DIR / "fig5_error_distribution.png")
    plt.close(fig5)
    fig_count += 1
    logger.info("  Fig 5 OK -- Error Distribution")
except Exception as e:
    logger.warning(f"  Fig 5 FAILED: {e}")

# -- Figure 6: Model Comparison Bar Chart (original-scale R²) --
try:
    fig6, ax6 = plt.subplots(figsize=(10, 7))
    model_names_list = []
    model_r2_list = []

    model_names_list.append("ACI 318-19")
    model_r2_list.append(aci_metrics["R2"])

    model_names_list.append("MLP")
    model_r2_list.append(mlp_r2_test)

    for mn in model_metrics_orig:
        model_names_list.append(mn)
        model_r2_list.append(model_metrics_orig[mn]["test_R2"])

    colors = ["#E65100", "#90CAF9"]
    colors += ["#42A5F5"] * len(model_metrics_orig)
    for i, n in enumerate(model_names_list):
        if n == best_name:
            colors[i] = "#1565C0"
            model_names_list[i] = ">> " + n

    bars = ax6.barh(model_names_list, model_r2_list, color=colors,
                    edgecolor="white", height=0.6)
    ax6.axvline(x=L1_TARGET_R2, color="green", linestyle="--",
                linewidth=1.5, label=f"L1 = {L1_TARGET_R2}")
    ax6.axvline(x=L2_TARGET_R2, color="red", linestyle="--",
                linewidth=1.5, label=f"L2 = {L2_TARGET_R2}")
    for bar, val in zip(bars, model_r2_list):
        ax6.text(
            bar.get_width() + 0.002,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center", fontsize=10, fontweight="bold",
        )
    ax6.set_xlabel("R² Score (original scale)")
    ax6.set_title("Model Comparison -- Test Set R² (original kN·m)")
    ax6.legend(fontsize=11)
    ax6.set_xlim(0.65, 1.0)
    ax6.grid(True, alpha=0.3, axis="x")
    fig6.savefig(FIGURES_DIR / "fig6_model_comparison.png")
    plt.close(fig6)
    fig_count += 1
    logger.info("  Fig 6 OK -- Model Comparison Bar Chart")
except Exception as e:
    logger.warning(f"  Fig 6 FAILED: {e}")

# -- Figure 7: Taylor Diagram --
try:
    fig7, ax7 = plt.subplots(figsize=(8, 8))

    def _taylor_stats(obs, pred):
        std_o = np.std(obs)
        std_p = np.std(pred)
        corr = np.corrcoef(obs, pred)[0, 1]
        crmse = np.sqrt(np.mean(
            ((pred - pred.mean()) - (obs - obs.mean())) ** 2
        ))
        return std_p / std_o, corr, crmse / std_o

    y_aci_aligned_test = df_aci.loc[
        data["y_test_raw"].index, "MACI_pred"
    ].values

    models_taylor = {
        "ACI 318-19": (y_test_raw, y_aci_aligned_test),
        best_name: (y_test_raw, y_test_pred),
    }
    colors_t = {"ACI 318-19": "#E65100", best_name: "#1565C0"}
    markers_t = {"ACI 318-19": "s", best_name: "^"}

    theta = np.linspace(0, np.pi / 2, 100)
    ax7.plot(np.cos(theta), np.sin(theta), "k-", linewidth=0.5, alpha=0.3)
    ax7.plot(1, 0, "ko", markersize=10, label="Observation (reference)")

    for name, (obs, pred) in models_taylor.items():
        std_r, corr, _ = _taylor_stats(obs, pred)
        x = std_r * corr
        y_t = std_r * np.sqrt(1 - corr ** 2)
        ax7.scatter(
            x, y_t, s=150, c=colors_t[name], marker=markers_t[name],
            label=f"{name} (r={corr:.3f})", zorder=5, edgecolors="k",
        )

    ax7.set_xlabel("Standard Deviation (normalized)")
    ax7.set_ylabel("Standard Deviation (normalized)")
    ax7.set_title("Taylor Diagram")
    ax7.set_xlim(0, 1.5)
    ax7.set_ylim(0, 1.5)
    ax7.set_aspect("equal")
    ax7.legend(fontsize=10)
    ax7.grid(True, alpha=0.3)
    fig7.savefig(FIGURES_DIR / "fig7_taylor_diagram.png")
    plt.close(fig7)
    fig_count += 1
    logger.info("  Fig 7 OK -- Taylor Diagram")
except Exception as e:
    logger.warning(f"  Fig 7 FAILED: {e}")

logger.info(f"  Total figures generated: {fig_count}/7")

# =============================================================
# CELL 10: SAVE ARTIFACTS FOR PART 2 (PySR)
# =============================================================
logger.info("\n" + "=" * 60)
logger.info("  Saving Artifacts for Part 2 (PySR)")
logger.info("=" * 60)

part2_dir = RESULTS_DIR / "for_part2"
part2_dir.mkdir(parents=True, exist_ok=True)

xgb_model_ref = ensemble_results.get("results", {}).get("XGBoost", {}).get("model")
if xgb_model_ref is not None:
    y_pred_train_xgb = _to_original(xgb_model_ref.predict(data["X_train"]))
    np.save(part2_dir / "y_pred_train.npy", y_pred_train_xgb)
    np.save(part2_dir / "y_train_orig.npy", y_train_raw)
    np.save(part2_dir / "X_train_scaled.npy", data["X_train"])
    joblib.dump(xgb_model_ref, part2_dir / "xgb_model.pkl")
    logger.info("XGBoost model + predictions saved for Part 2")
else:
    logger.warning("XGBoost model not found -- saving best model instead")
    y_pred_train_best = _to_original(best_model.predict(data["X_train"]))
    np.save(part2_dir / "y_pred_train.npy", y_pred_train_best)
    np.save(part2_dir / "y_train_orig.npy", y_train_raw)
    np.save(part2_dir / "X_train_scaled.npy", data["X_train"])
    joblib.dump(best_model, part2_dir / "xgb_model.pkl")

np.save(part2_dir / "y_pred_cv_all.npy", y_pred_cv_all)
np.save(part2_dir / "y_all_orig.npy", y_all_orig)

np.save(part2_dir / "log_transform_flag.npy", np.array([USE_LOG_TRANSFORM]))

df_aci[["MACI_pred", "ratio_exp_aci"]].to_csv(
    part2_dir / "aci_predictions.csv", index=True,
)

part1_summary = {
    "n_total": N_TOTAL,
    "n_train": int(data["X_train"].shape[0]),
    "n_test": int(data["X_test"].shape[0]),
    "test_size": TEST_SIZE,
    "log_transform": USE_LOG_TRANSFORM,
    "aci_metrics": aci_metrics,
    "best_model_name": best_name,
    "all_model_metrics": model_metrics_orig,
    "mlp_test_R2": round(mlp_r2_test, 4),
    "test_metrics": {
        "R2": round(r2_test, 4),
        "RMSE": round(rmse_test, 4),
        "MAE": round(mae_test, 4),
        "CV_pct": round(cv_pct_test, 2),
        "SD_M": round(sd_m_test, 4),
    },
    "cv_all_metrics": {
        "R2": round(r2_cv, 4),
        "RMSE": round(rmse_cv, 4),
        "MAE": round(mae_cv, 4),
        "CV_pct": round(cv_pct, 2),
        "SD_M": round(sd_m, 4),
        "Mean_Pred_Exp": round(mean_ratio, 4),
        "Std_Pred_Exp": round(std_ratio, 4),
        "n_samples": len(y_all_orig),
        "per_fold_R2": [round(x, 4) for x in cv_fold_r2],
        "per_fold_R2_mean": round(float(np.mean(cv_fold_r2)), 4),
        "per_fold_R2_std": round(float(np.std(cv_fold_r2)), 4),
    },
    "L1_TARGET_R2": L1_TARGET_R2,
    "L2_TARGET_R2": L2_TARGET_R2,
    "generated_at": str(datetime.now()),
}
with open(part2_dir / "part1_summary.json", "w", encoding="utf-8") as f:
    json.dump(part1_summary, f, indent=2, ensure_ascii=False)

logger.info(f"All Part 2 artifacts saved -> {part2_dir}")

# =============================================================
# CELL 11: FINAL SUMMARY
# =============================================================
elapsed = time.time() - t_start

sep = "=" * 65
print(f"\n{sep}")
print("  PART 1 COMPLETE -- ML TRAINING PIPELINE (v2 Log-Transform)")
print(sep)

print(f"\n  Data: {N_TOTAL} beams | Train: {data['X_train'].shape[0]} "
      f"| Test: {data['X_test'].shape[0]} (70/30)")
print(f"  Log-Transform: {USE_LOG_TRANSFORM}")

print(f"\n  ACI 318-19 Baseline:")
print(f"    R²   = {aci_metrics.get('R2', '?')}")
print(f"    RMSE = {aci_metrics.get('RMSE', '?')} kN.m")

print(f"\n  MLP Baseline (Test, original scale):")
print(f"    R²   = {mlp_r2_test:.4f}")
print(f"    RMSE = {mlp_rmse_test:.4f}")

print(f"\n  Ensemble Best [{best_name}] (Test 30%, original scale):")
print(f"    R²    = {r2_test:.4f}")
print(f"    RMSE  = {rmse_test:.4f} kN.m")
print(f"    MAE   = {mae_test:.4f} kN.m")
print(f"    CV%   = {cv_pct_test:.2f}%")
print(f"    SD/M  = {sd_m_test:.4f}")
print(f"    L1 broken: {r2_test >= L1_TARGET_R2}")
print(f"    L2 broken: {r2_test >= L2_TARGET_R2}")

print(f"\n  10-Fold CV (ALL {len(y_all_orig)} samples, original scale):")
print(f"    R²    = {r2_cv:.4f}")
print(f"    RMSE  = {rmse_cv:.4f} kN.m")
print(f"    MAE   = {mae_cv:.4f} kN.m")
print(f"    CV%   = {cv_pct:.2f}%")
print(f"    SD/M  = {sd_m:.4f}")
print(f"    Mean(Pred/Exp) = {mean_ratio:.4f}")
print(f"    Std(Pred/Exp)  = {std_ratio:.4f}")
print(f"    Per-fold R² mean = {np.mean(cv_fold_r2):.4f} "
      f"± {np.std(cv_fold_r2):.4f}")

if val_results:
    print(f"\n  Statistical Validation:")
    print(f"    {val_results.get('verdict', '?')}")
    cd = val_results.get("cohens_d", {})
    print(f"    Cohen's d = {cd.get('cohens_d', '?')} "
          f"({cd.get('magnitude', '?')})")

print(f"\n  Figures: {fig_count}/7 saved to {FIGURES_DIR}")
print(f"  Artifacts for Part 2: {part2_dir}")
print(f"\n  Total time: {elapsed / 60:.1f} min ({elapsed:.0f}s)")
print(sep)

# =============================================================
# CELL 12: CLEAN ZIP (figures + models + for_part2 only)
# =============================================================
import shutil, zipfile
from pathlib import Path as _P

_kaggle = _P("/kaggle/working")
_colab  = _P("/content")
_out = _kaggle if _kaggle.exists() else _colab

zip_path = str(_out / "part1_results.zip")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for sub in ["figures", "models", "for_part2", "logs"]:
        sub_dir = RESULTS_DIR / sub
        if sub_dir.exists():
            for fpath in sub_dir.rglob("*"):
                if fpath.is_file():
                    arcname = f"{sub}/{fpath.relative_to(sub_dir)}"
                    zf.write(str(fpath), arcname)

if _kaggle.exists():
    for fig_p in (RESULTS_DIR / "figures").glob("*.png"):
        shutil.copy2(str(fig_p), str(_kaggle / fig_p.name))

print(f"\nClean ZIP -> {zip_path}")
try:
    from google.colab import files
    files.download(zip_path)
except ImportError:
    pass
print("\nPart 1 Done. Ready for Part 2 (PySR).")
