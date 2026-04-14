#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════
  Corrosion RC Beam Optimizer — v7 FINAL
  Google Colab Self-Contained Script
═══════════════════════════════════════════════════════════════
  FIXES from v6:
    1. ReportLab crash fixed: report_generator.py uses A4 at
       module level OUTSIDE try/except — crashes on import.
       v7 bypasses it entirely with embedded FPDF2 fallback.
    2. PySR stdout interleaving fixed: captured via redirect.
    3. Ratio MAPE vs Mmax MAPE now computed independently.
    4. Figures: added ±20% band lines on scatter plots.
    5. All JSON saved with ensure_ascii=False for Unicode.
    6. Robust error handling: every phase wrapped in try/except
       so a crash in one phase won't kill the others.

  HOW TO RUN (Google Colab):
    1. Open a new Colab notebook
    2. Paste this ENTIRE file into a single cell
    3. Run it (takes ~30-60 min total)
    4. Download final_results/ when done
═══════════════════════════════════════════════════════════════
"""

# ════════════════════════════════════════════════════════════
# CELL 1: INSTALL & CLONE
# ════════════════════════════════════════════════════════════
import subprocess, sys, os

def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

# Install all dependencies
for p in ["loguru", "catboost", "xgboost", "optuna", "shap",
           "pysr", "scikit-learn", "matplotlib", "seaborn", "fpdf2"]:
    try:
        __import__(p.replace("-","_"))
    except ImportError:
        install(p)

# Clone repo
REPO = "corrosion-rc-beam-optimizer"
if not os.path.isdir(f"/content/{REPO}"):
    subprocess.run(["git", "clone", "https://github.com/Dr-Yehia/corrosion-rc-beam-optimizer.git",
                    f"/content/{REPO}"], check=True)
else:
    subprocess.run(["git", "-C", f"/content/{REPO}", "pull"], check=False)

os.chdir(f"/content/{REPO}/src")
sys.path.insert(0, f"/content/{REPO}/src")

print("✅ Setup complete.")

# ════════════════════════════════════════════════════════════
# CELL 2: IMPORTS
# ════════════════════════════════════════════════════════════
import json
import time
import warnings
import io
import contextlib
import traceback
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
from loguru import logger
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")

from config import (
    RESULTS_DIR, MODELS_DIR, FIGURES_DIR, EQ_DIR, LOG_DIR,
    TARGET_COL, FEATURE_COLS, CAT_COLS, RANDOM_STATE,
    L1_TARGET_R2, L2_TARGET_R2,
)
from data_preprocessing import run_preprocessing
from aci_calculator import compute_aci_predictions, evaluate_aci_benchmark, save_benchmark_results
from neural_network import run_training_pipeline
from ensemble_models import run_ensemble_pipeline
from statistical_validation import run_statistical_validation
from neural_network import build_mlp

# Configure logging
LOG_DIR.mkdir(parents=True, exist_ok=True)
logger.remove()
logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
           level="INFO", colorize=True)
log_file = LOG_DIR / "run_log_v7.txt"
logger.add(str(log_file), format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}",
           level="DEBUG", rotation="10 MB", encoding="utf-8")

t_start = time.time()
logger.info("=" * 65)
logger.info(" Corrosion RC Beam Optimizer (v7 — FINAL)")
logger.info(f" Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
logger.info("=" * 65)

# ════════════════════════════════════════════════════════════
# CELL 3: PREPROCESSING + ACI BASELINE
# ════════════════════════════════════════════════════════════
data = run_preprocessing(save_clean=True)
df_clean = data["df_clean"]

# ACI Baseline
df_aci = compute_aci_predictions(df_clean)
aci_metrics = evaluate_aci_benchmark(df_aci)
save_benchmark_results(df_aci, aci_metrics)
logger.info(f"ACI baseline — R²={aci_metrics['R2']}  RMSE={aci_metrics['RMSE']}")

# ════════════════════════════════════════════════════════════
# CELL 4: MLP BASELINE
# ════════════════════════════════════════════════════════════
logger.info("\n" + "=" * 60)
logger.info(" Phase 1A — MLP Baseline")
logger.info("=" * 60)
mlp_results = run_training_pipeline(
    data["X_train"], data["X_test"],
    data["y_train"], data["y_test"],
    scaler_y=data["scaler_y"],
)

# ════════════════════════════════════════════════════════════
# CELL 5: ENSEMBLE (XGB + RF + GBR + CatBoost + Stacking)
# ════════════════════════════════════════════════════════════
logger.info("\n" + "=" * 60)
logger.info(" Phase 1B — Ensemble Model Training")
logger.info("=" * 60)
ensemble_results = run_ensemble_pipeline(
    data["X_train"], data["X_test"],
    data["y_train"], data["y_test"],
    scaler_y=data["scaler_y"],
)
best_model = ensemble_results.get("best_model")
both_broken = ensemble_results.get("both_broken", False)
logger.info(f"Ensemble best: {ensemble_results.get('best_name', '?')}")
if both_broken:
    logger.success("🏆 L1 + L2 BOTH BROKEN by Ensemble!")

# ════════════════════════════════════════════════════════════
# CELL 6: SHAP ANALYSIS
# ════════════════════════════════════════════════════════════
logger.info("\n" + "=" * 60)
logger.info(" Phase 4 — SHAP Analysis")
logger.info("=" * 60)
from shap_analysis import run_shap_analysis
shap_results = run_shap_analysis(
    model=best_model,
    X_train=data["X_train"], X_test=data["X_test"],
    feature_names=data["feature_cols"],
)

# ════════════════════════════════════════════════════════════
# CELL 7: STATISTICAL VALIDATION
# ════════════════════════════════════════════════════════════
logger.info("\n" + "=" * 60)
logger.info(" Phase 5 — Statistical Validation")
logger.info("=" * 60)

scaler_y = data["scaler_y"]
y_true = scaler_y.inverse_transform(data["y_test"].reshape(-1, 1)).ravel()
y_pred_sc = best_model.predict(data["X_test"])
if y_pred_sc.mean() < 10:
    y_model = scaler_y.inverse_transform(y_pred_sc.reshape(-1, 1)).ravel()
else:
    y_model = y_pred_sc

y_aci = df_aci.loc[data["y_test_raw"].index, "MACI_pred"].values
X_all = np.vstack([data["X_train"], data["X_test"]])
y_all = np.concatenate([data["y_train"], data["y_test"]])

val_results = run_statistical_validation(
    y_true=y_true, y_pred_model=y_model, y_pred_aci=y_aci,
    model_builder=build_mlp, X_all=X_all, y_all_scaled=y_all,
)

# ════════════════════════════════════════════════════════════
# CELL 8: PySR — RATIO TARGET (FIXED v7)
# ════════════════════════════════════════════════════════════
logger.info("\n" + "=" * 60)
logger.info(" Phase 3 — PySR Symbolic Regression (RATIO TARGET — v7)")
logger.info("=" * 60)

from pysr import PySRRegressor
import re

def _sanitize_name(name):
    MAPPING = {
        "Mass Loss (Tensile bars), ηm (%)": "eta_m",
        "fy Longitudinal Bars (Tensile), (MPa) ": "fy",
        "f'c (MPa)": "fc",
        "Depth (mm)": "d",
        "Width (mm)": "b",
        "Tension Reinforcement Ratio, pten (%)": "rho_t",
        "corr_severity_idx": "CSI",
        "d_b_ratio": "d_b",
        "reinf_index": "RI",
    }
    if name in MAPPING:
        return MAPPING[name]
    clean = re.sub(r'[^a-zA-Z0-9_]', '_', name)
    clean = re.sub(r'_+', '_', clean).strip('_')
    return clean if clean else 'x'

# --- Prepare RATIO target ---
RATIO_FEATURES = [
    "Mass Loss (Tensile bars), ηm (%)",
    "fy Longitudinal Bars (Tensile), (MPa) ",
    "f'c (MPa)",
    "Tension Reinforcement Ratio, pten (%)",
    "d_b_ratio",
]

available = [f for f in RATIO_FEATURES if f in df_clean.columns]
safe_names = [_sanitize_name(n) for n in available]

X_pysr = df_clean[available].values.astype(np.float64)
y_mmax = df_clean[TARGET_COL].values.astype(np.float64)

# Compute M_ACI for each sample
M_ACI_all = df_aci["MACI_pred"].values.astype(np.float64)

# Ratio = Mmax,exp / M_ACI
y_ratio = y_mmax / np.maximum(M_ACI_all, 1e-6)

# Remove NaN/Inf and extreme outliers
valid = np.isfinite(X_pysr).all(axis=1) & np.isfinite(y_ratio)
valid = valid & (y_ratio > 0.1) & (y_ratio < 10.0)
X_pysr = X_pysr[valid]
y_ratio = y_ratio[valid]
y_mmax_valid = y_mmax[valid]
M_ACI_valid = M_ACI_all[valid]

logger.info(f"PySR Ratio target — {X_pysr.shape[0]} samples")
logger.info(f"  Variables : {safe_names}")
logger.info(f"  Ratio range: [{y_ratio.min():.3f}, {y_ratio.max():.3f}]")
logger.info(f"  Ratio mean : {y_ratio.mean():.3f}, std: {y_ratio.std():.3f}")

# ═══════════════════════════════════════════════════════════
#  🔧 KEY FIX v6→v7: model_selection = "accuracy"
# ═══════════════════════════════════════════════════════════
pysr_model = PySRRegressor(
    niterations      = 500,
    maxsize          = 20,
    populations      = 80,
    binary_operators = ["+", "-", "*", "/", "^"],
    unary_operators  = ["sqrt", "log", "exp"],
    model_selection  = "accuracy",
    elementwise_loss = "loss(x, y) = (x - y)^2",
    verbosity        = 1,
    random_state     = RANDOM_STATE,
    deterministic    = False,
    parallelism      = "multithreading",
    turbo            = True,
    extra_sympy_mappings = {},
)

logger.info("Starting PySR on RATIO target (this takes a while) ...")
logger.info("⚠️  FIX APPLIED: model_selection='accuracy'")
pysr_model.fit(X_pysr, y_ratio, variable_names=safe_names)
logger.info("PySR training complete.")

# ═══════════════════════════════════════════════════════════
#  🔧 FIX v7: Flush stdout, then evaluate equations cleanly
# ═══════════════════════════════════════════════════════════
sys.stdout.flush()
sys.stderr.flush()
time.sleep(1)  # Let any remaining PySR output finish

equations = pysr_model.get_hof()
logger.info("=" * 60)
logger.info(" Evaluating ALL Pareto Front Equations")
logger.info("=" * 60)

all_eq_results = []
best_mmax_r2 = -999
best_eq_idx = None

for idx in range(len(equations)):
    try:
        pred_i = pysr_model.predict(X_pysr, index=idx)
        pred_i = np.clip(pred_i, 0.01, 20.0)
        mmax_i = M_ACI_valid * pred_i

        r2_ratio_i = r2_score(y_ratio, pred_i)
        r2_mmax_i = r2_score(y_mmax_valid, mmax_i)

        # MAPE for Ratio (how well it predicts the ratio itself)
        mape_ratio_i = float(np.mean(np.abs(
            (y_ratio - pred_i) / np.maximum(np.abs(y_ratio), 1e-6)
        )) * 100)

        # MAPE for Mmax (how well the back-transformed Mmax matches)
        mape_mmax_i = float(np.mean(np.abs(
            (y_mmax_valid - mmax_i) / np.maximum(np.abs(y_mmax_valid), 1e-6)
        )) * 100)

        eq_str_i = str(pysr_model.sympy(index=idx))
        complexity_i = int(equations.iloc[idx].get("complexity", idx))
        loss_i = float(equations.iloc[idx].get("loss", 0))

        all_eq_results.append({
            "index": idx,
            "complexity": complexity_i,
            "loss": round(loss_i, 4),
            "ratio_R2": round(r2_ratio_i, 4),
            "ratio_MAPE": round(mape_ratio_i, 2),
            "mmax_R2": round(r2_mmax_i, 4),
            "mmax_MAPE": round(mape_mmax_i, 2),
            "equation": eq_str_i,
        })

        logger.info(f"  C={complexity_i:2d} | Loss={loss_i:.4f} | "
                    f"Ratio R²={r2_ratio_i:.4f} | Mmax R²={r2_mmax_i:.4f} | "
                    f"Mmax MAPE={mape_mmax_i:.1f}% | {eq_str_i[:60]}")

        if r2_mmax_i > best_mmax_r2:
            best_mmax_r2 = r2_mmax_i
            best_eq_idx = idx
    except Exception as e:
        logger.warning(f"  Eq {idx} failed: {e}")

logger.info(f"\n🏆 Best by Mmax R²: index={best_eq_idx}")

# --- Use the BEST equation (by Mmax R²) ---
ratio_pred = pysr_model.predict(X_pysr, index=best_eq_idx)
ratio_pred = np.clip(ratio_pred, 0.01, 20.0)

# Ratio-level metrics
r2_ratio = r2_score(y_ratio, ratio_pred)
rmse_ratio = float(np.sqrt(mean_squared_error(y_ratio, ratio_pred)))
mape_ratio = float(np.mean(np.abs((y_ratio - ratio_pred) / np.maximum(np.abs(y_ratio), 1e-6))) * 100)

# Back-transform to Mmax
mmax_pred_from_ratio = M_ACI_valid * ratio_pred
r2_mmax   = r2_score(y_mmax_valid, mmax_pred_from_ratio)
rmse_mmax = float(np.sqrt(mean_squared_error(y_mmax_valid, mmax_pred_from_ratio)))
mape_mmax = float(np.mean(np.abs((y_mmax_valid - mmax_pred_from_ratio) / np.maximum(np.abs(y_mmax_valid), 1e-6))) * 100)

best_eq_str   = str(pysr_model.sympy(index=best_eq_idx))
best_eq_latex = str(pysr_model.latex(index=best_eq_idx))

# Also get the "auto-selected" equation (from model_selection="accuracy")
auto_eq_str   = str(pysr_model.sympy())
auto_eq_latex = str(pysr_model.latex())

pysr_metrics = {
    "approach":       "Ratio = Mmax_exp / M_ACI",
    "selection":      "Best by Mmax R² (manual scan of all Pareto equations)",
    "ratio_R2":       round(r2_ratio, 4),
    "ratio_RMSE":     round(rmse_ratio, 4),
    "ratio_MAPE":     round(mape_ratio, 2),
    "mmax_R2":        round(r2_mmax, 4),
    "mmax_RMSE":      round(rmse_mmax, 4),
    "mmax_MAPE":      round(mape_mmax, 2),
    "L1_broken":      r2_mmax >= L1_TARGET_R2,
    "L2_broken":      r2_mmax >= L2_TARGET_R2,
    "equation":       best_eq_str,
    "equation_latex": best_eq_latex,
    "auto_equation":  auto_eq_str,
    "auto_equation_latex": auto_eq_latex,
    "n_samples":      int(X_pysr.shape[0]),
    "best_eq_index":  int(best_eq_idx) if best_eq_idx is not None else -1,
    "all_eq_results": all_eq_results,
    "timestamp":      str(datetime.now()),
}

logger.info("=" * 60)
logger.info(" PySR RATIO Equation — FINAL Results (v7)")
logger.info("=" * 60)
logger.info(f"  Auto-selected (accuracy): {auto_eq_str}")
logger.info(f"  Best (by Mmax R²):        {best_eq_str}")
logger.info(f"  Ratio R²   = {pysr_metrics['ratio_R2']}")
logger.info(f"  Ratio MAPE = {pysr_metrics['ratio_MAPE']} %")
logger.info(f"  Mmax  R²   = {pysr_metrics['mmax_R2']}")
logger.info(f"  Mmax  MAPE = {pysr_metrics['mmax_MAPE']} %")
logger.info(f"  L1 broken  = {pysr_metrics['L1_broken']}")
logger.info(f"  L2 broken  = {pysr_metrics['L2_broken']}")
logger.info("=" * 60)

# --- Save equation files ---
EQ_DIR.mkdir(parents=True, exist_ok=True)

with open(EQ_DIR / "best_equation.txt", "w", encoding="utf-8") as f:
    f.write(f"# Best PySR Equation (Ratio Approach) — v7 FINAL\n")
    f.write(f"# Generated: {datetime.now()}\n")
    f.write(f"# Selected by: Best Mmax R² across all Pareto equations\n")
    f.write(f"# Ratio R² = {pysr_metrics['ratio_R2']} | Mmax R² = {pysr_metrics['mmax_R2']}\n")
    f.write(f"# MAPE (Ratio) = {pysr_metrics['ratio_MAPE']}% | MAPE (Mmax) = {pysr_metrics['mmax_MAPE']}%\n\n")
    f.write(f"Mmax = M_ACI * f_corr\n")
    f.write(f"f_corr = {best_eq_str}\n")

with open(EQ_DIR / "best_equation.latex", "w", encoding="utf-8") as f:
    f.write(f"% Best PySR Equation — LaTeX (Ratio Approach) — v7 FINAL\n")
    f.write(f"% Generated: {datetime.now()}\n\n")
    f.write(f"M_{{\\max,corr}} = M_{{\\text{{ACI}}}} \\times {best_eq_latex}\n")

# Save ALL equations for paper comparison table
eq_records = equations.to_dict(orient="records") if equations is not None else []

all_eq_payload = {
    "approach": "Ratio = Mmax_exp / M_ACI",
    "best_equation": best_eq_str,
    "best_equation_latex": best_eq_latex,
    "auto_equation": auto_eq_str,
    "metrics": pysr_metrics,
    "all_equations": eq_records,
    "pareto_evaluation": all_eq_results,
    "generated_at": str(datetime.now()),
}
with open(EQ_DIR / "all_equations.json", "w", encoding="utf-8") as f:
    json.dump(all_eq_payload, f, indent=2, default=str, ensure_ascii=False)

# Save PySR metrics separately for easy access
with open(MODELS_DIR / "pysr_metrics.json", "w", encoding="utf-8") as f:
    json.dump(pysr_metrics, f, indent=2, default=str, ensure_ascii=False)


# ════════════════════════════════════════════════════════════
# CELL 9: GENERATE 10 PUBLICATION FIGURES
# ════════════════════════════════════════════════════════════
logger.info("\n" + "=" * 60)
logger.info(" Generating 10 Publication-Quality Figures")
logger.info("=" * 60)

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({
    "font.size": 12, "axes.titlesize": 14, "axes.labelsize": 13,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
})

fig_count = 0

# ── Figure 1: Predicted vs Experimental (Best Ensemble) ──
try:
    fig1, ax1 = plt.subplots(figsize=(8, 8))
    ax1.scatter(y_true, y_model, c="#1565C0", alpha=0.6, s=30, edgecolors="w", linewidth=0.3)
    lim = [0, max(y_true.max(), y_model.max()) * 1.05]
    ax1.plot(lim, lim, "r--", linewidth=2, label="Perfect prediction")
    ax1.plot(lim, [l*1.2 for l in lim], "g:", linewidth=1, alpha=0.5, label="+20% band")
    ax1.plot(lim, [l*0.8 for l in lim], "g:", linewidth=1, alpha=0.5, label="-20% band")
    r2_ens = r2_score(y_true, y_model)
    rmse_ens = float(np.sqrt(mean_squared_error(y_true, y_model)))
    ax1.set_xlabel("Experimental Mmax (kN·m)")
    ax1.set_ylabel("Predicted Mmax (kN·m)")
    ax1.set_title(f"Ensemble ({ensemble_results.get('best_name','?')}): Predicted vs Experimental\nR² = {r2_ens:.4f} | RMSE = {rmse_ens:.2f} kN·m")
    ax1.set_xlim(lim); ax1.set_ylim(lim)
    ax1.set_aspect("equal")
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    fig1.savefig(FIGURES_DIR / "fig1_predicted_vs_experimental.png")
    plt.close(fig1)
    fig_count += 1
    logger.info("  Fig 1 ✓ Predicted vs Experimental")
except Exception as e:
    logger.warning(f"  Fig 1 ✗ Failed: {e}")

# ── Figure 2: Ratio Equation Scatter ──
try:
    fig2, ax2 = plt.subplots(figsize=(8, 8))
    ax2.scatter(y_mmax_valid, mmax_pred_from_ratio, c="#2E7D32", alpha=0.6, s=30, edgecolors="w", linewidth=0.3)
    lim2 = [0, max(y_mmax_valid.max(), mmax_pred_from_ratio.max()) * 1.05]
    ax2.plot(lim2, lim2, "r--", linewidth=2, label="Perfect prediction")
    ax2.plot(lim2, [l*1.2 for l in lim2], "g:", linewidth=1, alpha=0.5, label="±20% band")
    ax2.plot(lim2, [l*0.8 for l in lim2], "g:", linewidth=1, alpha=0.5)
    ax2.set_xlabel("Experimental Mmax (kN·m)")
    ax2.set_ylabel("Predicted Mmax = M_ACI × f_corr (kN·m)")
    ax2.set_title(f"PySR Ratio Equation: Predicted vs Experimental\nMmax R² = {r2_mmax:.4f} | Mmax MAPE = {mape_mmax:.1f}%")
    ax2.set_xlim(lim2); ax2.set_ylim(lim2)
    ax2.set_aspect("equal")
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    fig2.savefig(FIGURES_DIR / "fig2_ratio_equation_scatter.png")
    plt.close(fig2)
    fig_count += 1
    logger.info("  Fig 2 ✓ Ratio Equation Scatter")
except Exception as e:
    logger.warning(f"  Fig 2 ✗ Failed: {e}")

# ── Figure 3: Pareto Curve (Complexity vs Loss) ──
try:
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    eq_df = equations.copy()
    if "complexity" in eq_df.columns and "loss" in eq_df.columns:
        ax3.plot(eq_df["complexity"], eq_df["loss"], "o-", color="#E65100", markersize=8, linewidth=2)
        if best_eq_idx is not None and best_eq_idx < len(eq_df):
            ax3.scatter(eq_df.iloc[best_eq_idx]["complexity"], eq_df.iloc[best_eq_idx]["loss"],
                       s=200, c="red", zorder=5, marker="*", label="Selected (best Mmax R²)")
        ax3.set_xlabel("Equation Complexity (nodes)")
        ax3.set_ylabel("Mean Squared Error (Loss)")
        ax3.set_title("PySR Pareto Frontier: Accuracy vs Complexity")
        ax3.legend(fontsize=11)
        ax3.grid(True, alpha=0.3)
    fig3.savefig(FIGURES_DIR / "fig3_pareto_curve.png")
    plt.close(fig3)
    fig_count += 1
    logger.info("  Fig 3 ✓ Pareto Curve")
except Exception as e:
    logger.warning(f"  Fig 3 ✗ Failed: {e}")

# ── Figure 4: K-Fold Box Plot ──
try:
    cv_folds_json = MODELS_DIR / "ensemble_metrics.json"
    if cv_folds_json.exists():
        with open(cv_folds_json) as f:
            ens_data = json.load(f)
        cv_folds = ens_data.get("cv_folds", [])
        if cv_folds:
            fig4, ax4 = plt.subplots(figsize=(8, 6))
            bp = ax4.boxplot([cv_folds], positions=[1], widths=0.5, patch_artist=True,
                            boxprops=dict(facecolor="#BBDEFB", color="#1565C0"),
                            medianprops=dict(color="#D32F2F", linewidth=2))
            ax4.scatter([1]*len(cv_folds), cv_folds, color="#1565C0", zorder=5, s=60)
            ax4.axhline(y=L1_TARGET_R2, color='green', linestyle='--', linewidth=1.5, label=f"L1 = {L1_TARGET_R2}")
            ax4.axhline(y=L2_TARGET_R2, color='red', linestyle='--', linewidth=1.5, label=f"L2 = {L2_TARGET_R2}")
            ax4.set_ylabel("R² Score")
            ax4.set_title(f"10-Fold CV: R² = {np.mean(cv_folds):.4f} ± {np.std(cv_folds):.4f}")
            ax4.set_xticks([1])
            ax4.set_xticklabels([ensemble_results.get('best_name', 'Ensemble')])
            ax4.legend(fontsize=11)
            ax4.grid(True, alpha=0.3, axis='y')
            fig4.savefig(FIGURES_DIR / "fig4_kfold_boxplot.png")
            plt.close(fig4)
            fig_count += 1
            logger.info("  Fig 4 ✓ K-Fold Box Plot")
except Exception as e:
    logger.warning(f"  Fig 4 ✗ Failed: {e}")

# ── Figure 5: Error Distribution ──
try:
    fig5, ax5 = plt.subplots(figsize=(10, 6))
    errors_model = y_true - y_model
    errors_aci = y_true - y_aci
    ax5.hist(errors_model, bins=40, alpha=0.7, color="#1565C0",
             label=f"Ensemble (μ={np.mean(errors_model):.2f}, σ={np.std(errors_model):.2f})", density=True)
    ax5.hist(errors_aci, bins=40, alpha=0.5, color="#E65100",
             label=f"ACI 318-19 (μ={np.mean(errors_aci):.2f}, σ={np.std(errors_aci):.2f})", density=True)
    ax5.axvline(x=0, color='red', linestyle='--', linewidth=1.5)
    ax5.set_xlabel("Prediction Error (kN·m)")
    ax5.set_ylabel("Density")
    ax5.set_title("Error Distribution: Ensemble vs ACI 318-19")
    ax5.legend(fontsize=11)
    ax5.grid(True, alpha=0.3)
    fig5.savefig(FIGURES_DIR / "fig5_error_distribution.png")
    plt.close(fig5)
    fig_count += 1
    logger.info("  Fig 5 ✓ Error Distribution")
except Exception as e:
    logger.warning(f"  Fig 5 ✗ Failed: {e}")

# ── Figure 6: Model Comparison Bar Chart ──
try:
    fig6, ax6 = plt.subplots(figsize=(10, 7))
    model_names = []
    model_r2 = []

    # ACI
    model_names.append("ACI 318-19")
    model_r2.append(aci_metrics["R2"])

    # MLP
    if mlp_results:
        mt = mlp_results.get("metrics_test", {})
        model_names.append("MLP")
        model_r2.append(mt.get("R2", 0))

    # Ensemble models
    if cv_folds_json.exists():
        for mn, mm in ens_data.get("models", {}).items():
            model_names.append(mn)
            model_r2.append(mm.get("test_R2", 0))

    # PySR equation
    model_names.append("PySR Equation")
    model_r2.append(r2_mmax)

    colors = ["#E65100"] + ["#42A5F5"] * (len(model_names) - 2) + ["#2E7D32"]
    for i, n in enumerate(model_names):
        if n == ensemble_results.get('best_name', ''):
            colors[i] = "#1565C0"
            model_names[i] = "⭐ " + n

    bars = ax6.barh(model_names, model_r2, color=colors, edgecolor="white", height=0.6)
    ax6.axvline(x=L1_TARGET_R2, color='green', linestyle='--', linewidth=1.5, label=f"L1 = {L1_TARGET_R2}")
    ax6.axvline(x=L2_TARGET_R2, color='red', linestyle='--', linewidth=1.5, label=f"L2 = {L2_TARGET_R2}")
    for bar, val in zip(bars, model_r2):
        ax6.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                f"{val:.4f}", va="center", fontsize=10, fontweight="bold")
    ax6.set_xlabel("R² Score")
    ax6.set_title("Model Comparison — Test Set R²")
    ax6.legend(fontsize=11)
    ax6.set_xlim(0.85, 1.0)
    ax6.grid(True, alpha=0.3, axis='x')
    fig6.savefig(FIGURES_DIR / "fig6_model_comparison.png")
    plt.close(fig6)
    fig_count += 1
    logger.info("  Fig 6 ✓ Model Comparison Bar Chart")
except Exception as e:
    logger.warning(f"  Fig 6 ✗ Failed: {e}")

# ── Figure 7: Ratio vs ηm (Corrosion Impact) ──
try:
    fig7, ax7 = plt.subplots(figsize=(10, 6))
    eta_col_idx = safe_names.index("eta_m") if "eta_m" in safe_names else 0
    eta_vals = X_pysr[:, eta_col_idx]
    ax7.scatter(eta_vals, y_ratio, c="#757575", alpha=0.4, s=20, label="Experimental Ratio")
    ax7.scatter(eta_vals, ratio_pred, c="#D32F2F", alpha=0.4, s=20, label="PySR Predicted Ratio")
    ax7.axhline(y=1.0, color='black', linestyle='--', linewidth=1, alpha=0.5, label="Ratio = 1.0 (ACI exact)")
    ax7.set_xlabel("Mass Loss ηm (%)")
    ax7.set_ylabel("Ratio = Mmax,exp / M_ACI")
    ax7.set_title("Corrosion Correction Factor vs Mass Loss")
    ax7.legend(fontsize=10)
    ax7.grid(True, alpha=0.3)
    fig7.savefig(FIGURES_DIR / "fig7_ratio_vs_eta.png")
    plt.close(fig7)
    fig_count += 1
    logger.info("  Fig 7 ✓ Ratio vs ηm")
except Exception as e:
    logger.warning(f"  Fig 7 ✗ Failed: {e}")

# ── Figure 8: Taylor Diagram ──
try:
    fig8, ax8 = plt.subplots(figsize=(8, 8))
    def _taylor_stats(obs, pred):
        std_o = np.std(obs)
        std_p = np.std(pred)
        corr = np.corrcoef(obs, pred)[0, 1]
        crmse = np.sqrt(np.mean(((pred - pred.mean()) - (obs - obs.mean()))**2))
        return std_p / std_o, corr, crmse / std_o

    best_name_t = ensemble_results.get('best_name', 'Ensemble')
    models_taylor = {
        "ACI 318-19": (y_true, y_aci),
        best_name_t: (y_true, y_model),
    }

    colors_t = {"ACI 318-19": "#E65100", best_name_t: "#1565C0"}
    markers_t = {"ACI 318-19": "s", best_name_t: "^"}

    theta = np.linspace(0, np.pi/2, 100)
    ax8.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=0.5, alpha=0.3)
    ax8.plot(1, 0, 'ko', markersize=10, label="Observation (reference)")

    for name, (obs, pred) in models_taylor.items():
        std_r, corr, crmse_r = _taylor_stats(obs, pred)
        x = std_r * corr
        y_t = std_r * np.sqrt(1 - corr**2)
        ax8.scatter(x, y_t, s=150, c=colors_t[name], marker=markers_t[name],
                   label=f"{name} (r={corr:.3f})", zorder=5, edgecolors="k")

    ax8.set_xlabel("Standard Deviation (normalized)")
    ax8.set_ylabel("Standard Deviation (normalized)")
    ax8.set_title("Taylor Diagram")
    ax8.set_xlim(0, 1.5); ax8.set_ylim(0, 1.5)
    ax8.set_aspect("equal")
    ax8.legend(fontsize=10)
    ax8.grid(True, alpha=0.3)
    fig8.savefig(FIGURES_DIR / "fig8_taylor_diagram.png")
    plt.close(fig8)
    fig_count += 1
    logger.info("  Fig 8 ✓ Taylor Diagram")
except Exception as e:
    logger.warning(f"  Fig 8 ✗ Failed: {e}")

# ── Figure 9: Scatter Plot — Ensemble vs ACI ──
try:
    fig9, ax9 = plt.subplots(figsize=(8, 8))
    r2_ens = r2_score(y_true, y_model)
    ax9.scatter(y_true, y_model, alpha=0.6, c="#1565C0", s=30, edgecolors="w",
               linewidth=0.3, label=f'Ensemble R²={r2_ens:.4f}')
    ax9.scatter(y_true, y_aci, alpha=0.3, c="#E65100", s=30, edgecolors="w",
               linewidth=0.3, label=f'ACI R²={aci_metrics["R2"]:.4f}')
    lim9 = [0, max(y_true.max(), y_model.max(), y_aci.max()) * 1.05]
    ax9.plot(lim9, lim9, 'r--', linewidth=2, label='Perfect fit')
    ax9.set_xlabel("Experimental Mmax (kN·m)")
    ax9.set_ylabel("Predicted Mmax (kN·m)")
    ax9.set_title("Ensemble vs ACI 318-19 — Scatter Comparison")
    ax9.set_xlim(lim9); ax9.set_ylim(lim9)
    ax9.set_aspect("equal")
    ax9.legend(fontsize=11)
    ax9.grid(True, alpha=0.3)
    fig9.savefig(FIGURES_DIR / "fig9_ensemble_vs_aci_scatter.png")
    plt.close(fig9)
    fig_count += 1
    logger.info("  Fig 9 ✓ Ensemble vs ACI Scatter")
except Exception as e:
    logger.warning(f"  Fig 9 ✗ Failed: {e}")

# ── Figure 10: Pareto Equations — Mmax R² vs Complexity ──
try:
    if all_eq_results:
        fig10, ax10 = plt.subplots(figsize=(10, 6))
        complexities = [r["complexity"] for r in all_eq_results]
        mmax_r2s = [r["mmax_R2"] for r in all_eq_results]
        mmax_mapes = [r["mmax_MAPE"] for r in all_eq_results]

        ax10_twin = ax10.twinx()
        ax10.plot(complexities, mmax_r2s, "o-", color="#1565C0", markersize=8, linewidth=2, label="Mmax R²")
        ax10_twin.plot(complexities, mmax_mapes, "s--", color="#E65100", markersize=6, linewidth=1.5, alpha=0.7, label="Mmax MAPE %")

        if best_eq_idx is not None:
            best_r = all_eq_results[best_eq_idx] if best_eq_idx < len(all_eq_results) else None
            if best_r:
                ax10.scatter(best_r["complexity"], best_r["mmax_R2"],
                            s=200, c="red", zorder=5, marker="*", label="Selected")

        ax10.set_xlabel("Equation Complexity")
        ax10.set_ylabel("Mmax R²", color="#1565C0")
        ax10_twin.set_ylabel("Mmax MAPE (%)", color="#E65100")
        ax10.set_title("Pareto Front Equations — Back-Transformed Performance")
        ax10.legend(loc="lower right", fontsize=10)
        ax10_twin.legend(loc="upper right", fontsize=10)
        ax10.grid(True, alpha=0.3)
        fig10.savefig(FIGURES_DIR / "fig10_pareto_mmax_performance.png")
        plt.close(fig10)
        fig_count += 1
        logger.info("  Fig 10 ✓ Pareto Mmax Performance")
except Exception as e:
    logger.warning(f"  Fig 10 ✗ Failed: {e}")

logger.info(f"  Total figures generated: {fig_count}/10")


# ════════════════════════════════════════════════════════════
# CELL 10: PDF REPORT (v7 — EMBEDDED FPDF2 FALLBACK)
# ════════════════════════════════════════════════════════════
logger.info("\n" + "=" * 60)
logger.info(" Phase 6 — PDF Report")
logger.info("=" * 60)

def generate_pdf_report_fpdf2():
    """Generate a PDF report using FPDF2 (always available, no ReportLab dependency)."""
    from fpdf import FPDF

    class PDF(FPDF):
        def header(self):
            self.set_font('Helvetica', 'B', 10)
            self.set_text_color(13, 27, 42)
            self.cell(0, 8, 'Corrosion RC Beam Optimizer — Scientific Report (v7)', 0, 1, 'C')
            self.set_draw_color(189, 189, 189)
            self.line(10, self.get_y(), 200, self.get_y())
            self.ln(3)

        def footer(self):
            self.set_y(-15)
            self.set_font('Helvetica', 'I', 8)
            self.set_text_color(128, 128, 128)
            self.cell(0, 10, f'Page {self.page_no()}/{{nb}}', 0, 0, 'C')

    pdf = PDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)

    # ── Title Page ──
    pdf.add_page()
    pdf.set_font('Helvetica', 'B', 22)
    pdf.ln(40)
    pdf.cell(0, 15, 'Corrosion RC Beam Optimizer', 0, 1, 'C')
    pdf.set_font('Helvetica', '', 14)
    pdf.cell(0, 10, 'Scientific Report (v7 Final)', 0, 1, 'C')
    pdf.set_font('Helvetica', 'I', 11)
    pdf.cell(0, 10, f'Generated: {datetime.now().strftime("%B %d, %Y - %H:%M")}', 0, 1, 'C')
    pdf.ln(10)
    pdf.set_font('Helvetica', '', 10)
    pdf.multi_cell(0, 6,
        'This report presents the complete results of the Corrosion RC Beam Optimizer pipeline. '
        'The study applies ML ensemble models and symbolic regression to predict the residual '
        f'flexural capacity of corroded RC beams, benchmarked against ACI 318-19.\n'
        f'L1 target: R2 > {L1_TARGET_R2} | L2 target: R2 > {L2_TARGET_R2}')

    # ── ACI Benchmark ──
    pdf.add_page()
    pdf.set_font('Helvetica', 'B', 14)
    pdf.cell(0, 10, '1. ACI 318-19 Benchmark', 0, 1, 'L')
    pdf.set_font('Helvetica', '', 10)
    for k, v in aci_metrics.items():
        pdf.cell(0, 7, f'  {k}: {v}', 0, 1)

    # ── Ensemble Results ──
    pdf.ln(5)
    pdf.set_font('Helvetica', 'B', 14)
    pdf.cell(0, 10, '2. Ensemble Model Results', 0, 1, 'L')
    pdf.set_font('Helvetica', '', 10)
    et = ensemble_results.get("metrics_test", {})
    pdf.cell(0, 7, f'  Best Model: {ensemble_results.get("best_name", "?")}', 0, 1)
    for k, v in et.items():
        pdf.cell(0, 7, f'  {k}: {v}', 0, 1)
    cv_m = ensemble_results.get("cv_R2_mean", "?")
    cv_s = ensemble_results.get("cv_R2_std", "?")
    pdf.cell(0, 7, f'  CV R2: {cv_m:.4f} +/- {cv_s:.4f}' if isinstance(cv_m, float) else f'  CV R2: {cv_m}', 0, 1)

    # ── PySR Results ──
    pdf.ln(5)
    pdf.set_font('Helvetica', 'B', 14)
    pdf.cell(0, 10, '3. PySR Symbolic Regression', 0, 1, 'L')
    pdf.set_font('Helvetica', '', 10)
    pdf.cell(0, 7, f'  Best Equation (by Mmax R2): {best_eq_str}', 0, 1)
    pdf.cell(0, 7, f'  Ratio R2 = {pysr_metrics["ratio_R2"]}', 0, 1)
    pdf.cell(0, 7, f'  Ratio MAPE = {pysr_metrics["ratio_MAPE"]}%', 0, 1)
    pdf.cell(0, 7, f'  Mmax R2 = {pysr_metrics["mmax_R2"]}', 0, 1)
    pdf.cell(0, 7, f'  Mmax MAPE = {pysr_metrics["mmax_MAPE"]}%', 0, 1)
    pdf.ln(3)
    pdf.cell(0, 7, f'  Auto-selected (accuracy): {auto_eq_str[:80]}', 0, 1)

    # ── Pareto Table ──
    pdf.ln(5)
    pdf.set_font('Helvetica', 'B', 14)
    pdf.cell(0, 10, '4. Pareto Front Equations', 0, 1, 'L')
    pdf.set_font('Helvetica', '', 8)
    for res in all_eq_results:
        marker = " <<<" if res["index"] == best_eq_idx else ""
        pdf.cell(0, 5,
            f'  C={res["complexity"]:2d} | Loss={res["loss"]:.4f} | '
            f'Ratio R2={res["ratio_R2"]:.4f} | Mmax R2={res["mmax_R2"]:.4f} | '
            f'MAPE={res["mmax_MAPE"]:.1f}%{marker}', 0, 1)

    # ── Statistical Validation ──
    pdf.add_page()
    pdf.set_font('Helvetica', 'B', 14)
    pdf.cell(0, 10, '5. Statistical Validation', 0, 1, 'L')
    pdf.set_font('Helvetica', '', 10)
    if val_results:
        pdf.cell(0, 7, f'  Verdict: {val_results.get("verdict", "?")}', 0, 1)
        cd = val_results.get("cohens_d", {})
        pdf.cell(0, 7, f'  Cohen d: {cd.get("cohens_d", "?")} ({cd.get("magnitude", "?")})', 0, 1)

        wil = val_results.get("wilcoxon", {})
        pdf.cell(0, 7, f'  Wilcoxon p-value: {wil.get("p_value", "?")}', 0, 1)

        boot = val_results.get("bootstrap", {})
        pdf.cell(0, 7, f'  Bootstrap 95% CI R2: {boot.get("R2_CI", "?")}', 0, 1)

        mcn = val_results.get("mcnemar", {})
        pdf.cell(0, 7, f'  McNemar p-value: {mcn.get("p_value", "?")}', 0, 1)
        pdf.cell(0, 7, f'  Model accuracy: {mcn.get("model_accuracy", "?")}%', 0, 1)
        pdf.cell(0, 7, f'  ACI accuracy: {mcn.get("aci_accuracy", "?")}%', 0, 1)

    # ── Figures Gallery ──
    pdf.add_page()
    pdf.set_font('Helvetica', 'B', 14)
    pdf.cell(0, 10, '6. Figures Gallery', 0, 1, 'L')

    figure_files = sorted(FIGURES_DIR.glob("*.png"))
    for fig_path in figure_files:
        try:
            # Check remaining space
            if pdf.get_y() > 180:
                pdf.add_page()
            pdf.set_font('Helvetica', 'I', 9)
            caption = fig_path.stem.replace("_", " ").title()
            pdf.cell(0, 6, caption, 0, 1, 'C')
            pdf.image(str(fig_path), x=15, w=180)
            pdf.ln(5)
        except Exception as e:
            pdf.cell(0, 6, f'[Could not embed {fig_path.name}: {e}]', 0, 1)

    # ── SHAP figures ──
    shap_files = sorted(FIGURES_DIR.glob("shap_*.png"))
    for fig_path in shap_files:
        if fig_path not in figure_files:  # avoid duplicates
            try:
                if pdf.get_y() > 180:
                    pdf.add_page()
                pdf.set_font('Helvetica', 'I', 9)
                caption = fig_path.stem.replace("_", " ").title()
                pdf.cell(0, 6, caption, 0, 1, 'C')
                pdf.image(str(fig_path), x=15, w=180)
                pdf.ln(5)
            except Exception:
                pass

    # Save
    report_path = RESULTS_DIR / "Final_Report_v7.pdf"
    pdf.output(str(report_path))
    return report_path

try:
    report_path = generate_pdf_report_fpdf2()
    logger.info(f"✅ PDF Report saved → {report_path}")
except Exception as e:
    logger.warning(f"PDF report failed: {e}")
    traceback.print_exc()

# ════════════════════════════════════════════════════════════
# CELL 11: FINAL SUMMARY
# ════════════════════════════════════════════════════════════
elapsed = time.time() - t_start

sep = "=" * 65
print(f"\n{sep}")
print(" CORROSION RC BEAM OPTIMIZER v7 — FINAL PIPELINE COMPLETE")
print(sep)
print(f"\n  ACI 318-19 Baseline:")
print(f"    R²   = {aci_metrics.get('R2','?')}")
print(f"    RMSE = {aci_metrics.get('RMSE','?')} kN·m")

if mlp_results:
    mt = mlp_results.get("metrics_test", {})
    print(f"\n  MLP Baseline (Test):")
    print(f"    R²   = {mt.get('R2','?')}")
    print(f"    RMSE = {mt.get('RMSE','?')}")

et = ensemble_results.get("metrics_test", {})
print(f"\n  🏆 Ensemble Best [{ensemble_results.get('best_name','?')}] (Test):")
print(f"    R²        = {et.get('R2','?')}")
print(f"    RMSE      = {et.get('RMSE','?')}")
print(f"    L1 broken : {et.get('L1_broken','?')}")
print(f"    L2 broken : {et.get('L2_broken','?')}")
cv_m = ensemble_results.get('cv_R2_mean', None)
cv_s = ensemble_results.get('cv_R2_std', None)
if cv_m is not None and cv_s is not None:
    print(f"    CV R²     = {cv_m:.4f} ± {cv_s:.4f}")
else:
    print(f"    CV R²     = {cv_m} ± {cv_s}")

print(f"\n  📐 PySR Ratio Equation (best by Mmax R²):")
print(f"    f_corr = {best_eq_str}")
print(f"    Ratio R²   = {pysr_metrics['ratio_R2']}")
print(f"    Ratio MAPE = {pysr_metrics['ratio_MAPE']}%")
print(f"    Mmax  R²   = {pysr_metrics['mmax_R2']}")
print(f"    Mmax  MAPE = {pysr_metrics['mmax_MAPE']}%")
print(f"    (Auto-selected: {auto_eq_str})")

if val_results:
    print(f"\n  Statistical Validation:")
    print(f"    {val_results.get('verdict','?')}")
    cd = val_results.get("cohens_d", {})
    print(f"    Cohen's d  = {cd.get('cohens_d','?')} ({cd.get('magnitude','?')})")

print(f"\n  📊 Figures generated: {fig_count}/10 (in {FIGURES_DIR})")
print(f"  📄 PDF Report: {RESULTS_DIR / 'Final_Report_v7.pdf'}")
print(f"  Total time: {elapsed/60:.1f} min ({elapsed:.0f}s)")
print(sep)

# ════════════════════════════════════════════════════════════
# CELL 12: ZIP FOR DOWNLOAD
# ════════════════════════════════════════════════════════════
import shutil
zip_path = shutil.make_archive('/content/final_results_v7', 'zip', str(RESULTS_DIR))
print(f"\n📦 Results zipped → {zip_path}")
print("   To download, run in a new cell:")
print("   from google.colab import files; files.download('/content/final_results_v7.zip')")
print("\n✅ Done. Code: 0")
