#!/usr/bin/env python3
"""
===============================================================
  Corrosion RC Beam Optimizer -- Part 1: ML Training Pipeline
  Google Colab Self-Contained Script
===============================================================
  PIPELINE:
    1. Load & preprocess data (791 clean beams)
    2. Split 70/30 (random_state=42)
    3. Train: MLP, XGBoost, RF, GBR, CatBoost+Optuna, Stacking
    4. 10-Fold CV -> predict ALL 791 points (cross_val_predict)
    5. Compute: R2, RMSE, MAE, CV%, SD/M
    6. SHAP Analysis
    7. Statistical Validation
    8. Publication scatter plot (ALL 791 points)
    9. Save artifacts for Part 2 (PySR)

  HOW TO RUN (Google Colab):
    1. Open a new Colab notebook
    2. Paste this ENTIRE file into a single cell
    3. Run it (takes ~10-20 min)
    4. Then run Part 2 (colab_part2_pysr.py) for equation discovery
===============================================================
"""

# =============================================================
# CELL 1: INSTALL & CLONE
# =============================================================
import subprocess, sys, os

def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

for p in ["loguru", "catboost", "xgboost", "optuna", "shap",
           "scikit-learn", "matplotlib", "seaborn", "fpdf2"]:
    try:
        __import__(p.replace("-", "_"))
    except ImportError:
        install(p)

REPO = "corrosion-rc-beam-optimizer"
if not os.path.isdir(f"/content/{REPO}"):
    subprocess.run(["git", "clone",
                    "https://github.com/Dr-Yehia/corrosion-rc-beam-optimizer.git",
                    f"/content/{REPO}"], check=True)
else:
    subprocess.run(["git", "-C", f"/content/{REPO}", "pull"], check=False)

# ============= CRITICAL: PATCH 70/30 SPLIT =============
config_path = f"/content/{REPO}/src/config.py"
with open(config_path, "r") as f:
    cfg_txt = f.read()
cfg_txt = cfg_txt.replace("TEST_SIZE    = 0.20", "TEST_SIZE    = 0.30")
with open(config_path, "w") as f:
    f.write(cfg_txt)
print("CONFIG PATCHED: TEST_SIZE = 0.30 (70/30 split)")

os.chdir(f"/content/{REPO}/src")
sys.path.insert(0, f"/content/{REPO}/src")
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

t_start = time.time()
logger.info("=" * 65)
logger.info("  Corrosion RC Beam Optimizer -- Part 1: ML Training")
logger.info(f"  Split: 70/30 (TEST_SIZE = {TEST_SIZE})")
logger.info(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
logger.info("=" * 65)

# =============================================================
# CELL 3: PREPROCESSING + ACI BASELINE
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

# =============================================================
# CELL 4: MLP BASELINE
# =============================================================
logger.info("\n" + "=" * 60)
logger.info("  Phase 1A -- MLP Baseline")
logger.info("=" * 60)
mlp_results = run_training_pipeline(
    data["X_train"],
    data["X_test"],
    data["y_train"],
    data["y_test"],
    scaler_y=data["scaler_y"],
)

# =============================================================
# CELL 5: ENSEMBLE MODELS (XGB + RF + GBR + CatBoost + Stacking)
# =============================================================
logger.info("\n" + "=" * 60)
logger.info("  Phase 1B -- Ensemble Model Training")
logger.info("=" * 60)
ensemble_results = run_ensemble_pipeline(
    data["X_train"],
    data["X_test"],
    data["y_train"],
    data["y_test"],
    scaler_y=data["scaler_y"],
)
best_model = ensemble_results.get("best_model")
best_name = ensemble_results.get("best_name", "?")
both_broken = ensemble_results.get("both_broken", False)
logger.info(f"Ensemble best: {best_name}")
if both_broken:
    logger.success("L1 + L2 BOTH BROKEN by Ensemble!")

# =============================================================
# CELL 6: 10-FOLD CV -- ALL SAMPLES (cross_val_predict)
# =============================================================
logger.info("\n" + "=" * 60)
logger.info("  10-Fold Cross-Validation -- ALL samples")
logger.info("=" * 60)

scaler_y = data["scaler_y"]

X_all_sc = np.vstack([data["X_train"], data["X_test"]])
y_all_orig = np.concatenate(
    [data["y_train_raw"].values, data["y_test_raw"].values]
)
all_original_idx = np.concatenate(
    [data["y_train_raw"].index.values, data["y_test_raw"].index.values]
)

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
y_pred_cv_all = cross_val_predict(
    cv_model, X_all_sc, y_all_orig, cv=kf_all, n_jobs=-1
)
logger.info("10-Fold CV predictions complete for ALL samples.")

# -- Metrics on ALL CV predictions --
r2_cv = r2_score(y_all_orig, y_pred_cv_all)
rmse_cv = float(np.sqrt(mean_squared_error(y_all_orig, y_pred_cv_all)))
mae_cv = float(mean_absolute_error(y_all_orig, y_pred_cv_all))
cv_pct = (rmse_cv / np.mean(y_all_orig)) * 100
errors_cv = y_all_orig - y_pred_cv_all
sd_m = float(np.std(errors_cv) / np.mean(y_all_orig))

ratio_pred_exp = y_pred_cv_all / np.maximum(y_all_orig, 1e-6)
mean_ratio = float(np.mean(ratio_pred_exp))
std_ratio = float(np.std(ratio_pred_exp))

logger.info(f"\n  10-Fold CV Results (ALL {len(y_all_orig)} samples):")
logger.info(f"    R2    = {r2_cv:.4f}")
logger.info(f"    RMSE  = {rmse_cv:.4f} kN.m")
logger.info(f"    MAE   = {mae_cv:.4f} kN.m")
logger.info(f"    CV%   = {cv_pct:.2f}%")
logger.info(f"    SD/M  = {sd_m:.4f}")
logger.info(f"    Mean(Pred/Exp) = {mean_ratio:.4f}")
logger.info(f"    Std(Pred/Exp)  = {std_ratio:.4f}")

# -- Test Set (30%) metrics --
y_test_true = scaler_y.inverse_transform(
    data["y_test"].reshape(-1, 1)
).ravel()
y_test_pred_sc = best_model.predict(data["X_test"])
if y_test_pred_sc.mean() < 10:
    y_test_pred = scaler_y.inverse_transform(
        y_test_pred_sc.reshape(-1, 1)
    ).ravel()
else:
    y_test_pred = y_test_pred_sc

r2_test = r2_score(y_test_true, y_test_pred)
rmse_test = float(np.sqrt(mean_squared_error(y_test_true, y_test_pred)))
mae_test = float(mean_absolute_error(y_test_true, y_test_pred))
cv_pct_test = (rmse_test / np.mean(y_test_true)) * 100
sd_m_test = float(
    np.std(y_test_true - y_test_pred) / np.mean(y_test_true)
)

logger.info(f"\n  Test Set (30%) Metrics:")
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
    y_true=y_test_true,
    y_pred_model=y_test_pred,
    y_pred_aci=y_aci_test,
    model_builder=build_mlp,
    X_all=X_all_sc,
    y_all_scaled=np.concatenate([data["y_train"], data["y_test"]]),
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

# -- Figure 1: MAIN SCATTER -- ALL 791 points (10-Fold CV) --
try:
    fig1, ax1 = plt.subplots(figsize=(8, 8))
    ax1.scatter(
        y_all_orig, y_pred_cv_all,
        c="#1565C0", alpha=0.5, s=25,
        edgecolors="w", linewidth=0.3, zorder=3,
    )
    lim = [0, max(y_all_orig.max(), y_pred_cv_all.max()) * 1.05]
    ax1.plot(lim, lim, "r--", linewidth=2, label="Perfect prediction")
    ax1.plot(lim, [v * 1.2 for v in lim], "g:", linewidth=1, alpha=0.5,
             label="+20% band")
    ax1.plot(lim, [v * 0.8 for v in lim], "g:", linewidth=1, alpha=0.5,
             label="-20% band")
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
    ax1.grid(True, alpha=0.3)
    textstr = (
        f"R2 = {r2_cv:.4f}\n"
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
    logger.info(f"  Fig 1 OK -- 10-Fold CV Scatter ({len(y_all_orig)} points)")
except Exception as e:
    logger.warning(f"  Fig 1 FAILED: {e}")

# -- Figure 2: Test Set Scatter (30%) --
try:
    fig2, ax2 = plt.subplots(figsize=(8, 8))
    ax2.scatter(
        y_test_true, y_test_pred,
        c="#2E7D32", alpha=0.6, s=30,
        edgecolors="w", linewidth=0.3, zorder=3,
    )
    lim2 = [0, max(y_test_true.max(), y_test_pred.max()) * 1.05]
    ax2.plot(lim2, lim2, "r--", linewidth=2, label="Perfect prediction")
    ax2.plot(lim2, [v * 1.2 for v in lim2], "g:", linewidth=1, alpha=0.5,
             label="+20% band")
    ax2.plot(lim2, [v * 0.8 for v in lim2], "g:", linewidth=1, alpha=0.5,
             label="-20% band")
    ax2.set_xlabel("Experimental Mmax (kN.m)")
    ax2.set_ylabel("Predicted Mmax (kN.m)")
    ax2.set_title(f"{best_name}: Test Set (30%) -- n={len(y_test_true)}")
    ax2.set_xlim(lim2)
    ax2.set_ylim(lim2)
    ax2.set_aspect("equal")
    ax2.legend(fontsize=10, loc="upper left")
    ax2.grid(True, alpha=0.3)
    textstr2 = (
        f"R2 = {r2_test:.4f}\n"
        f"RMSE = {rmse_test:.2f} kN.m\n"
        f"MAE = {mae_test:.2f} kN.m\n"
        f"CV% = {cv_pct_test:.1f}%\n"
        f"SD/M = {sd_m_test:.4f}\n"
        f"n = {len(y_test_true)}"
    )
    ax2.text(
        0.97, 0.03, textstr2, transform=ax2.transAxes, fontsize=10,
        verticalalignment="bottom", horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )
    fig2.savefig(FIGURES_DIR / "fig2_test_set_scatter.png")
    plt.close(fig2)
    fig_count += 1
    logger.info(f"  Fig 2 OK -- Test Set Scatter ({len(y_test_true)} points)")
except Exception as e:
    logger.warning(f"  Fig 2 FAILED: {e}")

# -- Figure 3: Ensemble vs ACI (all points) --
try:
    y_aci_aligned = df_aci.loc[all_original_idx, "MACI_pred"].values
    fig3, ax3 = plt.subplots(figsize=(8, 8))
    ax3.scatter(
        y_all_orig, y_pred_cv_all, alpha=0.5, c="#1565C0", s=25,
        edgecolors="w", linewidth=0.3, zorder=3,
        label=f"{best_name} (R2={r2_cv:.4f})",
    )
    r2_aci_full = r2_score(y_all_orig, y_aci_aligned)
    ax3.scatter(
        y_all_orig, y_aci_aligned, alpha=0.3, c="#E65100", s=20,
        edgecolors="w", linewidth=0.3, zorder=2,
        label=f"ACI 318-19 (R2={r2_aci_full:.4f})",
    )
    lim3 = [
        0,
        max(y_all_orig.max(), y_pred_cv_all.max(), y_aci_aligned.max()) * 1.05,
    ]
    ax3.plot(lim3, lim3, "r--", linewidth=2, label="Perfect fit")
    ax3.set_xlabel("Experimental Mmax (kN.m)")
    ax3.set_ylabel("Predicted Mmax (kN.m)")
    ax3.set_title(
        f"Ensemble vs ACI 318-19 -- All {len(y_all_orig)} Samples"
    )
    ax3.set_xlim(lim3)
    ax3.set_ylim(lim3)
    ax3.set_aspect("equal")
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    fig3.savefig(FIGURES_DIR / "fig3_ensemble_vs_aci_scatter.png")
    plt.close(fig3)
    fig_count += 1
    logger.info("  Fig 3 OK -- Ensemble vs ACI")
except Exception as e:
    logger.warning(f"  Fig 3 FAILED: {e}")

# -- Figure 4: K-Fold Box Plot --
try:
    cv_folds_json = MODELS_DIR / "ensemble_metrics.json"
    if cv_folds_json.exists():
        with open(cv_folds_json) as f:
            ens_data = json.load(f)
        cv_folds = ens_data.get("cv_folds", [])
        if cv_folds:
            fig4, ax4 = plt.subplots(figsize=(8, 6))
            bp = ax4.boxplot(
                [cv_folds], positions=[1], widths=0.5, patch_artist=True,
                boxprops=dict(facecolor="#BBDEFB", color="#1565C0"),
                medianprops=dict(color="#D32F2F", linewidth=2),
            )
            ax4.scatter([1] * len(cv_folds), cv_folds, color="#1565C0",
                        zorder=5, s=60)
            ax4.axhline(y=L1_TARGET_R2, color="green", linestyle="--",
                        linewidth=1.5, label=f"L1 = {L1_TARGET_R2}")
            ax4.axhline(y=L2_TARGET_R2, color="red", linestyle="--",
                        linewidth=1.5, label=f"L2 = {L2_TARGET_R2}")
            ax4.set_ylabel("R2 Score")
            ax4.set_title(
                f"10-Fold CV: R2 = {np.mean(cv_folds):.4f} "
                f"+/- {np.std(cv_folds):.4f}"
            )
            ax4.set_xticks([1])
            ax4.set_xticklabels([best_name])
            ax4.legend(fontsize=11)
            ax4.grid(True, alpha=0.3, axis="y")
            fig4.savefig(FIGURES_DIR / "fig4_kfold_boxplot.png")
            plt.close(fig4)
            fig_count += 1
            logger.info("  Fig 4 OK -- K-Fold Box Plot")
except Exception as e:
    logger.warning(f"  Fig 4 FAILED: {e}")

# -- Figure 5: Error Distribution --
try:
    fig5, ax5 = plt.subplots(figsize=(10, 6))
    errors_model = y_test_true - y_test_pred
    errors_aci = y_test_true - y_aci_test
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

# -- Figure 6: Model Comparison Bar Chart --
try:
    fig6, ax6 = plt.subplots(figsize=(10, 7))
    model_names = []
    model_r2 = []

    model_names.append("ACI 318-19")
    model_r2.append(aci_metrics["R2"])

    if mlp_results:
        mt = mlp_results.get("metrics_test", {})
        model_names.append("MLP")
        model_r2.append(mt.get("R2", 0))

    if cv_folds_json.exists():
        for mn, mm in ens_data.get("models", {}).items():
            model_names.append(mn)
            model_r2.append(mm.get("test_R2", 0))

    colors = ["#E65100"] + ["#42A5F5"] * (len(model_names) - 1)
    for i, n in enumerate(model_names):
        if n == best_name:
            colors[i] = "#1565C0"
            model_names[i] = ">> " + n

    bars = ax6.barh(model_names, model_r2, color=colors,
                    edgecolor="white", height=0.6)
    ax6.axvline(x=L1_TARGET_R2, color="green", linestyle="--",
                linewidth=1.5, label=f"L1 = {L1_TARGET_R2}")
    ax6.axvline(x=L2_TARGET_R2, color="red", linestyle="--",
                linewidth=1.5, label=f"L2 = {L2_TARGET_R2}")
    for bar, val in zip(bars, model_r2):
        ax6.text(
            bar.get_width() + 0.002,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center", fontsize=10, fontweight="bold",
        )
    ax6.set_xlabel("R2 Score")
    ax6.set_title("Model Comparison -- Test Set R2")
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
        "ACI 318-19": (y_test_true, y_aci_aligned_test),
        best_name: (y_test_true, y_test_pred),
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
    y_train_orig = data["y_train_raw"].values
    y_pred_train_xgb = xgb_model_ref.predict(data["X_train"])
    np.save(part2_dir / "y_pred_train.npy", y_pred_train_xgb)
    np.save(part2_dir / "y_train_orig.npy", y_train_orig)
    np.save(part2_dir / "X_train_scaled.npy", data["X_train"])
    joblib.dump(xgb_model_ref, part2_dir / "xgb_model.pkl")
    logger.info("XGBoost model + predictions saved for Part 2")
else:
    logger.warning("XGBoost model not found -- saving best model instead")
    y_train_orig = data["y_train_raw"].values
    y_pred_train_best = best_model.predict(data["X_train"])
    np.save(part2_dir / "y_pred_train.npy", y_pred_train_best)
    np.save(part2_dir / "y_train_orig.npy", y_train_orig)
    np.save(part2_dir / "X_train_scaled.npy", data["X_train"])
    joblib.dump(best_model, part2_dir / "xgb_model.pkl")

np.save(part2_dir / "y_pred_cv_all.npy", y_pred_cv_all)
np.save(part2_dir / "y_all_orig.npy", y_all_orig)

# ACI predictions aligned with df_clean
df_aci[["MACI_pred", "ratio_exp_aci"]].to_csv(
    part2_dir / "aci_predictions.csv", index=True,
)

part1_summary = {
    "n_total": N_TOTAL,
    "n_train": int(data["X_train"].shape[0]),
    "n_test": int(data["X_test"].shape[0]),
    "test_size": TEST_SIZE,
    "aci_metrics": aci_metrics,
    "best_model_name": best_name,
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
print("  PART 1 COMPLETE -- ML TRAINING PIPELINE")
print(sep)

print(f"\n  Data: {N_TOTAL} beams | Train: {data['X_train'].shape[0]} "
      f"| Test: {data['X_test'].shape[0]} (70/30)")

print(f"\n  ACI 318-19 Baseline:")
print(f"    R2   = {aci_metrics.get('R2', '?')}")
print(f"    RMSE = {aci_metrics.get('RMSE', '?')} kN.m")

if mlp_results:
    mt = mlp_results.get("metrics_test", {})
    print(f"\n  MLP Baseline (Test):")
    print(f"    R2   = {mt.get('R2', '?')}")
    print(f"    RMSE = {mt.get('RMSE', '?')}")

et = ensemble_results.get("metrics_test", {})
print(f"\n  Ensemble Best [{best_name}] (Test 30%):")
print(f"    R2    = {r2_test:.4f}")
print(f"    RMSE  = {rmse_test:.4f} kN.m")
print(f"    MAE   = {mae_test:.4f} kN.m")
print(f"    CV%   = {cv_pct_test:.2f}%")
print(f"    SD/M  = {sd_m_test:.4f}")
print(f"    L1 broken: {r2_test >= L1_TARGET_R2}")
print(f"    L2 broken: {r2_test >= L2_TARGET_R2}")

print(f"\n  10-Fold CV (ALL {len(y_all_orig)} samples):")
print(f"    R2    = {r2_cv:.4f}")
print(f"    RMSE  = {rmse_cv:.4f} kN.m")
print(f"    MAE   = {mae_cv:.4f} kN.m")
print(f"    CV%   = {cv_pct:.2f}%")
print(f"    SD/M  = {sd_m:.4f}")
print(f"    Mean(Pred/Exp) = {mean_ratio:.4f}")
print(f"    Std(Pred/Exp)  = {std_ratio:.4f}")

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
# CELL 12: ZIP FOR DOWNLOAD
# =============================================================
import shutil

zip_path = shutil.make_archive(
    "/content/part1_results", "zip", str(RESULTS_DIR)
)
print(f"\nResults zipped -> {zip_path}")
print("   To download:")
print("   from google.colab import files; "
      "files.download('/content/part1_results.zip')")
print("\nPart 1 Done. Ready for Part 2 (PySR).")
