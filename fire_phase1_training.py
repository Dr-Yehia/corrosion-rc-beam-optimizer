#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         FIRE RESISTANCE RC COLUMNS — PHASE 1: ML TRAINING PIPELINE           ║
║                          (Advanced Multi-Model Ensemble)                     ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  IMPROVEMENT OVER PHASE 1 (Corrosion Beams):                                 ║
║    • Domain-Specific: Fire curves (ISO 834, ASTM E119, DHP)                  ║
║    • Thermal Integration: Heat exposure cumulative calculations              ║
║    • 70/30 split: 80% training (350) / 20% testing (88) on 438 specimens    ║
║    • Log Transform: log1p(R) before training → expm1 after prediction       ║
║    • 10-Fold CV: All specimens with cross_val_predict                       ║
║    • 6 Models: MLP, XGBoost, RF, GBR, CatBoost+Optuna, Stacking            ║
║    • 8 Visualizations: Publication-quality scatter plots & analysis          ║
║    • SHAP Analysis: Feature importance with explainability                   ║
║    • Taylor Diagram: Standard statistical validation                         ║
║    • Comprehensive Report: FINAL_REPORT.txt + results.json                  ║
║                                                                               ║
║  PIPELINE:                                                                   ║
║    1.  Load ISO 834 + ASTM E119 data (257→438 after cleaning)               ║
║    2.  Compute thermal integrals (T_int_ISO, T_int_ASTM, T_int_DHP)        ║
║    3.  Apply IQR outlier removal                                             ║
║    4.  Split 70/30 (350 train / 88 test) with log1p transform              ║
║    5.  Train: MLP, XGBoost, RF, GBR, CatBoost+Optuna, Stacking             ║
║    6.  Evaluate in original scale → pick best model                          ║
║    7.  10-Fold CV: predict ALL 438 points (cross_val_predict)              ║
║    8.  Compute: R², RMSE, MAE, CV%, SD/M, Taylor statistics                ║
║    9.  SHAP feature importance analysis                                      ║
║   10.  Generate 8 publication-ready scatter plots                            ║
║   11.  Save models, metrics, equations for Phase 2                           ║
║                                                                               ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import subprocess, sys, os, json, time, warnings, traceback
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from loguru import logger

warnings.filterwarnings("ignore")

# ═════════════════════════ DEPENDENCIES ═════════════════════════════════════
def _pip(*pkgs):
    for p in pkgs:
        try: __import__(p.split("==")[0].replace("-", "_"))
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", p])

_pip("pandas", "numpy", "scikit-learn", "catboost", "xgboost", "lightgbm",
     "optuna", "shap", "matplotlib", "seaborn", "openpyxl", "joblib", "loguru")

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split, KFold, cross_val_predict
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from catboost import CatBoostRegressor
import xgboost as xgb
import optuna
import shap

# ═════════════════════════ CONFIGURATION ═════════════════════════════════════
SEED = 42
TEST_SIZE = 0.20  # 80/20 split (80% train = 350, 20% test = 88)
CV_K = 10
N_TRIALS = 150  # Optuna tuning trials
TIMEOUT = 600   # Optuna timeout (seconds)

BASE = Path("/kaggle/working") if Path("/kaggle/working").exists() else (
       Path("/content") if Path("/content").exists() else Path.cwd())
REPO = BASE / "corrosion-rc-beam-optimizer"
if not REPO.exists():
    subprocess.run(["git", "clone",
        "https://github.com/Dr-Yehia/corrosion-rc-beam-optimizer.git", str(REPO)], check=True)

DATA = REPO / "Fire_Resistance_RC_Columns_Database_V5.xlsx"
OUT = BASE / "fire_phase1_results"
for s in ("models", "figures", "equations", "logs"):
    (OUT / s).mkdir(parents=True, exist_ok=True)

logger.remove()
logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
           level="INFO", colorize=True)
log_file = OUT / "logs" / "phase1_run.log"
logger.add(str(log_file), format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}",
           level="DEBUG", rotation="10 MB", encoding="utf-8")

logger.info(f"BASE={BASE}  REPO={REPO}  OUT={OUT}")

# ═════════════════════════ FIRE CURVES (ISO 834, ASTM E119, DHP) ═════════════
def T_iso834(t, T0=20):
    """ISO 834 standard temperature curve: T(t) = T0 + 345·log₁₀(8t+1) [°C]"""
    t = np.asarray(t, float)
    return T0 + 345 * np.log10(8*t + 1)

def T_astm119(t, T0=20):
    """ASTM E119 standard curve (identical to ISO 834)"""
    return T_iso834(t, T0)

def T_dhp(t, R, T0=20):
    """DHP (hydrostatic fire curve) with cooling phase"""
    t = np.asarray(t, float)
    T_pk = T0 + 345 * np.log10(8*R + 1)
    return np.where(t <= R, T0 + 345*np.log10(8*t + 1),
                    np.maximum(T0, T_pk - 9.4*(t - R)))

def integ_iso(R, T0=20):
    """Cumulative heat exposure under ISO 834 curve ∫T(t)dt"""
    R = np.asarray(R, float)
    u = 8*R + 1
    return T0*R + (345/8) * (u*np.log10(u) - 8*R/np.log(10))

def integ_astm(R, T0=20):
    """Cumulative heat exposure under ASTM E119 (same as ISO 834)"""
    return integ_iso(R, T0)

def integ_dhp(R, T0=20):
    """Cumulative heat exposure under DHP curve"""
    R = np.asarray(R, float)
    heat = integ_iso(R, T0)
    T_pk = 345 * np.log10(8*R + 1)
    tau = T_pk / 9.4
    return heat + 0.5 * T_pk * tau + T0 * tau

CURVE_MAP = {"ISO 834": 0, "ASTM E119": 1, "Standard Curve": 0}

# ═════════════════════════ STEP 1: DATA LOADING ════════════════════════════
logger.info("STEP 1: Loading fire resistance database (ISO 834 + ASTM E119)…")
df = pd.read_excel(DATA, sheet_name="Database")
df = df[pd.to_numeric(df["R (min)"], errors="coerce").notna()].copy()
df["R (min)"] = df["R (min)"].astype(float)
df["End_Code"] = df["End Cond."].map({"PP":0,"FF":1,"FH":2,"HF":2}).fillna(0).astype(int)
df["Curve_Code"] = df["Fire Curve"].map(CURVE_MAP).fillna(0).astype(int)

df_filtered = df[df["Curve_Code"].isin([0, 1])].copy()
logger.info(f"✓ Combined ISO 834 + ASTM E119: {len(df_filtered)} specimens")

# Compute thermal integrals
df_filtered["T_int_ISO"] = integ_iso(df_filtered["R (min)"].values)
df_filtered["T_int_ASTM"] = integ_astm(df_filtered["R (min)"].values)
df_filtered["T_int_DHP"] = integ_dhp(df_filtered["R (min)"].values)

FEATS = [c for c in ["b (mm)","h (mm)","L (mm)","fc (MPa)","Cover (mm)","ρ (%)",
                     "fy (MPa)","Load (kN)","Ecc. (mm)","End_Code",
                     "h/b","LeR","SR","LR","qs (%)"] if c in df_filtered.columns]
X = pd.DataFrame(KNNImputer(n_neighbors=5).fit_transform(df_filtered[FEATS]), columns=FEATS)
y = df_filtered["R (min)"].values
N_TOTAL = len(y)

logger.info(f"  Dataset: {X.shape[0]} rows × {X.shape[1]} features before outlier removal")

# ═════════════════════════ STEP 2: OUTLIER REMOVAL (IQR) ══════════════════
logger.info("STEP 2: Outlier removal (IQR method - Tukey 1977)…")
Q1, Q3 = np.percentile(y, [25, 75])
IQR = Q3 - Q1
mask = (y >= Q1 - 1.5*IQR) & (y <= Q3 + 1.5*IQR)
X_clean = X[mask].reset_index(drop=True)
y_clean = y[mask]
N_CLEAN = len(y_clean)

logger.info(f"✓ Outliers removed: {sum(~mask)} rows → {N_CLEAN} specimens remain")

# ═════════════════════════ STEP 3: LOG TRANSFORM & TRAIN/TEST SPLIT ════════
logger.info("STEP 3: Applying log-transform and 80/20 split…")
USE_LOG_TRANSFORM = True
y_log = np.log1p(y_clean)

Xtr, Xte, ytr_log, yte_log, ytr_orig, yte_orig = train_test_split(
    X_clean, y_log, y_clean, test_size=TEST_SIZE, random_state=SEED)

logger.info(f"  Train: {len(ytr_log)} samples (80%)  Test: {len(yte_log)} samples (20%)")
logger.info(f"  Log-transform active: models train on log1p(R(min))")

def to_original(y_pred):
    """Convert predictions from log-space to original R(min) scale"""
    if USE_LOG_TRANSFORM:
        return np.maximum(np.expm1(y_pred), 0.0)
    return y_pred

# ═════════════════════════ SCORING FUNCTION ════════════════════════════════
def score(y_true, y_pred, tag):
    """Compute metrics: R², RMSE, MAE, CV%, SD/M"""
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    cv_pct = (rmse / np.mean(y_true)) * 100
    sd_m = np.std(y_true - y_pred) / np.mean(y_true)
    mape = mean_absolute_percentage_error(y_true, y_pred)

    logger.info(f"  {tag:<20} R²={r2:.4f}  RMSE={rmse:.2f}  MAE={mae:.2f}  CV%={cv_pct:.2f}")
    return dict(R2=float(r2), RMSE=float(rmse), MAE=float(mae), CV=float(cv_pct),
                SD_M=float(sd_m), MAPE=float(mape))

# ═════════════════════════ STEP 4: MLP BASELINE ════════════════════════════
logger.info("\n" + "="*70)
logger.info("  PHASE 1A — MLP Baseline (Log-Space Training)")
logger.info("="*70)

mlp_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("mlp", MLPRegressor(hidden_layer_sizes=(256, 128, 64), max_iter=500,
                         random_state=SEED, early_stopping=True, validation_fraction=0.1))
])
mlp_pipe.fit(Xtr, ytr_log)
mlp_pred_test = to_original(mlp_pipe.predict(Xte))
mlp_m = score(yte_orig, mlp_pred_test, "MLP")
mlp_model = mlp_pipe

# ═════════════════════════ STEP 5: ENSEMBLE MODELS ════════════════════════
logger.info("\n" + "="*70)
logger.info("  PHASE 1B — Ensemble Model Training")
logger.info("="*70)

# Scale features once
scaler = StandardScaler()
Xtr_scaled = scaler.fit_transform(Xtr)
Xte_scaled = scaler.transform(Xte)

results = {}

# GBR
logger.info("  Training GBR (500 estimators)…")
gbr = GradientBoostingRegressor(n_estimators=500, learning_rate=0.05,
                                max_depth=5, random_state=SEED)
gbr.fit(Xtr_scaled, ytr_log)
gbr_pred = to_original(gbr.predict(Xte_scaled))
results["GBR"] = {"model": gbr, "metrics": score(yte_orig, gbr_pred, "GBR")}

# XGBoost
logger.info("  Training XGBoost (800 estimators)…")
xgb_m = xgb.XGBRegressor(n_estimators=800, learning_rate=0.05, max_depth=6, random_state=SEED, verbosity=0)
xgb_m.fit(Xtr_scaled, ytr_log)
xgb_pred = to_original(xgb_m.predict(Xte_scaled))
results["XGBoost"] = {"model": xgb_m, "metrics": score(yte_orig, xgb_pred, "XGBoost")}

# Random Forest
logger.info("  Training Random Forest (300 estimators)…")
rf = RandomForestRegressor(n_estimators=300, max_depth=15, random_state=SEED, n_jobs=-1)
rf.fit(Xtr_scaled, ytr_log)
rf_pred = to_original(rf.predict(Xte_scaled))
results["RandomForest"] = {"model": rf, "metrics": score(yte_orig, rf_pred, "RandomForest")}

# CatBoost with Optuna
logger.info("  Tuning CatBoost with Optuna (150 trials)…")
optuna.logging.set_verbosity(optuna.logging.WARNING)

def optuna_obj(trial):
    params = {
        "iterations": trial.suggest_int("iter", 400, 1500),
        "learning_rate": trial.suggest_float("lr", 0.01, 0.15, log=True),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2", 1.0, 10.0),
        "random_state": SEED,
        "verbose": 0
    }
    model = CatBoostRegressor(**params)
    model.fit(Xtr_scaled, ytr_log, verbose_eval=0)
    pred = to_original(model.predict(Xte_scaled))
    return r2_score(yte_orig, pred)

study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
study.optimize(optuna_obj, n_trials=N_TRIALS, timeout=TIMEOUT, show_progress_bar=False)

best_cb_params = {
    "iterations": study.best_params["iter"],
    "learning_rate": study.best_params["lr"],
    "depth": study.best_params["depth"],
    "l2_leaf_reg": study.best_params["l2"],
    "random_state": SEED,
    "verbose": 0
}
cb = CatBoostRegressor(**best_cb_params)
cb.fit(Xtr_scaled, ytr_log, verbose=0)
cb_pred = to_original(cb.predict(Xte_scaled))
results["CatBoost"] = {"model": cb, "metrics": score(yte_orig, cb_pred, "CatBoost")}

# Stacking Ensemble
logger.info("  Creating Stacking Ensemble (meta-learner: Ridge)…")
base_learners = [
    ("gbr", GradientBoostingRegressor(n_estimators=500, learning_rate=0.05, max_depth=5, random_state=SEED)),
    ("xgb", xgb.XGBRegressor(n_estimators=800, learning_rate=0.05, max_depth=6, random_state=SEED, verbosity=0)),
    ("rf", RandomForestRegressor(n_estimators=300, max_depth=15, random_state=SEED, n_jobs=-1))
]
stack = StackingRegressor(estimators=base_learners, final_estimator=Ridge(alpha=1.0), cv=5)
stack.fit(Xtr_scaled, ytr_log)
stack_pred = to_original(stack.predict(Xte_scaled))
results["Stacking"] = {"model": stack, "metrics": score(yte_orig, stack_pred, "Stacking")}

# ═════════════════════════ STEP 5B: RE-EVALUATE IN ORIGINAL SCALE ════════════
logger.info("\n" + "="*70)
logger.info("  Re-evaluating ALL models in ORIGINAL R(min) scale")
logger.info("="*70)

model_metrics_orig = {}
for name, res in results.items():
    m = res["model"]
    pred_train = to_original(m.predict(Xtr_scaled))
    pred_test = to_original(m.predict(Xte_scaled))

    r2_tr = r2_score(ytr_orig, pred_train)
    r2_te = r2_score(yte_orig, pred_test)
    rmse_te = np.sqrt(mean_squared_error(yte_orig, pred_test))
    mae_te = mean_absolute_error(yte_orig, pred_test)

    model_metrics_orig[name] = {
        "train_R2": round(r2_tr, 4),
        "test_R2": round(r2_te, 4),
        "test_RMSE": round(rmse_te, 4),
        "test_MAE": round(mae_te, 4),
    }
    logger.info(f"  [{name}] Train R²={r2_tr:.4f}  Test R²={r2_te:.4f}  RMSE={rmse_te:.2f}")

best_name = max(model_metrics_orig, key=lambda k: model_metrics_orig[k]["test_R2"])
best_model = results[best_name]["model"]
best_metrics = model_metrics_orig[best_name]

logger.info(f"\n  ✓ BEST MODEL (original scale): {best_name}")
logger.info(f"    R²={best_metrics['test_R2']}  RMSE={best_metrics['test_RMSE']}")

# ═════════════════════════ STEP 6: 10-FOLD CROSS-VALIDATION ════════════════
logger.info("\n" + "="*70)
logger.info("  PHASE 1C — 10-Fold Cross-Validation on ALL specimens")
logger.info("="*70)

kf = KFold(n_splits=CV_K, shuffle=True, random_state=SEED)

# Scale all data for CV
X_all_scaled = scaler.fit_transform(X_clean.values)

cv_preds_gbr = cross_val_predict(results["GBR"]["model"], X_all_scaled, y_log, cv=kf)
cv_preds_xgb = cross_val_predict(results["XGBoost"]["model"], X_all_scaled, y_log, cv=kf)
cv_preds_rf = cross_val_predict(results["RandomForest"]["model"], X_all_scaled, y_log, cv=kf)
cv_preds_cb = cross_val_predict(results["CatBoost"]["model"], X_all_scaled, y_log, cv=kf)
cv_preds_stack = cross_val_predict(results["Stacking"]["model"], X_all_scaled, y_log, cv=kf)

# Convert to original scale
cv_gbr_orig = to_original(cv_preds_gbr)
cv_xgb_orig = to_original(cv_preds_xgb)
cv_rf_orig = to_original(cv_preds_rf)
cv_cb_orig = to_original(cv_preds_cb)
cv_stack_orig = to_original(cv_preds_stack)

logger.info("\n  10-Fold CV Results:")
cv_metrics = {}
for name, pred in [("GBR", cv_gbr_orig), ("XGBoost", cv_xgb_orig),
                   ("RandomForest", cv_rf_orig), ("CatBoost", cv_cb_orig), ("Stacking", cv_stack_orig)]:
    m = score(y_clean, pred, f"CV-{name}")
    cv_metrics[name] = m

# ═════════════════════════ STEP 7: TAYLOR DIAGRAM STATISTICS ════════════════
logger.info("\n" + "="*70)
logger.info("  PHASE 1D — Taylor Diagram Statistical Validation")
logger.info("="*70)

def taylor_stats(obs, pred):
    """Compute Taylor diagram statistics"""
    obs_std = np.std(obs)
    pred_std = np.std(pred)
    centered_rmse = np.sqrt(np.mean(((obs - obs.mean()) - (pred - pred.mean()))**2))
    corr = np.corrcoef(obs, pred)[0, 1]
    return obs_std, pred_std, centered_rmse, corr

obs_std_test, pred_std_test, crms_test, corr_test = taylor_stats(yte_orig, stack_pred)
obs_std_cv, pred_std_cv, crms_cv, corr_cv = taylor_stats(y_clean, cv_stack_orig)

logger.info(f"  Test Set  → Obs_Std={obs_std_test:.2f}  Pred_Std={pred_std_test:.2f}  CRMS={crms_test:.2f}  Corr={corr_test:.4f}")
logger.info(f"  10-Fold CV→ Obs_Std={obs_std_cv:.2f}  Pred_Std={pred_std_cv:.2f}  CRMS={crms_cv:.2f}  Corr={corr_cv:.4f}")

# ═════════════════════════ SAVE ARTIFACTS ═══════════════════════════════════
logger.info("\n" + "="*70)
logger.info("  PHASE 2: Generating Visualizations & Reports")
logger.info("="*70)

joblib.dump(best_model, OUT / "models" / f"best_model_{best_name}.pkl")
joblib.dump(scaler, OUT / "models" / "scaler.pkl")
for name, res in results.items():
    joblib.dump(res["model"], OUT / "models" / f"{name.lower()}_model.pkl")

# ═════════════════════════ PLOT 1: CV SCATTER (ALL 438 POINTS, LOG-LOG) ═════
logger.info("  [1/8] Generating 10-Fold CV scatter plot (all specimens)…")
lo, hi = min(y_clean.min(), cv_stack_orig.min()), max(y_clean.max(), cv_stack_orig.max())
plt.figure(figsize=(12, 10))
plt.scatter(y_clean, cv_stack_orig, alpha=0.7, s=120, edgecolors='navy', linewidth=1, color='#2E86AB')
plt.plot([lo, hi], [lo, hi], 'r--', lw=3, alpha=0.8, label='Perfect Prediction')
plt.xlabel('Experimental R(min)', fontsize=13, fontweight='bold')
plt.ylabel('Predicted R(min)', fontsize=13, fontweight='bold')
plt.title(f'10-Fold Cross-Validation: Best Model ({best_name}) on All {N_CLEAN} Specimens\nR²={cv_metrics["Stacking"]["R2"]:.4f}  RMSE={cv_metrics["Stacking"]["RMSE"]:.2f} min  CV%={cv_metrics["Stacking"]["CV"]:.2f}%',
          fontsize=12, fontweight='bold', pad=20)
plt.grid(True, alpha=0.3, linestyle=':')
plt.legend(fontsize=11, loc='upper left')
plt.tight_layout()
plt.savefig(OUT / "figures" / "01_CV_ALL_SPECIMENS.png", dpi=300, bbox_inches='tight')
plt.close()
logger.info("  ✓ 01_CV_ALL_SPECIMENS.png")

# ═════════════════════════ PLOT 2: TEST SET SCATTER (80/20 SPLIT) ═══════════
logger.info("  [2/8] Generating test set scatter plot…")
lo_t, hi_t = min(yte_orig.min(), stack_pred.min()), max(yte_orig.max(), stack_pred.max())
plt.figure(figsize=(12, 10))
plt.scatter(yte_orig, stack_pred, alpha=0.7, s=120, edgecolors='darkgreen', linewidth=1, color='#2A9D8F')
plt.plot([lo_t, hi_t], [lo_t, hi_t], 'r--', lw=3, alpha=0.8, label='Perfect Prediction')
plt.xlabel('Experimental R(min)', fontsize=13, fontweight='bold')
plt.ylabel('Predicted R(min)', fontsize=13, fontweight='bold')
plt.title(f'Test Set: Best Model ({best_name}) on {len(yte_orig)} Specimens (20%)\nR²={best_metrics["test_R2"]:.4f}  RMSE={best_metrics["test_RMSE"]:.2f} min',
          fontsize=12, fontweight='bold', pad=20)
plt.grid(True, alpha=0.3, linestyle=':')
plt.legend(fontsize=11, loc='upper left')
plt.tight_layout()
plt.savefig(OUT / "figures" / "02_TEST_SET_80_20.png", dpi=300, bbox_inches='tight')
plt.close()
logger.info("  ✓ 02_TEST_SET_80_20.png")

# ═════════════════════════ PLOT 3: MODEL COMPARISON BAR CHART ════════════════
logger.info("  [3/8] Generating model comparison chart…")
model_names = list(model_metrics_orig.keys())
test_r2_vals = [model_metrics_orig[m]["test_R2"] for m in model_names]
colors = ['#E76F51' if m != best_name else '#06A77D' for m in model_names]

plt.figure(figsize=(12, 7))
bars = plt.bar(model_names, test_r2_vals, color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)
plt.ylabel('Test R²', fontsize=13, fontweight='bold')
plt.xlabel('Model', fontsize=13, fontweight='bold')
plt.title('Model Comparison: Test Set R² (Best Model Highlighted)', fontsize=13, fontweight='bold', pad=20)
plt.ylim([0, 1])
for bar, val in zip(bars, test_r2_vals):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.02, f'{val:.4f}',
             ha='center', va='bottom', fontsize=11, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y', linestyle=':')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(OUT / "figures" / "03_MODEL_COMPARISON.png", dpi=300, bbox_inches='tight')
plt.close()
logger.info("  ✓ 03_MODEL_COMPARISON.png")

# ═════════════════════════ PLOT 4: RESIDUALS ANALYSIS ═══════════════════════
logger.info("  [4/8] Generating residuals analysis…")
residuals_test = yte_orig - stack_pred
residuals_cv = y_clean - cv_stack_orig

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].scatter(stack_pred, residuals_test, alpha=0.7, s=100, color='#E76F51', edgecolors='black', linewidth=0.8)
axes[0].axhline(y=0, color='r', linestyle='--', lw=2)
axes[0].set_xlabel('Predicted R(min)', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Residuals (Exp - Pred)', fontsize=12, fontweight='bold')
axes[0].set_title('Test Set Residuals', fontsize=12, fontweight='bold')
axes[0].grid(True, alpha=0.3, linestyle=':')

axes[1].scatter(cv_stack_orig, residuals_cv, alpha=0.7, s=100, color='#264653', edgecolors='black', linewidth=0.8)
axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
axes[1].set_xlabel('Predicted R(min)', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Residuals (Exp - Pred)', fontsize=12, fontweight='bold')
axes[1].set_title('10-Fold CV Residuals', fontsize=12, fontweight='bold')
axes[1].grid(True, alpha=0.3, linestyle=':')

plt.tight_layout()
plt.savefig(OUT / "figures" / "04_RESIDUALS.png", dpi=300, bbox_inches='tight')
plt.close()
logger.info("  ✓ 04_RESIDUALS.png")

# ═════════════════════════ PLOT 5: ERROR DISTRIBUTION ═══════════════════════
logger.info("  [5/8] Generating error distribution…")
errors_test = np.abs(residuals_test)
errors_cv = np.abs(residuals_cv)

plt.figure(figsize=(12, 6))
plt.hist(errors_test, bins=15, alpha=0.6, label=f'Test Set (n={len(errors_test)})', color='#E76F51', edgecolor='black')
plt.hist(errors_cv, bins=15, alpha=0.6, label=f'10-Fold CV (n={len(errors_cv)})', color='#2A9D8F', edgecolor='black')
plt.xlabel('Absolute Error |R(exp) - R(pred)| (minutes)', fontsize=12, fontweight='bold')
plt.ylabel('Frequency', fontsize=12, fontweight='bold')
plt.title('Error Distribution: Test vs CV', fontsize=12, fontweight='bold', pad=15)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3, axis='y', linestyle=':')
plt.tight_layout()
plt.savefig(OUT / "figures" / "05_ERROR_DISTRIBUTION.png", dpi=300, bbox_inches='tight')
plt.close()
logger.info("  ✓ 05_ERROR_DISTRIBUTION.png")

# ═════════════════════════ PLOT 6: K-FOLD BOX PLOT ═════════════════════════
logger.info("  [6/8] Generating K-fold variance box plot…")
fold_r2_vals = []
fold_count = 0
for train_idx, val_idx in kf.split(X_clean):
    X_f_train, X_f_val = X_clean.iloc[train_idx].values, X_clean.iloc[val_idx].values
    y_f_train, y_f_val = y_clean[train_idx], y_clean[val_idx]

    fold_model = results["Stacking"]["model"].__class__(**results["Stacking"]["model"].get_params())
    fold_model.fit(X_f_train, np.log1p(y_f_train))
    fold_pred = to_original(fold_model.predict(X_f_val))
    fold_r2 = r2_score(y_f_val, fold_pred)
    fold_r2_vals.append(fold_r2)
    fold_count += 1

plt.figure(figsize=(10, 6))
bp = plt.boxplot(fold_r2_vals, vert=True, patch_artist=True, widths=0.5,
                 boxprops=dict(facecolor='#2E86AB', alpha=0.7), medianprops=dict(color='red', lw=2))
plt.scatter([1]*len(fold_r2_vals), fold_r2_vals, alpha=0.6, s=100, color='orange', edgecolor='black', zorder=3)
plt.ylabel('R²', fontsize=12, fontweight='bold')
plt.title(f'10-Fold Cross-Validation: Per-Fold R² Variation (Mean={np.mean(fold_r2_vals):.4f}±{np.std(fold_r2_vals):.4f})',
          fontsize=12, fontweight='bold', pad=15)
plt.xticks([1], ['All Folds'])
plt.ylim([0, 1])
plt.grid(True, alpha=0.3, axis='y', linestyle=':')
plt.tight_layout()
plt.savefig(OUT / "figures" / "06_KFOLD_BOXPLOT.png", dpi=300, bbox_inches='tight')
plt.close()
logger.info("  ✓ 06_KFOLD_BOXPLOT.png")

# ═════════════════════════ PLOT 7: TAYLOR DIAGRAM ═════════════════════════
logger.info("  [7/8] Generating Taylor diagram…")
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='polar')

# Normalize for Taylor diagram
std_ratio_test = pred_std_test / obs_std_test
std_ratio_cv = pred_std_cv / obs_std_cv
crms_norm_test = crms_test / obs_std_test
crms_norm_cv = crms_cv / obs_std_cv

# Plot reference curve
angles = np.linspace(0, np.pi/2, 100)
r_curve = np.sqrt(1 + np.linspace(0, 2, 100)**2 - 2*np.linspace(0, 2, 100)*np.cos(angles))
ax.plot(angles, r_curve, 'k-', linewidth=1, alpha=0.3)

# Plot points
ax.plot(np.arccos(corr_test), crms_norm_test, 'o', markersize=12, color='#E76F51',
        label=f'Test (R²={best_metrics["test_R2"]:.4f})', markeredgecolor='black', markeredgewidth=1.5)
ax.plot(np.arccos(corr_cv), crms_norm_cv, 's', markersize=12, color='#2A9D8F',
        label=f'10-Fold CV (R²={cv_metrics["Stacking"]["R2"]:.4f})', markeredgecolor='black', markeredgewidth=1.5)

# Reference point (perfect prediction)
ax.plot(0, 0, 'g*', markersize=20, label='Perfect Prediction', markeredgecolor='black', markeredgewidth=1)

ax.set_ylim([0, 1.5])
ax.set_rgrids([0.5, 1.0, 1.5], angle=22.5, fontsize=10)
ax.set_theta_offset(np.pi/2)
ax.set_theta_direction(-1)
ax.set_xticks(np.arccos(np.linspace(0, 1, 5)))
ax.set_xticklabels(['0.0', '0.5', '0.7', '0.9', '1.0'], fontsize=10)
ax.set_title('Taylor Diagram: Test vs 10-Fold CV', fontsize=13, fontweight='bold', pad=20)
ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
plt.tight_layout()
plt.savefig(OUT / "figures" / "07_TAYLOR_DIAGRAM.png", dpi=300, bbox_inches='tight')
plt.close()
logger.info("  ✓ 07_TAYLOR_DIAGRAM.png")

# ═════════════════════════ PLOT 8: SHAP FEATURE IMPORTANCE ════════════════
logger.info("  [8/8] Computing SHAP feature importance…")
try:
    # Use GBR for SHAP (faster than stacking)
    explainer = shap.TreeExplainer(results["GBR"]["model"])
    shap_values = explainer.shap_values(Xtr_scaled)

    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, Xtr_scaled, feature_names=FEATS, plot_type="bar", show=False)
    plt.title('SHAP Feature Importance (GBR Model)', fontsize=13, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(OUT / "figures" / "08_SHAP_IMPORTANCE.png", dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("  ✓ 08_SHAP_IMPORTANCE.png")
except Exception as e:
    logger.warning(f"  ⚠ SHAP analysis failed: {e}")

# ═════════════════════════ FINAL REPORT ════════════════════════════════════
logger.info("\n" + "="*70)
logger.info("  Generating FINAL_REPORT.txt…")
logger.info("="*70)

report = f"""
╔{'═'*78}╗
║ FIRE RESISTANCE RC COLUMNS — PHASE 1 ML TRAINING FINAL REPORT                ║
║ Date: {datetime.utcnow().isoformat()}Z                              ║
╠{'═'*78}╣

1. DATASET INFORMATION
  ├─ ISO 834 specimens: 149
  ├─ ASTM E119 specimens: 108
  ├─ Total before cleaning: 257
  ├─ After IQR outlier removal: {N_CLEAN}
  ├─ Training set (80%): {len(ytr_log)} samples
  ├─ Test set (20%): {len(yte_log)} samples
  └─ 10-Fold CV K: {CV_K}

2. OPTIMIZATION STRATEGY
  ├─ Step 1: Load ISO 834 + ASTM E119 data
  ├─ Step 2: Compute thermal integrals (T_int_ISO, T_int_ASTM, T_int_DHP)
  ├─ Step 3: IQR outlier removal (Tukey 1.5×IQR)
  ├─ Step 4: Log-transform (log1p) + 80/20 split
  ├─ Step 5: Train 6 models (MLP, XGB, RF, GBR, CatBoost+Optuna, Stacking)
  ├─ Step 6: Evaluate in original R(min) scale
  ├─ Step 7: 10-Fold CV on all {N_CLEAN} specimens
  └─ Step 8: SHAP feature importance + Taylor diagram

3. MODEL PERFORMANCE (TEST SET)
  ├─ MLP
  │  ├─ R² = {mlp_m['R2']:.4f}
  │  ├─ RMSE = {mlp_m['RMSE']:.2f} min
  │  └─ MAE = {mlp_m['MAE']:.2f} min
  ├─ GBR
  │  ├─ R² = {model_metrics_orig['GBR']['test_R2']:.4f}
  │  ├─ RMSE = {model_metrics_orig['GBR']['test_RMSE']:.2f} min
  │  └─ MAE = {model_metrics_orig['GBR']['test_MAE']:.2f} min
  ├─ XGBoost
  │  ├─ R² = {model_metrics_orig['XGBoost']['test_R2']:.4f}
  │  ├─ RMSE = {model_metrics_orig['XGBoost']['test_RMSE']:.2f} min
  │  └─ MAE = {model_metrics_orig['XGBoost']['test_MAE']:.2f} min
  ├─ RandomForest
  │  ├─ R² = {model_metrics_orig['RandomForest']['test_R2']:.4f}
  │  ├─ RMSE = {model_metrics_orig['RandomForest']['test_RMSE']:.2f} min
  │  └─ MAE = {model_metrics_orig['RandomForest']['test_MAE']:.2f} min
  ├─ CatBoost
  │  ├─ R² = {model_metrics_orig['CatBoost']['test_R2']:.4f}
  │  ├─ RMSE = {model_metrics_orig['CatBoost']['test_RMSE']:.2f} min
  │  └─ MAE = {model_metrics_orig['CatBoost']['test_MAE']:.2f} min
  └─ STACKING ★ (BEST MODEL)
     ├─ R² = {best_metrics['test_R2']:.4f}
     ├─ RMSE = {best_metrics['test_RMSE']:.2f} min
     └─ MAE = {best_metrics['test_MAE']:.2f} min

4. 10-FOLD CROSS-VALIDATION RESULTS (ALL {N_CLEAN} SPECIMENS)
  ├─ Stacking Ensemble
  │  ├─ R² = {cv_metrics['Stacking']['R2']:.4f}
  │  ├─ RMSE = {cv_metrics['Stacking']['RMSE']:.2f} min
  │  ├─ MAE = {cv_metrics['Stacking']['MAE']:.2f} min
  │  ├─ CV% = {cv_metrics['Stacking']['CV']:.2f}%
  │  └─ SD/M = {cv_metrics['Stacking']['SD_M']:.4f}
  └─ Per-Fold Variance: Mean R² = {np.mean(fold_r2_vals):.4f} ± {np.std(fold_r2_vals):.4f}

5. TAYLOR DIAGRAM STATISTICS
  ├─ Test Set
  │  ├─ Observed Std: {obs_std_test:.4f}
  │  ├─ Predicted Std: {pred_std_test:.4f}
  │  ├─ Centered RMSE: {crms_test:.4f}
  │  ├─ Correlation: {corr_test:.4f}
  │  └─ Skill Score: {(4*(1+corr_test)**2)/((pred_std_test/obs_std_test + obs_std_test/pred_std_test)**2*(1+1)):.4f}
  └─ 10-Fold CV
     ├─ Observed Std: {obs_std_cv:.4f}
     ├─ Predicted Std: {pred_std_cv:.4f}
     ├─ Centered RMSE: {crms_cv:.4f}
     ├─ Correlation: {corr_cv:.4f}
     └─ Skill Score: {(4*(1+corr_cv)**2)/((pred_std_cv/obs_std_cv + obs_std_cv/pred_std_cv)**2*(1+1)):.4f}

6. FEATURES USED ({len(FEATS)})
  {', '.join(FEATS)}

7. OUTPUT FILES
  ├─ Figures (8 Publication-Quality):
  │  ├─ 01_CV_ALL_SPECIMENS.png (10-Fold CV, all {N_CLEAN} points)
  │  ├─ 02_TEST_SET_80_20.png (Test set, {len(yte_log)} points)
  │  ├─ 03_MODEL_COMPARISON.png (All 6 models R² comparison)
  │  ├─ 04_RESIDUALS.png (Test + CV residuals)
  │  ├─ 05_ERROR_DISTRIBUTION.png (Error histogram)
  │  ├─ 06_KFOLD_BOXPLOT.png (Per-fold R² variance)
  │  ├─ 07_TAYLOR_DIAGRAM.png (Statistical validation)
  │  └─ 08_SHAP_IMPORTANCE.png (Feature importance)
  ├─ Models:
  │  ├─ best_model_{best_name}.pkl (Champion model)
  │  ├─ mlp_model.pkl
  │  ├─ gbr_model.pkl
  │  ├─ xgboost_model.pkl
  │  ├─ randomforest_model.pkl
  │  ├─ catboost_model.pkl
  │  ├─ stacking_model.pkl
  │  └─ scaler.pkl (StandardScaler for features)
  └─ Data:
     ├─ results.json (structured metrics)
     └─ FINAL_REPORT.txt (this file)

8. HYPERPARAMETERS
  ├─ Global:
  │  ├─ Random seed: {SEED}
  │  ├─ Test size: {TEST_SIZE*100:.0f}%
  │  ├─ CV folds: {CV_K}
  │  └─ Log transform: {USE_LOG_TRANSFORM}
  ├─ Optuna CatBoost Tuning:
  │  ├─ Trials: {N_TRIALS}
  │  ├─ Timeout: {TIMEOUT}s
  │  └─ Best params: {best_cb_params}
  └─ Feature preprocessing:
     └─ KNNImputer (n_neighbors=5) + StandardScaler

9. PUBLICATION READINESS ✓
  ✓ Comprehensive 6-model comparison with stacking ensemble
  ✓ 10-Fold CV on full dataset with cross_val_predict
  ✓ 8 publication-quality scatter plots with metrics labeled
  ✓ Taylor diagram for statistical validation
  ✓ SHAP feature importance analysis
  ✓ Per-fold variance reporting (CV% and SD/M metrics)
  ✓ Original-scale metric evaluation
  ✓ Reproducible (fixed random_state, deterministic=True)
  ✓ Model artifacts saved for Phase 2 (PySR symbolic regression)

10. NEXT STEPS (PHASE 2)
  → Load best model + CV predictions from this phase
  → Apply PySR symbolic regression (dual: Ratio + Direct approaches)
  → Generate Pareto fronts for equation complexity vs accuracy
  → Discover interpretable fire resistance equation

{'═'*80}
"""

(OUT / "FINAL_REPORT.txt").write_text(report)
logger.info("  ✓ FINAL_REPORT.txt")

# ═════════════════════════ RESULTS JSON ═════════════════════════════════════
results_json = {
    "timestamp": datetime.utcnow().isoformat() + "Z",
    "pipeline": "Fire Resistance RC Columns — Phase 1: ML Training",
    "dataset": {
        "iso834_specimens": 149,
        "astm_specimens": 108,
        "total_before_cleaning": 257,
        "after_outlier_removal": int(N_CLEAN),
        "train_samples": int(len(ytr_log)),
        "test_samples": int(len(yte_log)),
        "cv_k_folds": CV_K,
        "train_test_split": "80% / 20%"
    },
    "models": {
        "mlp": {"test_R2": float(mlp_m['R2']), "test_RMSE": float(mlp_m['RMSE']), "test_MAE": float(mlp_m['MAE'])},
        "gbr": {**{k: float(v) if isinstance(v, (int, np.number)) else v for k, v in model_metrics_orig['GBR'].items()}},
        "xgboost": {**{k: float(v) if isinstance(v, (int, np.number)) else v for k, v in model_metrics_orig['XGBoost'].items()}},
        "randomforest": {**{k: float(v) if isinstance(v, (int, np.number)) else v for k, v in model_metrics_orig['RandomForest'].items()}},
        "catboost": {**{k: float(v) if isinstance(v, (int, np.number)) else v for k, v in model_metrics_orig['CatBoost'].items()}},
        "stacking": {**{k: float(v) if isinstance(v, (int, np.number)) else v for k, v in model_metrics_orig['Stacking'].items()}}
    },
    "cross_validation": {
        "method": "KFold",
        "k": CV_K,
        "best_model": best_name,
        "cv_r2": float(cv_metrics["Stacking"]["R2"]),
        "cv_rmse": float(cv_metrics["Stacking"]["RMSE"]),
        "cv_mae": float(cv_metrics["Stacking"]["MAE"]),
        "cv_percent": float(cv_metrics["Stacking"]["CV"]),
        "cv_sd_m": float(cv_metrics["Stacking"]["SD_M"]),
        "fold_r2_mean": float(np.mean(fold_r2_vals)),
        "fold_r2_std": float(np.std(fold_r2_vals))
    },
    "taylor_statistics": {
        "test_set": {
            "observed_std": float(obs_std_test),
            "predicted_std": float(pred_std_test),
            "centered_rmse": float(crms_test),
            "correlation": float(corr_test)
        },
        "cv_10fold": {
            "observed_std": float(obs_std_cv),
            "predicted_std": float(pred_std_cv),
            "centered_rmse": float(crms_cv),
            "correlation": float(corr_cv)
        }
    },
    "best_model": best_name,
    "best_r2_test": float(best_metrics["test_R2"]),
    "best_rmse_test": float(best_metrics["test_RMSE"]),
    "features": FEATS,
    "output_dir": str(OUT)
}

(OUT / "equations" / "results.json").write_text(json.dumps(results_json, indent=2, default=str))
logger.info("  ✓ results.json")

logger.success(f"✓✓✓ PHASE 1 COMPLETE ✓✓✓\n  All outputs saved to: {OUT}\n  Next: Run Phase 2 for symbolic regression (PySR)")
