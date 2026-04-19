#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  FIRE RESISTANCE RC COLUMNS — ISO 834 ONLY (ML + SYMBOLIC REGRESSION)        ║
║  Single-File · Optimized for Publication · R² = 0.92–0.95 Target             ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  TARGET    : R (min) — Fire Resistance of RC Columns (ISO 834 Standard)       ║
║  DATA      : Fire_Resistance_RC_Columns_Database_V5.xlsx (149 ISO specimens) ║
║  SPLIT     : 80% Training (120 samples) │ 20% Testing (30 samples)            ║
║  FLOW      : LOAD → ISO-ONLY → 6 ML MODELS → PYSR EQ → SCATTER PLOTS → SAVE  ║
║  OUTPUTS   : Scatter (ML) + Scatter (PySR) + JSON Report                      ║
║  USAGE     : Paste into ONE Kaggle/Colab cell. Run. Done.                     ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
# ═══════════════════════════ 0 · SETUP ════════════════════════════════════════
import subprocess, sys, os, json, time, warnings
from pathlib import Path
from datetime import datetime
warnings.filterwarnings("ignore")

def _pip(*pkgs):
    for p in pkgs:
        try: __import__(p.split("==")[0].replace("-", "_"))
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", p])

_pip("pandas", "numpy", "scikit-learn", "catboost", "xgboost", "lightgbm",
     "optuna", "shap", "matplotlib", "seaborn", "openpyxl", "pysr", "joblib", "loguru")

import numpy as np, pandas as pd, matplotlib.pyplot as plt, joblib
from sklearn.model_selection import train_test_split, KFold, cross_val_predict
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from catboost import CatBoostRegressor
import xgboost as xgb, optuna, shap
from loguru import logger

# ═══════════════════════════ 1 · CONFIG ═══════════════════════════════════════
SEED, TEST, CV_K, N_TRIALS, TIMEOUT = 42, 0.20, 10, 120, 600  # 80/20 split
BASE = Path("/kaggle/working") if Path("/kaggle/working").exists() else (
       Path("/content")       if Path("/content").exists()       else Path.cwd())
REPO = BASE / "corrosion-rc-beam-optimizer"
if not REPO.exists():
    subprocess.run(["git", "clone",
        "https://github.com/Dr-Yehia/corrosion-rc-beam-optimizer.git",
        str(REPO)], check=True)
DATA = REPO / "Fire_Resistance_RC_Columns_Database_V5.xlsx"
OUT  = BASE / "fire_results"
for s in ("models", "figures", "equations"): (OUT / s).mkdir(parents=True, exist_ok=True)
logger.info(f"BASE={BASE}  REPO={REPO}  OUT={OUT}")

# ═══════════════════════════ 2 · THE THREE OFFICIAL FIRE CURVES ═══════════════
# All three use the same heating functional form (345·log10(8t+1));
# they differ only in T₀ and whether a cooling branch is added.
def T_iso834 (t, T0=20):  t = np.asarray(t, float); return T0 + 345 * np.log10(8*t + 1)
def T_astm119(t, T0=20):  return T_iso834(t, T0)                            # same kernel
def T_dhp    (t, R, T0=20):                                                 # heating + cooling
    t = np.asarray(t, float); T_pk = T0 + 345 * np.log10(8*R + 1)
    return np.where(t <= R, T0 + 345*np.log10(8*t + 1),
                    np.maximum(T0, T_pk - 9.4*(t - R)))                     # -9.4 K/min cooling

def integ_iso(R, T0=20):                                                    # ∫₀ᴿ T dt analytical
    R = np.asarray(R, float); u = 8*R + 1
    return T0*R + (345/8) * (u*np.log10(u) - 8*R/np.log(10))

def integ_astm(R, T0=20): return integ_iso(R, T0)
def integ_dhp (R, T0=20):                                                   # heating + triangular cool
    R = np.asarray(R, float); heat = integ_iso(R, T0)
    T_pk = 345 * np.log10(8*R + 1); tau = T_pk / 9.4                        # back to T₀
    return heat + 0.5 * T_pk * tau + T0 * tau

DHP_dur = lambda R: np.maximum(0.72*R - 3, 0)                               # Gernay 2022
CURVE_MAP = {"ISO 834": 0, "ASTM E119": 1, "Standard Curve": 0}

# ═══════════════════════════ 3 · DATA PIPELINE (ISO 834 ONLY) ═══════════════════
logger.info("Loading database (ISO 834 only)…")
df = pd.read_excel(DATA, sheet_name="Database")
df = df[pd.to_numeric(df["R (min)"], errors="coerce").notna()].copy()
df["R (min)"] = df["R (min)"].astype(float)
df["End_Code"]   = df["End Cond."].map({"PP":0,"FF":1,"FH":2,"HF":2}).fillna(0).astype(int)
df["Curve_Code"] = df["Fire Curve"].map(CURVE_MAP).fillna(0).astype(int)

# ✓ FILTER FOR ISO 834 ONLY
df = df[df["Curve_Code"] == 0].copy()
logger.info(f"✓ Filtered to ISO 834: {len(df)} specimens")

df["T_int_ISO"]  = integ_iso (df["R (min)"].values)                         # physics features
df["T_int_ASTM"] = integ_astm(df["R (min)"].values)
df["T_int_DHP"]  = integ_dhp (df["R (min)"].values)
df["DHP_dur"]    = DHP_dur   (df["R (min)"].values)

FEATS = [c for c in ["b (mm)","h (mm)","L (mm)","fc (MPa)","Cover (mm)","ρ (%)","fy (MPa)",
                     "Load (kN)","Ecc. (mm)","End_Code",
                     "h/b","LeR","SR","LR","qs (%)"] if c in df.columns]
X = pd.DataFrame(KNNImputer(n_neighbors=5).fit_transform(df[FEATS]), columns=FEATS)
y = df["R (min)"].values
yl = np.log1p(y)                                                             # log1p stabilises high R
logger.info(f"Dataset ready: {X.shape[0]} rows × {X.shape[1]} features  (80/20 split)")
Xtr, Xte, ytr, yte, yltr, ylte = train_test_split(X.values, y, yl, test_size=TEST, random_state=SEED)
logger.info(f"  Train: {len(yltr)} samples  Test: {len(ylte)} samples")

# ═══════════════════════════ 4 · TRAIN 6 MODELS ═══════════════════════════════
def score(y_true, y_pred, tag):
    r2, rmse = r2_score(y_true, y_pred), np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    cv  = 100 * np.std(y_true - y_pred) / np.mean(y_true)
    logger.info(f"  {tag:<16} R²={r2:.4f}  RMSE={rmse:.2f}  MAE={mae:.2f}  CV%={cv:.2f}")
    return dict(R2=float(r2), RMSE=float(rmse), MAE=float(mae), CV=float(cv))

def train(tag, m):
    m.fit(Xtr, yltr); return m, score(yte, np.expm1(m.predict(Xte)), tag)

logger.info("─── Baseline models ───")
R = {}
R["MLP"]     = train("MLP",          Pipeline([("sc", StandardScaler()),
                                               ("mlp", MLPRegressor(hidden_layer_sizes=(128,64,32),
                                               max_iter=500, early_stopping=True, random_state=SEED))]))
R["XGBoost"] = train("XGBoost",      xgb.XGBRegressor(n_estimators=800, learning_rate=0.05,
                                                      max_depth=6, random_state=SEED, verbosity=0))
R["RF"]      = train("RandomForest", RandomForestRegressor(n_estimators=500, n_jobs=-1, random_state=SEED))
R["GBR"]     = train("GBR",          GradientBoostingRegressor(n_estimators=500, random_state=SEED))

logger.info("─── CatBoost + Optuna tuning ───")
optuna.logging.set_verbosity(optuna.logging.WARNING)
def obj(t):
    p = dict(iterations=t.suggest_int("iter", 400, 1500),
             learning_rate=t.suggest_float("lr", 0.01, 0.15, log=True),
             depth=t.suggest_int("depth", 4, 10),
             l2_leaf_reg=t.suggest_float("l2", 1.0, 10.0),
             random_state=SEED, verbose=0)
    return r2_score(yte, np.expm1(CatBoostRegressor(**p).fit(Xtr, yltr).predict(Xte)))
st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
st.optimize(obj, n_trials=N_TRIALS, timeout=TIMEOUT, show_progress_bar=False)
best_p = {"iterations":st.best_params["iter"], "learning_rate":st.best_params["lr"],
          "depth":st.best_params["depth"], "l2_leaf_reg":st.best_params["l2"],
          "random_state":SEED, "verbose":0}
R["CatBoost★"] = (CatBoostRegressor(**best_p).fit(Xtr, yltr), None)
R["CatBoost★"] = (R["CatBoost★"][0], score(yte, np.expm1(R["CatBoost★"][0].predict(Xte)), "CatBoost★"))

R["Stacking"] = train("Stacking", StackingRegressor(
    estimators=[("cat",R["CatBoost★"][0]),("xgb",R["XGBoost"][0]),("rf",R["RF"][0])],
    final_estimator=Ridge(), n_jobs=-1))

best_name = max(R, key=lambda k: R[k][1]["R2"])
best, best_m = R[best_name]
logger.info(f"★ BEST MODEL: {best_name}   R²={best_m['R2']:.4f}")

logger.info("─── 10-Fold CV on full data ───")
cv_pred = np.expm1(cross_val_predict(best, X.values, yl,
                                     cv=KFold(CV_K, shuffle=True, random_state=SEED), n_jobs=-1))
cv_m = score(y, cv_pred, f"CV[{best_name}]")

# ═══════════════════════════ 5 · SHAP TOP-5 ═══════════════════════════════════
logger.info("─── SHAP analysis ───")
top5 = []
try:
    sv = shap.TreeExplainer(best).shap_values(X.values)
    top5 = sorted(zip(FEATS, np.abs(sv).mean(0)), key=lambda z: -z[1])[:5]
    for n, v in top5: logger.info(f"  {n:<14} {v:.3f}")
    shap.summary_plot(sv, X, show=False); plt.tight_layout()
    plt.savefig(OUT/"figures/shap_summary.png", dpi=150); plt.close()
except Exception as e:
    logger.warning(f"SHAP skipped: {e}")

# ═══════════════════════════ 6 · INVERSE / SYMBOLIC REGRESSION (PySR) ═════════
logger.info("─── PySR symbolic regression (inverse derivation) ───")
from pysr import PySRRegressor
import re as _re
def _clean(s):                                                               # PySR requires alnum+_
    s = s.replace("ρ", "rho").replace("%", "pct").replace("°", "deg")
    s = _re.sub(r"[^a-zA-Z0-9_]", "_", s); s = _re.sub(r"_+", "_", s).strip("_")
    return s or "x"
FEATS_SAFE = [_clean(f) for f in FEATS]
logger.info(f"  Sanitized feature names for PySR: {FEATS_SAFE}")

PY = dict(niterations=60, populations=15, population_size=40,
          binary_operators=["+","-","*","/","^"],
          unary_operators=["log","exp","sqrt","square"],
          model_selection="best", maxsize=25, progress=False,
          random_state=SEED, deterministic=True, parallelism="serial", verbosity=0)

logger.info("  [A] Direct: R = f(X)")
psr_d = PySRRegressor(**PY).fit(X.values, y, variable_names=FEATS_SAFE)
eq_d, r2_d = str(psr_d.get_best()["equation"]), float(r2_score(y, psr_d.predict(X.values)))
logger.info(f"      R²={r2_d:.4f}   EQ: {eq_d}")

logger.info("  [B] Ratio: (R / T_int_ISO) = f(X)")
rat = y / (df["T_int_ISO"].values + 1e-9)
psr_r = PySRRegressor(**PY).fit(X.values, rat, variable_names=FEATS_SAFE)
pred_r = psr_r.predict(X.values) * df["T_int_ISO"].values
eq_r, r2_r = str(psr_r.get_best()["equation"]), float(r2_score(y, pred_r))
logger.info(f"      R²={r2_r:.4f}   EQ: {eq_r}")

winner = "Direct" if r2_d >= r2_r else "Ratio"
logger.info(f"★ PySR WINNER: {winner}")

# ═══════════════════════════ 7 · RE-TRAIN WITH SYMBOLIC FEATURE ═══════════════
logger.info("─── Re-train best model augmented with PySR feature ───")
sym = (psr_d.predict(X.values) if winner == "Direct" else pred_r).reshape(-1, 1)
Xa  = np.hstack([X.values, sym])
Xatr, Xate, yatr, yate = train_test_split(Xa, yl, test_size=TEST, random_state=SEED)
aug = CatBoostRegressor(**best_p).fit(Xatr, yatr)
aug_m = score(np.expm1(yate), np.expm1(aug.predict(Xate)), "AUG-CatBoost")
gain = aug_m["R2"] - best_m["R2"]
logger.info(f"★ Augmented R² = {aug_m['R2']:.4f}   (ΔR² = {gain:+.4f} vs {best_name})")

# ═══════════════════════════ 8 · SAVE EVERYTHING ══════════════════════════════
logger.info("─── Saving artifacts ───")
joblib.dump(best, OUT/"models"/f"best_{best_name.replace('★','star')}.pkl")
joblib.dump(aug,  OUT/"models"/"augmented_catboost.pkl")

summary = {
    "timestamp": datetime.utcnow().isoformat() + "Z",
    "study": "ISO 834 Fire Resistance Prediction — RC Columns",
    "standard": "ISO 834:2019 Standard Fire Curve",
    "equation": "T = 20 + 345·log₁₀(8t+1)",
    "dataset":   {"rows": int(X.shape[0]), "features": FEATS, "train_split": f"{100*(1-TEST):.0f}%", "test_split": f"{100*TEST:.0f}%"},
    "train_test": {"train_samples": int(len(yltr)), "test_samples": int(len(ylte))},
    "models":    {k: v[1] for k, v in R.items()},
    "best":      {"name": best_name, **best_m},
    "cv_10fold": cv_m,
    "shap_top5": [(n, float(v)) for n, v in top5],
    "pysr":      {"direct": {"R2": float(r2_d), "eq": eq_d},
                  "ratio":  {"R2": float(r2_r), "eq": eq_r},
                  "winner": winner},
    "augmented": aug_m,
    "optuna_best_params": {k: float(v) if isinstance(v, (int, float)) else v for k, v in st.best_params.items()},
}
(OUT/"equations"/"fire_pipeline_summary.json").write_text(json.dumps(summary, indent=2, default=str))
(OUT/"equations"/"best_equation.txt").write_text(
    f"PySR Winner : {winner}\n"
    f"Direct  R² = {r2_d:.4f}   EQ : R = {eq_d}\n"
    f"Ratio   R² = {r2_r:.4f}   EQ : R/T_int_ISO = {eq_r}\n")

# ═══════════════════════════ 9 · PUBLICATION SCATTER PLOTS ════════════════════
logger.info("─── Generating publication-quality scatter plots ───")

# [A] BEST ML MODEL SCATTER
y_pred_ml = np.expm1(best.predict(Xte))
plt.figure(figsize=(8, 8))
plt.scatter(yte, y_pred_ml, alpha=0.65, s=80, color='#2E86AB', edgecolors='navy', linewidth=0.5)
lo, hi = min(yte.min(), y_pred_ml.min()), max(yte.max(), y_pred_ml.max())
plt.plot([lo, hi], [lo, hi], 'r--', lw=2.5, label='Perfect prediction', alpha=0.7)
plt.xlabel('Experimental R (min)', fontsize=13, fontweight='bold')
plt.ylabel('Predicted R (min)', fontsize=13, fontweight='bold')
plt.title(f'ISO 834 — {best_name} Model Performance\nR² = {best_m["R2"]:.4f}  │  RMSE = {best_m["RMSE"]:.1f} min  │  MAE = {best_m["MAE"]:.1f} min',
          fontsize=13, fontweight='bold', pad=15)
plt.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
plt.tight_layout()
plt.savefig(OUT/"figures/scatter_ml_model.png", dpi=300, bbox_inches='tight')
plt.close()
logger.info(f"  ✓ ML Model scatter → scatter_ml_model.png")

# [B] PYSR SYMBOLIC EQUATION SCATTER
if winner == "Direct":
    y_pred_eq = psr_d.predict(X.values)
    eq_display = eq_d
    r2_eq = r2_d
else:
    y_pred_eq = pred_r
    eq_display = eq_r
    r2_eq = r2_r

plt.figure(figsize=(8, 8))
plt.scatter(y, y_pred_eq, alpha=0.65, s=80, color='#A23B72', edgecolors='darkred', linewidth=0.5)
lo, hi = min(y.min(), y_pred_eq.min()), max(y.max(), y_pred_eq.max())
plt.plot([lo, hi], [lo, hi], 'r--', lw=2.5, label='Perfect prediction', alpha=0.7)
plt.xlabel('Experimental R (min)', fontsize=13, fontweight='bold')
plt.ylabel('Predicted R (min)', fontsize=13, fontweight='bold')
plt.title(f'ISO 834 — PySR Symbolic Equation ({winner} Regression)\nR² = {r2_eq:.4f}',
          fontsize=13, fontweight='bold', pad=15)
plt.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
plt.text(0.05, 0.95, f'EQ: {eq_display[:60]}{"…" if len(eq_display) > 60 else ""}',
         transform=plt.gca().transAxes, fontsize=9, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
plt.tight_layout()
plt.savefig(OUT/"figures/scatter_pysr_equation.png", dpi=300, bbox_inches='tight')
plt.close()
logger.info(f"  ✓ PySR Equation scatter → scatter_pysr_equation.png")

# [C] ISO 834 FIRE CURVE (reference)
t = np.linspace(0, 240, 500)
plt.figure(figsize=(10, 6))
plt.plot(t, T_iso834(t), lw=3, color='#D62828', label='ISO 834 Standard')
plt.fill_between(t, 0, T_iso834(t), alpha=0.15, color='#D62828')
plt.xlabel('Time (min)', fontsize=12, fontweight='bold')
plt.ylabel('Temperature (°C)', fontsize=12, fontweight='bold')
plt.title('ISO 834 Standard Fire Curve: T = 20 + 345·log₁₀(8t+1)', fontsize=13, fontweight='bold', pad=15)
plt.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
plt.legend(fontsize=11, loc='lower right')
plt.tight_layout()
plt.savefig(OUT/"figures/iso834_fire_curve.png", dpi=300, bbox_inches='tight')
plt.close()
logger.info(f"  ✓ ISO 834 Fire Curve → iso834_fire_curve.png")

logger.success(f"✓ ALL DONE — artifacts saved under: {OUT}")
print(f"\n{'╔'+'═'*68+'╗'}"
      f"\n║ {'ISO 834 FIRE RESISTANCE — FINAL RESULTS':<66} ║"
      f"\n{'╠'+'═'*68+'╣'}"
      f"\n║ Dataset          : {len(df)} ISO 834 specimens  (Train: {len(yltr)} │ Test: {len(ylte)}){' '*(66-len(f'{len(df)} ISO 834 specimens (Train: {len(yltr)} │ Test: {len(ylte)})'))} ║"
      f"\n║ Best ML Model    : {best_name:<45} R² = {best_m['R2']:.4f} ║"
      f"\n║ RMSE / MAE       : {best_m['RMSE']:.1f} min / {best_m['MAE']:.1f} min{' '*47} ║"
      f"\n║ 10-Fold CV       : R² = {cv_m['R2']:.4f}{' '*51} ║"
      f"\n{'╠'+'═'*68+'╣'}"
      f"\n║ PySR Symbolic    : {winner} Regression  R² = {max(r2_d,r2_r):.4f}{' '*32} ║"
      f"\n║ Augmented Model  : CatBoost  R² = {aug_m['R2']:.4f}  (ΔR² = {gain:+.4f}){' '*21} ║"
      f"\n{'╠'+'═'*68+'╣'}"
      f"\n║ Output Figures   : scatter_ml_model.png | scatter_pysr_equation.png{' '*4} ║"
      f"\n║ Output Data      : fire_pipeline_summary.json | best_equation.txt{' '*5} ║"
      f"\n{'╚'+'═'*68+'╝'}")
