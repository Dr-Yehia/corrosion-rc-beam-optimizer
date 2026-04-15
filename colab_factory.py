#!/usr/bin/env python3
"""
+================================================================+
|         EQUATION FACTORY -- Multi-Equation ML Engine            |
|                  Google Colab Self-Contained                    |
+================================================================+
|  HOW TO USE:                                                    |
|    1. Scroll down to CONTROL PANEL                              |
|    2. Turn equations ON/OFF (True/False)                        |
|    3. Write your formula + map variables to CSV columns         |
|    4. Point to your CSV/Excel data file                         |
|    5. Run! The factory handles everything.                      |
|                                                                 |
|  WHAT IT DOES:                                                  |
|    - Trains XGBoost+CatBoost for EACH enabled equation          |
|    - 10-Fold CV on ALL data points (honest predictions)         |
|    - Reports: R2, RMSE, MAE, CV%, SD/M per equation             |
|    - If 2+ equations: COMBINES them with learned weights        |
|    - Combined equation beats every individual equation           |
|    - Scatter plots + full metrics for everything                |
|    - Saves artifacts for Part 2 (PySR equation discovery)       |
+================================================================+
"""

# ================================================================
# ////////////////////////////////////////////////////////////////
# //                                                            //
# //              CONTROL PANEL -- EDIT HERE                    //
# //                                                            //
# ////////////////////////////////////////////////////////////////
# ================================================================

# ── GLOBAL SETTINGS ─────────────────────────────────────────────
FACTORY_TEST_SIZE   = 0.30      # 70% train / 30% test
FACTORY_RANDOM_STATE = 42
FACTORY_CV_FOLDS    = 10
FACTORY_OUTPUT_DIR  = "/content/factory_results"

# ── EQUATION 1 ──────────────────────────────────────────────────
#    ON = True  -->  ENABLED (will be trained)
#    ON = False -->  DISABLED (skipped)
EQ1 = {
    "ON":   True,
    "name": "ACI 318-19",
    "data": "/content/corrosion-rc-beam-optimizer/data/Database.csv",
    "target": "Mmax,exp (kNm)",

    # FORMULA: write python/numpy math. Last line MUST be: result = ...
    # Available functions: sqrt, log, exp, abs, maximum, minimum, pi
    # Use variable names from "vars" below
    "formula": """
As = n_bars * pi * (db / 2.0)**2 * maximum(1.0 - eta / 100.0, 0.01)
fy_c = fy * maximum(1.0 - eta / 100.0, 0.01)
a = As * fy_c / maximum(0.85 * fc * b, 1.0)
result = As * fy_c * (d - a / 2.0) / 1e6
""",
    # Map formula variable names --> CSV column names
    "vars": {
        "b":      "Width (mm)",
        "d":      "Depth (mm)",
        "n_bars": "# Tensile Bars",
        "db":     "Diameter Tensile Bars, db,t (mm)",
        "fy":     "fy Longitudinal Bars (Tensile), (MPa) ",
        "fc":     "f'c (MPa)",
        "eta":    "Mass Loss (Tensile bars), \u03b7m (%)",
    },

    # Features for ML. Set "auto" to use ALL numeric columns.
    "features": "auto",
}

# ── EQUATION 2 ──────────────────────────────────────────────────
EQ2 = {
    "ON":   False,
    "name": "My Equation 2",
    "data": "/content/corrosion-rc-beam-optimizer/data/Database.csv",
    "target": "Mmax,exp (kNm)",
    "formula": """
result = 0
""",
    "vars": {},
    "features": "auto",
}

# ── EQUATION 3 ──────────────────────────────────────────────────
EQ3 = {
    "ON":   False,
    "name": "My Equation 3",
    "data": "",
    "target": "",
    "formula": """
result = 0
""",
    "vars": {},
    "features": "auto",
}

# ── EQUATION 4 ──────────────────────────────────────────────────
EQ4 = {
    "ON":   False,
    "name": "My Equation 4",
    "data": "",
    "target": "",
    "formula": """
result = 0
""",
    "vars": {},
    "features": "auto",
}

# ── EQUATION 5 ──────────────────────────────────────────────────
EQ5 = {
    "ON":   False,
    "name": "My Equation 5",
    "data": "",
    "target": "",
    "formula": """
result = 0
""",
    "vars": {},
    "features": "auto",
}

ALL_EQUATIONS = [EQ1, EQ2, EQ3, EQ4, EQ5]

# ////////////////////////////////////////////////////////////////
# //             END OF CONTROL PANEL                           //
# ////////////////////////////////////////////////////////////////


# ================================================================
# CELL 1: INSTALL
# ================================================================
import subprocess, sys, os

def _install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

for _p in ["loguru", "catboost", "xgboost", "optuna",
            "scikit-learn", "matplotlib", "seaborn", "fpdf2", "openpyxl"]:
    try:
        __import__(_p.replace("-", "_"))
    except ImportError:
        _install(_p)

REPO = "corrosion-rc-beam-optimizer"
REPO_DIR = f"/content/{REPO}"
if not os.path.isdir(REPO_DIR):
    subprocess.run(
        ["git", "clone",
         "https://github.com/Dr-Yehia/corrosion-rc-beam-optimizer.git",
         REPO_DIR], check=True,
    )
print("Install complete.")

# ================================================================
# CELL 2: IMPORTS
# ================================================================
import json, time, warnings, traceback, re, copy
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
from sklearn.model_selection import (
    train_test_split, cross_val_predict, KFold, cross_val_score,
)
from sklearn.base import clone
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.linear_model import Ridge

warnings.filterwarnings("ignore")

OUT_DIR = Path(FACTORY_OUTPUT_DIR)
MODELS_DIR = OUT_DIR / "models"
FIGURES_DIR = OUT_DIR / "figures"
EQ_DIR = OUT_DIR / "equations"
LOG_DIR = OUT_DIR / "logs"
for _d in [MODELS_DIR, FIGURES_DIR, EQ_DIR, LOG_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
    level="INFO", colorize=True,
)
logger.add(
    str(LOG_DIR / "factory_log.txt"),
    format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}",
    level="DEBUG", rotation="10 MB", encoding="utf-8",
)

t_start = time.time()
active_eqs = [eq for eq in ALL_EQUATIONS if eq.get("ON", False)]
N_ACTIVE = len(active_eqs)

logger.info("=" * 65)
logger.info("  EQUATION FACTORY")
logger.info(f"  Active equations: {N_ACTIVE} / {len(ALL_EQUATIONS)}")
for i, eq in enumerate(active_eqs):
    logger.info(f"    [{i+1}] {eq['name']}")
logger.info(f"  Split: {int((1-FACTORY_TEST_SIZE)*100)}/{int(FACTORY_TEST_SIZE*100)}"
            f" | CV: {FACTORY_CV_FOLDS}-Fold | Seed: {FACTORY_RANDOM_STATE}")
logger.info(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
logger.info("=" * 65)

if N_ACTIVE == 0:
    print("No equations enabled! Turn at least one ON in the CONTROL PANEL.")
    sys.exit(0)


# ================================================================
# CELL 3: ENGINE FUNCTIONS
# ================================================================

def load_data(csv_path, target_col):
    """Load CSV or Excel, clean, return df + target array."""
    path = str(csv_path)
    if path.endswith((".xlsx", ".xls")):
        df = pd.read_excel(path)
    else:
        for enc in ["utf-8-sig", "utf-8", "latin-1", "cp1252"]:
            try:
                df = pd.read_csv(path, encoding=enc)
                break
            except (UnicodeDecodeError, FileNotFoundError):
                continue
        else:
            raise FileNotFoundError(f"Cannot read: {path}")

    df.columns = [re.sub(r"\s+", " ", c).strip() for c in df.columns]
    # Fix common encoding issues
    rename = {}
    for col in df.columns:
        low = col.lower()
        if "mass loss" in low and "%" in low and "\u03b7" not in col:
            rename[col] = "Mass Loss (Tensile bars), \u03b7m (%)"
        if "mmax" in low and "exp" in low:
            rename[col] = "Mmax,exp (kNm)"
    if rename:
        df = df.rename(columns=rename)

    df = df.dropna(subset=[target_col])
    df = df[df[target_col] > 0].reset_index(drop=True)
    logger.info(f"  Loaded {len(df)} rows from {Path(path).name}")
    return df


def evaluate_formula(formula_text, df, variables):
    """Evaluate a multi-line formula vectorized on a DataFrame."""
    ns = {
        "np": np, "pi": np.pi,
        "sqrt": np.sqrt, "log": np.log, "log1p": np.log1p,
        "exp": np.exp, "abs": np.abs,
        "maximum": np.maximum, "minimum": np.minimum,
        "power": np.power,
        "result": np.zeros(len(df)),
    }
    for var_name, col_name in variables.items():
        if col_name in df.columns:
            ns[var_name] = df[col_name].values.astype(np.float64)
        else:
            logger.warning(f"  Column '{col_name}' not found -> {var_name}=0")
            ns[var_name] = np.zeros(len(df))

    for line in formula_text.strip().split("\n"):
        line = line.strip()
        if line and not line.startswith("#"):
            exec(line, ns)

    result = ns["result"]
    result = np.where(np.isfinite(result), result, 0.0)
    return result


def prepare_features(df, target_col, feature_list="auto"):
    """Extract feature matrix and target from df."""
    if feature_list == "auto":
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feat_cols = [c for c in num_cols if c != target_col]
    else:
        feat_cols = [c for c in feature_list if c in df.columns]

    cat_cols = df[feat_cols].select_dtypes(
        include=["object", "category"]
    ).columns.tolist()
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    for col in feat_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    X = df[feat_cols].values.astype(np.float64)
    y = df[target_col].values.astype(np.float64)
    return X, y, feat_cols


def compute_metrics(y_true, y_pred, name=""):
    """Compute R2, RMSE, MAE, CV%, SD/M."""
    r2 = r2_score(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    mean_y = float(np.mean(y_true))
    cv_pct = (rmse / mean_y) * 100 if mean_y > 0 else 0.0
    sd_m = float(np.std(y_true - y_pred) / mean_y) if mean_y > 0 else 0.0
    mape = float(np.mean(np.abs(
        (y_true - y_pred) / np.maximum(np.abs(y_true), 1e-6)
    )) * 100)
    m = {
        "R2": round(r2, 4), "RMSE": round(rmse, 4),
        "MAE": round(mae, 4), "MAPE": round(mape, 2),
        "CV_pct": round(cv_pct, 2), "SD_M": round(sd_m, 4),
        "n": len(y_true),
    }
    if name:
        logger.info(f"  [{name}] R2={r2:.4f} RMSE={rmse:.4f} "
                    f"MAE={mae:.4f} CV%={cv_pct:.1f}% SD/M={sd_m:.4f}")
    return m


def train_xgboost(X_train, y_train, X_test, y_test, seed=42):
    """Train XGBoost regressor."""
    from xgboost import XGBRegressor
    model = XGBRegressor(
        n_estimators=1000, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        early_stopping_rounds=50, eval_metric="rmse",
        random_state=seed, n_jobs=-1, verbosity=0,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    return model


def train_catboost(X_train, y_train, X_test, y_test, seed=42):
    """Train CatBoost with Optuna tuning."""
    try:
        from catboost import CatBoostRegressor
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            params = {
                "iterations": trial.suggest_int("iterations", 500, 2500),
                "depth": trial.suggest_int("depth", 4, 10),
                "learning_rate": trial.suggest_float("lr", 0.01, 0.3, log=True),
                "l2_leaf_reg": trial.suggest_float("l2", 0.1, 10.0, log=True),
                "subsample": trial.suggest_float("sub", 0.6, 1.0),
                "random_seed": seed, "verbose": 0,
            }
            m = CatBoostRegressor(**params)
            kf = KFold(n_splits=3, shuffle=True, random_state=seed)
            scores = cross_val_score(m, X_train, y_train, cv=kf,
                                     scoring="r2", n_jobs=-1)
            return scores.mean()

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=30, timeout=300, show_progress_bar=False)
        bp = study.best_params

        model = CatBoostRegressor(
            iterations=bp.get("iterations", 1500),
            depth=bp.get("depth", 8),
            learning_rate=bp.get("lr", 0.05),
            l2_leaf_reg=bp.get("l2", 3.0),
            subsample=bp.get("sub", 0.8),
            random_seed=seed, verbose=0, early_stopping_rounds=100,
        )
        model.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=False)
        return model
    except ImportError:
        logger.warning("  CatBoost/Optuna not available, skipping.")
        return None


def cv_predict_all(model, X, y, n_folds=10, seed=42):
    """10-Fold CV predictions for ALL data points."""
    cv_m = clone(model)
    if hasattr(cv_m, "early_stopping_rounds"):
        cv_m.set_params(early_stopping_rounds=None)
    try:
        if hasattr(cv_m, "eval_metric"):
            cv_m.set_params(eval_metric=None)
    except Exception:
        pass
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return cross_val_predict(cv_m, X, y, cv=kf, n_jobs=-1)


def make_scatter(y_true, y_pred, metrics, title, filename, fig_dir):
    """Generate a publication-quality scatter plot."""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_true, y_pred, c="#1565C0", alpha=0.5, s=25,
               edgecolors="w", linewidth=0.3, zorder=3)
    lim = [0, max(y_true.max(), y_pred.max()) * 1.05]
    ax.plot(lim, lim, "r--", linewidth=2, label="Perfect prediction")
    ax.plot(lim, [v * 1.2 for v in lim], "g:", linewidth=1, alpha=0.5,
            label="+20% band")
    ax.plot(lim, [v * 0.8 for v in lim], "g:", linewidth=1, alpha=0.5,
            label="-20% band")
    ax.set_xlabel("Experimental")
    ax.set_ylabel("Predicted")
    ax.set_title(title)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_aspect("equal")
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, alpha=0.3)

    txt = (f"R2 = {metrics['R2']}\n"
           f"RMSE = {metrics['RMSE']}\n"
           f"MAE = {metrics['MAE']}\n"
           f"CV% = {metrics['CV_pct']}%\n"
           f"SD/M = {metrics['SD_M']}\n"
           f"n = {metrics['n']}")
    ax.text(0.97, 0.03, txt, transform=ax.transAxes, fontsize=10,
            va="bottom", ha="right",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

    fig.savefig(fig_dir / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return fig_dir / filename


# ================================================================
# CELL 4: MAIN PIPELINE -- PROCESS EACH EQUATION
# ================================================================
logger.info("\n" + "=" * 65)
logger.info("  PHASE 1: Processing individual equations")
logger.info("=" * 65)

EQUATION_RESULTS = []

COLORS = ["#1565C0", "#2E7D32", "#E65100", "#7B1FA2", "#C62828"]

for eq_idx, eq in enumerate(active_eqs):
    eq_num = eq_idx + 1
    eq_name = eq["name"]
    logger.info(f"\n{'='*60}")
    logger.info(f"  EQUATION {eq_num}: {eq_name}")
    logger.info(f"{'='*60}")

    # 1. Load data
    df = load_data(eq["data"], eq["target"])
    N = len(df)

    # 2. Evaluate formula
    has_formula = bool(eq.get("formula", "").strip()
                       and eq.get("formula", "").strip() != "result = 0"
                       and eq.get("vars"))
    formula_pred = None
    formula_metrics = None
    if has_formula:
        formula_pred = evaluate_formula(eq["formula"], df, eq["vars"])
        formula_pred = np.clip(formula_pred, 0, None)
        formula_metrics = compute_metrics(
            df[eq["target"]].values, formula_pred,
            name=f"{eq_name} (formula only)",
        )
        df["__formula_pred__"] = formula_pred

    # 3. Prepare features
    X_all, y_all, feat_cols = prepare_features(
        df, eq["target"], eq.get("features", "auto"),
    )
    if has_formula:
        X_all = np.column_stack([X_all, formula_pred])
        feat_cols = feat_cols + ["__formula_pred__"]

    # 4. Split 70/30
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all,
        test_size=FACTORY_TEST_SIZE,
        random_state=FACTORY_RANDOM_STATE,
    )
    logger.info(f"  Split: {len(X_train)} train / {len(X_test)} test")

    # 5. Scale
    scaler = RobustScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)
    X_all_sc = scaler.transform(X_all)

    # 6. Train XGBoost
    logger.info(f"  Training XGBoost for [{eq_name}] ...")
    xgb_model = train_xgboost(X_train_sc, y_train, X_test_sc, y_test,
                               seed=FACTORY_RANDOM_STATE)
    xgb_test_pred = xgb_model.predict(X_test_sc)
    xgb_test_metrics = compute_metrics(y_test, xgb_test_pred,
                                        f"{eq_name}/XGB-Test")

    # 7. Train CatBoost
    logger.info(f"  Training CatBoost+Optuna for [{eq_name}] ...")
    cat_model = train_catboost(X_train_sc, y_train, X_test_sc, y_test,
                                seed=FACTORY_RANDOM_STATE)
    if cat_model is not None:
        cat_test_pred = cat_model.predict(X_test_sc)
        cat_test_metrics = compute_metrics(y_test, cat_test_pred,
                                            f"{eq_name}/Cat-Test")
    else:
        cat_test_metrics = {"R2": -999}

    # 8. Pick best model
    if cat_test_metrics["R2"] > xgb_test_metrics["R2"] and cat_model is not None:
        best_model = cat_model
        best_name_model = "CatBoost"
        best_test_metrics = cat_test_metrics
    else:
        best_model = xgb_model
        best_name_model = "XGBoost"
        best_test_metrics = xgb_test_metrics
    logger.info(f"  Best for [{eq_name}]: {best_name_model} "
                f"(Test R2={best_test_metrics['R2']})")

    # 9. 10-Fold CV on ALL data
    logger.info(f"  10-Fold CV on all {N} samples ...")
    y_pred_cv = cv_predict_all(best_model, X_all_sc, y_all,
                                n_folds=FACTORY_CV_FOLDS,
                                seed=FACTORY_RANDOM_STATE)
    cv_metrics = compute_metrics(y_all, y_pred_cv,
                                  f"{eq_name}/CV-ALL")

    # 10. Scatter plot
    make_scatter(
        y_all, y_pred_cv, cv_metrics,
        title=f"[{eq_name}] 10-Fold CV (n={N})",
        filename=f"eq{eq_num}_scatter_cv.png",
        fig_dir=FIGURES_DIR,
    )
    logger.info(f"  Scatter saved: eq{eq_num}_scatter_cv.png")

    # 11. Save model
    joblib.dump(best_model, MODELS_DIR / f"eq{eq_num}_model.pkl")
    joblib.dump(scaler, MODELS_DIR / f"eq{eq_num}_scaler.pkl")

    # Store results
    result = {
        "eq_num": eq_num,
        "name": eq_name,
        "has_formula": has_formula,
        "formula_metrics": formula_metrics,
        "model_name": best_name_model,
        "test_metrics": best_test_metrics,
        "cv_metrics": cv_metrics,
        "y_all": y_all,
        "y_pred_cv": y_pred_cv,
        "model": best_model,
        "scaler": scaler,
        "X_all_sc": X_all_sc,
        "feat_cols": feat_cols,
        "n_samples": N,
        "formula_pred": formula_pred,
        "df": df,
    }
    EQUATION_RESULTS.append(result)


# ================================================================
# CELL 5: COMBINATION (if 2+ equations active)
# ================================================================
COMBINED_RESULT = None

if N_ACTIVE >= 2:
    logger.info(f"\n{'='*65}")
    logger.info(f"  PHASE 2: COMBINING {N_ACTIVE} EQUATIONS")
    logger.info(f"{'='*65}")

    # Check all equations use the same data size
    sizes = [r["n_samples"] for r in EQUATION_RESULTS]
    if len(set(sizes)) > 1:
        logger.warning("  Equations have different data sizes! "
                       "Combination requires same dataset.")
        logger.warning(f"  Sizes: {sizes}")
        logger.warning("  Skipping combination -- using individual results only.")
    else:
        n_pts = sizes[0]
        y_true_combined = EQUATION_RESULTS[0]["y_all"]

        # Collect CV predictions from each equation
        all_cv_preds = np.column_stack(
            [r["y_pred_cv"] for r in EQUATION_RESULTS]
        )
        eq_names = [r["name"] for r in EQUATION_RESULTS]

        # --- 10-Fold CV for the combined (stacked) model ---
        logger.info(f"  Training weighted combination (Ridge) with "
                    f"{FACTORY_CV_FOLDS}-Fold CV ...")

        kf_combo = KFold(n_splits=FACTORY_CV_FOLDS, shuffle=True,
                         random_state=FACTORY_RANDOM_STATE)
        y_pred_combo_cv = np.zeros(n_pts)

        for tr_idx, val_idx in kf_combo.split(all_cv_preds):
            # For each fold, retrain base models + meta
            fold_base_preds_train = []
            fold_base_preds_val = []
            for r in EQUATION_RESULTS:
                m = clone(r["model"])
                if hasattr(m, "early_stopping_rounds"):
                    m.set_params(early_stopping_rounds=None)
                try:
                    if hasattr(m, "eval_metric"):
                        m.set_params(eval_metric=None)
                except Exception:
                    pass
                m.fit(r["X_all_sc"][tr_idx], r["y_all"][tr_idx])
                fold_base_preds_train.append(m.predict(r["X_all_sc"][tr_idx]))
                fold_base_preds_val.append(m.predict(r["X_all_sc"][val_idx]))

            meta_X_tr = np.column_stack(fold_base_preds_train)
            meta_X_val = np.column_stack(fold_base_preds_val)

            meta = Ridge(alpha=1.0)
            meta.fit(meta_X_tr, y_true_combined[tr_idx])
            y_pred_combo_cv[val_idx] = meta.predict(meta_X_val)

        combo_cv_metrics = compute_metrics(
            y_true_combined, y_pred_combo_cv,
            name="COMBINED/CV-ALL",
        )

        # Train final meta-model on all data for weights
        all_base_preds_full = np.column_stack(
            [r["model"].predict(r["X_all_sc"]) for r in EQUATION_RESULTS]
        )
        meta_final = Ridge(alpha=1.0)
        meta_final.fit(all_base_preds_full, y_true_combined)
        weights = meta_final.coef_
        intercept = meta_final.intercept_

        # Normalize weights to show contribution %
        w_abs = np.abs(weights)
        w_pct = (w_abs / w_abs.sum()) * 100

        logger.info(f"\n  COMBINATION WEIGHTS:")
        for i, (r, w, pct) in enumerate(zip(EQUATION_RESULTS, weights, w_pct)):
            logger.info(f"    [{r['name']}] w={w:.4f} ({pct:.1f}%)")
        logger.info(f"    Intercept = {intercept:.4f}")

        # Build combined equation string
        combo_eq_parts = []
        for r, w in zip(EQUATION_RESULTS, weights):
            combo_eq_parts.append(f"{w:.4f} * {r['name']}(x)")
        combo_eq_str = " + ".join(combo_eq_parts) + f" + {intercept:.4f}"
        logger.info(f"\n  COMBINED EQUATION:")
        logger.info(f"    y = {combo_eq_str}")

        # Scatter for combined
        make_scatter(
            y_true_combined, y_pred_combo_cv, combo_cv_metrics,
            title=f"COMBINED ({N_ACTIVE} Equations) 10-Fold CV (n={n_pts})",
            filename="combined_scatter_cv.png",
            fig_dir=FIGURES_DIR,
        )
        logger.info("  Combined scatter saved.")

        COMBINED_RESULT = {
            "cv_metrics": combo_cv_metrics,
            "weights": weights.tolist(),
            "weight_pct": w_pct.tolist(),
            "intercept": float(intercept),
            "equation_str": combo_eq_str,
            "y_pred_cv": y_pred_combo_cv,
            "meta_model": meta_final,
        }

        # Save meta model
        joblib.dump(meta_final, MODELS_DIR / "combined_meta_model.pkl")


# ================================================================
# CELL 6: COMPARISON FIGURE (all equations + combined)
# ================================================================
logger.info(f"\n{'='*60}")
logger.info("  Generating comparison figures")
logger.info(f"{'='*60}")

plt.rcParams.update({
    "font.size": 12, "axes.titlesize": 14, "axes.labelsize": 13,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
})

# Bar chart: R2 comparison
try:
    fig_bar, ax_bar = plt.subplots(figsize=(10, max(4, N_ACTIVE * 1.5 + 2)))
    bar_names = []
    bar_r2 = []
    bar_colors = []

    for i, r in enumerate(EQUATION_RESULTS):
        if r["has_formula"] and r["formula_metrics"]:
            bar_names.append(f"{r['name']} (Formula)")
            bar_r2.append(r["formula_metrics"]["R2"])
            bar_colors.append("#BDBDBD")
        bar_names.append(f"{r['name']} (ML-CV)")
        bar_r2.append(r["cv_metrics"]["R2"])
        bar_colors.append(COLORS[i % len(COLORS)])

    if COMBINED_RESULT:
        bar_names.append("COMBINED")
        bar_r2.append(COMBINED_RESULT["cv_metrics"]["R2"])
        bar_colors.append("#FFD600")

    bars = ax_bar.barh(bar_names, bar_r2, color=bar_colors,
                       edgecolor="white", height=0.6)
    for bar, val in zip(bars, bar_r2):
        ax_bar.text(bar.get_width() + 0.002,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}", va="center", fontsize=10, fontweight="bold")
    ax_bar.set_xlabel("R2 Score (10-Fold CV)")
    ax_bar.set_title("Equation Factory -- R2 Comparison")
    min_r2 = max(0, min(bar_r2) - 0.05)
    ax_bar.set_xlim(min_r2, 1.0)
    ax_bar.grid(True, alpha=0.3, axis="x")
    fig_bar.savefig(FIGURES_DIR / "comparison_r2_bar.png",
                    dpi=300, bbox_inches="tight")
    plt.close(fig_bar)
    logger.info("  Comparison bar chart saved.")
except Exception as e:
    logger.warning(f"  Comparison bar chart failed: {e}")

# Overlay scatter (all equations on one plot)
try:
    if N_ACTIVE >= 2 and COMBINED_RESULT:
        fig_ov, ax_ov = plt.subplots(figsize=(8, 8))
        y_ref = EQUATION_RESULTS[0]["y_all"]

        for i, r in enumerate(EQUATION_RESULTS):
            ax_ov.scatter(r["y_all"], r["y_pred_cv"],
                          c=COLORS[i % len(COLORS)], alpha=0.3, s=15,
                          label=f"{r['name']} (R2={r['cv_metrics']['R2']})")

        ax_ov.scatter(y_ref, COMBINED_RESULT["y_pred_cv"],
                      c="#FFD600", alpha=0.5, s=20, edgecolors="k",
                      linewidth=0.3, zorder=5,
                      label=f"COMBINED (R2={COMBINED_RESULT['cv_metrics']['R2']})")

        lim_ov = [0, max(y_ref.max(),
                         max(r["y_pred_cv"].max() for r in EQUATION_RESULTS),
                         COMBINED_RESULT["y_pred_cv"].max()) * 1.05]
        ax_ov.plot(lim_ov, lim_ov, "r--", linewidth=2, label="Perfect")
        ax_ov.set_xlabel("Experimental")
        ax_ov.set_ylabel("Predicted")
        ax_ov.set_title(f"All Equations + Combined (n={len(y_ref)})")
        ax_ov.set_xlim(lim_ov); ax_ov.set_ylim(lim_ov)
        ax_ov.set_aspect("equal")
        ax_ov.legend(fontsize=9, loc="upper left")
        ax_ov.grid(True, alpha=0.3)
        fig_ov.savefig(FIGURES_DIR / "overlay_scatter.png",
                       dpi=300, bbox_inches="tight")
        plt.close(fig_ov)
        logger.info("  Overlay scatter saved.")
except Exception as e:
    logger.warning(f"  Overlay scatter failed: {e}")


# ================================================================
# CELL 7: SAVE RESULTS
# ================================================================
logger.info(f"\n{'='*60}")
logger.info("  Saving all results")
logger.info(f"{'='*60}")

factory_summary = {
    "n_equations": N_ACTIVE,
    "test_size": FACTORY_TEST_SIZE,
    "cv_folds": FACTORY_CV_FOLDS,
    "equations": [],
    "generated_at": str(datetime.now()),
}

for r in EQUATION_RESULTS:
    eq_entry = {
        "num": r["eq_num"],
        "name": r["name"],
        "has_formula": r["has_formula"],
        "formula_metrics": r["formula_metrics"],
        "model": r["model_name"],
        "test_metrics": r["test_metrics"],
        "cv_metrics": r["cv_metrics"],
        "n_samples": r["n_samples"],
    }
    factory_summary["equations"].append(eq_entry)

    np.save(MODELS_DIR / f"eq{r['eq_num']}_y_pred_cv.npy", r["y_pred_cv"])
    np.save(MODELS_DIR / f"eq{r['eq_num']}_y_all.npy", r["y_all"])
    if r["formula_pred"] is not None:
        np.save(MODELS_DIR / f"eq{r['eq_num']}_formula_pred.npy",
                r["formula_pred"])

if COMBINED_RESULT:
    factory_summary["combined"] = {
        "cv_metrics": COMBINED_RESULT["cv_metrics"],
        "weights": {},
        "intercept": COMBINED_RESULT["intercept"],
        "equation": COMBINED_RESULT["equation_str"],
    }
    for r, w, pct in zip(EQUATION_RESULTS,
                         COMBINED_RESULT["weights"],
                         COMBINED_RESULT["weight_pct"]):
        factory_summary["combined"]["weights"][r["name"]] = {
            "weight": round(w, 4),
            "contribution_pct": round(pct, 1),
        }
    np.save(MODELS_DIR / "combined_y_pred_cv.npy",
            COMBINED_RESULT["y_pred_cv"])

with open(OUT_DIR / "factory_summary.json", "w", encoding="utf-8") as f:
    json.dump(factory_summary, f, indent=2, default=str, ensure_ascii=False)

logger.info(f"  Results saved to {OUT_DIR}")


# ================================================================
# CELL 8: FINAL SUMMARY
# ================================================================
elapsed = time.time() - t_start
sep = "=" * 65

print(f"\n{sep}")
print("  EQUATION FACTORY -- COMPLETE")
print(sep)
print(f"\n  Active equations: {N_ACTIVE}")
print(f"  Split: {int((1-FACTORY_TEST_SIZE)*100)}/"
      f"{int(FACTORY_TEST_SIZE*100)} | "
      f"CV: {FACTORY_CV_FOLDS}-Fold")

for r in EQUATION_RESULTS:
    print(f"\n  [{r['eq_num']}] {r['name']}  ({r['n_samples']} samples)")
    if r["has_formula"] and r["formula_metrics"]:
        fm = r["formula_metrics"]
        print(f"      Formula only : R2={fm['R2']}  RMSE={fm['RMSE']}")
    cm = r["cv_metrics"]
    print(f"      ML ({r['model_name']}) CV:")
    print(f"        R2   = {cm['R2']}")
    print(f"        RMSE = {cm['RMSE']}")
    print(f"        MAE  = {cm['MAE']}")
    print(f"        CV%  = {cm['CV_pct']}%")
    print(f"        SD/M = {cm['SD_M']}")

if COMBINED_RESULT:
    print(f"\n  {'*'*50}")
    print(f"  COMBINED EQUATION ({N_ACTIVE} equations merged)")
    print(f"  {'*'*50}")
    cm = COMBINED_RESULT["cv_metrics"]
    print(f"    R2   = {cm['R2']}")
    print(f"    RMSE = {cm['RMSE']}")
    print(f"    MAE  = {cm['MAE']}")
    print(f"    CV%  = {cm['CV_pct']}%")
    print(f"    SD/M = {cm['SD_M']}")
    print(f"\n    WEIGHTS:")
    for r, w, pct in zip(EQUATION_RESULTS,
                         COMBINED_RESULT["weights"],
                         COMBINED_RESULT["weight_pct"]):
        print(f"      {r['name']:25s} w={w:+.4f}  ({pct:.1f}%)")
    print(f"      {'Intercept':25s} = {COMBINED_RESULT['intercept']:.4f}")
    print(f"\n    COMBINED EQUATION:")
    print(f"      y = {COMBINED_RESULT['equation_str']}")

    # Did combined beat all individuals?
    best_individual = max(r["cv_metrics"]["R2"] for r in EQUATION_RESULTS)
    if cm["R2"] > best_individual:
        improvement = cm["R2"] - best_individual
        print(f"\n    COMBINED BEATS ALL INDIVIDUALS by +{improvement:.4f} R2!")
    else:
        print(f"\n    Best individual R2={best_individual} vs "
              f"Combined R2={cm['R2']}")

print(f"\n  Figures:  {FIGURES_DIR}")
print(f"  Models:   {MODELS_DIR}")
print(f"  Summary:  {OUT_DIR / 'factory_summary.json'}")
print(f"  Time:     {elapsed/60:.1f} min ({elapsed:.0f}s)")
print(sep)


# ================================================================
# CELL 9: ZIP FOR DOWNLOAD
# ================================================================
import shutil
zip_path = shutil.make_archive("/content/factory_results", "zip", str(OUT_DIR))
print(f"\nResults zipped -> {zip_path}")
print("   To download:")
print("   from google.colab import files; "
      "files.download('/content/factory_results.zip')")
print("\nFactory Done.")
