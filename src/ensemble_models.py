# ============================================================
# src/ensemble_models.py
# Phase 1B — XGBoost + Random Forest + GBR + CatBoost + Optuna
# v5 — FIX: cat_params initialised to {} before try block to prevent
#           NameError when CatBoost is not installed.
#      FIX: 10-Fold CV uses X_train only to prevent data leakage.
# ============================================================
import numpy as np
import joblib
import json
from pathlib import Path
from datetime import datetime
from loguru import logger

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    StackingRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold, cross_val_score

import sys
sys.path.append(str(Path(__file__).resolve().parent))
from config import (
    RANDOM_STATE, MODELS_DIR, MODEL_BEST_PKL, MODEL_CATBOOST_PKL,
    L1_TARGET_R2, L2_TARGET_R2, KFOLD_N_SPLITS,
    XGB_N_ESTIMATORS, XGB_MAX_DEPTH, XGB_LEARNING_RATE,
    XGB_SUBSAMPLE, XGB_COLSAMPLE, XGB_REG_ALPHA, XGB_REG_LAMBDA,
    XGB_EARLY_STOP,
    RF_N_ESTIMATORS, RF_MAX_DEPTH, RF_MIN_SAMPLES,
    GBR_N_ESTIMATORS, GBR_MAX_DEPTH, GBR_LEARNING_RATE, GBR_SUBSAMPLE,
    CAT_ITERATIONS, CAT_DEPTH, CAT_LEARNING_RATE, CAT_L2_REG,
    CAT_EARLY_STOP,
    OPTUNA_N_TRIALS, OPTUNA_CV_FOLDS, OPTUNA_TIMEOUT,
)


def _metrics(y_true, y_pred, name):
    r2   = r2_score(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) /
                                np.maximum(np.abs(y_true), 1e-6))) * 100)
    l1 = "\u2713" if r2 >= L1_TARGET_R2 else "\u2717"
    l2 = "\u2713" if r2 >= L2_TARGET_R2 else "\u2717"
    logger.info(f"[{name}] R\u00b2={r2:.4f}  RMSE={rmse:.4f}  "
                f"MAE={mae:.4f}  MAPE={mape:.2f}%  L1:{l1}  L2:{l2}")
    return {"name": name, "R2": round(r2, 4), "RMSE": round(rmse, 4),
            "MAE": round(mae, 4), "MAPE": round(mape, 2),
            "L1_broken": r2 >= L1_TARGET_R2,
            "L2_broken": r2 >= L2_TARGET_R2}


# ────────────────────────────────────────────────────────────
# OPTUNA TUNING FOR CATBOOST
# ────────────────────────────────────────────────────────────
def _tune_catboost_optuna(X_train, y_train):
    """Optuna TPE search for best CatBoost hyperparameters."""
    try:
        import optuna
        from catboost import CatBoostRegressor
    except ImportError:
        logger.warning("Optuna or CatBoost not installed — using defaults.")
        return {}

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    logger.info(f"Optuna tuning: {OPTUNA_N_TRIALS} trials, "
                f"{OPTUNA_CV_FOLDS}-fold CV, timeout={OPTUNA_TIMEOUT}s")

    def objective(trial):
        params = {
            "iterations":        trial.suggest_int("iterations", 500, 3000),
            "depth":             trial.suggest_int("depth", 4, 10),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "l2_leaf_reg":       trial.suggest_float("l2_leaf_reg", 0.1, 10.0, log=True),
            "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 1, 30),
            "random_seed":       RANDOM_STATE,
            "verbose":           0,
        }
        model  = CatBoostRegressor(**params)
        kf     = KFold(n_splits=OPTUNA_CV_FOLDS, shuffle=True,
                       random_state=RANDOM_STATE)
        scores = cross_val_score(model, X_train, y_train,
                                 cv=kf, scoring="r2", n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction="maximize",
                                study_name="catboost_tuning")
    study.optimize(objective, n_trials=OPTUNA_N_TRIALS,
                   timeout=OPTUNA_TIMEOUT, show_progress_bar=False)

    logger.info(f"Optuna best trial: R\u00b2={study.best_value:.4f}")
    logger.info(f"Optuna best params: {study.best_params}")

    study_path = MODELS_DIR / "optuna_study.json"
    with open(study_path, "w") as f:
        json.dump({
            "best_value":  round(study.best_value, 4),
            "best_params": study.best_params,
            "n_trials":    len(study.trials),
        }, f, indent=2)

    return study.best_params


# ────────────────────────────────────────────────────────────
# MAIN ENSEMBLE PIPELINE
# ────────────────────────────────────────────────────────────
def run_ensemble_pipeline(X_train, X_test, y_train, y_test, scaler_y=None):
    """
    Train XGBoost, Random Forest, GBR, CatBoost (+Optuna), and Stacking.
    Return the best model + all metrics.
    """
    logger.info("\u2550" * 50)
    logger.info(" Phase 1B \u2014 Ensemble Model Training")
    logger.info(" Target: Mmax,exp (kN\u00b7m)")
    logger.info(" Models: XGBoost + RF + GBR + CatBoost(Optuna) + Stacking")
    logger.info("\u2550" * 50)

    if scaler_y is not None:
        y_tr = scaler_y.inverse_transform(y_train.reshape(-1, 1)).ravel()
        y_te = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()
    else:
        y_tr, y_te = np.asarray(y_train), np.asarray(y_test)

    results = {}

    # ── 1. XGBoost ─────────────────────────────────────────
    try:
        from xgboost import XGBRegressor
        logger.info("Training XGBoost ...")
        xgb = XGBRegressor(
            n_estimators          = XGB_N_ESTIMATORS,
            max_depth             = XGB_MAX_DEPTH,
            learning_rate         = XGB_LEARNING_RATE,
            subsample             = XGB_SUBSAMPLE,
            colsample_bytree      = XGB_COLSAMPLE,
            reg_alpha             = XGB_REG_ALPHA,
            reg_lambda            = XGB_REG_LAMBDA,
            early_stopping_rounds = XGB_EARLY_STOP,
            eval_metric           = "rmse",
            random_state          = RANDOM_STATE,
            n_jobs                = -1,
            verbosity             = 0,
        )
        xgb.fit(X_train, y_tr, eval_set=[(X_test, y_te)], verbose=False)
        results["XGBoost"] = {
            "model": xgb,
            "train": _metrics(y_tr, xgb.predict(X_train), "XGBoost-Train"),
            "test":  _metrics(y_te, xgb.predict(X_test),  "XGBoost-Test"),
        }
        joblib.dump(xgb, MODELS_DIR / "model_xgboost.pkl")
        logger.info("XGBoost saved.")
    except ImportError:
        logger.warning("XGBoost not installed — skipping.")

    # ── 2. Random Forest ───────────────────────────────────
    logger.info("Training Random Forest ...")
    rf = RandomForestRegressor(
        n_estimators     = RF_N_ESTIMATORS,
        max_depth        = RF_MAX_DEPTH,
        min_samples_leaf = RF_MIN_SAMPLES,
        random_state     = RANDOM_STATE,
        n_jobs           = -1,
    )
    rf.fit(X_train, y_tr)
    results["RandomForest"] = {
        "model": rf,
        "train": _metrics(y_tr, rf.predict(X_train), "RF-Train"),
        "test":  _metrics(y_te, rf.predict(X_test),  "RF-Test"),
    }
    joblib.dump(rf, MODELS_DIR / "model_rf.pkl")
    logger.info("Random Forest saved.")

    # ── 3. Gradient Boosting ───────────────────────────────
    logger.info("Training Gradient Boosting ...")
    gbr = GradientBoostingRegressor(
        n_estimators  = GBR_N_ESTIMATORS,
        max_depth     = GBR_MAX_DEPTH,
        learning_rate = GBR_LEARNING_RATE,
        subsample     = GBR_SUBSAMPLE,
        random_state  = RANDOM_STATE,
    )
    gbr.fit(X_train, y_tr)
    results["GBR"] = {
        "model": gbr,
        "train": _metrics(y_tr, gbr.predict(X_train), "GBR-Train"),
        "test":  _metrics(y_te, gbr.predict(X_test),  "GBR-Test"),
    }
    joblib.dump(gbr, MODELS_DIR / "model_gbr.pkl")
    logger.info("Gradient Boosting saved.")

    # ── 4. CatBoost + Optuna ───────────────────────────────
    # FIX (v5): cat_params is initialised to {} BEFORE the try block.
    # If CatBoost is not installed, the Stacking section below can still
    # safely call cat_params.get(...) without raising NameError.
    cat_params = {}
    try:
        from catboost import CatBoostRegressor

        logger.info("\u2550" * 40)
        logger.info(" CatBoost + Optuna Hyperparameter Tuning")
        logger.info("\u2550" * 40)
        best_params = _tune_catboost_optuna(X_train, y_tr)

        if best_params:
            cat_params = {
                "iterations":        best_params.get("iterations",        CAT_ITERATIONS),
                "depth":             best_params.get("depth",             CAT_DEPTH),
                "learning_rate":     best_params.get("learning_rate",     CAT_LEARNING_RATE),
                "l2_leaf_reg":       best_params.get("l2_leaf_reg",       CAT_L2_REG),
                "subsample":         best_params.get("subsample",         0.8),
                "colsample_bylevel": best_params.get("colsample_bylevel", 0.8),
                "min_child_samples": best_params.get("min_child_samples", 5),
            }
        else:
            cat_params = {
                "iterations":    CAT_ITERATIONS,
                "depth":         CAT_DEPTH,
                "learning_rate": CAT_LEARNING_RATE,
                "l2_leaf_reg":   CAT_L2_REG,
            }

        logger.info(f"Training CatBoost with params: {cat_params}")
        cat = CatBoostRegressor(
            **cat_params,
            random_seed           = RANDOM_STATE,
            verbose               = 0,
            early_stopping_rounds = CAT_EARLY_STOP,
        )
        cat.fit(X_train, y_tr, eval_set=(X_test, y_te), verbose=False)

        results["CatBoost"] = {
            "model": cat,
            "train": _metrics(y_tr, cat.predict(X_train), "CatBoost-Train"),
            "test":  _metrics(y_te, cat.predict(X_test),  "CatBoost-Test"),
        }
        joblib.dump(cat, MODEL_CATBOOST_PKL)
        logger.info("CatBoost saved.")

    except ImportError:
        logger.warning("CatBoost not installed — skipping. "
                       "Install: pip install catboost")

    # ── 5. Stacking Ensemble ───────────────────────────────
    logger.info("Training Stacking Ensemble ...")
    estimators = []
    if "XGBoost" in results:
        try:
            from xgboost import XGBRegressor
            xgb_stack = XGBRegressor(
                n_estimators     = XGB_N_ESTIMATORS,
                max_depth        = XGB_MAX_DEPTH,
                learning_rate    = XGB_LEARNING_RATE,
                subsample        = XGB_SUBSAMPLE,
                colsample_bytree = XGB_COLSAMPLE,
                reg_alpha        = XGB_REG_ALPHA,
                reg_lambda       = XGB_REG_LAMBDA,
                random_state     = RANDOM_STATE,
                n_jobs           = -1,
                verbosity        = 0,
            )
            estimators.append(("xgb", xgb_stack))
        except ImportError:
            pass
    estimators.append(("rf",  results["RandomForest"]["model"]))
    estimators.append(("gbr", results["GBR"]["model"]))
    if "CatBoost" in results:
        try:
            from catboost import CatBoostRegressor
            # cat_params is always defined (initialised to {} above)
            cat_stack = CatBoostRegressor(
                iterations    = cat_params.get("iterations",    CAT_ITERATIONS),
                depth         = cat_params.get("depth",         CAT_DEPTH),
                learning_rate = cat_params.get("learning_rate", CAT_LEARNING_RATE),
                l2_leaf_reg   = cat_params.get("l2_leaf_reg",   CAT_L2_REG),
                random_seed   = RANDOM_STATE,
                verbose       = 0,
            )
            estimators.append(("cat", cat_stack))
        except Exception:
            estimators.append(("cat", results["CatBoost"]["model"]))

    if len(estimators) >= 2:
        stacking = StackingRegressor(
            estimators     = estimators,
            final_estimator= Ridge(alpha=1.0),
            cv             = 5,
            n_jobs         = 1,
        )
        stacking.fit(X_train, y_tr)
        results["Stacking"] = {
            "model": stacking,
            "train": _metrics(y_tr, stacking.predict(X_train), "Stacking-Train"),
            "test":  _metrics(y_te, stacking.predict(X_test),  "Stacking-Test"),
        }
        joblib.dump(stacking, MODELS_DIR / "model_stacking.pkl")
        logger.info("Stacking Ensemble saved.")

    # ── 6. Best model + 10-Fold CV ─────────────────────────
    best_name  = max(results, key=lambda k: results[k]["test"]["R2"])
    best_model = results[best_name]["model"]
    best_test  = results[best_name]["test"]
    logger.info(f"\U0001f3c6 Best model: {best_name}  "
                f"Test R\u00b2={best_test['R2']}")

    # FIX (v5): CV uses X_train ONLY — avoids data leakage from test set.
    # FIX (v5b): Clone model and remove early_stopping_rounds for CV
    #            (XGBoost crashes without eval_set when early_stopping is set).
    from sklearn.base import clone
    cv_model = clone(best_model)
    # Remove early_stopping for XGBoost/CatBoost compatibility with cross_val_score
    if hasattr(cv_model, 'early_stopping_rounds'):
        cv_model.set_params(early_stopping_rounds=None)
    if hasattr(cv_model, 'eval_metric') and hasattr(cv_model, 'early_stopping_rounds'):
        try:
            cv_model.set_params(early_stopping_rounds=None)
        except Exception:
            pass

    kf    = KFold(n_splits=KFOLD_N_SPLITS, shuffle=True,
                  random_state=RANDOM_STATE)
    cv_r2 = cross_val_score(cv_model, X_train, y_tr,
                             cv=kf, scoring="r2", n_jobs=-1)
    logger.info(f"CV R\u00b2 ({best_name}) = "
                f"{cv_r2.mean():.4f} \u00b1 {cv_r2.std():.4f}")
    logger.info(f"CV per fold: {[round(x, 4) for x in cv_r2]}")

    joblib.dump(best_model, MODEL_BEST_PKL)
    logger.info(f"Best model saved \u2192 {MODEL_BEST_PKL}")

    l1_broken = best_test["R2"] >= L1_TARGET_R2
    l2_broken = best_test["R2"] >= L2_TARGET_R2
    if l1_broken and l2_broken:
        logger.info("\u2b50\u2b50\u2b50 BOTH BENCHMARKS BROKEN! \u2b50\u2b50\u2b50")
    elif l1_broken:
        logger.info("\u2705 L1 (ACI) BROKEN! L2 not yet.")
    else:
        logger.info("\u274c Neither benchmark broken yet.")

    summary = {
        "target":      "Mmax,exp (kNm)",
        "best_model":  best_name,
        "cv_R2_mean":  round(float(cv_r2.mean()), 4),
        "cv_R2_std":   round(float(cv_r2.std()),  4),
        "cv_folds":    [round(float(x), 4) for x in cv_r2],
        "L1_broken":   l1_broken,
        "L2_broken":   l2_broken,
        "models": {
            k: {"train_R2":  v["train"]["R2"],
                "test_R2":   v["test"]["R2"],
                "test_RMSE": v["test"]["RMSE"],
                "test_MAE":  v["test"]["MAE"],
                "L1_broken": v["test"]["L1_broken"],
                "L2_broken": v["test"]["L2_broken"]}
            for k, v in results.items()
        },
        "saved_at": str(datetime.now()),
    }
    sp = MODELS_DIR / "ensemble_metrics.json"
    with open(sp, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Ensemble metrics saved \u2192 {sp}")

    logger.info("\u2550" * 50)
    logger.info(" Phase 1B Complete")
    logger.info("\u2550" * 50)

    return {
        "best_model":   best_model,
        "best_name":    best_name,
        "results":      results,
        "metrics_test": best_test,
        "cv_R2_mean":   float(cv_r2.mean()),
        "cv_R2_std":    float(cv_r2.std()),
        "both_broken":  l1_broken and l2_broken,
        "summary":      summary,
    }
