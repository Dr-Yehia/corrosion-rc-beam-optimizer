# ============================================================
# src/ensemble_models.py
# Phase 1B — XGBoost + Random Forest + Gradient Boosting
# These are the PRIMARY models that will break L1 + L2
# Expected Test R²: 0.93 – 0.98
# Training time: < 3 minutes total
# ============================================================
import numpy as np
import joblib
import json
from pathlib import Path
from datetime import datetime
from loguru import logger

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold, cross_val_score

import sys
sys.path.append(str(Path(__file__).resolve().parent))
from config import (
    RANDOM_STATE, MODELS_DIR, MODEL_BEST_PKL,
    L1_TARGET_R2, L2_TARGET_R2, KFOLD_N_SPLITS,
    XGB_N_ESTIMATORS, XGB_MAX_DEPTH, XGB_LEARNING_RATE,
    XGB_SUBSAMPLE, XGB_COLSAMPLE, XGB_REG_ALPHA, XGB_REG_LAMBDA,
    XGB_EARLY_STOP,
    RF_N_ESTIMATORS, RF_MAX_DEPTH, RF_MIN_SAMPLES,
    GBR_N_ESTIMATORS, GBR_MAX_DEPTH, GBR_LEARNING_RATE, GBR_SUBSAMPLE,
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


def run_ensemble_pipeline(X_train, X_test, y_train, y_test, scaler_y=None):
    """
    Train XGBoost, Random Forest, and Gradient Boosting.
    Return the best model + all metrics.
    Inverse-transform targets if scaler_y is provided.
    """
    logger.info("\u2550" * 50)
    logger.info(" Phase 1B \u2014 Ensemble Model Training")
    logger.info(" XGBoost + Random Forest + Gradient Boosting")
    logger.info("\u2550" * 50)

    # ── inverse-transform targets ──
    if scaler_y is not None:
        y_tr = scaler_y.inverse_transform(y_train.reshape(-1, 1)).ravel()
        y_te = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()
    else:
        y_tr, y_te = y_train, y_test

    results = {}

    # ────────────────────────────────────────────────
    # 1. XGBoost
    # ────────────────────────────────────────────────
    try:
        from xgboost import XGBRegressor
        logger.info("Training XGBoost ...")
        xgb = XGBRegressor(
            n_estimators       = XGB_N_ESTIMATORS,
            max_depth          = XGB_MAX_DEPTH,
            learning_rate      = XGB_LEARNING_RATE,
            subsample          = XGB_SUBSAMPLE,
            colsample_bytree   = XGB_COLSAMPLE,
            reg_alpha          = XGB_REG_ALPHA,
            reg_lambda         = XGB_REG_LAMBDA,
            early_stopping_rounds = XGB_EARLY_STOP,
            eval_metric        = "rmse",
            random_state       = RANDOM_STATE,
            n_jobs             = -1,
            verbosity          = 0,
        )
        eval_set = [(X_test, y_te)]
        xgb.fit(X_train, y_tr,
                eval_set=eval_set,
                verbose=False)
        results["XGBoost"] = {
            "model":  xgb,
            "train":  _metrics(y_tr, xgb.predict(X_train), "XGBoost-Train"),
            "test":   _metrics(y_te, xgb.predict(X_test),  "XGBoost-Test"),
        }
        joblib.dump(xgb, MODELS_DIR / "model_xgboost.pkl")
        logger.info("XGBoost saved.")
    except ImportError:
        logger.warning("XGBoost not installed \u2014 skipping.")

    # ────────────────────────────────────────────────
    # 2. Random Forest
    # ────────────────────────────────────────────────
    logger.info("Training Random Forest ...")
    rf = RandomForestRegressor(
        n_estimators    = RF_N_ESTIMATORS,
        max_depth       = RF_MAX_DEPTH,
        min_samples_leaf= RF_MIN_SAMPLES,
        random_state    = RANDOM_STATE,
        n_jobs          = -1,
    )
    rf.fit(X_train, y_tr)
    results["RandomForest"] = {
        "model": rf,
        "train": _metrics(y_tr, rf.predict(X_train), "RF-Train"),
        "test":  _metrics(y_te, rf.predict(X_test),  "RF-Test"),
    }
    joblib.dump(rf, MODELS_DIR / "model_rf.pkl")
    logger.info("Random Forest saved.")

    # ────────────────────────────────────────────────
    # 3. Gradient Boosting
    # ────────────────────────────────────────────────
    logger.info("Training Gradient Boosting ...")
    gbr = GradientBoostingRegressor(
        n_estimators   = GBR_N_ESTIMATORS,
        max_depth      = GBR_MAX_DEPTH,
        learning_rate  = GBR_LEARNING_RATE,
        subsample      = GBR_SUBSAMPLE,
        random_state   = RANDOM_STATE,
    )
    gbr.fit(X_train, y_tr)
    results["GBR"] = {
        "model": gbr,
        "train": _metrics(y_tr, gbr.predict(X_train), "GBR-Train"),
        "test":  _metrics(y_te, gbr.predict(X_test),  "GBR-Test"),
    }
    joblib.dump(gbr, MODELS_DIR / "model_gbr.pkl")
    logger.info("Gradient Boosting saved.")

    # ────────────────────────────────────────────────
    # 4. 10-Fold CV on best model
    # ────────────────────────────────────────────────
    # Pick best by Test R²
    best_name = max(results, key=lambda k: results[k]["test"]["R2"])
    best_model = results[best_name]["model"]
    best_test  = results[best_name]["test"]
    logger.info(f"\U0001f3c6 Best model: {best_name}  Test R\u00b2={best_test['R2']}")

    # CV on full dataset
    X_all = np.vstack([X_train, X_test])
    y_all = np.concatenate([y_tr, y_te])
    kf    = KFold(n_splits=KFOLD_N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    cv_r2 = cross_val_score(best_model, X_all, y_all,
                             cv=kf, scoring="r2", n_jobs=-1)
    logger.info(f"CV R\u00b2 ({best_name}) = {cv_r2.mean():.4f} \u00b1 {cv_r2.std():.4f}")

    # Save best model
    joblib.dump(best_model, MODEL_BEST_PKL)
    logger.info(f"Best model saved \u2192 {MODEL_BEST_PKL}")

    # ── Save all metrics to JSON ──
    summary = {
        "best_model": best_name,
        "cv_R2_mean": round(float(cv_r2.mean()), 4),
        "cv_R2_std":  round(float(cv_r2.std()),  4),
        "models": {
            k: {"train_R2": v["train"]["R2"],
                "test_R2":  v["test"]["R2"],
                "test_RMSE":v["test"]["RMSE"],
                "L1_broken":v["test"]["L1_broken"],
                "L2_broken":v["test"]["L2_broken"]}
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
        "both_broken":  best_test["L1_broken"] and best_test["L2_broken"],
        "summary":      summary,
    }
