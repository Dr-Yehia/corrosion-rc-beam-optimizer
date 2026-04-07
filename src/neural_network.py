# ============================================================
# src/neural_network.py  —  MLP (used by GA fitness evaluator)
# ============================================================
import numpy as np
import joblib
import json
from pathlib import Path
from datetime import datetime
from loguru import logger

from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold

import sys
sys.path.append(str(Path(__file__).resolve().parent))
from config import (
    NN_HIDDEN_LAYERS, NN_LEARNING_RATE, NN_EPOCHS,
    NN_BATCH_SIZE, NN_PATIENCE, NN_L2_ALPHA, NN_VALIDATION_FRAC,
    RANDOM_STATE, MODEL_MLP_PKL, MODELS_DIR,
    L1_TARGET_R2, L2_TARGET_R2, KFOLD_N_SPLITS,
)


def build_mlp() -> MLPRegressor:
    model = MLPRegressor(
        hidden_layer_sizes  = tuple(NN_HIDDEN_LAYERS),
        activation          = "relu",
        solver              = "adam",
        alpha               = NN_L2_ALPHA,
        learning_rate       = "adaptive",
        learning_rate_init  = NN_LEARNING_RATE,
        max_iter            = NN_EPOCHS,
        batch_size          = NN_BATCH_SIZE,
        early_stopping      = True,
        validation_fraction = NN_VALIDATION_FRAC,
        n_iter_no_change    = NN_PATIENCE,
        random_state        = RANDOM_STATE,
        verbose             = False,
    )
    logger.info(f"MLP built \u2014 layers: {NN_HIDDEN_LAYERS}, lr: {NN_LEARNING_RATE}")
    return model


def train_mlp(model, X_train, y_train):
    logger.info("Training MLP ...")
    model.fit(X_train, y_train)
    logger.info(f"MLP training complete \u2014 iterations: {model.n_iter_}")
    logger.info(f"Final training loss: {model.loss_:.6f}")
    return model


def evaluate_model(model, X, y, scaler_y=None, split_name="Test"):
    y_pred_sc = model.predict(X)
    if scaler_y is not None:
        y_true = scaler_y.inverse_transform(y.reshape(-1, 1)).ravel()
        y_pred = scaler_y.inverse_transform(y_pred_sc.reshape(-1, 1)).ravel()
    else:
        y_true, y_pred = y, y_pred_sc

    r2   = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-6))) * 100

    metrics = {
        "split": split_name, "R2": round(r2, 4),
        "RMSE": round(rmse, 4), "MAE": round(mae, 4),
        "MAPE": round(mape, 2),
        "L1_broken": r2 >= L1_TARGET_R2,
        "L2_broken": r2 >= L2_TARGET_R2,
        "timestamp": str(datetime.now()),
    }
    l1 = "\u2713" if metrics["L1_broken"] else "\u2717"
    l2 = "\u2713" if metrics["L2_broken"] else "\u2717"
    logger.info(f"[{split_name}] R\u00b2={r2:.4f}  RMSE={rmse:.4f}  "
                f"MAE={mae:.4f}  MAPE={mape:.2f}%  L1:{l1}  L2:{l2}")
    return metrics


def cross_validate_mlp(X, y, n_splits=KFOLD_N_SPLITS):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    r2s, rmses = [], []
    logger.info(f"Running {n_splits}-Fold CV (MLP) ...")
    for fold, (tr, va) in enumerate(kf.split(X, y), 1):
        m = build_mlp()
        m.fit(X[tr], y[tr])
        yp = m.predict(X[va])
        r2s.append(r2_score(y[va], yp))
        rmses.append(np.sqrt(mean_squared_error(y[va], yp)))
        logger.info(f"  Fold {fold:2d} \u2014 R\u00b2={r2s[-1]:.4f}  RMSE={rmses[-1]:.4f}")
    cv = {
        "cv_R2_mean": round(float(np.mean(r2s)), 4),
        "cv_R2_std":  round(float(np.std(r2s)),  4),
        "cv_RMSE_mean": round(float(np.mean(rmses)), 4),
        "cv_RMSE_std":  round(float(np.std(rmses)),  4),
        "n_folds": n_splits,
    }
    logger.info(f"CV complete \u2014 R\u00b2 = {cv['cv_R2_mean']:.4f} \u00b1 {cv['cv_R2_std']:.4f}")
    return cv


def save_model(model, metrics_train, metrics_test, cv_results):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_MLP_PKL)
    logger.info(f"Model saved \u2192 {MODEL_MLP_PKL}")
    all_metrics = {"train": metrics_train, "test": metrics_test,
                   "cv": cv_results, "saved_at": str(datetime.now())}
    mp = MODELS_DIR / "mlp_metrics.json"
    with open(mp, "w") as f:
        json.dump(all_metrics, f, indent=2)
    logger.info(f"Metrics saved \u2192 {mp}")


def load_model(path: Path = MODEL_MLP_PKL):
    m = joblib.load(path)
    logger.info(f"Model loaded \u2190 {path}")
    return m


def predict(model, X, scaler_y=None):
    yp = model.predict(X)
    if scaler_y is not None:
        return scaler_y.inverse_transform(yp.reshape(-1, 1)).ravel()
    return yp


def run_training_pipeline(X_train, X_test, y_train, y_test, scaler_y=None):
    """
    Phase 1A: MLP baseline (fast, used as GA fitness proxy).
    Primary model selection is in ensemble_models.py (Phase 1B).
    """
    logger.info("\u2550" * 38)
    logger.info(" Phase 1A \u2014 MLP Baseline")
    logger.info("\u2550" * 38)
    model         = build_mlp()
    model         = train_mlp(model, X_train, y_train)
    metrics_train = evaluate_model(model, X_train, y_train, scaler_y, "Train")
    metrics_test  = evaluate_model(model, X_test,  y_test,  scaler_y, "Test")
    cv_results    = cross_validate_mlp(
        np.vstack([X_train, X_test]),
        np.concatenate([y_train, y_test]),
    )
    save_model(model, metrics_train, metrics_test, cv_results)
    both_broken = metrics_test["L1_broken"] and metrics_test["L2_broken"]
    if both_broken:
        logger.success("\U0001f3c6 Both benchmarks broken by MLP!")
    elif metrics_test["L1_broken"]:
        logger.info("\u2713 L1 broken. Ensemble expected to break L2.")
    else:
        logger.info("\u2717 MLP baseline \u2014 proceeding to Ensemble + GA.")
    logger.info("\u2550" * 38)
    logger.info(" Phase 1A Complete")
    logger.info("\u2550" * 38)
    return {
        "model": model,
        "metrics_train": metrics_train,
        "metrics_test":  metrics_test,
        "cv":            cv_results,
        "both_broken":   both_broken,
    }
