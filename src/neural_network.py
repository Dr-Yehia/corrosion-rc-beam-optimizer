# ============================================================
# src/neural_network.py
# Corrosion RC Beam Optimizer
# Shallow MLP — build, train, evaluate, save
# Used as Phase-1 baseline AND as fitness evaluator inside GA
# ============================================================

import numpy as np
import pandas as pd
import joblib
import json
from pathlib import Path
from datetime import datetime
from loguru import logger

from sklearn.neural_network import MLPRegressor
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error
)
from sklearn.model_selection import KFold

import sys
sys.path.append(str(Path(__file__).resolve().parent))
from config import (
    NN_HIDDEN_LAYERS, NN_DROPOUT, NN_LEARNING_RATE,
    NN_EPOCHS, NN_BATCH_SIZE, NN_PATIENCE,
    RANDOM_STATE, MODEL_MLP_PKL, MODELS_DIR,
    L1_TARGET_R2, L2_TARGET_R2,
    KFOLD_N_SPLITS,
)


# ────────────────────────────────────────────────────────────
# 1. BUILD MODEL
# ────────────────────────────────────────────────────────────
def build_mlp() -> MLPRegressor:
    """
    Build a shallow MLP using scikit-learn MLPRegressor.
    Architecture: Input → Dense(64, ReLU) → Dense(32, ReLU) → Output(1)
    """
    model = MLPRegressor(
        hidden_layer_sizes = tuple(NN_HIDDEN_LAYERS),
        activation         = "relu",
        solver             = "adam",
        alpha              = 1e-4,           # L2 regularisation
        learning_rate_init = NN_LEARNING_RATE,
        max_iter           = NN_EPOCHS,
        batch_size         = NN_BATCH_SIZE,
        early_stopping     = True,
        validation_fraction= 0.10,
        n_iter_no_change   = NN_PATIENCE,
        random_state       = RANDOM_STATE,
        verbose            = False,
    )
    logger.info(f"MLP built — layers: {NN_HIDDEN_LAYERS}, lr: {NN_LEARNING_RATE}")
    return model


# ────────────────────────────────────────────────────────────
# 2. TRAIN
# ────────────────────────────────────────────────────────────
def train_mlp(
    model: MLPRegressor,
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> MLPRegressor:
    """
    Fit the MLP on the training set.
    Returns the trained model.
    """
    logger.info("Training MLP ...")
    model.fit(X_train, y_train)
    logger.info(f"MLP training complete — iterations: {model.n_iter_}")
    logger.info(f"Final training loss: {model.loss_:.6f}")
    return model


# ────────────────────────────────────────────────────────────
# 3. EVALUATE
# ────────────────────────────────────────────────────────────
def evaluate_model(
    model: MLPRegressor,
    X: np.ndarray,
    y: np.ndarray,
    scaler_y=None,
    split_name: str = "Test",
) -> dict:
    """
    Compute R², RMSE, MAE, MAPE on the given split.
    If scaler_y is provided, metrics are computed in original R(%) scale.

    Parameters
    ----------
    model      : trained MLPRegressor
    X          : scaled feature array
    y          : scaled or raw target array
    scaler_y   : fitted StandardScaler for target (optional)
    split_name : label for logging ('Train' / 'Test')

    Returns
    -------
    dict of metrics
    """
    y_pred_sc = model.predict(X)

    if scaler_y is not None:
        y_true = scaler_y.inverse_transform(y.reshape(-1, 1)).ravel()
        y_pred = scaler_y.inverse_transform(y_pred_sc.reshape(-1, 1)).ravel()
    else:
        y_true = y
        y_pred = y_pred_sc

    r2   = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-6))) * 100

    l1_ok = r2 >= L1_TARGET_R2
    l2_ok = r2 >= L2_TARGET_R2

    metrics = {
        "split"   : split_name,
        "R2"      : round(r2,   4),
        "RMSE"    : round(rmse, 4),
        "MAE"     : round(mae,  4),
        "MAPE"    : round(mape, 2),
        "L1_broken": l1_ok,
        "L2_broken": l2_ok,
        "timestamp": str(datetime.now()),
    }

    l1_sym = "✓" if l1_ok else "✗"
    l2_sym = "✓" if l2_ok else "✗"

    logger.info(f"[{split_name}] R²={r2:.4f}  RMSE={rmse:.4f}  "
                f"MAE={mae:.4f}  MAPE={mape:.2f}%  "
                f"L1:{l1_sym}  L2:{l2_sym}")
    return metrics


# ────────────────────────────────────────────────────────────
# 4. K-FOLD CROSS-VALIDATION
# ────────────────────────────────────────────────────────────
def cross_validate_mlp(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = KFOLD_N_SPLITS,
) -> dict:
    """
    Stratified K-Fold cross-validation of the MLP architecture.
    Reports mean ± std of R², RMSE, MAE across folds.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    r2_scores, rmse_scores, mae_scores = [], [], []

    logger.info(f"Running {n_splits}-Fold CV ...")
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), 1):
        m = build_mlp()
        m.fit(X[train_idx], y[train_idx])
        y_pred = m.predict(X[val_idx])

        r2_scores.append(r2_score(y[val_idx], y_pred))
        rmse_scores.append(np.sqrt(mean_squared_error(y[val_idx], y_pred)))
        mae_scores.append(mean_absolute_error(y[val_idx], y_pred))
        logger.info(f"  Fold {fold:2d} — R²={r2_scores[-1]:.4f}  "
                    f"RMSE={rmse_scores[-1]:.4f}")

    cv_results = {
        "cv_R2_mean"   : round(float(np.mean(r2_scores)),   4),
        "cv_R2_std"    : round(float(np.std(r2_scores)),    4),
        "cv_RMSE_mean" : round(float(np.mean(rmse_scores)), 4),
        "cv_RMSE_std"  : round(float(np.std(rmse_scores)),  4),
        "cv_MAE_mean"  : round(float(np.mean(mae_scores)),  4),
        "cv_MAE_std"   : round(float(np.std(mae_scores)),   4),
        "n_folds"      : n_splits,
    }

    logger.info(f"CV complete — R² = {cv_results['cv_R2_mean']:.4f} "
                f"± {cv_results['cv_R2_std']:.4f}")
    return cv_results


# ────────────────────────────────────────────────────────────
# 5. SAVE MODEL
# ────────────────────────────────────────────────────────────
def save_model(
    model: MLPRegressor,
    metrics_train: dict,
    metrics_test:  dict,
    cv_results:    dict,
) -> None:
    """
    Persist trained model and all metrics to results/models/.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Save model
    joblib.dump(model, MODEL_MLP_PKL)
    logger.info(f"Model saved → {MODEL_MLP_PKL}")

    # Save metrics
    all_metrics = {
        "train"    : metrics_train,
        "test"     : metrics_test,
        "cv"       : cv_results,
        "saved_at" : str(datetime.now()),
    }
    metrics_path = MODELS_DIR / "mlp_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    logger.info(f"Metrics saved → {metrics_path}")


# ────────────────────────────────────────────────────────────
# 6. LOAD MODEL
# ────────────────────────────────────────────────────────────
def load_model(path: Path = MODEL_MLP_PKL) -> MLPRegressor:
    """Load a previously saved MLP model."""
    model = joblib.load(path)
    logger.info(f"Model loaded ← {path}")
    return model


# ────────────────────────────────────────────────────────────
# 7. PREDICT (inference)
# ────────────────────────────────────────────────────────────
def predict(
    model: MLPRegressor,
    X: np.ndarray,
    scaler_y=None,
) -> np.ndarray:
    """
    Run inference and return predictions in original R(%) scale.
    """
    y_pred_sc = model.predict(X)
    if scaler_y is not None:
        y_pred = scaler_y.inverse_transform(
            y_pred_sc.reshape(-1, 1)
        ).ravel()
    else:
        y_pred = y_pred_sc
    return y_pred


# ────────────────────────────────────────────────────────────
# 8. FULL TRAINING PIPELINE
# ────────────────────────────────────────────────────────────
def run_training_pipeline(
    X_train: np.ndarray,
    X_test:  np.ndarray,
    y_train: np.ndarray,
    y_test:  np.ndarray,
    scaler_y=None,
) -> dict:
    """
    End-to-end MLP training, evaluation, CV, and save.
    Returns dict with model, metrics, and CV results.
    """
    logger.info("══════════════════════════════════════")
    logger.info(" Phase 1 — Baseline MLP Training")
    logger.info("══════════════════════════════════════")

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
        logger.success("🏆 Both benchmarks broken by MLP baseline!")
    elif metrics_test["L1_broken"]:
        logger.info("✓ L1 (ACI) broken. L2 (SOTA) not yet.")
    else:
        logger.info("✗ Neither benchmark broken yet — proceeding to GA.")

    logger.info("══════════════════════════════════════")
    logger.info(" Phase 1 Complete")
    logger.info("══════════════════════════════════════")

    return {
        "model"        : model,
        "metrics_train": metrics_train,
        "metrics_test" : metrics_test,
        "cv"           : cv_results,
        "both_broken"  : both_broken,
    }


# ────────────────────────────────────────────────────────────
# CLI entry point
# ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from data_preprocessing import run_preprocessing

    data    = run_preprocessing(save_clean=True)
    results = run_training_pipeline(
        data["X_train"], data["X_test"],
        data["y_train"], data["y_test"],
        scaler_y = data["scaler_y"],
    )

    print("\n✅ MLP baseline training complete.")
    print(f"   Test R²   = {results['metrics_test']['R2']}")
    print(f"   Test RMSE = {results['metrics_test']['RMSE']}")
    print(f"   CV R²     = {results['cv']['cv_R2_mean']} ± {results['cv']['cv_R2_std']}")
    print(f"   L1 broken : {results['metrics_test']['L1_broken']}")
    print(f"   L2 broken : {results['metrics_test']['L2_broken']}")
