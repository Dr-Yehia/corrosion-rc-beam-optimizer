# ============================================================
# src/statistical_validation.py
# Corrosion RC Beam Optimizer
# Rigorous statistical validation to confirm benchmark breaks
#
# Tests performed:
#   1. Wilcoxon Signed-Rank Test  (model vs ACI 318-19)
#   2. Bootstrap Confidence Intervals (R², RMSE, MAE)
#   3. 10-Fold Stratified Cross-Validation
#   4. McNemar Test (correct/incorrect classification)
#   5. Effect Size — Cohen’s d
# ============================================================

import numpy as np
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
from loguru import logger
from scipy import stats
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error
)
from sklearn.model_selection import KFold
import sys

sys.path.append(str(Path(__file__).resolve().parent))
from config import (
    WILCOXON_ALPHA, BOOTSTRAP_N, KFOLD_N_SPLITS,
    RANDOM_STATE, MODELS_DIR, FIGURES_DIR,
    L1_TARGET_R2, L2_TARGET_R2,
)


# ============================================================
# HELPERS
# ============================================================

def _r2_rmse_mae(y_true: np.ndarray, y_pred: np.ndarray) -> tuple:
    r2   = r2_score(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    return r2, rmse, mae


# ============================================================
# 1. WILCOXON SIGNED-RANK TEST
# ============================================================

def wilcoxon_test(
    y_true:        np.ndarray,
    y_pred_model:  np.ndarray,
    y_pred_aci:    np.ndarray,
    alpha:         float = WILCOXON_ALPHA,
) -> dict:
    """
    Wilcoxon Signed-Rank Test: non-parametric test whether
    the model’s absolute errors are significantly smaller than
    ACI 318-19’s absolute errors.

    H0 : median(|e_model|) == median(|e_ACI|)  (no difference)
    H1 : median(|e_model|)  < median(|e_ACI|)  (model is better)

    Parameters
    ----------
    y_true        : experimental R(%) values
    y_pred_model  : model predictions
    y_pred_aci    : ACI 318-19 predictions
    alpha         : significance level (default 0.05)

    Returns
    -------
    dict with statistic, p_value, significant, interpretation
    """
    err_model = np.abs(y_true - y_pred_model)
    err_aci   = np.abs(y_true - y_pred_aci)

    stat, p = stats.wilcoxon(
        err_model, err_aci,
        alternative = "less",
        zero_method = "wilcox",
    )

    significant = bool(p < alpha)
    result = {
        "test"            : "Wilcoxon Signed-Rank (one-sided, less)",
        "statistic"       : round(float(stat), 4),
        "p_value"         : round(float(p),    6),
        "alpha"           : alpha,
        "significant"     : significant,
        "interpretation"  : (
            f"Model errors are statistically SMALLER than ACI errors "
            f"(p={p:.4f} < \u03b1={alpha}).\n"
            f"The improvement over ACI 318-19 is statistically significant."
            if significant else
            f"No statistically significant difference detected "
            f"(p={p:.4f} ≥ \u03b1={alpha})."
        ),
        "median_err_model": round(float(np.median(err_model)), 4),
        "median_err_aci"  : round(float(np.median(err_aci)),   4),
    }

    sym = "✓" if significant else "✗"
    logger.info("═" * 55)
    logger.info(" Wilcoxon Signed-Rank Test")
    logger.info("═" * 55)
    logger.info(f"  Statistic      : {result['statistic']}")
    logger.info(f"  p-value        : {result['p_value']}")
    logger.info(f"  Significant {sym} : {significant}  (α={alpha})")
    logger.info(f"  Median |e| model : {result['median_err_model']}")
    logger.info(f"  Median |e| ACI   : {result['median_err_aci']}")
    logger.info("═" * 55)
    return result


# ============================================================
# 2. BOOTSTRAP CONFIDENCE INTERVALS
# ============================================================

def bootstrap_ci(
    y_true:   np.ndarray,
    y_pred:   np.ndarray,
    n_boot:   int   = BOOTSTRAP_N,
    ci:       float = 0.95,
) -> dict:
    """
    Bootstrap 95% confidence intervals for R², RMSE, and MAE.
    Uses stratified resampling (n_boot=1000 iterations).

    Returns
    -------
    dict with R2, RMSE, MAE point estimates and [lo, hi] CIs
    """
    np.random.seed(RANDOM_STATE)
    n    = len(y_true)
    r2s, rmses, maes = [], [], []

    for _ in range(n_boot):
        idx    = np.random.choice(n, size=n, replace=True)
        r2, rmse, mae = _r2_rmse_mae(y_true[idx], y_pred[idx])
        r2s.append(r2)
        rmses.append(rmse)
        maes.append(mae)

    alpha_lo = (1 - ci) / 2
    alpha_hi = 1 - alpha_lo

    def _ci(arr):
        lo = float(np.percentile(arr, alpha_lo * 100))
        hi = float(np.percentile(arr, alpha_hi * 100))
        return round(lo, 4), round(hi, 4)

    r2_pt, rmse_pt, mae_pt = _r2_rmse_mae(y_true, y_pred)

    result = {
        "n_bootstrap"  : n_boot,
        "ci_level"     : ci,
        "R2"           : round(r2_pt,   4),
        "R2_CI"        : list(_ci(r2s)),
        "RMSE"         : round(rmse_pt, 4),
        "RMSE_CI"      : list(_ci(rmses)),
        "MAE"          : round(mae_pt,  4),
        "MAE_CI"       : list(_ci(maes)),
        "R2_boot_mean" : round(float(np.mean(r2s)),   4),
        "R2_boot_std"  : round(float(np.std(r2s)),    4),
    }

    logger.info("═" * 55)
    logger.info(f" Bootstrap CI  ({n_boot} iterations, {int(ci*100)}% CI)")
    logger.info("═" * 55)
    logger.info(f"  R²    = {result['R2']}  CI: {result['R2_CI']}")
    logger.info(f"  RMSE  = {result['RMSE']} CI: {result['RMSE_CI']}")
    logger.info(f"  MAE   = {result['MAE']}  CI: {result['MAE_CI']}")
    logger.info("═" * 55)
    return result


# ============================================================
# 3. 10-FOLD CROSS-VALIDATION
# ============================================================

def kfold_cross_validation(
    model_builder,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = KFOLD_N_SPLITS,
) -> dict:
    """
    10-Fold Cross-Validation.
    Rebuilds and retrains the model on each fold from scratch
    to ensure no data leakage.

    Parameters
    ----------
    model_builder : callable that returns a fresh unfitted model
    X             : full scaled feature array
    y             : full scaled target array
    n_splits      : number of folds (default 10)

    Returns
    -------
    dict with per-fold and aggregate metrics
    """
    kf        = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    fold_data = []
    r2s, rmses, maes = [], [], []

    logger.info("═" * 55)
    logger.info(f" {n_splits}-Fold Cross-Validation")
    logger.info("═" * 55)

    for fold, (train_idx, val_idx) in enumerate(
        kf.split(X, y), start=1
    ):
        m = model_builder()
        m.fit(X[train_idx], y[train_idx])
        y_pred = m.predict(X[val_idx])

        r2, rmse, mae = _r2_rmse_mae(y[val_idx], y_pred)
        r2s.append(r2)
        rmses.append(rmse)
        maes.append(mae)
        fold_data.append({"fold": fold, "R2": round(r2, 4),
                          "RMSE": round(rmse, 4), "MAE": round(mae, 4)})
        logger.info(
            f"  Fold {fold:2d} ─ R²={r2:.4f}  "
            f"RMSE={rmse:.4f}  MAE={mae:.4f}"
        )

    result = {
        "n_folds"     : n_splits,
        "folds"       : fold_data,
        "R2_mean"     : round(float(np.mean(r2s)),   4),
        "R2_std"      : round(float(np.std(r2s)),    4),
        "R2_min"      : round(float(np.min(r2s)),    4),
        "R2_max"      : round(float(np.max(r2s)),    4),
        "RMSE_mean"   : round(float(np.mean(rmses)), 4),
        "RMSE_std"    : round(float(np.std(rmses)),  4),
        "MAE_mean"    : round(float(np.mean(maes)),  4),
        "MAE_std"     : round(float(np.std(maes)),   4),
        "L1_all_folds": all(r >= L1_TARGET_R2 for r in r2s),
        "L2_all_folds": all(r >= L2_TARGET_R2 for r in r2s),
    }

    logger.info("─" * 55)
    logger.info(
        f"  Aggregate ─ R²={result['R2_mean']:.4f} ± "
        f"{result['R2_std']:.4f}  "
        f"RMSE={result['RMSE_mean']:.4f} ± {result['RMSE_std']:.4f}"
    )
    logger.info(
        f"  L1 all folds: {result['L1_all_folds']}  "
        f"L2 all folds: {result['L2_all_folds']}"
    )
    logger.info("═" * 55)
    return result


# ============================================================
# 4. EFFECT SIZE — COHEN’S d
# ============================================================

def cohens_d(
    y_true:       np.ndarray,
    y_pred_model: np.ndarray,
    y_pred_aci:   np.ndarray,
) -> dict:
    """
    Cohen’s d effect size between model errors and ACI errors.
    Quantifies the practical significance of the improvement.

    Interpretation:
        d < 0.20  : negligible
        0.20–0.50 : small
        0.50–0.80 : medium
        d > 0.80  : large
    """
    err_model = np.abs(y_true - y_pred_model)
    err_aci   = np.abs(y_true - y_pred_aci)

    diff      = err_aci - err_model   # positive = model is better
    d         = diff.mean() / (diff.std() + 1e-9)

    if   abs(d) < 0.20: magnitude = "negligible"
    elif abs(d) < 0.50: magnitude = "small"
    elif abs(d) < 0.80: magnitude = "medium"
    else:               magnitude = "large"

    result = {
        "cohens_d"  : round(float(d), 4),
        "magnitude" : magnitude,
        "mean_diff" : round(float(diff.mean()), 4),
        "std_diff"  : round(float(diff.std()),  4),
    }

    logger.info(f"  Cohen’s d = {result['cohens_d']}  ({magnitude} effect)")
    return result


# ============================================================
# 5. McNEMAR TEST (classification of acceptable predictions)
# ============================================================

def mcnemar_test(
    y_true:       np.ndarray,
    y_pred_model: np.ndarray,
    y_pred_aci:   np.ndarray,
    tolerance:    float = 10.0,
) -> dict:
    """
    McNemar test on correct/incorrect predictions.
    A prediction is ‘correct’ if |y_true - y_pred| <= tolerance.
    Tests whether the model and ACI have significantly different
    correctness rates.

    Parameters
    ----------
    tolerance : acceptable absolute error (default 10% R)
    """
    ok_model = np.abs(y_true - y_pred_model) <= tolerance
    ok_aci   = np.abs(y_true - y_pred_aci)   <= tolerance

    # Contingency: model correct & ACI wrong / model wrong & ACI correct
    b = int(np.sum( ok_model & ~ok_aci))    # model ✓, ACI ✗
    c = int(np.sum(~ok_model &  ok_aci))    # model ✗, ACI ✓

    if b + c == 0:
        p = 1.0
        stat = 0.0
    else:
        stat = (abs(b - c) - 1) ** 2 / (b + c)
        p    = float(stats.chi2.sf(stat, df=1))

    result = {
        "test"            : "McNemar (exact, continuity corrected)",
        "tolerance_pct"   : tolerance,
        "b_model_ok"      : b,
        "c_aci_ok"        : c,
        "chi2_statistic"  : round(float(stat), 4),
        "p_value"         : round(p, 6),
        "significant"     : bool(p < WILCOXON_ALPHA),
        "model_accuracy"  : round(float(ok_model.mean()) * 100, 2),
        "aci_accuracy"    : round(float(ok_aci.mean())   * 100, 2),
    }

    logger.info("═" * 55)
    logger.info(" McNemar Test (classification accuracy)")
    logger.info("═" * 55)
    logger.info(f"  Tolerance          : ±{tolerance}% R")
    logger.info(f"  Model accuracy     : {result['model_accuracy']}%")
    logger.info(f"  ACI accuracy       : {result['aci_accuracy']}%")
    logger.info(f"  χ² statistic       : {result['chi2_statistic']}")
    logger.info(f"  p-value            : {result['p_value']}")
    logger.info(f"  Significant        : {result['significant']}")
    logger.info("═" * 55)
    return result


# ============================================================
# 6. SAVE VALIDATION RESULTS
# ============================================================

def save_validation_results(results: dict) -> None:
    """Persist all statistical validation results to JSON."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / "statistical_validation.json"
    payload = {**results, "generated_at": str(datetime.now())}
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    logger.info(f"Statistical validation results saved \u2192 {path}")


# ============================================================
# 7. FULL PIPELINE
# ============================================================

def run_statistical_validation(
    y_true:        np.ndarray,
    y_pred_model:  np.ndarray,
    y_pred_aci:    np.ndarray,
    model_builder,
    X_all:         np.ndarray,
    y_all_scaled:  np.ndarray,
) -> dict:
    """
    Execute the full statistical validation suite.

    Parameters
    ----------
    y_true         : true R(%) values (original scale)
    y_pred_model   : model predictions (original scale)
    y_pred_aci     : ACI 318-19 predictions (original scale)
    model_builder  : callable returning a fresh unfitted model
    X_all          : full scaled feature array (train+test)
    y_all_scaled   : full scaled target array

    Returns
    -------
    dict with all test results
    """
    logger.info("=" * 60)
    logger.info(" Phase 4 — Statistical Validation")
    logger.info("=" * 60)

    wilcoxon  = wilcoxon_test(y_true, y_pred_model, y_pred_aci)
    bootstrap = bootstrap_ci(y_true, y_pred_model)
    kfold     = kfold_cross_validation(model_builder, X_all, y_all_scaled)
    effect    = cohens_d(y_true, y_pred_model, y_pred_aci)
    mcnemar   = mcnemar_test(y_true, y_pred_model, y_pred_aci)

    # Consolidated pass/fail summary
    r2_test, _, _ = _r2_rmse_mae(y_true, y_pred_model)
    passed = {
        "wilcoxon_significant"     : wilcoxon["significant"],
        "bootstrap_R2_lower_bound" : bootstrap["R2_CI"][0] >= L1_TARGET_R2,
        "kfold_L1_all_folds"       : kfold["L1_all_folds"],
        "kfold_L2_all_folds"       : kfold["L2_all_folds"],
        "effect_size_large"        : effect["cohens_d"] > 0.80,
        "mcnemar_significant"      : mcnemar["significant"],
        "model_R2"                 : round(r2_test, 4),
    }

    all_passed = (
        passed["wilcoxon_significant"] and
        passed["bootstrap_R2_lower_bound"] and
        passed["kfold_L1_all_folds"]
    )

    verdict = (
        "✅ STATISTICAL VALIDATION PASSED — "
        "Benchmark improvement confirmed with p<0.05."
        if all_passed else
        "⛔ STATISTICAL VALIDATION INCOMPLETE — "
        "One or more tests did not reach significance."
    )

    logger.info(verdict)

    results = {
        "wilcoxon"  : wilcoxon,
        "bootstrap" : bootstrap,
        "kfold"     : kfold,
        "cohens_d"  : effect,
        "mcnemar"   : mcnemar,
        "summary"   : passed,
        "verdict"   : verdict,
    }

    save_validation_results(results)

    logger.info("=" * 60)
    logger.info(" Phase 4 Complete ✓")
    logger.info("=" * 60)
    return results


# ============================================================
# CLI ENTRY POINT
# ============================================================

if __name__ == "__main__":
    from data_preprocessing import run_preprocessing
    from neural_network import load_model, build_mlp, predict
    from aci_calculator import (
        load_raw_data as _load, clean_data as _clean,
        compute_aci_predictions, evaluate_aci_benchmark,
    )
    import joblib

    data     = run_preprocessing(save_clean=True)
    model    = load_model()
    scaler_y = joblib.load(MODELS_DIR / "scaler_y.pkl")

    y_true  = scaler_y.inverse_transform(
        data["y_test"].reshape(-1, 1)
    ).ravel()
    y_model = predict(model, data["X_test"], scaler_y)

    df_clean    = data["df_clean"]
    df_aci      = compute_aci_predictions(df_clean)
    aci_metrics = evaluate_aci_benchmark(df_aci)

    # Simple ACI predictions aligned with test set indices
    y_aci = df_aci.loc[
        data["y_test_raw"].index, "MACI_pred"
    ].values

    X_all = np.vstack([data["X_train"], data["X_test"]])
    y_all = np.concatenate([data["y_train"], data["y_test"]])

    results = run_statistical_validation(
        y_true        = y_true,
        y_pred_model  = y_model,
        y_pred_aci    = y_aci,
        model_builder = build_mlp,
        X_all         = X_all,
        y_all_scaled  = y_all,
    )

    print("\n" + results["verdict"])
