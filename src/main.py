# ============================================================
# src/main.py  —  Corrosion RC Beam Optimizer
# Master Orchestrator — v4
# Target: Mmax,exp (kN·m) — matching Zhang et al. (2025)
#
# Phase 0  ─ ACI 318-19 baseline
# Phase 1A ─ MLP baseline
# Phase 1B ─ Ensemble: XGBoost + RF + GBR + CatBoost(Optuna) + Stacking
# Phase 2  ─ NSGA-III GA optimisation
# Phase 3  ─ PySR symbolic regression
# Phase 4  ─ SHAP feature importance
# Phase 5  ─ Statistical validation
# Phase 6  ─ PDF report generation
# ============================================================

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from loguru import logger
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    LOG_FILE, LOG_DIR, MODELS_DIR, RESULTS_DIR,
    L1_TARGET_R2, L2_TARGET_R2, BREAK_BOTH,
    RANDOM_STATE, MODEL_BEST_PKL,
)


# ============================================================
# LOGGING
# ============================================================
def _configure_logging():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(sys.stderr,
               format="<green>{time:HH:mm:ss}</green> | "
                      "<level>{level:<8}</level> | {message}",
               level="INFO", colorize=True)
    logger.add(str(LOG_FILE),
               format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}",
               level="DEBUG", rotation="10 MB", retention=5, encoding="utf-8")
    logger.info(f"Logging configured \u2192 {LOG_FILE}")


# ============================================================
# ARGUMENT PARSER
# ============================================================
def _parse_args():
    parser = argparse.ArgumentParser(prog="main.py")
    parser.add_argument("--skip-pysr",   action="store_true")
    parser.add_argument("--skip-shap",   action="store_true")
    parser.add_argument("--skip-ga",     action="store_true",
                        help="Skip Phase 2 GA (use ensemble result directly).")
    parser.add_argument("--phase",       nargs="+", type=int)
    parser.add_argument("--report-only", action="store_true")
    return parser.parse_args()


# ============================================================
# PHASES
# ============================================================
def phase_0_aci(data):
    from aci_calculator import (
        compute_aci_predictions, evaluate_aci_benchmark, save_benchmark_results)
    logger.info("\n" + "=" * 60)
    logger.info(" Phase 0 \u2500 ACI 318-19 Benchmark")
    logger.info("=" * 60)
    df_aci      = compute_aci_predictions(data["df_clean"])
    aci_metrics = evaluate_aci_benchmark(df_aci)
    save_benchmark_results(df_aci, aci_metrics)
    logger.info(f"ACI baseline \u2500 R\u00b2={aci_metrics['R2']}  "
                f"RMSE={aci_metrics['RMSE']}  Ratio={aci_metrics['ratio_mean']}")
    return {"df_aci": df_aci, "aci_metrics": aci_metrics}


def phase_1_mlp(data):
    """Phase 1A: MLP (fast baseline, also used by GA)."""
    from neural_network import run_training_pipeline
    logger.info("\n" + "=" * 60)
    logger.info(" Phase 1A \u2500 MLP Baseline Training")
    logger.info("=" * 60)
    return run_training_pipeline(
        data["X_train"], data["X_test"],
        data["y_train"], data["y_test"],
        scaler_y=data["scaler_y"],
    )


def phase_1b_ensemble(data):
    """Phase 1B: XGBoost + RF + GBR + CatBoost(Optuna) + Stacking."""
    from ensemble_models import run_ensemble_pipeline
    logger.info("\n" + "=" * 60)
    logger.info(" Phase 1B \u2500 Ensemble Model Training")
    logger.info("=" * 60)
    return run_ensemble_pipeline(
        data["X_train"], data["X_test"],
        data["y_train"], data["y_test"],
        scaler_y=data["scaler_y"],
    )


def phase_2_ga(data, aci_metrics):
    from genetic_algorithm import run_nsga3
    logger.info("\n" + "=" * 60)
    logger.info(" Phase 2 \u2500 NSGA-III Optimisation")
    logger.info("=" * 60)
    log_lines  = []
    ga_results = run_nsga3(
        X_train=data["X_train"], y_train=data["y_train"],
        X_test=data["X_test"],   y_test=data["y_test"],
        scaler_y=data["scaler_y"], aci_rmse=aci_metrics["RMSE"],
        aci_mae=aci_metrics["MAE"], log_lines=log_lines,
    )
    ga_results["log_lines"] = log_lines
    return ga_results


def phase_3_pysr(data, aci_metrics):
    from symbolic_regression import run_symbolic_regression, PYSR_FEATURES
    logger.info("\n" + "=" * 60)
    logger.info(" Phase 3 \u2500 PySR Symbolic Regression")
    logger.info("=" * 60)
    return run_symbolic_regression(
        df=data["df_clean"], aci_rmse=aci_metrics["RMSE"],
        feature_list=PYSR_FEATURES,
    )


def phase_4_shap(data, model):
    from shap_analysis import run_shap_analysis
    logger.info("\n" + "=" * 60)
    logger.info(" Phase 4 \u2500 SHAP Analysis")
    logger.info("=" * 60)
    return run_shap_analysis(
        model=model, X_train=data["X_train"], X_test=data["X_test"],
        feature_names=data["feature_cols"],
    )


def phase_5_validation(data, model, aci_metrics):
    from statistical_validation import run_statistical_validation
    from neural_network import predict, build_mlp
    from aci_calculator import compute_aci_predictions
    import joblib
    logger.info("\n" + "=" * 60)
    logger.info(" Phase 5 \u2500 Statistical Validation")
    logger.info("=" * 60)
    scaler_y = data["scaler_y"]
    y_true   = scaler_y.inverse_transform(
        data["y_test"].reshape(-1, 1)).ravel()
    # Use model.predict directly (works for sklearn & xgb)
    y_pred_sc = model.predict(data["X_test"])
    # If predictions look scaled, inverse-transform
    if y_pred_sc.mean() < 10:   # scaled range ~[-2, 2]
        y_model = scaler_y.inverse_transform(
            y_pred_sc.reshape(-1, 1)).ravel()
    else:
        y_model = y_pred_sc
    df_aci = compute_aci_predictions(data["df_clean"])
    y_aci  = df_aci.loc[data["y_test_raw"].index, "MACI_pred"].values
    X_all  = np.vstack([data["X_train"], data["X_test"]])
    y_all  = np.concatenate([data["y_train"], data["y_test"]])
    return run_statistical_validation(
        y_true=y_true, y_pred_model=y_model, y_pred_aci=y_aci,
        model_builder=build_mlp, X_all=X_all, y_all_scaled=y_all,
    )


def phase_6_report(
    mlp_results=None, ensemble_results=None, ga_results=None,
    aci_metrics=None, shap_results=None, pysr_results=None,
    val_results=None, log_lines=None,
):
    from report_generator import generate_report
    logger.info("\n" + "=" * 60)
    logger.info(" Phase 6 \u2500 PDF Report Generation")
    logger.info("=" * 60)
    # Pass best available metrics to report
    best_metrics = None
    if ensemble_results:
        best_metrics = ensemble_results.get("metrics_test")
    elif mlp_results:
        best_metrics = mlp_results.get("metrics_test")
    return generate_report(
        mlp_metrics        = best_metrics,
        ga_results         = ga_results,
        aci_metrics        = aci_metrics,
        shap_results       = shap_results,
        pysr_results       = pysr_results,
        validation_results = val_results,
        log_lines          = log_lines or [],
    )


# ============================================================
# SUMMARY PRINTER
# ============================================================
def _print_summary(aci_metrics, mlp_results, ensemble_results,
                   ga_results, val_results, report_path, elapsed):
    sep = "=" * 65
    print(f"\n{sep}")
    print(" CORROSION RC BEAM OPTIMIZER \u2014 PIPELINE COMPLETE")
    print(sep)
    print(f"\n  ACI 318-19 Baseline:")
    print(f"    R\u00b2   = {aci_metrics.get('R2','?')}")
    print(f"    RMSE = {aci_metrics.get('RMSE','?')} kN\u00b7m")

    if mlp_results:
        mt = mlp_results.get("metrics_test", {})
        print(f"\n  MLP Baseline (Test):")
        print(f"    R\u00b2   = {mt.get('R2','?')}")
        print(f"    RMSE = {mt.get('RMSE','?')}")

    if ensemble_results:
        et = ensemble_results.get("metrics_test", {})
        print(f"\n  \U0001f3c6 Ensemble Best [{ensemble_results.get('best_name','?')}] (Test):")
        print(f"    R\u00b2        = {et.get('R2','?')}")
        print(f"    RMSE      = {et.get('RMSE','?')}")
        print(f"    L1 broken : {et.get('L1_broken','?')}")
        print(f"    L2 broken : {et.get('L2_broken','?')}")
        print(f"    CV R\u00b2     = {ensemble_results.get('cv_R2_mean','?'):.4f} "
              f"\u00b1 {ensemble_results.get('cv_R2_std','?'):.4f}")

    if ga_results and ga_results.get("best_individual"):
        best = ga_results["best_individual"]
        print(f"\n  NSGA-III Best:")
        print(f"    R\u00b2 = {best.metrics.get('R2','?')}  "
              f"Fitness = {best.fitness:.4f}")

    if val_results:
        print(f"\n  Statistical Validation:")
        print(f"    {val_results.get('verdict','?')}")

    print(f"\n  Report \u2192 {report_path}")
    print(f"  Total time: {elapsed/60:.1f} min ({elapsed:.0f}s)")
    print(sep + "\n")


# ============================================================
# MAIN
# ============================================================
def main():
    args    = _parse_args()
    _configure_logging()
    t_start = time.time()

    logger.info("=" * 65)
    logger.info(" Corrosion RC Beam Optimizer (v4)")
    logger.info(" Target: Mmax,exp (kN\u00b7m)")
    logger.info(f" Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f" Python : {sys.version.split()[0]}")
    logger.info("=" * 65)

    all_phases    = [0, 1, 2, 3, 4, 5, 6]
    phases_to_run = args.phase if args.phase else all_phases
    if args.skip_pysr and 3 in phases_to_run:
        phases_to_run.remove(3)
    if args.skip_shap and 4 in phases_to_run:
        phases_to_run.remove(4)
    if args.report_only:
        phases_to_run = [6]

    logger.info(f"Phases to run: {phases_to_run}")

    from data_preprocessing import run_preprocessing
    data = run_preprocessing(save_clean=True)

    aci_metrics      = {}
    mlp_results      = {}
    ensemble_results = {}
    ga_results       = {}
    pysr_results     = {}
    shap_results     = {}
    val_results      = {}
    log_lines        = []
    best_model       = None

    if 0 in phases_to_run:
        r = phase_0_aci(data)
        aci_metrics = r["aci_metrics"]

    if 1 in phases_to_run:
        # 1A — MLP baseline
        mlp_results = phase_1_mlp(data)
        best_model  = mlp_results.get("model")

        # 1B — Ensemble (primary)
        ensemble_results = phase_1b_ensemble(data)
        if ensemble_results.get("best_model") is not None:
            best_model = ensemble_results["best_model"]

        # Log overall status
        both = ensemble_results.get("both_broken", False)
        if both:
            logger.success("\U0001f3c6 L1 + L2 BOTH BROKEN by Ensemble!")
        else:
            et = ensemble_results.get("metrics_test", {})
            logger.info(f"Ensemble Test R\u00b2={et.get('R2','?')}  "
                        f"L1:{et.get('L1_broken','?')}  "
                        f"L2:{et.get('L2_broken','?')}")

    if 2 in phases_to_run and not args.skip_ga:
        if not aci_metrics:
            r = phase_0_aci(data)
            aci_metrics = r["aci_metrics"]
        ga_results = phase_2_ga(data, aci_metrics)
        log_lines  = ga_results.get("log_lines", [])
        import joblib
        from config import MODEL_GA_PKL
        try:
            best_model = joblib.load(MODEL_GA_PKL)
        except Exception:
            pass

    if 3 in phases_to_run:
        pysr_results = phase_3_pysr(data, aci_metrics)

    if 4 in phases_to_run:
        if best_model is None:
            import joblib
            best_model = joblib.load(MODEL_BEST_PKL)
        shap_results = phase_4_shap(data, best_model)

    if 5 in phases_to_run:
        if best_model is None:
            import joblib
            best_model = joblib.load(MODEL_BEST_PKL)
        val_results = phase_5_validation(data, best_model, aci_metrics)

    if 6 in phases_to_run:
        report_path = phase_6_report(
            mlp_results=mlp_results, ensemble_results=ensemble_results,
            ga_results=ga_results, aci_metrics=aci_metrics,
            shap_results=shap_results, pysr_results=pysr_results,
            val_results=val_results, log_lines=log_lines,
        )
    else:
        report_path = RESULTS_DIR / "Final_Report.pdf"

    elapsed = time.time() - t_start
    _print_summary(
        aci_metrics=aci_metrics, mlp_results=mlp_results,
        ensemble_results=ensemble_results, ga_results=ga_results,
        val_results=val_results, report_path=report_path, elapsed=elapsed,
    )
    logger.info(f"Pipeline finished in {elapsed/60:.1f} min.")


if __name__ == "__main__":
    main()
