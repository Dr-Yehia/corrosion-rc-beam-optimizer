# ============================================================
# src/main.py
# Corrosion RC Beam Optimizer
# Master Orchestrator — runs the complete research pipeline
#
# Execution order:
#   Phase 0  ─ ACI 318-19 baseline benchmark
#   Phase 1  ─ MLP baseline training & evaluation
#   Phase 2  ─ NSGA-III GA optimisation (multi-run)
#   Phase 3  ─ PySR symbolic regression
#   Phase 4  ─ SHAP feature importance analysis
#   Phase 5  ─ Statistical validation (Wilcoxon, Bootstrap, CV …)
#   Phase 6  ─ Automated PDF report generation
#
# Usage:
#   python src/main.py              # full run
#   python src/main.py --skip-pysr  # skip symbolic regression
#   python src/main.py --phase 0 1  # run specific phases only
# ============================================================

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from loguru import logger
import numpy as np

# ─ ensure src/ is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    LOG_FILE, LOG_DIR, MODELS_DIR, RESULTS_DIR,
    L1_TARGET_R2, L2_TARGET_R2, BREAK_BOTH,
    RANDOM_STATE,
)


# ============================================================
# LOGGING SETUP
# ============================================================

def _configure_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger.remove()   # remove default stderr sink
    logger.add(
        sys.stderr,
        format  = "<green>{time:HH:mm:ss}</green> | "
                  "<level>{level:<8}</level> | {message}",
        level   = "INFO",
        colorize= True,
    )
    logger.add(
        str(LOG_FILE),
        format  = "{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}",
        level   = "DEBUG",
        rotation= "10 MB",
        retention= 5,
        encoding= "utf-8",
    )
    logger.info(f"Logging configured → {LOG_FILE}")


# ============================================================
# ARGUMENT PARSER
# ============================================================

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog        = "main.py",
        description = "Corrosion RC Beam Optimizer — Full Pipeline",
    )
    parser.add_argument(
        "--skip-pysr", action="store_true",
        help="Skip Phase 3 (PySR symbolic regression).",
    )
    parser.add_argument(
        "--skip-shap", action="store_true",
        help="Skip Phase 4 (SHAP analysis).",
    )
    parser.add_argument(
        "--phase", nargs="+", type=int,
        help="Run only the specified phase numbers (0-6).",
    )
    parser.add_argument(
        "--report-only", action="store_true",
        help="Skip all computation — regenerate PDF report from saved results.",
    )
    return parser.parse_args()


# ============================================================
# PHASE RUNNERS
# ============================================================

def phase_0_aci(data: dict) -> dict:
    """
    Phase 0 ─ ACI 318-19 benchmark.
    Establishes the official baseline to beat.
    """
    from aci_calculator import (
        compute_aci_predictions, evaluate_aci_benchmark,
        save_benchmark_results,
    )
    logger.info("\n" + "=" * 60)
    logger.info(" Phase 0 ─ ACI 318-19 Benchmark")
    logger.info("=" * 60)

    df_aci      = compute_aci_predictions(data["df_clean"])
    aci_metrics = evaluate_aci_benchmark(df_aci)
    save_benchmark_results(df_aci, aci_metrics)

    logger.info(
        f"ACI baseline ─ R²={aci_metrics['R2']}  "
        f"RMSE={aci_metrics['RMSE']}  "
        f"Ratio={aci_metrics['ratio_mean']}"
    )
    return {"df_aci": df_aci, "aci_metrics": aci_metrics}


def phase_1_mlp(data: dict) -> dict:
    """
    Phase 1 ─ Baseline MLP training.
    Provides the starting point for GA optimisation.
    """
    from neural_network import run_training_pipeline

    logger.info("\n" + "=" * 60)
    logger.info(" Phase 1 ─ MLP Baseline Training")
    logger.info("=" * 60)

    mlp_results = run_training_pipeline(
        data["X_train"], data["X_test"],
        data["y_train"], data["y_test"],
        scaler_y = data["scaler_y"],
    )
    return mlp_results


def phase_2_ga(data: dict, aci_metrics: dict) -> dict:
    """
    Phase 2 ─ NSGA-III Genetic Algorithm optimisation.
    Multi-run with restart-on-convergence strategy.
    """
    from genetic_algorithm import run_nsga3

    logger.info("\n" + "=" * 60)
    logger.info(" Phase 2 ─ NSGA-III Optimisation")
    logger.info("=" * 60)

    log_lines  = []
    ga_results = run_nsga3(
        X_train   = data["X_train"],
        y_train   = data["y_train"],
        X_test    = data["X_test"],
        y_test    = data["y_test"],
        scaler_y  = data["scaler_y"],
        aci_rmse  = aci_metrics["RMSE"],
        aci_mae   = aci_metrics["MAE"],
        log_lines = log_lines,
    )
    ga_results["log_lines"] = log_lines
    return ga_results


def phase_3_pysr(data: dict, aci_metrics: dict) -> dict:
    """
    Phase 3 ─ PySR symbolic regression.
    Discovers a closed-form equation for R(%).
    """
    from symbolic_regression import run_symbolic_regression, PYSR_FEATURES

    logger.info("\n" + "=" * 60)
    logger.info(" Phase 3 ─ PySR Symbolic Regression")
    logger.info("=" * 60)

    pysr_results = run_symbolic_regression(
        df          = data["df_clean"],
        aci_rmse    = aci_metrics["RMSE"],
        feature_list= PYSR_FEATURES,
    )
    return pysr_results


def phase_4_shap(data: dict, model) -> dict:
    """
    Phase 4 ─ SHAP feature importance analysis.
    Identifies the variables most influential on R(%).
    """
    from shap_analysis import run_shap_analysis

    logger.info("\n" + "=" * 60)
    logger.info(" Phase 4 ─ SHAP Analysis")
    logger.info("=" * 60)

    shap_results = run_shap_analysis(
        model         = model,
        X_train       = data["X_train"],
        X_test        = data["X_test"],
        feature_names = data["feature_cols"],
    )
    return shap_results


def phase_5_validation(
    data:        dict,
    model,
    aci_metrics: dict,
) -> dict:
    """
    Phase 5 ─ Statistical validation.
    Confirms benchmark improvements are statistically significant.
    """
    from statistical_validation import run_statistical_validation
    from neural_network import predict, build_mlp
    from aci_calculator import compute_aci_predictions
    import joblib

    logger.info("\n" + "=" * 60)
    logger.info(" Phase 5 ─ Statistical Validation")
    logger.info("=" * 60)

    scaler_y = data["scaler_y"]
    y_true   = scaler_y.inverse_transform(
        data["y_test"].reshape(-1, 1)
    ).ravel()
    y_model  = predict(model, data["X_test"], scaler_y)

    # ACI predictions aligned with test indices
    df_aci   = compute_aci_predictions(data["df_clean"])
    y_aci    = df_aci.loc[
        data["y_test_raw"].index, "MACI_pred"
    ].values

    X_all = np.vstack([data["X_train"], data["X_test"]])
    y_all = np.concatenate([data["y_train"], data["y_test"]])

    val_results = run_statistical_validation(
        y_true        = y_true,
        y_pred_model  = y_model,
        y_pred_aci    = y_aci,
        model_builder = build_mlp,
        X_all         = X_all,
        y_all_scaled  = y_all,
    )
    return val_results


def phase_6_report(
    mlp_results:    dict = None,
    ga_results:     dict = None,
    aci_metrics:    dict = None,
    shap_results:   dict = None,
    pysr_results:   dict = None,
    val_results:    dict = None,
    log_lines:      list = None,
) -> Path:
    """
    Phase 6 ─ PDF report generation.
    """
    from report_generator import generate_report

    logger.info("\n" + "=" * 60)
    logger.info(" Phase 6 ─ PDF Report Generation")
    logger.info("=" * 60)

    path = generate_report(
        mlp_metrics        = mlp_results.get("metrics_test") if mlp_results else None,
        ga_results         = ga_results,
        aci_metrics        = aci_metrics,
        shap_results       = shap_results,
        pysr_results       = pysr_results,
        validation_results = val_results,
        log_lines          = log_lines,
    )
    return path


# ============================================================
# PIPELINE SUMMARY PRINTER
# ============================================================

def _print_summary(
    aci_metrics:  dict,
    mlp_results:  dict,
    ga_results:   dict,
    val_results:  dict,
    report_path:  Path,
    elapsed:      float,
) -> None:
    """Print a colour-coded final summary to console."""
    sep = "=" * 65
    print(f"\n{sep}")
    print(" CORROSION RC BEAM OPTIMIZER — PIPELINE COMPLETE")
    print(sep)

    # ACI
    print(f"\n  ACI 318-19 Baseline:")
    print(f"    R\u00b2   = {aci_metrics.get('R2',  '?')}")
    print(f"    RMSE = {aci_metrics.get('RMSE','?')} kN\u00b7m")

    # MLP
    mt = mlp_results.get("metrics_test", {}) if mlp_results else {}
    print(f"\n  MLP Baseline (Test):")
    print(f"    R\u00b2   = {mt.get('R2',  '?')}")
    print(f"    RMSE = {mt.get('RMSE','?')}")
    print(f"    L1 broken : {mt.get('L1_broken', '?')}")
    print(f"    L2 broken : {mt.get('L2_broken', '?')}")

    # GA
    if ga_results and ga_results.get("best_individual"):
        best = ga_results["best_individual"]
        print(f"\n  NSGA-III Best:")
        print(f"    R\u00b2       = {best.metrics.get('R2',   '?')}")
        print(f"    RMSE     = {best.metrics.get('RMSE', '?')}")
        print(f"    CV-R\u00b2    = {best.metrics.get('CV_R2','?')}")
        print(f"    Fitness  = {best.fitness:.4f}")
        print(f"    Success  = {ga_results.get('success', '?')}")
        print(f"    Best Run = {ga_results.get('best_run', '?')} "
              f"| Gen = {ga_results.get('best_gen', '?')}")

    # Statistical
    if val_results:
        print(f"\n  Statistical Validation:")
        print(f"    {val_results.get('verdict', '?')}")

    print(f"\n  Report → {report_path}")
    print(f"  Total time: {elapsed/60:.1f} min  ({elapsed:.0f}s)")
    print(sep + "\n")


# ============================================================
# MAIN ENTRY POINT
# ============================================================

def main() -> None:
    args    = _parse_args()
    _configure_logging()
    t_start = time.time()

    logger.info("=" * 65)
    logger.info(" Corrosion RC Beam Optimizer")
    logger.info(f" Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f" Python : {sys.version.split()[0]}")
    logger.info("=" * 65)

    # Determine which phases to run
    all_phases   = [0, 1, 2, 3, 4, 5, 6]
    phases_to_run = args.phase if args.phase else all_phases
    if args.skip_pysr and 3 in phases_to_run:
        phases_to_run.remove(3)
    if args.skip_shap and 4 in phases_to_run:
        phases_to_run.remove(4)
    if args.report_only:
        phases_to_run = [6]

    logger.info(f"Phases to run: {phases_to_run}")

    # Preprocessing (always required)
    from data_preprocessing import run_preprocessing
    data = run_preprocessing(save_clean=True)

    # Results containers
    aci_metrics  = {}
    mlp_results  = {}
    ga_results   = {}
    pysr_results = {}
    shap_results = {}
    val_results  = {}
    log_lines    = []
    best_model   = None

    # ────────────────────────────────────────────────────
    if 0 in phases_to_run:
        r = phase_0_aci(data)
        aci_metrics = r["aci_metrics"]

    if 1 in phases_to_run:
        mlp_results = phase_1_mlp(data)
        best_model  = mlp_results.get("model")

    if 2 in phases_to_run:
        if not aci_metrics:
            logger.warning("ACI metrics not available — running Phase 0 first.")
            r = phase_0_aci(data)
            aci_metrics = r["aci_metrics"]
        ga_results  = phase_2_ga(data, aci_metrics)
        log_lines   = ga_results.get("log_lines", [])
        # Use GA model if it is better than MLP baseline
        from neural_network import load_model
        import joblib
        from config import MODEL_GA_PKL, MODEL_MLP_PKL
        try:
            ga_model   = joblib.load(MODEL_GA_PKL)
            best_model = ga_model
        except Exception:
            pass

    if 3 in phases_to_run:
        pysr_results = phase_3_pysr(data, aci_metrics)

    if 4 in phases_to_run:
        if best_model is None:
            from neural_network import load_model
            best_model = load_model()
        shap_results = phase_4_shap(data, best_model)

    if 5 in phases_to_run:
        if best_model is None:
            from neural_network import load_model
            best_model = load_model()
        val_results = phase_5_validation(data, best_model, aci_metrics)

    if 6 in phases_to_run:
        report_path = phase_6_report(
            mlp_results  = mlp_results,
            ga_results   = ga_results,
            aci_metrics  = aci_metrics,
            shap_results = shap_results,
            pysr_results = pysr_results,
            val_results  = val_results,
            log_lines    = log_lines,
        )
    else:
        report_path = RESULTS_DIR / "Final_Report.pdf"

    # Final summary
    elapsed = time.time() - t_start
    _print_summary(
        aci_metrics  = aci_metrics,
        mlp_results  = mlp_results,
        ga_results   = ga_results,
        val_results  = val_results,
        report_path  = report_path,
        elapsed      = elapsed,
    )

    logger.info(f"Pipeline finished in {elapsed/60:.1f} min.")


if __name__ == "__main__":
    main()
