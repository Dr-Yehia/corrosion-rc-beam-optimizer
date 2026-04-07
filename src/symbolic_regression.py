# ============================================================
# src/symbolic_regression.py
# Corrosion RC Beam Optimizer
# PySR Symbolic Regression Pipeline
# Goal : Discover a closed-form equation for Mmax,exp (kNm)
#        that outperforms ACI 318-19
# Output: best_equation.txt + best_equation.latex
# ============================================================
import re

import numpy as np
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
from loguru import logger
import sys

sys.path.append(str(Path(__file__).resolve().parent))
from config import (
    PYSR_NITERATIONS, PYSR_MAXSIZE, PYSR_POPULATIONS,
    PYSR_BINARY_OPS, PYSR_UNARY_OPS,
    PYSR_OUTPUT_FILE, PYSR_LATEX_FILE,
    FEATURE_COLS, TARGET_COL, RANDOM_STATE,
    EQ_DIR, FIGURES_DIR, MODELS_DIR,
    L1_TARGET_R2, L2_TARGET_R2,
)

try:
    from pysr import PySRRegressor
    PYSR_AVAILABLE = True
except ImportError:
    PYSR_AVAILABLE = False
    logger.error(
        "PySR not installed. Run: pip install pysr\n"
        "Julia is required: https://julialang.org/downloads/"
    )

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Sanitize variable names for PySR (only alphanumeric + underscore)
def _sanitize_name(name: str) -> str:
    """Convert column name to PySR-safe variable name."""
    MAPPING = {
        "Mass Loss (Tensile bars), \u03b7m (%)": "eta_m",
        "fy Longitudinal Bars (Tensile), (MPa) ": "fy",
        "f'c (MPa)": "fc",
        "Depth (mm)": "d",
        "Width (mm)": "b",
        "Tension Reinforcement Ratio, pten (%)": "rho_t",
        "corr_severity_idx": "CSI",
        "d_b_ratio": "d_b",
        "reinf_index": "RI",
    }
    if name in MAPPING:
        return MAPPING[name]
    # Fallback: replace non-alphanumeric with underscore
    clean = re.sub(r'[^a-zA-Z0-9_]', '_', name)
    clean = re.sub(r'_+', '_', clean).strip('_')
    return clean if clean else 'x'


# ============================================================
# 1. FEATURE SUBSET FOR SYMBOLIC REGRESSION
# ============================================================

# PySR works best with 5–8 features — select the most physically relevant
# (confirmed by SHAP analysis in shap_analysis.py)
PYSR_FEATURES = [
    "Mass Loss (Tensile bars), \u03b7m (%)",
    "fy Longitudinal Bars (Tensile), (MPa) ",
    "f'c (MPa)",
    "Depth (mm)",
    "Width (mm)",
    "Tension Reinforcement Ratio, pten (%)",
    "corr_severity_idx",
    "d_b_ratio",
]


# ============================================================
# 2. BUILD PySR MODEL
# ============================================================

def build_pysr_model() -> "PySRRegressor":
    """
    Configure PySRRegressor with physics-appropriate constraints.

    Key settings:
    - binary_operators  : +, -, *, /, ^
    - unary_operators   : sqrt, log, exp
    - maxsize           : 20 nodes (interpretable yet expressive)
    - populations       : 30 independent evolutionary populations
    - constraints       : penalise division-by-zero, enforce positive terms
    """
    if not PYSR_AVAILABLE:
        raise ImportError("PySR is not installed.")

    model = PySRRegressor(
        niterations      = PYSR_NITERATIONS,
        maxsize          = PYSR_MAXSIZE,
        populations      = PYSR_POPULATIONS,
        binary_operators = PYSR_BINARY_OPS,
        unary_operators  = PYSR_UNARY_OPS,
        model_selection  = "best",        # best complexity-accuracy trade-off
        loss             = "loss(x, y) = (x - y)^2",   # MSE
        verbosity        = 1,
        random_state     = RANDOM_STATE,
        deterministic    = True,
        parallelism      = "multithreading",
        turbo            = True,          # Julia-level speed optimisation
        output_jax_format= False,
        extra_sympy_mappings={},
    )
    logger.info("PySR model configured.")
    logger.info(f"  Iterations  : {PYSR_NITERATIONS}")
    logger.info(f"  Max size    : {PYSR_MAXSIZE}")
    logger.info(f"  Populations : {PYSR_POPULATIONS}")
    logger.info(f"  Binary ops  : {PYSR_BINARY_OPS}")
    logger.info(f"  Unary ops   : {PYSR_UNARY_OPS}")
    return model


# ============================================================
# 3. PREPARE FEATURES
# ============================================================

def prepare_pysr_features(
    df: pd.DataFrame,
    feature_list: list = None,
) -> tuple:
    """
    Extract and validate the feature matrix and target vector
    for PySR training.

    Parameters
    ----------
    df           : clean DataFrame (output of data_preprocessing)
    feature_list : list of column names to use (defaults to PYSR_FEATURES)

    Returns
    -------
    X (np.ndarray), y (np.ndarray), feature_names (list)
    """
    if feature_list is None:
        feature_list = PYSR_FEATURES

    available = [f for f in feature_list if f in df.columns]
    missing   = set(feature_list) - set(available)
    if missing:
        logger.warning(f"PySR: missing features {missing} — using available.")

    X = df[available].values.astype(np.float64)
    y = df[TARGET_COL].values.astype(np.float64)

    # Remove any rows with NaN or Inf
    valid_mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X, y = X[valid_mask], y[valid_mask]

    # Sanitize names for PySR
    safe_names = [_sanitize_name(n) for n in available]
    logger.info(f"PySR feature matrix: {X.shape[0]} samples, "
                f"{X.shape[1]} features")
    logger.info(f"PySR variable names: {safe_names}")
    logger.info(f"Target Mmax — mean={y.mean():.2f}, std={y.std():.2f}")
    return X, y, safe_names


# ============================================================
# 4. TRAIN
# ============================================================

def train_pysr(
    model: "PySRRegressor",
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list,
) -> "PySRRegressor":
    """
    Fit PySR on the full clean dataset.
    PySR searches for the symbolic equation that best maps
    {\u03b7m, fy, f’c, d, b, ...} → R(%)
    """
    logger.info("Starting PySR symbolic regression ...")
    logger.info("This may take several minutes — Julia JIT on first run.")

    model.fit(X, y, variable_names=feature_names)

    logger.info("PySR training complete.")
    return model


# ============================================================
# 5. EXTRACT & RANK EQUATIONS
# ============================================================

def extract_best_equations(model: "PySRRegressor") -> pd.DataFrame:
    """
    Extract the Pareto-optimal equations discovered by PySR.
    Returns a DataFrame ranked by score (accuracy vs complexity).
    """
    equations = model.get_hof()   # Hall of Fame dataframe
    logger.info(f"PySR discovered {len(equations)} Pareto-optimal equations.")

    if "sympy_format" in equations.columns:
        for i, row in equations.iterrows():
            logger.info(
                f"  Eq {i:2d} | Complexity={row.get('complexity', '?'):3d} | "
                f"Loss={row.get('loss', 0):.4f} | "
                f"{row.get('sympy_format', row.get('equation',''))}"
            )
    return equations


# ============================================================
# 6. EVALUATE DISCOVERED EQUATION
# ============================================================

def evaluate_pysr_equation(
    model: "PySRRegressor",
    X: np.ndarray,
    y: np.ndarray,
    aci_rmse: float = None,
) -> dict:
    """
    Evaluate best PySR equation on the full dataset.
    Computes R², RMSE, MAE, MAPE and ACI improvement.
    """
    y_pred = model.predict(X)
    y_pred = np.clip(y_pred, 0, 500)   # physical bounds (kNm)

    r2   = r2_score(y, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y, y_pred)))
    mae  = float(mean_absolute_error(y, y_pred))
    mape = float(np.mean(np.abs((y - y_pred) / np.maximum(np.abs(y), 1e-6))) * 100)

    l1_broken = r2 >= L1_TARGET_R2
    l2_broken = r2 >= L2_TARGET_R2

    metrics = {
        "R2"        : round(r2,   4),
        "RMSE"      : round(rmse, 4),
        "MAE"       : round(mae,  4),
        "MAPE"      : round(mape, 2),
        "L1_broken" : l1_broken,
        "L2_broken" : l2_broken,
        "timestamp" : str(datetime.now()),
    }
    if aci_rmse:
        metrics["RMSE_vs_ACI_pct"] = round(
            (aci_rmse - rmse) / aci_rmse * 100, 2
        )

    logger.info("=" * 50)
    logger.info(" PySR Best Equation — Evaluation")
    logger.info("=" * 50)
    logger.info(f"  R²   = {metrics['R2']}")
    logger.info(f"  RMSE = {metrics['RMSE']}")
    logger.info(f"  MAE  = {metrics['MAE']}")
    logger.info(f"  MAPE = {metrics['MAPE']} %")
    if aci_rmse:
        logger.info(f"  RMSE improvement vs ACI: {metrics['RMSE_vs_ACI_pct']} %")
    logger.info(f"  L1 broken : {l1_broken}")
    logger.info(f"  L2 broken : {l2_broken}")
    logger.info("=" * 50)

    return metrics


# ============================================================
# 7. SAVE EQUATIONS
# ============================================================

def save_equations(
    model: "PySRRegressor",
    equations: pd.DataFrame,
    metrics: dict,
) -> None:
    """
    Save the best discovered equation in three formats:
    1. Plain text (.txt)
    2. LaTeX (.latex)
    3. Full equations table + metrics (.json)
    """
    EQ_DIR.mkdir(parents=True, exist_ok=True)

    # ─ Best equation string
    best_eq_str   = str(model.sympy())
    best_eq_latex = str(model.latex())

    # Plain text
    with open(PYSR_OUTPUT_FILE, "w") as f:
        f.write(f"# Best PySR Equation\n")
        f.write(f"# Generated: {datetime.now()}\n")
        f.write(f"# R\u00b2 = {metrics['R2']} | RMSE = {metrics['RMSE']}\n\n")
        f.write(f"Mmax = {best_eq_str}\n")
    logger.info(f"Equation saved \u2192 {PYSR_OUTPUT_FILE}")

    # LaTeX
    with open(PYSR_LATEX_FILE, "w") as f:
        f.write(f"% Best PySR Equation — LaTeX format\n")
        f.write(f"% Generated: {datetime.now()}\n")
        f.write(f"% R\u00b2 = {metrics['R2']} | RMSE = {metrics['RMSE']}\n\n")
        f.write(f"M_{{max}} = {best_eq_latex}\n")
    logger.info(f"LaTeX equation saved \u2192 {PYSR_LATEX_FILE}")

    # Full JSON
    json_path = EQ_DIR / "all_equations.json"
    eq_records = equations.to_dict(orient="records") if equations is not None else []
    payload = {
        "best_equation"      : best_eq_str,
        "best_equation_latex": best_eq_latex,
        "metrics"            : metrics,
        "all_equations"      : eq_records,
        "generated_at"       : str(datetime.now()),
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    logger.info(f"Full equations JSON saved \u2192 {json_path}")

    # Print to console
    print(f"\n{'='*60}")
    print(f" DISCOVERED EQUATION:")
    print(f" Mmax = {best_eq_str}")
    print(f" LaTeX: M_max = {best_eq_latex}")
    print(f" R\u00b2 = {metrics['R2']} | RMSE = {metrics['RMSE']} kNm")
    print(f"{'='*60}")


# ============================================================
# 8. FULL PIPELINE
# ============================================================

def run_symbolic_regression(
    df: pd.DataFrame,
    aci_rmse: float = None,
    feature_list: list = None,
) -> dict:
    """
    End-to-end symbolic regression pipeline.

    Parameters
    ----------
    df          : clean DataFrame from data_preprocessing
    aci_rmse    : ACI RMSE baseline for improvement reporting
    feature_list: override default PYSR_FEATURES if needed

    Returns
    -------
    dict with 'model', 'equations', 'metrics', 'best_eq_str'
    """
    logger.info("=" * 60)
    logger.info(" Phase 3 — PySR Symbolic Regression")
    logger.info("=" * 60)

    X, y, feature_names = prepare_pysr_features(df, feature_list)
    model               = build_pysr_model()
    model               = train_pysr(model, X, y, feature_names)
    equations           = extract_best_equations(model)
    metrics             = evaluate_pysr_equation(model, X, y, aci_rmse)
    save_equations(model, equations, metrics)

    logger.info("=" * 60)
    logger.info(" Phase 3 Complete ✓")
    logger.info("=" * 60)

    return {
        "model"       : model,
        "equations"   : equations,
        "metrics"     : metrics,
        "best_eq_str" : str(model.sympy()),
    }


# ============================================================
# CLI ENTRY POINT
# ============================================================

if __name__ == "__main__":
    from data_preprocessing import run_preprocessing
    from aci_calculator import (
        load_raw_data as _load, clean_data as _clean,
        compute_aci_predictions, evaluate_aci_benchmark,
    )

    data     = run_preprocessing(save_clean=True)
    df_clean = data["df_clean"]

    df_aci      = compute_aci_predictions(df_clean)
    aci_metrics = evaluate_aci_benchmark(df_aci)

    results = run_symbolic_regression(
        df_clean,
        aci_rmse     = aci_metrics["RMSE"],
        feature_list = PYSR_FEATURES,
    )

    print("\n\u2705 Symbolic regression complete.")
    print(f"   Best equation : {results['best_eq_str']}")
    print(f"   R\u00b2            : {results['metrics']['R2']}")
    print(f"   RMSE          : {results['metrics']['RMSE']}")
