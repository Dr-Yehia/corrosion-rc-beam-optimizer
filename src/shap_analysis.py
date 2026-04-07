# ============================================================
# src/shap_analysis.py
# Corrosion RC Beam Optimizer
# SHAP (SHapley Additive exPlanations) Feature Importance
# Explains which variables drive the model's R(%) predictions
# Outputs: shap_importance.png, shap_beeswarm.png, shap_values.csv
# ============================================================

import numpy as np
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
from loguru import logger
import matplotlib
matplotlib.use("Agg")   # non-interactive backend
import matplotlib.pyplot as plt
import sys

sys.path.append(str(Path(__file__).resolve().parent))
from config import (
    SHAP_N_SAMPLES, SHAP_FIGURE_DPI,
    FIGURES_DIR, MODELS_DIR, TARGET_COL, RANDOM_STATE,
)

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.error("SHAP not installed. Run: pip install shap")


# ============================================================
# 1. BUILD SHAP EXPLAINER
# ============================================================

def build_explainer(model, X_background: np.ndarray):
    """
    Build a SHAP KernelExplainer for scikit-learn models.
    Uses a background sample (kmeans-summarised) for efficiency.

    Parameters
    ----------
    model        : trained scikit-learn model (MLPRegressor or GA model)
    X_background : representative background samples (scaled)

    Returns
    -------
    shap.KernelExplainer
    """
    if not SHAP_AVAILABLE:
        raise ImportError("SHAP is not installed.")

    # Summarise background with k-means (fast approximate explainer)
    background = shap.kmeans(X_background, min(50, len(X_background)))
    explainer  = shap.KernelExplainer(model.predict, background)
    logger.info(f"SHAP KernelExplainer built with "
                f"{min(50, len(X_background))} background clusters.")
    return explainer


# ============================================================
# 2. COMPUTE SHAP VALUES
# ============================================================

def compute_shap_values(
    explainer,
    X_explain:     np.ndarray,
    n_samples:     int = SHAP_N_SAMPLES,
) -> np.ndarray:
    """
    Compute SHAP values for a random subsample of X_explain.

    Parameters
    ----------
    explainer : SHAP explainer object
    X_explain : full test/train array (scaled)
    n_samples : number of samples to explain (default 200)

    Returns
    -------
    shap_values : np.ndarray of shape (n_samples, n_features)
    X_sample    : np.ndarray of shape (n_samples, n_features)
    """
    np.random.seed(RANDOM_STATE)
    idx       = np.random.choice(len(X_explain),
                                 size=min(n_samples, len(X_explain)),
                                 replace=False)
    X_sample  = X_explain[idx]

    logger.info(f"Computing SHAP values for {len(X_sample)} samples ...")
    shap_values = explainer.shap_values(X_sample, silent=True)
    logger.info("SHAP values computed.")
    return shap_values, X_sample


# ============================================================
# 3. RANK FEATURE IMPORTANCE
# ============================================================

def rank_feature_importance(
    shap_values:   np.ndarray,
    feature_names: list,
) -> pd.DataFrame:
    """
    Rank features by mean absolute SHAP value.

    Returns
    -------
    DataFrame with columns: feature, mean_abs_shap, rank
    """
    mean_abs = np.abs(shap_values).mean(axis=0)
    df_imp   = pd.DataFrame({
        "feature"       : feature_names,
        "mean_abs_shap" : mean_abs,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    df_imp["rank"] = df_imp.index + 1

    logger.info("Feature Importance Ranking (SHAP):")
    for _, row in df_imp.iterrows():
        logger.info(f"  #{row['rank']:2d}  {row['feature']:<45}  "
                    f"SHAP={row['mean_abs_shap']:.4f}")

    return df_imp


# ============================================================
# 4. PLOT — BAR CHART (Mean |SHAP|)
# ============================================================

def plot_shap_bar(
    df_importance: pd.DataFrame,
    top_n: int = 15,
    save: bool = True,
) -> None:
    """
    Horizontal bar chart of top-N features by mean |SHAP| value.
    Publication-ready (300 DPI, clean style).
    """
    df_plot = df_importance.head(top_n).sort_values(
        "mean_abs_shap", ascending=True
    )

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(
        df_plot["feature"],
        df_plot["mean_abs_shap"],
        color="#2196F3",
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set_xlabel("Mean |SHAP Value| — Impact on R(%) Prediction",
                  fontsize=12)
    ax.set_title(
        f"Top {top_n} Feature Importances (SHAP Analysis)\n"
        f"Corrosion RC Beam Optimizer",
        fontsize=13, fontweight="bold",
    )
    ax.tick_params(axis="y", labelsize=10)
    ax.tick_params(axis="x", labelsize=10)
    ax.spines[["top", "right"]].set_visible(False)

    # Value labels on bars
    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + 0.001, bar.get_y() + bar.get_height() / 2,
            f"{width:.3f}",
            va="center", ha="left", fontsize=9,
        )

    plt.tight_layout()
    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        path = FIGURES_DIR / "shap_importance.png"
        plt.savefig(path, dpi=SHAP_FIGURE_DPI, bbox_inches="tight")
        logger.info(f"SHAP bar chart saved \u2192 {path}")
    plt.close()


# ============================================================
# 5. PLOT — BEESWARM
# ============================================================

def plot_shap_beeswarm(
    shap_values:   np.ndarray,
    X_sample:      np.ndarray,
    feature_names: list,
    top_n: int = 15,
    save: bool = True,
) -> None:
    """
    SHAP beeswarm plot — shows distribution of SHAP values per feature.
    Colour = feature value (blue=low, red=high).
    """
    if not SHAP_AVAILABLE:
        return

    X_df = pd.DataFrame(X_sample, columns=feature_names)

    plt.figure(figsize=(11, 8))
    shap.summary_plot(
        shap_values,
        X_df,
        max_display = top_n,
        show        = False,
        plot_type   = "dot",
    )
    plt.title(
        "SHAP Beeswarm Plot — Feature Impact on R(%) Prediction",
        fontsize=13, fontweight="bold", pad=12,
    )
    plt.tight_layout()

    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        path = FIGURES_DIR / "shap_beeswarm.png"
        plt.savefig(path, dpi=SHAP_FIGURE_DPI, bbox_inches="tight")
        logger.info(f"SHAP beeswarm saved \u2192 {path}")
    plt.close()


# ============================================================
# 6. PLOT — DEPENDENCE (top feature vs target)
# ============================================================

def plot_shap_dependence(
    shap_values:   np.ndarray,
    X_sample:      np.ndarray,
    feature_names: list,
    top_feature:   str = None,
    save: bool = True,
) -> None:
    """
    SHAP dependence plot for the single most important feature.
    Shows how that feature's value affects predictions.
    """
    if not SHAP_AVAILABLE:
        return

    if top_feature is None:
        top_feature = feature_names[0]   # most important by default

    if top_feature not in feature_names:
        logger.warning(f"Feature '{top_feature}' not found — skipping.")
        return

    feat_idx = feature_names.index(top_feature)
    X_df     = pd.DataFrame(X_sample, columns=feature_names)

    plt.figure(figsize=(9, 6))
    shap.dependence_plot(
        feat_idx, shap_values, X_df,
        interaction_index = "auto",
        show              = False,
    )
    plt.title(
        f"SHAP Dependence: '{top_feature}' vs R(%) Prediction",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()

    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fname = top_feature.replace(" ", "_").replace("/", "_")[:40]
        path  = FIGURES_DIR / f"shap_dependence_{fname}.png"
        plt.savefig(path, dpi=SHAP_FIGURE_DPI, bbox_inches="tight")
        logger.info(f"SHAP dependence plot saved \u2192 {path}")
    plt.close()


# ============================================================
# 7. SAVE SHAP RESULTS
# ============================================================

def save_shap_results(
    df_importance: pd.DataFrame,
    shap_values:   np.ndarray,
    feature_names: list,
) -> None:
    """Save SHAP values and importance table to results/."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Importance table
    imp_path = MODELS_DIR / "shap_importance.csv"
    df_importance.to_csv(imp_path, index=False)
    logger.info(f"SHAP importance table saved \u2192 {imp_path}")

    # Raw SHAP values
    shap_df   = pd.DataFrame(shap_values, columns=feature_names)
    shap_path = MODELS_DIR / "shap_values.csv"
    shap_df.to_csv(shap_path, index=False)
    logger.info(f"SHAP values saved \u2192 {shap_path}")

    # Top-5 features for PySR guidance
    top5 = df_importance.head(5)["feature"].tolist()
    top5_path = MODELS_DIR / "top5_shap_features.json"
    with open(top5_path, "w") as f:
        json.dump({
            "top5_features" : top5,
            "generated_at"  : str(datetime.now()),
        }, f, indent=2)
    logger.info(f"Top-5 features saved \u2192 {top5_path}")
    logger.info(f"Top-5 features for PySR: {top5}")


# ============================================================
# 8. FULL PIPELINE
# ============================================================

def run_shap_analysis(
    model,
    X_train:       np.ndarray,
    X_test:        np.ndarray,
    feature_names: list,
    top_n:         int = 15,
) -> dict:
    """
    End-to-end SHAP analysis pipeline.

    Parameters
    ----------
    model         : trained scikit-learn model
    X_train       : scaled training array (used as background)
    X_test        : scaled test array (samples to explain)
    feature_names : list of feature column names
    top_n         : number of top features to plot

    Returns
    -------
    dict with 'df_importance', 'shap_values', 'top5_features'
    """
    logger.info("=" * 60)
    logger.info(" SHAP Analysis — Feature Importance")
    logger.info("=" * 60)

    explainer              = build_explainer(model, X_train)
    shap_values, X_sample  = compute_shap_values(
        explainer, X_test, SHAP_N_SAMPLES
    )
    df_importance          = rank_feature_importance(shap_values, feature_names)

    plot_shap_bar(df_importance, top_n=top_n)
    plot_shap_beeswarm(shap_values, X_sample, feature_names, top_n=top_n)

    top_feature = df_importance.iloc[0]["feature"]
    plot_shap_dependence(shap_values, X_sample, feature_names, top_feature)

    save_shap_results(df_importance, shap_values, feature_names)

    logger.info("=" * 60)
    logger.info(" SHAP Analysis Complete ✓")
    logger.info("=" * 60)

    return {
        "df_importance" : df_importance,
        "shap_values"   : shap_values,
        "top5_features" : df_importance.head(5)["feature"].tolist(),
    }


# ============================================================
# CLI ENTRY POINT
# ============================================================

if __name__ == "__main__":
    from data_preprocessing import run_preprocessing
    from neural_network import load_model
    import joblib

    data    = run_preprocessing(save_clean=True)
    model   = load_model()
    scaler_X = joblib.load(MODELS_DIR / "scaler_X.pkl")

    results = run_shap_analysis(
        model         = model,
        X_train       = data["X_train"],
        X_test        = data["X_test"],
        feature_names = data["feature_cols"],
    )

    print("\n\u2705 SHAP analysis complete.")
    print(f"   Top 5 features: {results['top5_features']}")
