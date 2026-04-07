# ============================================================
# src/aci_calculator.py
# Corrosion RC Beam Optimizer
# ACI 318-19 Section 22.2.2 — Official Benchmark Implementation
# Eq. A1:  Mn = As,corr · fy,corr · (d - a/2)
#          a  = As,corr · fy,corr / (0.85 · f'c · b)
# ============================================================

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from loguru import logger
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent))
from config import (
    ACI_COLS, TARGET_COL, FIGURES_DIR, RESULTS_DIR, L1_TARGET_R2
)


# ────────────────────────────────────────────────────────────
# CONSTANTS
# ────────────────────────────────────────────────────────────
PHI_FLEXURE   = 0.90   # ACI strength reduction factor
BETA_1        = 0.85   # Equivalent stress block factor (f'c ≤ 28 MPa)
RHO_STEEL     = 7850.0 # kg/m³ — for corrosion area calculation


# ────────────────────────────────────────────────────────────
# 1. CORRODED STEEL AREA
# ────────────────────────────────────────────────────────────
def corroded_area(n_bars: float, db_mm: float, eta_m: float) -> float:
    """
    Compute residual tensile steel area after corrosion.

    Parameters
    ----------
    n_bars : number of tensile bars
    db_mm  : nominal bar diameter (mm)
    eta_m  : gravimetric mass loss (%)  — 0 to 64

    Returns
    -------
    As_corr : corroded steel area (mm²)
    """
    As_nominal = n_bars * np.pi * (db_mm / 2.0) ** 2   # mm²
    As_corr    = As_nominal * (1.0 - eta_m / 100.0)
    return np.maximum(As_corr, 1e-6)   # avoid zero division


# ────────────────────────────────────────────────────────────
# 2. CORRODED YIELD STRENGTH
# ────────────────────────────────────────────────────────────
def corroded_fy(fy_nominal: float, eta_m: float) -> float:
    """
    Estimate residual yield strength after corrosion.
    Uses the linear degradation model (Du et al., 2005):
        fy,corr = fy · (1 - 0.005 · η_m / π_cross)
    Simplified form for uniform corrosion:
        fy,corr ≈ fy · (1 - eta_m / 100)

    Parameters
    ----------
    fy_nominal : nominal yield strength (MPa)
    eta_m      : gravimetric mass loss (%)

    Returns
    -------
    fy_corr : corroded yield strength (MPa)
    """
    fy_corr = fy_nominal * (1.0 - eta_m / 100.0)
    return np.maximum(fy_corr, 1.0)


# ────────────────────────────────────────────────────────────
# 3. ACI 318-19 Mn — SINGLE SPECIMEN
# ────────────────────────────────────────────────────────────
def aci_moment_capacity(
    b: float,
    d: float,
    n_bars: float,
    db_mm: float,
    fy: float,
    fc: float,
    eta_m: float,
) -> float:
    """
    Compute nominal flexural capacity Mn (kN·m) per ACI 318-19.

    Parameters
    ----------
    b      : section width (mm)
    d      : effective depth (mm)
    n_bars : number of tensile bars
    db_mm  : bar diameter (mm)
    fy     : nominal yield strength (MPa)
    fc     : concrete compressive strength (MPa)
    eta_m  : mass loss (%)

    Returns
    -------
    Mn_kNm : nominal moment capacity (kN·m)
    """
    As   = corroded_area(n_bars, db_mm, eta_m)       # mm²
    fy_c = corroded_fy(fy, eta_m)                    # MPa
    a    = (As * fy_c) / (0.85 * fc * b)            # mm  — stress block depth
    Mn   = As * fy_c * (d - a / 2.0)                # N·mm
    Mn_kNm = Mn / 1e6                               # → kN·m
    return Mn_kNm


# ────────────────────────────────────────────────────────────
# 4. VECTORISED — FULL DATAFRAME
# ────────────────────────────────────────────────────────────
def compute_aci_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply ACI 318-19 equation to every row in the database.
    Adds columns: MACI_pred (kN·m), R_ACI_pred (%), ratio_exp_aci

    Expects columns (from ACI_COLS in config.py):
        Width (mm), Depth (mm), Diameter Tensile Bars db,t (mm),
        # Tensile Bars, fy Longitudinal Bars (MPa), f'c (MPa),
        Mass Loss ηm (%), Mmax,exp (kNm)
    """
    df = df.copy()

    col_b    = "Width (mm)"
    col_d    = "Depth (mm)"
    col_db   = "Diameter Tensile Bars, db,t (mm)"
    col_n    = "# Tensile Bars"
    col_fy   = "fy Longitudinal Bars (Tensile), (MPa) "
    col_fc   = "f'c (MPa)"
    col_eta  = "Mass Loss (Tensile bars), ηm (%)"
    col_mexp = "Mmax,exp (kNm)"

    required = [col_b, col_d, col_db, col_n, col_fy, col_fc, col_eta, col_mexp]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"ACI calculator: missing columns {missing}")

    df["MACI_pred"] = df.apply(
        lambda r: aci_moment_capacity(
            b      = r[col_b],
            d      = r[col_d],
            n_bars = r[col_n],
            db_mm  = r[col_db],
            fy     = r[col_fy],
            fc     = r[col_fc],
            eta_m  = r[col_eta],
        ),
        axis=1,
    )

    # Ratio: experimental / ACI prediction
    df["ratio_exp_aci"] = df[col_mexp] / df["MACI_pred"]

    # ACI-predicted residual capacity R_ACI = (MACI_pred / M_control) × 100
    # Approximation: use ratio relative to median control beam
    df["R_ACI_pred"] = df["ratio_exp_aci"] * 100.0

    return df


# ────────────────────────────────────────────────────────────
# 5. BENCHMARK METRICS
# ────────────────────────────────────────────────────────────
def evaluate_aci_benchmark(df: pd.DataFrame) -> dict:
    """
    Compute ACI 318-19 benchmark metrics vs experimental data.
    Returns dict with R², RMSE, MAE, MAPE, mean ratio, std ratio.
    """
    col_mexp = "Mmax,exp (kNm)"
    y_true   = df[col_mexp].values
    y_pred   = df["MACI_pred"].values

    r2   = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-6))) * 100

    ratio_mean = df["ratio_exp_aci"].mean()
    ratio_std  = df["ratio_exp_aci"].std()
    ratio_min  = df["ratio_exp_aci"].min()
    ratio_max  = df["ratio_exp_aci"].max()

    underest_pct = ((df["ratio_exp_aci"] < 1.0).sum() / len(df)) * 100

    metrics = {
        "R2"               : round(r2,   4),
        "RMSE"             : round(rmse, 4),
        "MAE"              : round(mae,  4),
        "MAPE"             : round(mape, 2),
        "ratio_mean"       : round(ratio_mean, 4),
        "ratio_std"        : round(ratio_std,  4),
        "ratio_min"        : round(ratio_min,  4),
        "ratio_max"        : round(ratio_max,  4),
        "underestimate_pct": round(underest_pct, 1),
        "n_specimens"      : len(df),
    }

    logger.info("══════════════════════════════════════")
    logger.info(" ACI 318-19 Benchmark Results")
    logger.info("══════════════════════════════════════")
    logger.info(f"  Specimens     : {metrics['n_specimens']}")
    logger.info(f"  R²            : {metrics['R2']}")
    logger.info(f"  RMSE          : {metrics['RMSE']} kN·m")
    logger.info(f"  MAE           : {metrics['MAE']} kN·m")
    logger.info(f"  MAPE          : {metrics['MAPE']} %")
    logger.info(f"  Ratio mean    : {metrics['ratio_mean']}  (target = 1.0)")
    logger.info(f"  Ratio std     : {metrics['ratio_std']}")
    logger.info(f"  Ratio range   : {metrics['ratio_min']} – {metrics['ratio_max']}")
    logger.info(f"  Underestimates: {metrics['underestimate_pct']} % of specimens")
    logger.info("══════════════════════════════════════")

    return metrics


# ────────────────────────────────────────────────────────────
# 6. SAVE BENCHMARK RESULTS
# ────────────────────────────────────────────────────────────
def save_benchmark_results(df: pd.DataFrame, metrics: dict) -> None:
    """Save ACI prediction results and metrics to results/models/."""
    import json
    out_csv  = RESULTS_DIR / "models" / "aci_benchmark_predictions.csv"
    out_json = RESULTS_DIR / "models" / "aci_benchmark_metrics.json"

    df[["MACI_pred", "ratio_exp_aci"]].to_csv(out_csv, index=False)
    with open(out_json, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"ACI results saved → {out_csv}")
    logger.info(f"ACI metrics saved → {out_json}")


# ────────────────────────────────────────────────────────────
# CLI entry point
# ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from data_preprocessing import load_raw_data, clean_data

    df_raw   = load_raw_data()
    df_clean = clean_data(df_raw)
    df_aci   = compute_aci_predictions(df_clean)
    metrics  = evaluate_aci_benchmark(df_aci)
    save_benchmark_results(df_aci, metrics)

    print("\n✅ ACI 318-19 benchmark complete.")
    print(f"   R²   = {metrics['R2']}")
    print(f"   RMSE = {metrics['RMSE']} kN·m")
    print(f"   Ratio mean = {metrics['ratio_mean']}  (ACI underestimates by {round((1 - metrics['ratio_mean']) * 100, 1)}%)")
