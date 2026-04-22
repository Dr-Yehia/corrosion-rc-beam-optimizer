#!/usr/bin/env python3
"""
Build a physically-filtered symbolic equation from the trained Stacking model.

Pipeline:
1) Load and preprocess full dataset (clean + engineered features).
2) Load trained Stacking model from resultss/models/model_stacking.pkl.
3) Build dimensionless target ratio: R = M_stack / M_ACI.
4) Run PySR on dimensionless inputs only.
5) Score candidate equations with many objectives:
   - 1-R2, MAPE, RMSE_norm
   - endpoint(eta=0) error to 1.0
   - endpoint(eta=100) error to 0.0
   - monotonicity violation wrt eta
   - complexity
6) Perform MOEA/D-style decomposition selection (Tchebycheff on reference vectors).
7) Save best equation, metrics, ranked candidates, and diagnostic plots.

Designed for Kaggle / Colab execution.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    import sympy as sp
except Exception:  # pragma: no cover
    sp = None

# --------------------------------------------------------------------------------------
# Paths and imports
# --------------------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config import FEATURE_COLS, CAT_COLS, TARGET_COL  # noqa: E402
from data_preprocessing import load_raw_data, clean_data, engineer_features  # noqa: E402
from aci_calculator import compute_aci_predictions  # noqa: E402

RESULTSS_DIR = ROOT / "resultss"
MODELS_DIR = RESULTSS_DIR / "models"
FIG_DIR = RESULTSS_DIR / "figures"
EQ_DIR = RESULTSS_DIR / "equations"

for d in (MODELS_DIR, FIG_DIR, EQ_DIR):
    d.mkdir(parents=True, exist_ok=True)


@dataclass
class Candidate:
    equation: str
    complexity: float
    metrics: Dict[str, float]
    objectives: Dict[str, float]
    score: float


def setup_logger() -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
        level="INFO",
        colorize=True,
    )
    logger.add(str(RESULTSS_DIR / "logs" / "run_log_pysr_moead.txt"), level="DEBUG")


def encode_with_saved_mapping(df: pd.DataFrame, enc_path: Path) -> pd.DataFrame:
    """Encode categoricals using saved mapping from training run."""
    out = df.copy()
    if not enc_path.exists():
        logger.warning(f"Encoder mapping not found at {enc_path}. Falling back to on-the-fly category codes.")
        for c in CAT_COLS:
            if c in out.columns:
                out[c] = out[c].astype("category").cat.codes
        return out

    mapping = json.loads(enc_path.read_text())
    for c in CAT_COLS:
        if c not in out.columns:
            continue
        classes = mapping.get(c, [])
        idx = {v: i for i, v in enumerate(classes)}
        # Unknown categories -> -1
        out[c] = out[c].astype(str).map(idx).fillna(-1).astype(int)
    return out


def prepare_full_dataframe() -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Prepare full dataset and return (df, X_scaled, y_true, m_aci)."""
    logger.info("Loading + preprocessing full dataset ...")
    df_raw = load_raw_data()
    df_clean = clean_data(df_raw)
    df_feat = engineer_features(df_clean)
    df_enc = encode_with_saved_mapping(df_feat, MODELS_DIR / "cat_encoders.json")

    engineered = [
        "eta_log",
        "corr_severity_idx",
        "d_b_ratio",
        "eta_d_interaction",
        "reinf_index",
        "Mnom_proxy",
        "M_corr_reduced",
        "ductility_corr",
    ]
    feature_cols = [c for c in FEATURE_COLS if c in df_enc.columns] + [
        c for c in CAT_COLS if c in df_enc.columns
    ] + [c for c in engineered if c in df_enc.columns]

    X = df_enc[feature_cols].copy()
    y_true = df_enc[TARGET_COL].to_numpy(dtype=float)

    scaler_path = MODELS_DIR / "scaler_X.pkl"
    if not scaler_path.exists():
        raise FileNotFoundError(f"Missing scaler: {scaler_path}")
    scaler_x = joblib.load(scaler_path)
    X_scaled = scaler_x.transform(X)

    df_aci = compute_aci_predictions(df_enc)
    m_aci = df_aci["MACI_pred"].to_numpy(dtype=float)
    m_aci = np.maximum(m_aci, 1e-9)

    logger.info(f"Prepared dataset: n={len(df_enc)}, features={X.shape[1]}")
    return df_enc, X_scaled, y_true, m_aci


def get_stacking_predictions(X_scaled: np.ndarray) -> np.ndarray:
    model_path = MODELS_DIR / "model_stacking.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model: {model_path}")
    model = joblib.load(model_path)
    pred = model.predict(X_scaled)
    pred = np.asarray(pred, dtype=float)
    pred = np.maximum(pred, 0.0)
    return pred


def build_symbolic_inputs(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """Dimensionless symbolic inputs only."""
    eps = 1e-9

    eta = df["Mass Loss (Tensile bars), ηm (%)"].to_numpy(dtype=float)
    rho_t = df["Tension Reinforcement Ratio, pten (%)"].to_numpy(dtype=float)
    d = df["Depth (mm)"].to_numpy(dtype=float)
    b = df["Width (mm)"].to_numpy(dtype=float)

    # Already engineered in pipeline
    csi = df["corr_severity_idx"].to_numpy(dtype=float)
    ri = df["reinf_index"].to_numpy(dtype=float)
    d_b = d / np.maximum(b, eps)

    # Normalize to reduce scale pathologies
    X_sym = pd.DataFrame(
        {
            "eta": eta / 100.0,
            "rho": rho_t / 100.0,
            "d_b": d_b,
            "csi": csi / np.maximum(np.median(np.abs(csi)), eps),
            "ri": ri / np.maximum(np.median(np.abs(ri)), eps),
        }
    )

    data_dict = {c: X_sym[c].to_numpy(dtype=float) for c in X_sym.columns}
    return X_sym, data_dict


def run_pysr(
    X_sym: pd.DataFrame,
    y_ratio: np.ndarray,
    niterations: int,
    populations: int,
    maxsize: int,
    random_state: int,
):
    try:
        from pysr import PySRRegressor
    except Exception as exc:  # pragma: no cover
        raise ImportError("PySR is required. Install with: pip install pysr") from exc

    logger.info(
        f"Running PySR: niterations={niterations}, populations={populations}, maxsize={maxsize}"
    )

    model = PySRRegressor(
        niterations=niterations,
        populations=populations,
        maxsize=maxsize,
        binary_operators=["+", "-", "*", "/", "^"],
        unary_operators=["sqrt", "log"],
        nested_constraints={
            "sqrt": {"sqrt": 0, "log": 1},
            "log": {"log": 0, "sqrt": 1},
        },
        constraints={"^": (-1, 1), "sqrt": 8, "log": 8},
        model_selection="accuracy",
        elementwise_loss="loss(x, y) = abs(x - y)",
        random_state=random_state,
        deterministic=False,
        parallelism="multithreading",
        turbo=True,
        verbosity=1,
    )

    model.fit(X_sym.to_numpy(dtype=float), y_ratio, variable_names=list(X_sym.columns))

    eq_df = model.equations_.copy()
    return model, eq_df


def safe_sympify(expr: str):
    if sp is None:
        raise ImportError("sympy is required. Install with: pip install sympy")
    expr = expr.replace("^", "**")
    return sp.sympify(
        expr,
        locals={
            "sqrt": sp.sqrt,
            "log": sp.log,
            "exp": sp.exp,
            "abs": sp.Abs,
        },
    )


def evaluate_endpoint_ratio(expr_sp, med: Dict[str, float], eta_value: float) -> float:
    subs = dict(med)
    subs["eta"] = eta_value
    val = float(expr_sp.evalf(subs=subs))
    if not np.isfinite(val):
        return np.nan
    return float(val)


def monotonic_violation(expr_sp, med: Dict[str, float], n_grid: int = 60) -> float:
    eta_vals = np.linspace(0.0, 1.0, n_grid)
    vals = []
    for e in eta_vals:
        subs = dict(med)
        subs["eta"] = float(e)
        try:
            v = float(expr_sp.evalf(subs=subs))
        except Exception:
            return 1.0
        vals.append(v)
    vals = np.asarray(vals, dtype=float)
    if not np.all(np.isfinite(vals)):
        return 1.0
    diffs = np.diff(vals)
    # degradation ratio should generally decrease with eta
    return float(np.mean(diffs > 0.0))


def estimate_complexity(expr: str) -> float:
    # Simple token-based complexity for stable ranking
    ops = len(re.findall(r"[\+\-\*/\^]", expr))
    funcs = len(re.findall(r"sqrt|log|exp|abs", expr))
    terms = len(re.findall(r"eta|rho|d_b|csi|ri", expr))
    return float(ops + 1.5 * funcs + 0.5 * terms)


def evaluate_candidates(
    eq_df: pd.DataFrame,
    data_dict: Dict[str, np.ndarray],
    y_true: np.ndarray,
    m_aci: np.ndarray,
) -> List[Candidate]:
    cands: List[Candidate] = []

    med = {k: float(np.median(v)) for k, v in data_dict.items()}

    for _, row in eq_df.iterrows():
        expr = str(row.get("sympy_format", "")).strip()
        if not expr:
            continue

        try:
            expr_sp = safe_sympify(expr)
            fn = sp.lambdify(tuple(data_dict.keys()), expr_sp, modules=["numpy"])
            args = [data_dict[k] for k in data_dict.keys()]
            ratio_pred = np.asarray(fn(*args), dtype=float)
        except Exception:
            continue

        if ratio_pred.ndim != 1 or len(ratio_pred) != len(y_true):
            continue

        ratio_pred = np.nan_to_num(ratio_pred, nan=0.0, posinf=10.0, neginf=0.0)
        ratio_pred = np.clip(ratio_pred, 0.0, 5.0)

        y_pred = ratio_pred * m_aci
        y_pred = np.maximum(y_pred, 0.0)

        r2 = float(r2_score(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae = float(mean_absolute_error(y_true, y_pred))
        mape = float(np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-6))) * 100.0)

        # endpoint conditions for ratio target
        r0 = evaluate_endpoint_ratio(expr_sp, med, 0.0)
        r100 = evaluate_endpoint_ratio(expr_sp, med, 1.0)
        end0 = abs(r0 - 1.0) if np.isfinite(r0) else 1.0
        end100 = abs(r100 - 0.0) if np.isfinite(r100) else 1.0

        mono = monotonic_violation(expr_sp, med)
        comp = estimate_complexity(expr)

        metrics = {
            "R2": r2,
            "RMSE": rmse,
            "MAE": mae,
            "MAPE": mape,
            "ratio_eta0": float(r0) if np.isfinite(r0) else float("nan"),
            "ratio_eta100": float(r100) if np.isfinite(r100) else float("nan"),
        }
        objectives = {
            "obj_r2": 1.0 - r2,
            "obj_mape": mape / 100.0,
            "obj_rmse": rmse / max(float(np.mean(y_true)), 1e-6),
            "obj_end0": end0,
            "obj_end100": end100,
            "obj_mono": mono,
            "obj_comp": comp / 50.0,
        }

        cands.append(
            Candidate(
                equation=expr,
                complexity=comp,
                metrics=metrics,
                objectives=objectives,
                score=float("inf"),
            )
        )

    logger.info(f"Evaluated {len(cands)} candidate equations")
    return cands


def normalize_objectives(mat: np.ndarray) -> np.ndarray:
    lo = mat.min(axis=0)
    hi = mat.max(axis=0)
    rng = np.where((hi - lo) < 1e-12, 1.0, hi - lo)
    return (mat - lo) / rng


def moead_style_select(cands: List[Candidate], n_vectors: int = 64, seed: int = 42) -> List[int]:
    if not cands:
        return []

    obj_names = list(cands[0].objectives.keys())
    mat = np.array([[c.objectives[k] for k in obj_names] for c in cands], dtype=float)
    nmat = normalize_objectives(mat)

    rng = np.random.default_rng(seed)
    refs = rng.dirichlet(alpha=np.ones(nmat.shape[1]), size=n_vectors)

    chosen = set()
    for w in refs:
        # Tchebycheff scalarization: min max_i w_i * f_i
        scal = np.max(w * nmat, axis=1)
        idx = int(np.argmin(scal))
        chosen.add(idx)

    return sorted(chosen)


def choose_final(cands: List[Candidate], selected_idx: List[int]) -> int:
    idxs = selected_idx if selected_idx else list(range(len(cands)))

    # Final weighted score: prioritize accuracy + endpoint physics + monotonicity
    w = {
        "obj_r2": 0.25,
        "obj_mape": 0.20,
        "obj_rmse": 0.10,
        "obj_end0": 0.15,
        "obj_end100": 0.20,
        "obj_mono": 0.08,
        "obj_comp": 0.02,
    }

    best_idx = idxs[0]
    best_score = float("inf")

    for i in idxs:
        c = cands[i]
        sc = sum(w[k] * c.objectives[k] for k in w)
        c.score = float(sc)
        if sc < best_score:
            best_score = sc
            best_idx = i

    return best_idx


def save_outputs(cands: List[Candidate], best_idx: int, y_true: np.ndarray, m_aci: np.ndarray, data_dict: Dict[str, np.ndarray]):
    best = cands[best_idx]

    # Save ranked candidates
    ranked = sorted(
        [
            {
                "equation": c.equation,
                "complexity": c.complexity,
                "metrics": c.metrics,
                "objectives": c.objectives,
                "score": c.score,
            }
            for c in cands
        ],
        key=lambda x: x["score"],
    )

    out_rank = MODELS_DIR / "pysr_candidates_ranked.json"
    out_rank.write_text(json.dumps(ranked, indent=2))

    # Recompute best predictions for plotting
    expr_sp = safe_sympify(best.equation)
    fn = sp.lambdify(tuple(data_dict.keys()), expr_sp, modules=["numpy"])
    ratio = np.asarray(fn(*[data_dict[k] for k in data_dict.keys()]), dtype=float)
    ratio = np.nan_to_num(ratio, nan=0.0, posinf=10.0, neginf=0.0)
    ratio = np.clip(ratio, 0.0, 5.0)
    y_pred = ratio * m_aci

    r2 = float(r2_score(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-6))) * 100.0)

    # Save metrics
    out_metrics = {
        "approach": "Stacking-to-PySR ratio distillation with MOEA/D-style candidate selection",
        "equation": best.equation,
        "R2": round(r2, 4),
        "RMSE": round(rmse, 4),
        "MAE": round(mae, 4),
        "MAPE": round(mape, 2),
        "complexity": round(best.complexity, 4),
        "selection_score": round(best.score, 6),
        "n_candidates": len(cands),
    }
    (MODELS_DIR / "pysr_stacking_metrics.json").write_text(json.dumps(out_metrics, indent=2))

    # Save equation text + latex
    (EQ_DIR / "best_equation_stacking.txt").write_text(
        "# Best Equation from Stacking->PySR\n"
        f"# R2={r2:.4f} RMSE={rmse:.4f} MAE={mae:.4f} MAPE={mape:.2f}%\n\n"
        f"ratio = {best.equation}\n"
        "Mmax = ratio * M_ACI\n"
    )
    latex_eq = sp.latex(expr_sp)
    (EQ_DIR / "best_equation_stacking.latex").write_text(
        "% Best Equation from Stacking->PySR\n"
        f"R = {latex_eq}\\n"
        "M_{max} = R \cdot M_{ACI}\\n"
    )

    # Figure 1: predicted vs true
    fig1 = FIG_DIR / "pysr_stacking_scatter.png"
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, s=16, alpha=0.6)
    lo, hi = float(min(y_true.min(), y_pred.min())), float(max(y_true.max(), y_pred.max()))
    plt.plot([lo, hi], [lo, hi], "r--", lw=1.5)
    plt.xlabel("Experimental Mmax")
    plt.ylabel("Equation Mmax")
    plt.title(f"Stacking->PySR | R2={r2:.4f} | MAPE={mape:.2f}%")
    plt.tight_layout()
    plt.savefig(fig1, dpi=250)
    plt.close()

    # Figure 2: endpoint behavior (median fixed params)
    med = {k: float(np.median(v)) for k, v in data_dict.items()}
    eta_grid = np.linspace(0.0, 1.0, 120)
    ratio_grid = []
    for e in eta_grid:
        subs = dict(med)
        subs["eta"] = float(e)
        ratio_grid.append(float(expr_sp.evalf(subs=subs)))
    ratio_grid = np.asarray(ratio_grid, dtype=float)

    fig2 = FIG_DIR / "pysr_stacking_endpoints.png"
    plt.figure(figsize=(7, 4))
    plt.plot(eta_grid * 100.0, ratio_grid, lw=2)
    plt.axhline(1.0, color="g", ls="--", lw=1, label="target @0% ~ 1")
    plt.axhline(0.0, color="r", ls="--", lw=1, label="target @100% ~ 0")
    plt.xlabel("eta_m (%)")
    plt.ylabel("Predicted Ratio (M/M_ACI)")
    plt.title("Endpoint + trend diagnostic")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig2, dpi=250)
    plt.close()

    logger.success(f"Saved best equation -> {EQ_DIR / 'best_equation_stacking.txt'}")
    logger.success(f"Saved metrics -> {MODELS_DIR / 'pysr_stacking_metrics.json'}")
    logger.success(f"Saved ranked candidates -> {out_rank}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stacking to PySR symbolic distillation with MOEA/D-style selection")
    p.add_argument("--niterations", type=int, default=220)
    p.add_argument("--populations", type=int, default=40)
    p.add_argument("--maxsize", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ref-vectors", type=int, default=64)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logger()

    if sp is None:
        raise ImportError("Missing dependency: sympy. Install with: pip install sympy")

    df, X_scaled, y_true, m_aci = prepare_full_dataframe()
    m_stack = get_stacking_predictions(X_scaled)

    # Distillation target (dimensionless)
    y_ratio = np.maximum(m_stack / np.maximum(m_aci, 1e-9), 0.0)

    X_sym, data_dict = build_symbolic_inputs(df)

    _, eq_df = run_pysr(
        X_sym=X_sym,
        y_ratio=y_ratio,
        niterations=args.niterations,
        populations=args.populations,
        maxsize=args.maxsize,
        random_state=args.seed,
    )

    cands = evaluate_candidates(eq_df, data_dict, y_true, m_aci)
    if not cands:
        raise RuntimeError("No valid candidate equations generated. Increase niterations/populations.")

    selected = moead_style_select(cands, n_vectors=args.ref_vectors, seed=args.seed)
    best_idx = choose_final(cands, selected)
    save_outputs(cands, best_idx, y_true, m_aci, data_dict)

    logger.success("Done. Best equation pipeline finished successfully.")


if __name__ == "__main__":
    main()
