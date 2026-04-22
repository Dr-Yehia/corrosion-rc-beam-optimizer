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
    log_dir = RESULTSS_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.add(str(log_dir / "run_log_pysr_moead.txt"), level="DEBUG")


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
    feature_cols = (
        [c for c in FEATURE_COLS if c in df_enc.columns]
        + [c for c in CAT_COLS if c in df_enc.columns]
        + [c for c in engineered if c in df_enc.columns]
    )

    X = df_enc[feature_cols].copy()
    y_true = df_enc[TARGET_COL].to_numpy(dtype=float)

    scaler_path = MODELS_DIR / "scaler_X.pkl"
    if not scaler_path.exists():
        raise FileNotFoundError(f"Missing scaler: {scaler_path}")
    scaler_x = joblib.load(scaler_path)

    # FIX #1: Validate feature count matches scaler expectation before transform
    expected_n = scaler_x.n_features_in_
    if X.shape[1] != expected_n:
        raise ValueError(
            f"Feature count mismatch: scaler expects {expected_n} features, "
            f"but got {X.shape[1]}. Check FEATURE_COLS and engineered feature list."
        )
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
    """Dimensionless symbolic inputs — uses robust column lookup."""
    eps = 1e-9

    # FIX #2: Robust column lookup — handle encoding issues in column names
    def find_col(df: pd.DataFrame, candidates: List[str]) -> np.ndarray:
        for name in candidates:
            if name in df.columns:
                return df[name].to_numpy(dtype=float)
        # Partial match fallback
        for name in candidates:
            matches = [c for c in df.columns if name.lower() in c.lower()]
            if matches:
                logger.warning(f"Column '{name}' not found exactly; using '{matches[0]}'")
                return df[matches[0]].to_numpy(dtype=float)
        raise KeyError(f"None of {candidates} found in DataFrame columns: {list(df.columns)}")

    eta = find_col(df, ["Mass Loss (Tensile bars), ηm (%)", "Mass Loss", "eta_m", "ηm (%)"])
    rho_t = find_col(df, ["Tension Reinforcement Ratio, pten (%)", "pten (%)", "rho_t"])
    d = find_col(df, ["Depth (mm)", "d (mm)", "depth"])
    b = find_col(df, ["Width (mm)", "b (mm)", "width"])

    csi = df["corr_severity_idx"].to_numpy(dtype=float)
    ri = df["reinf_index"].to_numpy(dtype=float)
    d_b = d / np.maximum(b, eps)

    csi_med = np.median(np.abs(csi))
    ri_med = np.median(np.abs(ri))

    X_sym = pd.DataFrame(
        {
            "eta": eta / 100.0,
            "rho": rho_t / 100.0,
            "d_b": d_b,
            "csi": csi / max(csi_med, eps),
            "ri": ri / max(ri_med, eps),
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
        unary_operators=["sqrt", "log", "exp"],  # FIX #3: Added exp — physical decay with eta
        nested_constraints={
            "sqrt": {"sqrt": 0, "log": 1, "exp": 1},
            "log": {"log": 0, "sqrt": 1, "exp": 0},
            "exp": {"exp": 0, "log": 1, "sqrt": 1},
        },
        # FIX #4: Use positive complexity limits (not -1) so ^ operator is actually usable
        constraints={"^": (3, 3), "sqrt": 8, "log": 8, "exp": 6},
        model_selection="accuracy",
        elementwise_loss="loss(x, y) = abs(x - y)",
        random_state=random_state,
        # FIX #5: serial + deterministic=True for full reproducibility
        deterministic=True,
        parallelism="serial",
        # FIX #6: turbo removed — not guaranteed on Kaggle/Colab Julia environments
        verbosity=1,
    )

    model.fit(X_sym.to_numpy(dtype=float), y_ratio, variable_names=list(X_sym.columns))

    eq_df = model.equations_.copy()
    return model, eq_df


def safe_sympify(expr: str):
    if sp is None:
        raise ImportError("sympy is required. Install with: pip install sympy")
    # PySR sympy_format already uses ** not ^; only replace if ^ present
    expr_clean = expr.replace("^", "**")
    return sp.sympify(
        expr_clean,
        locals={
            "sqrt": sp.sqrt,
            "log": sp.log,
            "exp": sp.exp,
            "abs": sp.Abs,
        },
    )


def _get_equation_string(row: pd.Series) -> str:
    """
    FIX #7: Robust equation extraction — PySR column names differ across versions.
    Try sympy_format first (older PySR), then equation (newer PySR).
    """
    for col in ("sympy_format", "equation", "lambda_format"):
        val = row.get(col, "")
        if val and str(val).strip() and str(val).strip() not in ("nan", "None", ""):
            s = str(val).strip()
            # lambda_format looks like "PySRFunction(X=>...)" — skip it
            if "PySRFunction" in s:
                continue
            return s
    return ""


def evaluate_endpoint_ratio(expr_sp, med: Dict[str, float], eta_value: float) -> float:
    subs = dict(med)
    subs["eta"] = eta_value
    try:
        val = float(expr_sp.evalf(subs=subs))
    except Exception:
        return float("nan")
    if not np.isfinite(val):
        return float("nan")
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
    # Ratio should decrease (or stay flat) as corrosion increases
    return float(np.mean(diffs > 0.0))


def estimate_complexity(row: pd.Series, expr: str) -> float:
    # FIX #8: Use PySR's built-in complexity when available; fallback to token count
    if "complexity" in row and pd.notna(row["complexity"]):
        return float(row["complexity"])
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
    mean_y = float(np.mean(y_true))

    for _, row in eq_df.iterrows():
        expr = _get_equation_string(row)  # FIX #7 applied here
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
        # FIX: Clip MAPE per-sample ratio to avoid explosion near zero
        mape = float(
            np.mean(
                np.clip(
                    np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1.0)),
                    0.0,
                    10.0,
                )
            )
            * 100.0
        )

        r0 = evaluate_endpoint_ratio(expr_sp, med, 0.0)
        r100 = evaluate_endpoint_ratio(expr_sp, med, 1.0)
        end0 = abs(r0 - 1.0) if np.isfinite(r0) else 1.0
        end100 = abs(r100 - 0.0) if np.isfinite(r100) else 1.0

        mono = monotonic_violation(expr_sp, med)
        comp = estimate_complexity(row, expr)  # FIX #8 applied here

        metrics = {
            "R2": round(r2, 4),
            "RMSE": round(rmse, 4),
            "MAE": round(mae, 4),
            "MAPE": round(mape, 2),
            "ratio_eta0": round(float(r0), 4) if np.isfinite(r0) else None,
            "ratio_eta100": round(float(r100), 4) if np.isfinite(r100) else None,
        }
        objectives = {
            "obj_r2": max(0.0, 1.0 - r2),
            "obj_mape": mape / 100.0,
            "obj_rmse": rmse / max(mean_y, 1e-6),
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
    n_obj = len(obj_names)
    mat = np.array([[c.objectives[k] for k in obj_names] for c in cands], dtype=float)
    nmat = normalize_objectives(mat)

    # Use uniform simplex grid instead of random Dirichlet for deterministic coverage
    rng = np.random.default_rng(seed)
    refs = rng.dirichlet(alpha=np.ones(n_obj), size=n_vectors)

    chosen = set()
    for w in refs:
        # Tchebycheff scalarization: min max_i(w_i * f_i)
        scal = np.max(w * nmat, axis=1)
        idx = int(np.argmin(scal))
        chosen.add(idx)

    logger.info(f"MOEA/D selected {len(chosen)} unique candidates from {len(cands)} total")
    return sorted(chosen)


def choose_final(
    cands: List[Candidate],
    selected_idx: List[int],
    w_accuracy: float,
    w_physics: float,
    w_complexity: float,
) -> int:
    """
    FIX #9: Weights are now passed as arguments (not hardcoded) so CLI controls them.
    w_accuracy  -> accuracy objectives (R2, MAPE, RMSE)
    w_physics   -> physics objectives (endpoint0, endpoint100, monotonicity)
    w_complexity-> complexity objective
    """
    total = w_accuracy + w_physics + w_complexity
    wa = w_accuracy / total
    wp = w_physics / total
    wc = w_complexity / total

    w = {
        "obj_r2":    wa * 0.45,
        "obj_mape":  wa * 0.35,
        "obj_rmse":  wa * 0.20,
        "obj_end0":  wp * 0.40,
        "obj_end100": wp * 0.40,
        "obj_mono":  wp * 0.20,
        "obj_comp":  wc * 1.00,
    }

    idxs = selected_idx if selected_idx else list(range(len(cands)))
    best_idx = idxs[0]
    best_score = float("inf")

    for i in idxs:
        c = cands[i]
        sc = sum(w[k] * c.objectives[k] for k in w)
        c.score = float(sc)
        if sc < best_score:
            best_score = sc
            best_idx = i

    logger.info(f"Final winner: index={best_idx}, score={best_score:.6f}")
    return best_idx


def save_outputs(
    cands: List[Candidate],
    best_idx: int,
    y_true: np.ndarray,
    m_aci: np.ndarray,
    data_dict: Dict[str, np.ndarray],
) -> None:
    best = cands[best_idx]

    # Ranked candidates
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

    # Recompute best predictions
    expr_sp = safe_sympify(best.equation)
    fn = sp.lambdify(tuple(data_dict.keys()), expr_sp, modules=["numpy"])
    ratio = np.asarray(fn(*[data_dict[k] for k in data_dict.keys()]), dtype=float)
    ratio = np.nan_to_num(ratio, nan=0.0, posinf=10.0, neginf=0.0)
    # FIX #10: clip ratio to physically sane range
    ratio = np.clip(ratio, 0.0, 5.0)
    y_pred = ratio * m_aci

    r2 = float(r2_score(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    mape = float(
        np.mean(
            np.clip(
                np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1.0)),
                0.0,
                10.0,
            )
        )
        * 100.0
    )

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

    # Equation text
    (EQ_DIR / "best_equation_stacking.txt").write_text(
        "# Best Equation from Stacking->PySR\n"
        f"# R2={r2:.4f} RMSE={rmse:.4f} MAE={mae:.4f} MAPE={mape:.2f}%\n\n"
        f"ratio = {best.equation}\n"
        "Mmax = ratio * M_ACI\n"
    )

    # FIX #11: LaTeX file — use real newlines, not escaped \\n
    latex_eq = sp.latex(expr_sp)
    (EQ_DIR / "best_equation_stacking.latex").write_text(
        "% Best Equation from Stacking->PySR\n"
        f"R = {latex_eq}\n"
        r"M_{\max} = R \cdot M_{\mathrm{ACI}}" + "\n"
    )

    # Figure 1: predicted vs true
    fig1 = FIG_DIR / "pysr_stacking_scatter.png"
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, s=16, alpha=0.6)
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    plt.plot([lo, hi], [lo, hi], "r--", lw=1.5)
    plt.xlabel("Experimental $M_{max}$ (kN·m)")
    plt.ylabel("Equation $M_{max}$ (kN·m)")
    plt.title(f"Stacking→PySR | R²={r2:.4f} | MAPE={mape:.2f}%")
    plt.tight_layout()
    plt.savefig(fig1, dpi=250)
    plt.close()

    # Figure 2: endpoint + monotonicity trend
    med = {k: float(np.median(v)) for k, v in data_dict.items()}
    eta_grid = np.linspace(0.0, 1.0, 120)
    ratio_grid = []
    for e in eta_grid:
        subs = dict(med)
        subs["eta"] = float(e)
        try:
            ratio_grid.append(float(expr_sp.evalf(subs=subs)))
        except Exception:
            ratio_grid.append(float("nan"))
    ratio_grid = np.asarray(ratio_grid, dtype=float)

    fig2 = FIG_DIR / "pysr_stacking_endpoints.png"
    plt.figure(figsize=(7, 4))
    plt.plot(eta_grid * 100.0, ratio_grid, lw=2, label="Equation ratio R(η)")
    plt.axhline(1.0, color="g", ls="--", lw=1, label="Target @0% = 1.0")
    plt.axhline(0.0, color="r", ls="--", lw=1, label="Target @100% = 0.0")
    plt.xlabel("Mass Loss η (%)")
    plt.ylabel("Ratio R = M / M_ACI")
    plt.title("Endpoint + Monotonicity Diagnostic")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig2, dpi=250)
    plt.close()

    logger.success(f"Equation  → {EQ_DIR / 'best_equation_stacking.txt'}")
    logger.success(f"LaTeX     → {EQ_DIR / 'best_equation_stacking.latex'}")
    logger.success(f"Metrics   → {MODELS_DIR / 'pysr_stacking_metrics.json'}")
    logger.success(f"Ranked    → {out_rank}")
    logger.success(f"Fig1      → {fig1}")
    logger.success(f"Fig2      → {fig2}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stacking to PySR symbolic distillation with MOEA/D-style selection"
    )
    p.add_argument("--niterations", type=int, default=220)
    p.add_argument("--populations", type=int, default=40)
    p.add_argument("--maxsize", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ref-vectors", type=int, default=64)
    # FIX #9: expose objective weights via CLI
    p.add_argument(
        "--w-accuracy", type=float, default=0.55,
        help="Weight for accuracy objectives (R2, MAPE, RMSE). Default=0.55"
    )
    p.add_argument(
        "--w-physics", type=float, default=0.35,
        help="Weight for physics objectives (endpoints, monotonicity). Default=0.35"
    )
    p.add_argument(
        "--w-complexity", type=float, default=0.10,
        help="Weight for equation complexity penalty. Default=0.10"
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logger()

    if sp is None:
        raise ImportError("Missing dependency: sympy. Install with: pip install sympy")

    df, X_scaled, y_true, m_aci = prepare_full_dataframe()
    m_stack = get_stacking_predictions(X_scaled)

    # Distillation target (dimensionless, clipped to physical range)
    y_ratio = np.clip(m_stack / np.maximum(m_aci, 1e-9), 0.0, 5.0)  # FIX #10

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
        raise RuntimeError(
            "No valid candidate equations generated. "
            "Try increasing --niterations or --populations."
        )

    selected = moead_style_select(cands, n_vectors=args.ref_vectors, seed=args.seed)
    best_idx = choose_final(
        cands,
        selected,
        w_accuracy=args.w_accuracy,
        w_physics=args.w_physics,
        w_complexity=args.w_complexity,
    )
    save_outputs(cands, best_idx, y_true, m_aci, data_dict)

    logger.success("Done. Best equation pipeline finished successfully.")


if __name__ == "__main__":
    main()
