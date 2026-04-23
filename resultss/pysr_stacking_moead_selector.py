#!/usr/bin/env python3
"""
Discover a physics-constrained symbolic correction factor for ACI predictions.

This script trains PySR on the dimensionless experimental ratio:

    R = Mexp / MACI

and then selects the final equation with a strict post-search pipeline:
1) dimensionless inputs only
2) anchor samples at eta=0 and eta=100%
3) hard physics feasibility gates
4) elite intersection across R2, RMSE, MAE, and MAPE
5) Pareto filtering
6) weighted Tchebycheff minimax selection

Final form:

    Mpred = MACI * R_hat
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

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

from config import TARGET_COL  # noqa: E402
from data_preprocessing import load_raw_data, clean_data, engineer_features  # noqa: E402
from aci_calculator import compute_aci_predictions  # noqa: E402

RESULTSS_DIR = ROOT / "resultss"
MODELS_DIR = RESULTSS_DIR / "models"
FIG_DIR = RESULTSS_DIR / "figures"
EQ_DIR = RESULTSS_DIR / "equations"
LOG_DIR = RESULTSS_DIR / "logs"

for directory in (MODELS_DIR, FIG_DIR, EQ_DIR, LOG_DIR):
    directory.mkdir(parents=True, exist_ok=True)

EPS = 1e-9
TARGET_MAX_RATIO = 6.0
ENDPOINT_QUANTILES = (0.25, 0.50, 0.75)
STRICT_THRESHOLDS = {
    "end0": 0.10,
    "end100": 0.15,
    "mono": 0.05,
    "separation": 0.25,
}


@dataclass
class Candidate:
    equation: str
    complexity: float
    metrics: Dict[str, float]
    objectives: Dict[str, float]
    score: float = float("inf")
    feasible: bool = False
    failures: List[str] = field(default_factory=list)


def setup_logger() -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
        level="INFO",
        colorize=True,
    )
    logger.add(str(LOG_DIR / "run_log_pysr_moead.txt"), level="DEBUG")


def find_column(df: pd.DataFrame, *candidates: str) -> str:
    """Find a column robustly, tolerating Unicode/name variations."""
    for name in candidates:
        if name in df.columns:
            return name

    lowered = {col.lower(): col for col in df.columns}
    for name in candidates:
        key = name.lower()
        if key in lowered:
            return lowered[key]

    for name in candidates:
        key = name.lower()
        for col in df.columns:
            if key in col.lower():
                logger.warning(f"Column '{name}' not found exactly; using '{col}'")
                return col

    raise KeyError(f"None of {candidates} found in DataFrame columns.")


def prepare_full_dataframe() -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Prepare full dataset and return (df, y_true, m_aci)."""
    logger.info("Loading + preprocessing full dataset ...")
    df_raw = load_raw_data()
    df_clean = clean_data(df_raw)
    df_feat = engineer_features(df_clean)

    y_true = df_feat[TARGET_COL].to_numpy(dtype=float)
    df_aci = compute_aci_predictions(df_feat)
    m_aci = np.maximum(df_aci["MACI_pred"].to_numpy(dtype=float), EPS)

    logger.info(f"Prepared dataset: n={len(df_feat)}")
    return df_feat, y_true, m_aci


def build_ratio_target(y_true: np.ndarray, m_aci: np.ndarray) -> np.ndarray:
    """Dimensionless target based on experimental capacity, not stacking predictions."""
    ratio = y_true / np.maximum(m_aci, EPS)
    ratio = np.nan_to_num(ratio, nan=0.0, posinf=TARGET_MAX_RATIO, neginf=0.0)
    ratio = np.clip(ratio, 0.0, TARGET_MAX_RATIO)
    logger.info(
        "Experimental ratio target prepared: "
        f"min={ratio.min():.3f}, max={ratio.max():.3f}, mean={ratio.mean():.3f}"
    )
    return ratio


def build_symbolic_inputs(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], pd.DataFrame]:
    """Build explicit dimensionless groups for symbolic regression."""
    eta_col = find_column(
        df,
        "Mass Loss (Tensile bars), ¦Çm (%)",
        "Mass Loss (Tensile bars), Î·m (%)",
        "Mass Loss",
    )
    rho_col = find_column(
        df,
        "Tension Reinforcement Ratio, pten (%)",
        "pten (%)",
        "rho_t",
    )
    d_col = find_column(df, "Depth (mm)", "depth")
    b_col = find_column(df, "Width (mm)", "width")
    db_col = find_column(df, "Diameter Tensile Bars, db,t (mm)", "db,t")
    fy_col = find_column(df, "fy Longitudinal Bars (Tensile), (MPa) ", "fy")
    fc_col = find_column(df, "f'c (MPa)", "fc")

    eta = np.clip(df[eta_col].to_numpy(dtype=float) / 100.0, 0.0, 1.0)
    rho = np.maximum(df[rho_col].to_numpy(dtype=float) / 100.0, EPS)
    depth = np.maximum(df[d_col].to_numpy(dtype=float), EPS)
    width = np.maximum(df[b_col].to_numpy(dtype=float), EPS)
    db_t = np.maximum(df[db_col].to_numpy(dtype=float), EPS)
    fy = np.maximum(df[fy_col].to_numpy(dtype=float), EPS)
    fc = np.maximum(df[fc_col].to_numpy(dtype=float), EPS)

    base_df = pd.DataFrame(
        {
            "eta": eta,
            "rho": rho,
            "lam": depth / width,
            "delta": db_t / width,
            "phi": fy / fc,
        }
    )

    x_sym = base_df.copy()
    x_sym["csi"] = x_sym["eta"] * x_sym["phi"]
    x_sym["ri"] = x_sym["rho"] * x_sym["phi"] * x_sym["lam"]

    data_dict = {name: x_sym[name].to_numpy(dtype=float) for name in x_sym.columns}
    logger.info(f"Symbolic inputs prepared: {list(x_sym.columns)}")
    return x_sym, data_dict, base_df


def build_endpoint_profiles(base_df: pd.DataFrame) -> List[Dict[str, float]]:
    profiles: List[Dict[str, float]] = []
    for q in ENDPOINT_QUANTILES:
        profiles.append(
            {
                "rho": float(base_df["rho"].quantile(q)),
                "lam": float(base_df["lam"].quantile(q)),
                "delta": float(base_df["delta"].quantile(q)),
                "phi": float(base_df["phi"].quantile(q)),
            }
        )
    return profiles


def augment_with_anchor_samples(base_df: pd.DataFrame, anchor_repeats: int) -> Tuple[pd.DataFrame, np.ndarray]:
    """Add endpoint anchor samples so eta=0 -> 1 and eta=1 -> 0 are seen during search."""
    rows: List[Dict[str, float]] = []
    targets: List[float] = []

    for profile in build_endpoint_profiles(base_df):
        for eta_val, target in ((0.0, 1.0), (1.0, 0.0)):
            row = {
                "eta": eta_val,
                "rho": profile["rho"],
                "lam": profile["lam"],
                "delta": profile["delta"],
                "phi": profile["phi"],
                "csi": eta_val * profile["phi"],
                "ri": profile["rho"] * profile["phi"] * profile["lam"],
            }
            for _ in range(anchor_repeats):
                rows.append(dict(row))
                targets.append(target)

    anchor_df = pd.DataFrame(rows, columns=["eta", "rho", "lam", "delta", "phi", "csi", "ri"])
    anchor_y = np.asarray(targets, dtype=float)
    logger.info(f"Anchor samples added: {len(anchor_df)} rows")
    return anchor_df, anchor_y


def run_pysr(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
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
        "Running PySR: "
        f"niterations={niterations}, populations={populations}, maxsize={maxsize}"
    )

    model = PySRRegressor(
        niterations=niterations,
        populations=populations,
        maxsize=maxsize,
        binary_operators=["+", "-", "*", "/", "^"],
        unary_operators=["sqrt", "log", "exp"],
        nested_constraints={
            "sqrt": {"sqrt": 0, "log": 1, "exp": 1},
            "log": {"log": 0, "sqrt": 1, "exp": 0},
            "exp": {"exp": 0, "log": 1, "sqrt": 1},
        },
        constraints={"^": (-1, 1), "sqrt": 8, "log": 8, "exp": 6},
        model_selection="accuracy",
        elementwise_loss=(
            "loss(x, y) = "
            "0.60 * abs(x - y) / (abs(y) + 0.1) + "
            "0.25 * abs(x - y) + "
            "0.15 * (x - y)^2"
        ),
        random_state=random_state,
        deterministic=False,
        parallelism="multithreading",
        turbo=True,
        verbosity=1,
    )

    model.fit(x_train.to_numpy(dtype=float), y_train, variable_names=list(x_train.columns))
    return model, model.equations_.copy()


def safe_sympify(expr: str):
    if sp is None:
        raise ImportError("sympy is required. Install with: pip install sympy")

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


def get_equation_string(row: pd.Series) -> str:
    for col in ("sympy_format", "equation", "lambda_format"):
        val = row.get(col, "")
        if not val:
            continue
        expr = str(val).strip()
        if not expr or expr in {"nan", "None"} or "PySRFunction" in expr:
            continue
        return expr
    return ""


def build_substitution(profile: Dict[str, float], eta_value: float) -> Dict[str, float]:
    rho = float(profile["rho"])
    lam = float(profile["lam"])
    delta = float(profile["delta"])
    phi = float(profile["phi"])
    eta = float(eta_value)
    return {
        "eta": eta,
        "rho": rho,
        "lam": lam,
        "delta": delta,
        "phi": phi,
        "csi": eta * phi,
        "ri": rho * phi * lam,
    }


def evaluate_endpoint_ratio(expr_sp, profile: Dict[str, float], eta_value: float) -> float:
    try:
        value = float(expr_sp.evalf(subs=build_substitution(profile, eta_value)))
    except Exception:
        return float("nan")
    return value if np.isfinite(value) else float("nan")


def monotonic_violation(expr_sp, profile: Dict[str, float], n_grid: int = 80) -> float:
    eta_vals = np.linspace(0.0, 1.0, n_grid)
    values = []
    for eta_val in eta_vals:
        try:
            value = float(expr_sp.evalf(subs=build_substitution(profile, float(eta_val))))
        except Exception:
            return 1.0
        values.append(value)

    values = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(values)):
        return 1.0

    diffs = np.diff(values)
    return float(np.mean(diffs > 0.0))


def estimate_complexity(row: pd.Series, expr: str) -> float:
    if "complexity" in row and pd.notna(row["complexity"]):
        return float(row["complexity"])

    ops = len(re.findall(r"[\+\-\*/\^]", expr))
    funcs = len(re.findall(r"sqrt|log|exp|abs", expr))
    terms = len(re.findall(r"eta|rho|lam|delta|phi|csi|ri", expr))
    return float(ops + 1.5 * funcs + 0.5 * terms)


def candidate_failures(
    expr: str,
    r0_vals: np.ndarray,
    r100_vals: np.ndarray,
    mono_vals: np.ndarray,
    metrics: Dict[str, float],
) -> List[str]:
    failures: List[str] = []

    if "eta" not in expr.lower():
        failures.append("missing_eta")

    if not np.all(np.isfinite(r0_vals)) or not np.all(np.isfinite(r100_vals)):
        failures.append("non_finite_endpoint")
        return failures

    if not all(np.isfinite(metrics[name]) for name in ("R2", "RMSE", "MAE", "MAPE")):
        failures.append("non_finite_metric")
        return failures

    end0_max = float(np.max(np.abs(r0_vals - 1.0)))
    end100_max = float(np.max(np.abs(r100_vals - 0.0)))
    mono_max = float(np.max(mono_vals))
    separation = float(np.mean(r0_vals) - np.mean(r100_vals))

    if end0_max > STRICT_THRESHOLDS["end0"]:
        failures.append("eta0_not_one")
    if end100_max > STRICT_THRESHOLDS["end100"]:
        failures.append("eta100_not_zero")
    if mono_max > STRICT_THRESHOLDS["mono"]:
        failures.append("non_monotone")
    if separation < STRICT_THRESHOLDS["separation"]:
        failures.append("weak_endpoint_separation")

    return failures


def evaluate_candidates(
    eq_df: pd.DataFrame,
    data_dict: Dict[str, np.ndarray],
    base_df: pd.DataFrame,
    y_true: np.ndarray,
    m_aci: np.ndarray,
) -> List[Candidate]:
    candidates: List[Candidate] = []
    profiles = build_endpoint_profiles(base_df)
    mean_y = max(float(np.mean(y_true)), EPS)

    for _, row in eq_df.iterrows():
        expr = get_equation_string(row)
        if not expr:
            continue

        try:
            expr_sp = safe_sympify(expr)
            fn = sp.lambdify(tuple(data_dict.keys()), expr_sp, modules=["numpy"])
            ratio_pred = np.asarray(fn(*[data_dict[name] for name in data_dict]), dtype=float)
        except Exception:
            continue

        if ratio_pred.ndim != 1 or len(ratio_pred) != len(y_true):
            continue

        ratio_pred = np.nan_to_num(ratio_pred, nan=0.0, posinf=TARGET_MAX_RATIO, neginf=0.0)
        ratio_pred = np.clip(ratio_pred, 0.0, TARGET_MAX_RATIO)
        y_pred = np.maximum(ratio_pred * m_aci, 0.0)

        r2 = float(r2_score(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae = float(mean_absolute_error(y_true, y_pred))
        mape = float(
            np.mean(
                np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1.0))
            )
            * 100.0
        )

        r0_vals = np.asarray(
            [evaluate_endpoint_ratio(expr_sp, profile, 0.0) for profile in profiles],
            dtype=float,
        )
        r100_vals = np.asarray(
            [evaluate_endpoint_ratio(expr_sp, profile, 1.0) for profile in profiles],
            dtype=float,
        )
        mono_vals = np.asarray(
            [monotonic_violation(expr_sp, profile) for profile in profiles],
            dtype=float,
        )

        end0_max = float(np.nanmax(np.abs(r0_vals - 1.0))) if np.all(np.isfinite(r0_vals)) else float("inf")
        end100_max = float(np.nanmax(np.abs(r100_vals - 0.0))) if np.all(np.isfinite(r100_vals)) else float("inf")
        mono_max = float(np.nanmax(mono_vals)) if np.all(np.isfinite(mono_vals)) else float("inf")
        separation = float(np.nanmean(r0_vals) - np.nanmean(r100_vals))
        complexity = estimate_complexity(row, expr)

        metrics = {
            "R2": round(r2, 4),
            "RMSE": round(rmse, 4),
            "MAE": round(mae, 4),
            "MAPE": round(mape, 2),
            "ratio_eta0_mean": round(float(np.nanmean(r0_vals)), 4),
            "ratio_eta100_mean": round(float(np.nanmean(r100_vals)), 4),
            "end0_max": round(end0_max, 4),
            "end100_max": round(end100_max, 4),
            "mono_max": round(mono_max, 4),
            "endpoint_separation": round(separation, 4),
        }
        objectives = {
            "obj_r2": max(0.0, 1.0 - r2),
            "obj_rmse": rmse / mean_y,
            "obj_mae": mae / mean_y,
            "obj_mape": mape / 100.0,
            "obj_end0": end0_max,
            "obj_end100": end100_max,
            "obj_mono": mono_max,
            "obj_comp": complexity / 50.0,
        }

        failures = candidate_failures(expr, r0_vals, r100_vals, mono_vals, metrics)
        candidates.append(
            Candidate(
                equation=expr,
                complexity=complexity,
                metrics=metrics,
                objectives=objectives,
                feasible=(len(failures) == 0),
                failures=failures,
            )
        )

    feasible_count = sum(int(candidate.feasible) for candidate in candidates)
    logger.info(f"Evaluated {len(candidates)} candidate equations ({feasible_count} strict-physics feasible)")
    return candidates


def select_rankable_indices(candidates: List[Candidate]) -> Tuple[List[int], str]:
    strict = [idx for idx, candidate in enumerate(candidates) if candidate.feasible]
    if strict:
        return strict, "strict"

    relaxed = [
        idx
        for idx, candidate in enumerate(candidates)
        if "missing_eta" not in candidate.failures
        and "non_finite_endpoint" not in candidate.failures
        and "non_finite_metric" not in candidate.failures
    ]
    if relaxed:
        logger.warning("No candidate passed strict physics gates; falling back to eta-preserving finite candidates.")
        return relaxed, "relaxed"

    return [], "empty"


def elite_intersection(candidates: List[Candidate], idxs: Sequence[int], r2_gap: float, error_pct: float) -> List[int]:
    if not idxs:
        return []

    best_r2 = max(candidates[idx].metrics["R2"] for idx in idxs)
    best_rmse = min(candidates[idx].metrics["RMSE"] for idx in idxs)
    best_mae = min(candidates[idx].metrics["MAE"] for idx in idxs)
    best_mape = min(candidates[idx].metrics["MAPE"] for idx in idxs)

    elite = []
    for idx in idxs:
        metrics = candidates[idx].metrics
        if metrics["R2"] < best_r2 - r2_gap:
            continue
        if metrics["RMSE"] > best_rmse * (1.0 + error_pct):
            continue
        if metrics["MAE"] > best_mae * (1.0 + error_pct):
            continue
        if metrics["MAPE"] > best_mape * (1.0 + error_pct):
            continue
        elite.append(idx)

    logger.info(f"Elite intersection size: {len(elite)}")
    return elite


def pareto_front_indices(
    candidates: List[Candidate],
    idxs: Sequence[int],
    objective_names: Sequence[str],
) -> List[int]:
    front: List[int] = []
    for idx in idxs:
        dominated = False
        values_i = np.asarray([candidates[idx].objectives[name] for name in objective_names], dtype=float)
        for other_idx in idxs:
            if other_idx == idx:
                continue
            values_j = np.asarray([candidates[other_idx].objectives[name] for name in objective_names], dtype=float)
            if np.all(values_j <= values_i + 1e-12) and np.any(values_j < values_i - 1e-12):
                dominated = True
                break
        if not dominated:
            front.append(idx)
    return front


def normalize_objectives(mat: np.ndarray) -> np.ndarray:
    if mat.shape[0] == 1:
        return np.zeros_like(mat)
    lo = mat.min(axis=0)
    hi = mat.max(axis=0)
    rng = np.where((hi - lo) < 1e-12, 1.0, hi - lo)
    return (mat - lo) / rng


def build_objective_weights(
    w_accuracy: float,
    w_physics: float,
    w_complexity: float,
) -> Dict[str, float]:
    weights = {
        "obj_r2": w_accuracy / 4.0,
        "obj_rmse": w_accuracy / 4.0,
        "obj_mae": w_accuracy / 4.0,
        "obj_mape": w_accuracy / 4.0,
        "obj_end0": w_physics * 0.40,
        "obj_end100": w_physics * 0.40,
        "obj_mono": w_physics * 0.20,
        "obj_comp": w_complexity,
    }
    total = sum(weights.values())
    return {name: value / total for name, value in weights.items()}


def assign_tchebycheff_scores(
    candidates: List[Candidate],
    idxs: Sequence[int],
    objective_names: Sequence[str],
    weights: Dict[str, float],
) -> None:
    if not idxs:
        return

    mat = np.asarray(
        [[candidates[idx].objectives[name] for name in objective_names] for idx in idxs],
        dtype=float,
    )
    norm = normalize_objectives(mat)
    weight_vec = np.asarray([weights[name] for name in objective_names], dtype=float)
    worst = np.max(norm * weight_vec, axis=1)
    tie = np.sum(norm * weight_vec, axis=1)

    for loc, idx in enumerate(idxs):
        candidates[idx].score = float(worst[loc] + 1e-6 * tie[loc])


def choose_final(
    candidates: List[Candidate],
    rankable_idx: Sequence[int],
    w_accuracy: float,
    w_physics: float,
    w_complexity: float,
    elite_r2_gap: float,
    elite_error_pct: float,
) -> Tuple[int, List[int], List[int]]:
    objective_names = [
        "obj_r2",
        "obj_rmse",
        "obj_mae",
        "obj_mape",
        "obj_end0",
        "obj_end100",
        "obj_mono",
        "obj_comp",
    ]
    weights = build_objective_weights(w_accuracy, w_physics, w_complexity)

    assign_tchebycheff_scores(candidates, rankable_idx, objective_names, weights)

    elite_idx = elite_intersection(candidates, rankable_idx, elite_r2_gap, elite_error_pct)
    selection_pool = elite_idx if elite_idx else list(rankable_idx)
    pareto_idx = pareto_front_indices(candidates, selection_pool, objective_names)
    if not pareto_idx:
        pareto_idx = list(selection_pool)

    best_idx = min(
        pareto_idx,
        key=lambda idx: (
            candidates[idx].score,
            candidates[idx].objectives["obj_r2"],
            candidates[idx].objectives["obj_rmse"],
            candidates[idx].objectives["obj_mae"],
            candidates[idx].objectives["obj_mape"],
            candidates[idx].objectives["obj_comp"],
        ),
    )

    logger.info(
        f"Selection pool: rankable={len(rankable_idx)}, elite={len(elite_idx)}, pareto={len(pareto_idx)}"
    )
    return best_idx, list(elite_idx), list(pareto_idx)


def save_outputs(
    candidates: List[Candidate],
    best_idx: int,
    rankable_idx: Sequence[int],
    elite_idx: Sequence[int],
    pareto_idx: Sequence[int],
    y_true: np.ndarray,
    m_aci: np.ndarray,
    data_dict: Dict[str, np.ndarray],
    selection_mode: str,
) -> None:
    best = candidates[best_idx]

    expr_sp = safe_sympify(best.equation)
    fn = sp.lambdify(tuple(data_dict.keys()), expr_sp, modules=["numpy"])
    ratio = np.asarray(fn(*[data_dict[name] for name in data_dict]), dtype=float)
    ratio = np.nan_to_num(ratio, nan=0.0, posinf=TARGET_MAX_RATIO, neginf=0.0)
    ratio = np.clip(ratio, 0.0, TARGET_MAX_RATIO)
    y_pred = np.maximum(ratio * m_aci, 0.0)

    r2 = float(r2_score(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    mape = float(
        np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1.0))) * 100.0
    )

    ranked = sorted(
        [
            {
                "equation": candidate.equation,
                "complexity": candidate.complexity,
                "metrics": candidate.metrics,
                "objectives": candidate.objectives,
                "score": candidate.score,
                "feasible": candidate.feasible,
                "failures": candidate.failures,
            }
            for candidate in candidates
        ],
        key=lambda item: (
            not item["feasible"],
            item["score"],
            item["objectives"]["obj_r2"],
            item["objectives"]["obj_rmse"],
            item["objectives"]["obj_mae"],
            item["objectives"]["obj_mape"],
        ),
    )

    out_rank = MODELS_DIR / "pysr_candidates_ranked.json"
    out_rank.write_text(json.dumps(ranked, indent=2))

    out_metrics = {
        "approach": "Experimental ratio PySR with physics-constrained Pareto/Tchebycheff selection",
        "equation": best.equation,
        "R2": round(r2, 4),
        "RMSE": round(rmse, 4),
        "MAE": round(mae, 4),
        "MAPE": round(mape, 2),
        "complexity": round(best.complexity, 4),
        "selection_score": round(best.score, 6),
        "selection_mode": selection_mode,
        "n_candidates": len(candidates),
        "n_rankable": len(rankable_idx),
        "n_elite": len(elite_idx),
        "n_pareto": len(pareto_idx),
        "failures": best.failures,
    }
    (MODELS_DIR / "pysr_stacking_metrics.json").write_text(json.dumps(out_metrics, indent=2))

    (EQ_DIR / "best_equation_stacking.txt").write_text(
        "# Best Equation from Experimental-Ratio PySR\n"
        f"# R2={r2:.4f} RMSE={rmse:.4f} MAE={mae:.4f} MAPE={mape:.2f}%\n"
        f"# Selection mode={selection_mode} | rankable={len(rankable_idx)} "
        f"| elite={len(elite_idx)} | pareto={len(pareto_idx)}\n\n"
        f"ratio = {best.equation}\n"
        "Mmax = ratio * M_ACI\n"
    )

    (EQ_DIR / "best_equation_stacking.latex").write_text(
        "% Best Equation from Experimental-Ratio PySR\n"
        f"R = {sp.latex(expr_sp)}\n"
        r"M_{\max} = R \cdot M_{\mathrm{ACI}}" + "\n"
    )

    fig1 = FIG_DIR / "pysr_stacking_scatter.png"
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, s=16, alpha=0.6)
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    plt.plot([lo, hi], [lo, hi], "r--", lw=1.5)
    plt.xlabel("Experimental Mmax (kN.m)")
    plt.ylabel("Equation Mmax (kN.m)")
    plt.title(f"Experimental-Ratio PySR | R2={r2:.4f} | MAPE={mape:.2f}%")
    plt.tight_layout()
    plt.savefig(fig1, dpi=250)
    plt.close()

    median_profile = {
        "rho": float(np.median(data_dict["rho"])),
        "lam": float(np.median(data_dict["lam"])),
        "delta": float(np.median(data_dict["delta"])),
        "phi": float(np.median(data_dict["phi"])),
    }
    eta_grid = np.linspace(0.0, 1.0, 120)
    ratio_grid = []
    for eta_val in eta_grid:
        ratio_grid.append(evaluate_endpoint_ratio(expr_sp, median_profile, float(eta_val)))
    ratio_grid = np.asarray(ratio_grid, dtype=float)

    fig2 = FIG_DIR / "pysr_stacking_endpoints.png"
    plt.figure(figsize=(7, 4))
    plt.plot(eta_grid * 100.0, ratio_grid, lw=2)
    plt.axhline(1.0, color="g", ls="--", lw=1, label="target @0% = 1")
    plt.axhline(0.0, color="r", ls="--", lw=1, label="target @100% = 0")
    plt.xlabel("eta_m (%)")
    plt.ylabel("Predicted ratio (M/M_ACI)")
    plt.title("Endpoint + monotonicity diagnostic")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig2, dpi=250)
    plt.close()

    logger.success(f"Saved best equation -> {EQ_DIR / 'best_equation_stacking.txt'}")
    logger.success(f"Saved metrics -> {MODELS_DIR / 'pysr_stacking_metrics.json'}")
    logger.success(f"Saved ranked candidates -> {out_rank}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experimental-ratio PySR symbolic discovery with hard physics gates"
    )
    parser.add_argument("--niterations", type=int, default=220)
    parser.add_argument("--populations", type=int, default=40)
    parser.add_argument("--maxsize", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ref-vectors", type=int, default=64, help="Deprecated; kept for CLI compatibility.")
    parser.add_argument("--anchor-repeats", type=int, default=24)
    parser.add_argument("--elite-r2-gap", type=float, default=0.01)
    parser.add_argument("--elite-error-pct", type=float, default=0.05)
    parser.add_argument("--w-accuracy", type=float, default=0.65)
    parser.add_argument("--w-physics", type=float, default=0.30)
    parser.add_argument("--w-complexity", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logger()

    if sp is None:
        raise ImportError("Missing dependency: sympy. Install with: pip install sympy")

    df, y_true, m_aci = prepare_full_dataframe()
    y_ratio = build_ratio_target(y_true, m_aci)
    x_sym, data_dict, base_df = build_symbolic_inputs(df)
    anchor_df, anchor_y = augment_with_anchor_samples(base_df, args.anchor_repeats)

    x_train = pd.concat([x_sym, anchor_df], ignore_index=True)
    y_train = np.concatenate([y_ratio, anchor_y])
    logger.info(f"PySR training matrix: {len(x_train)} rows ({len(anchor_df)} anchors)")

    _, eq_df = run_pysr(
        x_train=x_train,
        y_train=y_train,
        niterations=args.niterations,
        populations=args.populations,
        maxsize=args.maxsize,
        random_state=args.seed,
    )

    candidates = evaluate_candidates(eq_df, data_dict, base_df, y_true, m_aci)
    if not candidates:
        raise RuntimeError("No valid candidate equations were produced by PySR.")

    rankable_idx, selection_mode = select_rankable_indices(candidates)
    if not rankable_idx:
        raise RuntimeError("All candidate equations failed hard sanity checks.")

    best_idx, elite_idx, pareto_idx = choose_final(
        candidates=candidates,
        rankable_idx=rankable_idx,
        w_accuracy=args.w_accuracy,
        w_physics=args.w_physics,
        w_complexity=args.w_complexity,
        elite_r2_gap=args.elite_r2_gap,
        elite_error_pct=args.elite_error_pct,
    )
    save_outputs(
        candidates=candidates,
        best_idx=best_idx,
        rankable_idx=rankable_idx,
        elite_idx=elite_idx,
        pareto_idx=pareto_idx,
        y_true=y_true,
        m_aci=m_aci,
        data_dict=data_dict,
        selection_mode=selection_mode,
    )

    logger.success("Done. Physics-constrained PySR pipeline finished successfully.")


if __name__ == "__main__":
    main()
