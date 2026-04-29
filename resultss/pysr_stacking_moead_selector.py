#!/usr/bin/env python3
"""
Physics-Guided Residual Symbolic Regression (PG-RSR) — NSGA-III selection.

Pipeline:
1) Load + preprocess full dataset.
2) Compute physics baseline M0 = As_c fy (d - ac/2) / 1e6  [kN·m]
   where As_c = As(1-η),  ac = As_c fy / (0.85 fc b).
3) Target: z = log(M_exp / M0)  — small log-residual (~±0.3).
4) PySR learns z using dimensionless features:
   eta, cr, rho, d_b, a_d, fc, fy, fy_fc, rho_g, cr_rho, csi, ri.
5) Final prediction: M_pred = M0 · exp(z_pred).
6) NSGA-III (Das-Dennis + fast non-dominated sort) selects Pareto front.
7) Equation selected on VALIDATION set only — test untouched until end.
8) Saves equation, LaTeX, metrics JSON, ranked candidates, and plots.

Published form:  M_max = M0 · exp(f)
where f is the symbolic correction discovered by PySR.
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
from sklearn.model_selection import train_test_split

try:
    import sympy as sp
except Exception:  # pragma: no cover
    sp = None

# Auto-install pykan if missing (needed on Kaggle/Colab)
try:
    import kan as _kan_check  # noqa: F401
except ImportError:
    import subprocess
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "pykan==0.2.4", "--quiet"],
        check=False,
    )

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
        # Mnom_proxy, M_corr_reduced, ductility_corr excluded:
        # scaler_X.pkl was saved with 23 features (before physics extras were added)
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

    # Trim or pad columns to match scaler exactly
    expected_n = scaler_x.n_features_in_
    if X.shape[1] != expected_n:
        # Drop any extra columns from the right to match scaler
        X = X.iloc[:, :expected_n]
    X_scaled = scaler_x.transform(X)

    df_aci = compute_aci_predictions(df_enc)
    m_aci = df_aci["MACI_pred"].to_numpy(dtype=float)
    m_aci = np.maximum(m_aci, 1e-9)

    logger.info(f"Prepared dataset: n={len(df_enc)}, features={X.shape[1]}")
    return df_enc, X_scaled, y_true, m_aci


def get_stacking_predictions(X_scaled: np.ndarray) -> np.ndarray:
    """Return Stacking predictions in kN·m (inverse-transforms log1p if needed)."""
    model_path = MODELS_DIR / "model_stacking.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model: {model_path}")
    model = joblib.load(model_path)
    pred  = np.asarray(model.predict(X_scaled), dtype=float)

    # part1_summary.json records whether the model was trained with log_transform
    summary_path = RESULTSS_DIR / "for_part2" / "part1_summary.json"
    log_transform = False
    if summary_path.exists():
        try:
            log_transform = json.loads(summary_path.read_text()).get("log_transform", False)
        except Exception:
            pass

    if log_transform:
        pred = np.expm1(pred)   # convert log1p-space predictions → kN·m
        logger.info("Stacking model outputs log1p-space → applied expm1 to get kN·m")

    return np.maximum(pred, 0.0)


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

    eta    = find_col(df, ["Mass Loss (Tensile bars), ηm (%)", "Mass Loss", "eta_m", "ηm (%)"])
    rho_t  = find_col(df, ["Tension Reinforcement Ratio, pten (%)", "pten (%)", "rho_t"])
    d      = find_col(df, ["Depth (mm)", "d (mm)", "depth"])
    b      = find_col(df, ["Width (mm)", "b (mm)", "width"])
    fc     = find_col(df, ["f'c (MPa)", "fc (MPa)", "fc"])
    fy     = find_col(df, ["fy Longitudinal Bars (Tensile), (MPa) ", "fy (MPa)", "fy"])
    n_bars = find_col(df, ["# Tensile Bars", "n_bars", "num_bars"])
    db_t   = find_col(df, ["Diameter Tensile Bars, db,t (mm)", "db,t (mm)", "db_t"])

    # --- Optional features (wrapped — may be absent in some datasets) ---
    def try_col(candidates):
        try:
            return find_col(df, candidates)
        except KeyError:
            return None

    cover_raw  = try_col(["Bottom Cover to Ctr of Tension Bar (mm)", "Bottom Cover", "cover"])
    s_stir_raw = try_col(["Stirrup Spacing, s (mm) ", "Stirrup Spacing", "sv", "s_mm"])
    d_stir_raw = try_col(["Stirrup Diameter, ds (mm)", "Stirrup Diameter", "ds"])
    fy_s_raw   = try_col(["fy,s Stirrup Bars", "fy_s", "fys"])
    a_sv_raw   = try_col(["Shear Span, x (mm)", "Shear Span", "shear_span", "x_mm"])
    wc_raw     = try_col(["W/C Ratio", "wc", "w_c"])

    # Steel area As = n * π * (db/2)² — direct structural capacity driver
    As = n_bars * np.pi * (db_t / 2.0) ** 2

    eta_frac = np.clip(eta / 100.0, 0.0, 0.80)
    As_corr  = As * np.maximum(1.0 - eta_frac, 0.0)

    # Physics baseline M0: ACI formula with corroded steel area
    # Target z = log(M_exp/M0) is small (~±0.3) — PySR learns only the residual
    a_corr = As_corr * fy / np.maximum(0.85 * fc * b, eps)
    arm    = np.maximum(d - 0.5 * a_corr, 0.10 * d)
    M0     = np.maximum(As_corr * fy * arm / 1e6, eps)

    csi = df["corr_severity_idx"].to_numpy(dtype=float)
    ri  = df["reinf_index"].to_numpy(dtype=float)

    csi_med = np.median(np.abs(csi))
    ri_med  = np.median(np.abs(ri))

    # Trimmed to 12 features that consistently appear in winning equations.
    # Removed: d_b, a_d, fy_fc, csi — never appeared in top candidates.
    base_feats = {
        "eta":    eta_frac,
        "fc":     fc / 40.0,
        "fy":     fy / 500.0,
        "rho_g":  As / np.maximum(b * d, eps) * 100.0,
        "cr_rho": np.maximum(1.0 - eta_frac, 0.0) * rho_t / 100.0,
        "ri":     ri  / max(float(ri_med),  eps),
    }

    # Add optional physics features — only cover (confirmed in winner)
    extra_feats: dict = {}
    cover_d = None
    if cover_raw is not None:
        cover_d = cover_raw / np.maximum(d, eps)
        extra_feats["cover_d"] = cover_d
        logger.info("Added extra physics feature: cover_d")

    # Pre-computed interaction features — free complexity budget for deeper relationships.
    fy_norm = base_feats["fy"]
    inter_feats: dict = {
        "eta_sq":  eta_frac ** 2,
        "eta_fy":  eta_frac * fy_norm,
        "log_fy":  np.log(np.maximum(fy_norm, 1e-9)),
    }
    if cover_d is not None:
        inter_feats["eta_cover"] = eta_frac * cover_d
        inter_feats["cover_sq"]  = cover_d ** 2
    logger.info(f"Final feature set ({len(base_feats)+len(extra_feats)+len(inter_feats)} total): "
                f"{list(base_feats)+list(extra_feats)+list(inter_feats)}")

    X_sym = pd.DataFrame({**base_feats, **extra_feats, **inter_feats})

    data_dict = {c: X_sym[c].to_numpy(dtype=float) for c in X_sym.columns}
    return X_sym, data_dict, M0


def augment_with_anchor_samples(
    X_sym: pd.DataFrame,
    y_target: np.ndarray,
    n_anchors: int = 150,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Inject physics anchors: near-zero-eta rows reinforce no-corrosion behaviour."""
    eta = X_sym["eta"].to_numpy()
    near_zero = eta < 0.05
    if near_zero.sum() < 3:
        logger.warning("Too few near-zero eta samples for anchoring; skipping augmentation")
        return X_sym, y_target
    # z = log(M/M0) ≈ 0 at η≈0: M0 already captures full capacity with no corrosion
    anchor_target = 0.0
    med_row = {c: float(np.median(X_sym[c])) for c in X_sym.columns}
    med_row["eta"] = 0.0
    if "cr" in med_row:
        med_row["cr"] = 1.0
    if "cr_rho" in med_row:
        med_row["cr_rho"] = med_row.get("rho", 0.02)
    anchors_X = pd.DataFrame([med_row] * n_anchors)
    anchors_y = np.full(n_anchors, anchor_target)
    X_aug = pd.concat([X_sym, anchors_X], ignore_index=True)
    y_aug = np.concatenate([y_target, anchors_y])
    logger.info(f"Anchors: {n_anchors} at eta=0 (z_target=0.0)")
    return X_aug, y_aug


PYSR_MODEL_PATH = MODELS_DIR / "pysr_model.pkl"

def _make_pysr_model(niterations, populations, maxsize, random_state):
    from pysr import PySRRegressor
    return PySRRegressor(
        niterations=niterations,
        populations=populations,
        maxsize=maxsize,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=[
            "safe_sqrt(x::T) where T = sqrt(abs(x))",
            "safe_log(x::T) where T = log(abs(x) + T(1e-6))",
            "square",
        ],
        extra_sympy_mappings={
            "safe_sqrt": lambda x: sp.sqrt(sp.Abs(x)),
            "safe_log":  lambda x: sp.log(sp.Abs(x) + 1e-6),
            "square":    lambda x: x**2,
        },
        nested_constraints={
            "safe_sqrt": {"safe_sqrt": 0, "safe_log": 1},
            "safe_log":  {"safe_log":  0, "safe_sqrt": 1},
            "square":    {"square": 0},
        },
        constraints={"safe_sqrt": 8, "safe_log": 8, "square": 8},
        model_selection="accuracy",
        # Huber on relative error: |exp(z_pred - z_true) - 1| ≈ MAPE directly
        elementwise_loss=(
            "loss(x, y) = begin\n"
            "  r = exp(x - y) - 1.0f0\n"
            "  ar = abs(r)\n"
            "  ar < 0.10f0 ? 0.5f0*r^2/0.10f0 : ar - 0.05f0\n"
            "end"
        ),
        random_state=random_state,
        deterministic=True,
        parallelism="serial",
        verbosity=1,
    )


def run_pysr(
    X_sym: pd.DataFrame,
    y_target: np.ndarray,
    niterations: int,
    populations: int,
    maxsize: int,
    random_state: int,
):
    try:
        from pysr import PySRRegressor  # noqa: F401
    except Exception as exc:
        raise ImportError("PySR is required. Install with: pip install pysr") from exc

    # Multi-seed search: 3 independent populations explore different optima.
    # Julia is precompiled only on the first call — subsequent seeds cost ~5-10s overhead.
    # Each seed runs niterations//3 iterations, preventing premature convergence
    # that plagued single-seed runs (HOF frozen after ~500 of 29,500 iterations).
    seeds = [random_state, random_state + 1000, random_state + 2000]
    iter_each = max(niterations // len(seeds), 40)

    all_dfs: List[pd.DataFrame] = []
    last_model = None
    X_np = X_sym.to_numpy(dtype=float)
    var_names = list(X_sym.columns)

    for seed in seeds:
        logger.info(f"PySR multi-seed: seed={seed}, iterations={iter_each}, populations={populations}")
        try:
            m = _make_pysr_model(iter_each, populations, maxsize, seed)
            m.fit(X_np, y_target, variable_names=var_names)
            all_dfs.append(m.equations_.copy())
            logger.info(f"  seed={seed} → {len(m.equations_)} equations in HOF")
            last_model = m
        except Exception as exc:
            logger.warning(f"  seed={seed} failed: {exc}")

    if not all_dfs:
        raise RuntimeError("All PySR seeds failed — try reducing maxsize or niterations")

    eq_df = pd.concat(all_dfs, ignore_index=True)
    logger.info(f"Multi-seed PySR complete: {len(eq_df)} total equations from {len(all_dfs)} seeds")

    if last_model is not None:
        joblib.dump(last_model, PYSR_MODEL_PATH)

    return last_model, eq_df


# ── KAN (Kolmogorov-Arnold Networks, MIT 2024) ─────────────────────────────
def _kan_top_features(X_sym: pd.DataFrame, k: int = 6) -> List[str]:
    """Select top-k KAN features from shap_importance.csv (keyword matching)."""
    shap_path = MODELS_DIR / "shap_importance.csv"
    if not shap_path.exists():
        return list(X_sym.columns)[:k]
    try:
        df_s = pd.read_csv(shap_path)
        feat_col = next((c for c in df_s.columns if "feature" in c.lower()), df_s.columns[0])
        imp_col  = next(
            (c for c in df_s.columns if any(x in c.lower() for x in ["importance", "shap", "mean"])),
            df_s.columns[1],
        )
        # keyword map: original feature name → X_sym column
        KEYWORD_MAP = {
            "depth":   "d_mm", "width": "b_mm",
            "mass loss": "eta", "ηm":   "eta",  "eta": "eta",
            "f'c":     "fc",   "fc":    "fc",
            "fy":      "fy",
            "pten":    "rho",  "reinf ratio": "rho",
            "reinf_index": "ri", " ri":  "ri",
            "corr_severity": "csi", "csi": "csi",
            "as_corr": "As_corr", "corroded": "As_corr",
            "as":      "As",
        }
        scored: dict[str, float] = {}
        for _, row in df_s.iterrows():
            fname = str(row[feat_col]).lower()
            imp   = abs(float(row[imp_col]))
            for kw, col in KEYWORD_MAP.items():
                if kw in fname and col in X_sym.columns:
                    scored[col] = scored.get(col, 0.0) + imp
        if not scored:
            return list(X_sym.columns)[:k]
        top = sorted(scored, key=scored.get, reverse=True)[:k]
        # fill up to k if not enough matched
        for c in X_sym.columns:
            if len(top) >= k:
                break
            if c not in top:
                top.append(c)
        logger.info(f"KAN top-{k} features (SHAP): {top}")
        return top
    except Exception as exc:
        logger.warning(f"KAN SHAP feature selection failed ({exc}) — using all features")
        return list(X_sym.columns)[:k]


def run_kan_symbolic(
    X_sym: pd.DataFrame,
    y_target: np.ndarray,
    random_state: int = 42,
) -> List[str]:
    """Train KAN on SHAP top-6 features, auto-convert to symbolic, return candidates."""
    try:
        from kan.KAN import KAN   # works across all pykan versions
        import torch
    except ImportError:
        try:
            from kan import KAN   # fallback for older pykan
            import torch
        except ImportError:
            logger.warning("pykan not installed — skipping KAN. (pip install pykan torch)")
            return []
    try:
        # ── 1. SHAP top-6 feature selection ───────────────────────────────
        top_feats = _kan_top_features(X_sym, k=6)
        X_kan = X_sym[top_feats]

        torch.manual_seed(random_state)
        X_t = torch.tensor(X_kan.to_numpy(dtype=float), dtype=torch.float32)

        # Train in log-space: log(R) has zero mean and real variance,
        # preventing KAN from collapsing to the trivial constant R≈1 solution.
        y_t   = torch.tensor(y_target, dtype=torch.float32).unsqueeze(1)
        dataset = {"train_input": X_t, "train_label": y_t,
                   "test_input":  X_t, "test_label":  y_t}

        n_feat = X_kan.shape[1]
        # Single-layer additive KAN: log(R) = f1(η) + f2(d_mm) + ... + f6(csi)
        # No hidden layer → no second-layer weights going to zero.
        # auto_symbolic fits each of the 6 edges independently → guaranteed non-constant.
        model  = KAN(width=[n_feat, 1], grid=5, k=3, seed=random_state)

        # ── 2. Force lamb=0 on model object before any training call ────────
        # Kaggle's pykan ignores the lamb kwarg in train() — setting it directly
        # on the model prevents the regularizer from killing gradients after step 35.
        for _attr in ("lamb", "reg_metric", "penalty", "l1_penalty"):
            try:
                setattr(model, _attr, 0.0)
            except Exception:
                pass

        # ── 3. Train with LBFGS ≥300 steps — robust fallback chain ──────────
        # Try configs in order: most-preferred (LBFGS, 500 steps) → least (default 100).
        # Each config is tried on both fit() and train() to maximise compatibility.
        trained = False
        step_configs = [
            {"opt": "LBFGS", "steps": 500, "lamb": 0.0, "verbose": False},
            {"opt": "LBFGS", "steps": 500, "lamb": 0.0},
            {"opt": "LBFGS", "steps": 500},
            {"opt": "LBFGS", "steps": 300, "lamb": 0.0},
            {"opt": "LBFGS", "steps": 300},
            {"steps": 500},
            {"steps": 300},
        ]
        for cfg in step_configs:
            for method_name in ["fit", "train"]:
                try:
                    getattr(model, method_name)(dataset, **cfg)
                    trained = True
                    logger.info(f"KAN: {method_name}(steps={cfg.get('steps','-')}) succeeded")
                    break
                except (TypeError, AttributeError, Exception):
                    continue
            if trained:
                break

        if not trained:
            # Absolute last resort — whatever API the installed version supports
            for method_name in ["fit", "train"]:
                try:
                    getattr(model, method_name)(dataset)
                    trained = True
                    logger.warning("KAN: using default steps (100) — all explicit-step calls failed")
                    break
                except (TypeError, AttributeError):
                    continue
        if not trained:
            raise RuntimeError("KAN: no compatible train/fit API found in installed pykan")

        # ── 3. Skip pruning — it kills neurons when network is not fully converged ──

        # ── 4. Safe auto_symbolic (no log/x^a → prevents NaN) ─────────────
        # 5 elements — must be < n_features(6) to avoid pykan off-by-one bug
        SAFE_LIB = ["x", "x^2", "x^3", "sqrt", "tanh"]
        try:
            model.auto_symbolic(lib=SAFE_LIB, r2_threshold=0.10)
        except (TypeError, IndexError):
            try:
                model.auto_symbolic(lib=SAFE_LIB)
            except IndexError:
                pass

        # ── 4. Extract formula safely ──────────────────────────────────────
        try:
            with torch.no_grad():
                raw_formulas = model.symbolic_formula(var_names=top_feats)
        except (IndexError, TypeError):
            try:
                with torch.no_grad():
                    raw_formulas = model.symbolic_formula()
            except IndexError as ie:
                logger.warning(f"KAN symbolic_formula IndexError — skipping: {ie}")
                return []

        # pykan returns (formulas, coeffs) tuple OR [[formula]] OR [formula]
        # Unwrap completely to get flat list of sympy-expression strings
        if isinstance(raw_formulas, tuple):
            raw_formulas = raw_formulas[0]

        formula_strs: List[str] = []
        def _flatten(obj) -> None:
            if isinstance(obj, list):
                for item in obj:
                    _flatten(item)
            else:
                formula_strs.append(str(obj))
        _flatten(raw_formulas)

        results: List[str] = []
        for s in formula_strs:
            if not s or s in ("nan", "0", "None", ""):
                continue
            logger.debug(f"KAN raw formula: {s}")
            for i, name in enumerate(top_feats, 0):
                s = s.replace(f"x_{i}", name)
            for i, name in enumerate(top_feats, 1):
                s = s.replace(f"x_{i}", name)
            if not any(name in s for name in top_feats):
                logger.warning(f"KAN formula constant — skipping: {s}")
                continue
            # KAN output is z directly (no exp wrapping — outer formula is M0*exp(z))
            try:
                test_sp = safe_sympify(s)
                free = {str(sym) for sym in test_sp.free_symbols}
                if not any(name in free for name in top_feats):
                    logger.warning(f"KAN sympify gave no feature symbols — skipping")
                    continue
            except Exception as parse_err:
                logger.warning(f"KAN formula failed sympify: {parse_err} — raw: {s[:80]}")
                continue
            results.append(s)

        logger.info(f"KAN produced {len(results)} symbolic candidate(s)")
        return results
    except Exception as exc:
        logger.warning(f"KAN symbolic extraction failed: {exc}")
        return []


def safe_sympify(expr: str):
    if sp is None:
        raise ImportError("sympy is required. Install with: pip install sympy")
    # PySR sympy_format already uses ** not ^; only replace if ^ present
    expr_clean = expr.replace("^", "**")
    return sp.sympify(
        expr_clean,
        locals={
            "sqrt":   sp.sqrt,
            "log":    sp.log,
            "exp":    sp.exp,
            "abs":    sp.Abs,
            "tanh":   sp.tanh,
            "sin":    sp.sin,
            "cos":    sp.cos,
            "square": lambda x: x**2,
            "cube":   lambda x: x**3,
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


def _apply_eta_subs(subs: Dict[str, float], eta_value: float) -> Dict[str, float]:
    """Update subs dict so cr and related features are physically consistent with eta."""
    subs["eta"] = eta_value
    cr_val = max(0.0, 1.0 - eta_value)
    if "cr" in subs:
        subs["cr"] = cr_val
    if "corr_ratio" in subs:
        subs["corr_ratio"] = cr_val
    if "cr_rho" in subs:
        subs["cr_rho"] = cr_val * subs.get("rho", 0.02)
    return subs


def evaluate_endpoint_ratio(expr_sp, med: Dict[str, float], eta_value: float) -> float:
    subs = _apply_eta_subs(dict(med), eta_value)
    try:
        val = float(expr_sp.evalf(subs=subs))
    except Exception:
        return float("nan")
    if not np.isfinite(val):
        return float("nan")
    return float(val)


def monotonic_violation(expr_sp, med: Dict[str, float], n_grid: int = 60) -> float:
    # Check that M_pred(η) = M0(η) * exp(z(η)) decreases with η — not z alone
    # z is a correction term; it needn't be monotone by itself.
    free_s = {str(s) for s in expr_sp.free_symbols}
    if "eta" not in free_s and "cr" not in free_s:
        return 1.0
    eta_vals = np.linspace(0.0, 0.64, n_grid)
    cr_med   = max(float(med.get("cr",  0.80)), 1e-9)
    a_d_med  = max(float(med.get("a_d", 0.10)), 1e-9)

    log_M_pred = []
    for e in eta_vals:
        subs = _apply_eta_subs(dict(med), float(e))
        try:
            z_val = float(expr_sp.evalf(subs=subs))
        except Exception:
            return 1.0
        # Approximate log(M0(η)) using dimensionless ratios at median geometry
        cr_i   = max(1.0 - e, 1e-9)
        a_d_i  = a_d_med * cr_i / cr_med
        arm_i  = max(1.0 - 0.5 * a_d_i, 0.10)
        arm_med = max(1.0 - 0.5 * a_d_med, 0.10)
        log_M0_rel = np.log(cr_i / cr_med) + np.log(arm_i / arm_med)
        log_M_pred.append(log_M0_rel + z_val)

    vals = np.asarray(log_M_pred, dtype=float)
    if not np.all(np.isfinite(vals)):
        return 1.0
    diffs = np.diff(vals)
    return float(np.mean(diffs > 0.0))


def estimate_complexity(row: pd.Series, expr: str) -> float:
    # FIX #8: Use PySR's built-in complexity when available; fallback to token count
    if "complexity" in row and pd.notna(row["complexity"]):
        return float(row["complexity"])
    ops = len(re.findall(r"[\+\-\*/\^]", expr))
    funcs = len(re.findall(r"sqrt|log|exp|abs", expr))
    terms = len(re.findall(r"eta|cr|rho|d_b|a_d|fc|fy|fy_fc|rho_g|csi|ri", expr))
    return float(ops + 1.5 * funcs + 0.5 * terms)


def evaluate_candidates(
    eq_df: pd.DataFrame,
    data_dict: Dict[str, np.ndarray],
    y_true: np.ndarray,
    y_target: np.ndarray,          # z = log(M_exp/M0)
    M0_eval: np.ndarray | None = None,
) -> List[Candidate]:
    cands: List[Candidate] = []

    med    = {k: float(np.median(v)) for k, v in data_dict.items()}
    mean_y = float(np.mean(y_true))

    for _, row in eq_df.iterrows():
        expr = _get_equation_string(row)
        if not expr:
            continue

        try:
            expr_sp = safe_sympify(expr)
            fn   = sp.lambdify(tuple(data_dict.keys()), expr_sp, modules=["numpy"])
            args = [data_dict[k] for k in data_dict.keys()]
            log_pred = np.asarray(fn(*args), dtype=float)
        except Exception:
            continue

        if log_pred.ndim != 1 or len(log_pred) != len(y_true):
            continue

        finite_mask = np.isfinite(log_pred)
        z_pred = np.nan_to_num(log_pred, nan=0.0, posinf=1.0, neginf=-1.0)
        z_pred = np.clip(z_pred, -1.0, 1.0)
        _M0 = M0_eval if M0_eval is not None else np.ones(len(y_true))
        y_pred = _M0 * np.exp(z_pred)   # M0 × exp(z) → kN·m

        sign_frac = float(np.mean(y_pred <= 0.0))  # always 0 with M0*exp(z), kept for safety

        r2   = float(r2_score(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae  = float(mean_absolute_error(y_true, y_pred))
        mape = float(
            np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-9))) * 100.0
        )

        # Endpoint physics in z-space: z ≈ 0 at both endpoints since M0 handles corrosion
        z0   = evaluate_endpoint_ratio(expr_sp, med, 0.0)
        z100 = evaluate_endpoint_ratio(expr_sp, med, 0.64)
        end0   = abs(z0   - 0.0) if np.isfinite(z0)   else 1.0
        end100 = abs(z100 - 0.0) if np.isfinite(z100) else 1.0

        mono = monotonic_violation(expr_sp, med)
        comp = estimate_complexity(row, expr)
        free_syms = {str(s) for s in expr_sp.free_symbols}
        has_eta = "eta" in free_syms or "cr" in free_syms
        has_d   = "d_b" in free_syms or "a_d" in free_syms or "rho_g" in free_syms
        has_fc  = "fc"  in free_syms or "fy_fc" in free_syms
        _nonsense_exp = bool(
            "exp" in expr and re.search(r"exp\s*\(\s*(fy|rho|d_b|fc|fy_fc)\s*\)", expr)
        )

        metrics = {
            "R2": round(r2, 4), "RMSE": round(rmse, 4),
            "MAE": round(mae, 4), "MAPE": round(mape, 2),
            "z_eta0":   round(float(z0),   4) if np.isfinite(z0)   else None,
            "z_eta064": round(float(z100),  4) if np.isfinite(z100) else None,
            "has_d": has_d, "has_fc": has_fc, "sign_ok": sign_frac < 0.05,
        }
        objectives = {
            "obj_r2":     max(0.0, 1.0 - r2),
            "obj_mape":   mape / 100.0,
            "obj_rmse":   rmse / max(mean_y, 1e-6),
            "obj_end0":   end0,
            "obj_end100": end100,
            "obj_mono":   mono,
            "obj_comp":   comp / 50.0,
            "obj_no_eta": 0.5 if not has_eta else 0.0,
            "obj_no_d":   1.0 if not has_d   else 0.0,
            "obj_no_fc":  1.0 if not has_fc  else 0.0,
            "obj_sign":   sign_frac + (0.5 if _nonsense_exp else 0.0),
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


# ── SHAP-guided objective weights ──────────────────────────────────────────
def load_shap_weights(shap_path: Path) -> Dict[str, float]:
    """Read shap_importance.csv and map importance to per-objective multipliers."""
    if not shap_path.exists():
        logger.warning("shap_importance.csv not found — using uniform objective weights")
        return {}
    try:
        df_s = pd.read_csv(shap_path)
        feat_col = next((c for c in df_s.columns if "feature" in c.lower()), df_s.columns[0])
        imp_col  = next(
            (c for c in df_s.columns if any(x in c.lower() for x in ["importance", "shap", "mean"])),
            df_s.columns[1],
        )
        total = df_s[imp_col].abs().sum()
        if total < 1e-9:
            return {}
        depth_imp = eta_imp = 0.0
        for _, row in df_s.iterrows():
            fname = str(row[feat_col]).lower()
            imp   = float(row[imp_col]) / total
            if "depth" in fname or "d (mm)" in fname:
                depth_imp += imp
            if "mass loss" in fname or "eta" in fname or "ηm" in fname:
                eta_imp += imp
        boost = 1.0 + depth_imp + eta_imp          # ~1.6 for this dataset
        weights = {
            "obj_r2":     boost,
            "obj_mape":   boost * 0.8,
            "obj_rmse":   boost * 0.6,
            "obj_end0":   1.0,
            "obj_end100": 1.0,
            "obj_mono":   1.2,
            "obj_comp":   0.5,
            "obj_no_eta": 1.5,
            "obj_no_d":   2.5,   # heavy penalty: depth is #1 SHAP feature (47%)
            "obj_no_fc":  2.0,   # penalize missing concrete strength (critical material property)
            "obj_sign":   1.5,   # penalize equations that predict negative capacity
        }
        logger.info(f"SHAP weights loaded — accuracy boost={boost:.3f} (depth={depth_imp:.2%}, eta={eta_imp:.2%})")
        return weights
    except Exception as exc:
        logger.warning(f"Could not load SHAP weights: {exc}")
        return {}


# ── NSGA-III core (Das-Dennis + fast non-dominated sort) ───────────────────
def _das_dennis(n_obj: int, n_partitions: int) -> np.ndarray:
    """Generate structured simplex reference points (Das & Dennis 1998)."""
    def _recurse(left: int, n: int, cur: list, out: list) -> None:
        if n == 1:
            out.append(cur + [left])
        else:
            for i in range(left + 1):
                _recurse(left - i, n - 1, cur + [i], out)
    out: list = []
    _recurse(n_partitions, n_obj, [], out)
    return np.array(out, dtype=float) / n_partitions


def _fast_nondom_sort(obj_mat: np.ndarray) -> List[List[int]]:
    """O(MN²) fast non-dominated sort (Deb 2002)."""
    n = len(obj_mat)
    dominates  = [[] for _ in range(n)]
    n_dom      = np.zeros(n, dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            ij = (obj_mat[i] <= obj_mat[j]).all() and (obj_mat[i] < obj_mat[j]).any()
            ji = (obj_mat[j] <= obj_mat[i]).all() and (obj_mat[j] < obj_mat[i]).any()
            if ij:
                dominates[i].append(j); n_dom[j] += 1
            elif ji:
                dominates[j].append(i); n_dom[i] += 1
    fronts: List[List[int]] = [[i for i in range(n) if n_dom[i] == 0]]
    k = 0
    while fronts[k]:
        nxt: List[int] = []
        for i in fronts[k]:
            for j in dominates[i]:
                n_dom[j] -= 1
                if n_dom[j] == 0:
                    nxt.append(j)
        k += 1
        fronts.append(nxt)
    return [f for f in fronts if f]


def nsga3_select(
    cands: List[Candidate],
    n_partitions: int = 8,
    shap_obj_weights: Dict[str, float] | None = None,
) -> List[int]:
    """NSGA-III reference-point association over PySR Pareto fronts."""
    if not cands:
        return []
    obj_names = list(cands[0].objectives.keys())
    n_obj = len(obj_names)

    # Auto-scale n_partitions to keep reference points manageable (target ~700-2000 refs)
    # For n_obj=10, n_partitions=8 → C(17,9)=24310 refs (too slow); n_partitions=4 → C(13,9)=715
    if n_obj >= 9 and n_partitions > 4:
        n_partitions = 4
        logger.info(f"Auto-scaled n_partitions to 4 for {n_obj}-objective problem")

    # SHAP-weighted objectives before normalisation
    w_vec = np.array([
        (shap_obj_weights or {}).get(k, 1.0) for k in obj_names
    ], dtype=float)
    raw  = np.array([[c.objectives[k] for k in obj_names] for c in cands], dtype=float)
    raw  = raw * w_vec
    nmat = normalize_objectives(raw)

    fronts = _fast_nondom_sort(nmat)
    refs   = _das_dennis(n_obj, n_partitions)       # structured simplex grid

    chosen: set = set()
    for front in fronts:
        for idx in front:
            # Associate each candidate with its nearest reference point
            dists = np.linalg.norm(refs - nmat[idx], axis=1)
            _ = int(np.argmin(dists))               # niche index (for logging)
            chosen.add(idx)
        if len(chosen) >= len(refs):
            break

    if not chosen:
        chosen = set(fronts[0])
    logger.info(
        f"NSGA-III selected {len(chosen)} candidates "
        f"from {len(cands)} total | fronts={len(fronts)} | refs={len(refs)}"
    )
    return sorted(chosen)


def choose_final(
    cands: List[Candidate],
    selected_idx: List[int],
    w_accuracy: float,
    w_physics: float,
    w_complexity: float,
    shap_obj_weights: Dict[str, float] | None = None,
) -> int:
    """Select best equation via SHAP-scaled weighted scoring over NSGA-III front."""
    total = w_accuracy + w_physics + w_complexity
    wa, wp, wc = w_accuracy / total, w_physics / total, w_complexity / total

    base = {
        "obj_r2":     wa * 0.30,   # reduced — MAPE is more important for publication
        "obj_mape":   wa * 0.55,   # boosted — directly targets < 15% MAPE criterion
        "obj_rmse":   wa * 0.15,
        "obj_end0":   wp * 0.15,   # endpoint η=0 → 1.0
        "obj_end100": wp * 0.15,   # endpoint η=0.64 → 0.95
        "obj_mono":   wp * 0.10,   # monotonic decrease with corrosion
        "obj_comp":   wc * 1.00,
        "obj_no_eta": wp * 0.10,   # penalize missing mass-loss variable
        "obj_no_d":   wp * 0.25,   # penalize missing depth (47% SHAP — most critical)
        "obj_no_fc":  wp * 0.15,   # penalize missing concrete strength
        "obj_sign":   wp * 0.10,   # penalize negative capacity ratio predictions
    }
    # Apply SHAP multipliers to final scoring weights
    sw = shap_obj_weights or {}
    w  = {k: base[k] * sw.get(k, 1.0) for k in base}

    idxs = selected_idx if selected_idx else list(range(len(cands)))
    best_idx, best_score = idxs[0], float("inf")
    for i in idxs:
        c  = cands[i]
        sc = sum(w[k] * c.objectives[k] for k in w)
        c.score = float(sc)
        if sc < best_score:
            best_score, best_idx = sc, i

    logger.info(f"Final winner: index={best_idx}, score={best_score:.6f}")
    return best_idx


def compute_cv_metrics(
    best_eq: str,
    data_dict: Dict[str, np.ndarray],
    y_true: np.ndarray,
    M0: np.ndarray,
    n_splits: int = 5,
) -> Dict[str, float]:
    """5-fold cross-validation of the best equation — required for top-journal publication."""
    from sklearn.model_selection import KFold
    expr_sp = safe_sympify(best_eq)
    fn = sp.lambdify(tuple(data_dict.keys()), expr_sp, modules=["numpy"])
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    n = len(y_true)
    indices = np.arange(n)
    fold_r2, fold_mape, fold_rmse, fold_mae = [], [], [], []
    for _, test_idx in kf.split(indices):
        dd_fold = {k: v[test_idx] for k, v in data_dict.items()}
        yt_fold = y_true[test_idx]
        M0_fold = M0[test_idx]
        z = np.nan_to_num(
            np.asarray(fn(*[dd_fold[k] for k in data_dict.keys()]), dtype=float),
            nan=0.0, posinf=1.0, neginf=-1.0,
        )
        z = np.clip(z, -1.0, 1.0)
        yp = M0_fold * np.exp(z)
        fold_r2.append(float(r2_score(yt_fold, yp)))
        fold_rmse.append(float(np.sqrt(mean_squared_error(yt_fold, yp))))
        fold_mae.append(float(mean_absolute_error(yt_fold, yp)))
        fold_mape.append(float(
            np.mean(np.abs((yt_fold - yp) / np.maximum(np.abs(yt_fold), 1e-9))) * 100.0
        ))
    result = {
        "cv_R2_mean":   round(float(np.mean(fold_r2)),   4),
        "cv_R2_std":    round(float(np.std(fold_r2)),    4),
        "cv_RMSE_mean": round(float(np.mean(fold_rmse)), 4),
        "cv_RMSE_std":  round(float(np.std(fold_rmse)),  4),
        "cv_MAE_mean":  round(float(np.mean(fold_mae)),  4),
        "cv_MAE_std":   round(float(np.std(fold_mae)),   4),
        "cv_MAPE_mean": round(float(np.mean(fold_mape)), 2),
        "cv_MAPE_std":  round(float(np.std(fold_mape)),  2),
    }
    logger.info(
        f"5-fold CV: R²={result['cv_R2_mean']}±{result['cv_R2_std']}  "
        f"MAPE={result['cv_MAPE_mean']}±{result['cv_MAPE_std']}%"
    )
    return result


def save_outputs(
    cands: List[Candidate],
    best_idx: int,
    y_true: np.ndarray,
    M0: np.ndarray,
    data_dict: Dict[str, np.ndarray],
    m_stack: np.ndarray | None = None,
    y_true_test: np.ndarray | None = None,
    M0_test: np.ndarray | None = None,
    data_dict_test: Dict[str, np.ndarray] | None = None,
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

    # Recompute best predictions: Mmax = M0 * exp(z)
    expr_sp = safe_sympify(best.equation)
    fn      = sp.lambdify(tuple(data_dict.keys()), expr_sp, modules=["numpy"])
    z_pred  = np.asarray(fn(*[data_dict[k] for k in data_dict.keys()]), dtype=float)
    z_pred  = np.clip(np.nan_to_num(z_pred, nan=0.0, posinf=1.0, neginf=-1.0), -1.0, 1.0)
    y_pred  = M0 * np.exp(z_pred)

    r2   = float(r2_score(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    mape = float(
        np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-9))) * 100.0
    )

    stack_r2 = float(r2_score(y_true, m_stack)) if m_stack is not None else None

    # ── Test-set metrics (20% holdout — scientific validation) ────────────
    test_r2 = test_rmse = test_mae = test_mape = None
    if y_true_test is not None and data_dict_test is not None and M0_test is not None:
        z_t = np.asarray(fn(*[data_dict_test[k] for k in data_dict.keys()]), dtype=float)
        z_t = np.clip(np.nan_to_num(z_t, nan=0.0, posinf=1.0, neginf=-1.0), -1.0, 1.0)
        y_pred_t  = M0_test * np.exp(z_t)
        test_r2   = float(r2_score(y_true_test, y_pred_t))
        test_rmse = float(np.sqrt(mean_squared_error(y_true_test, y_pred_t)))
        test_mae  = float(mean_absolute_error(y_true_test, y_pred_t))
        test_mape = float(
            np.mean(np.abs((y_true_test - y_pred_t) / np.maximum(np.abs(y_true_test), 1e-9))) * 100.0
        )
        logger.info(
            f"Test-set (20%): R²={test_r2:.4f}  RMSE={test_rmse:.4f}  "
            f"MAE={test_mae:.4f}  MAPE={test_mape:.2f}%"
        )

    # ── 5-fold cross-validation (required for top-journal submission) ────────
    cv = compute_cv_metrics(best.equation, data_dict, y_true, M0, n_splits=5)

    # ── Publication gate ────────────────────────────────────────────────────
    _val_mape  = best.metrics.get("MAPE", float("inf"))   # validation MAPE (selection set)
    _test_mape = test_mape if test_mape is not None else float("inf")
    _test_r2   = test_r2   if test_r2   is not None else -float("inf")
    _accepted  = (
        _val_mape  < 10.0
        and _test_mape < 10.0
        and _test_r2   > 0.95
        and best.complexity <= 18.0
        and best.metrics.get("sign_ok", False)
    )
    if not _accepted:
        logger.warning(
            f"Publication gate NOT passed: "
            f"val_MAPE={_val_mape:.2f}% | test_MAPE={_test_mape:.2f}% | "
            f"test_R²={_test_r2:.4f} | complexity={best.complexity:.0f}"
        )
    else:
        logger.success("Publication gate PASSED: val_MAPE<10%, test_MAPE<10%, R²>0.95, complexity≤18")

    out_metrics = {
        "approach": "PG-RSR: Physics-Guided Residual Symbolic Regression — M_pred = M0 * exp(z)",
        "equation": best.equation,
        "published_form": "Mmax = M0 * exp(z)  where M0 = As_c*fy*(d - ac/2)/1e6",
        "has_d": best.metrics.get("has_d", None),
        "has_fc": best.metrics.get("has_fc", None),
        "sign_ok": best.metrics.get("sign_ok", None),
        "R2":   round(r2, 4),
        "RMSE": round(rmse, 4),
        "MAE":  round(mae, 4),
        "MAPE": round(mape, 2),
        "test_R2":   round(test_r2,   4) if test_r2   is not None else None,
        "test_RMSE": round(test_rmse, 4) if test_rmse is not None else None,
        "test_MAE":  round(test_mae,  4) if test_mae  is not None else None,
        "test_MAPE": round(test_mape, 2) if test_mape is not None else None,
        **cv,
        "complexity": round(best.complexity, 4),
        "selection_score": round(best.score, 6),
        "n_candidates": len(cands),
        "stacking_R2_reference": round(stack_r2, 4) if stack_r2 is not None else None,
        "accepted_for_publication": _accepted,
        "acceptance_rule": "val_MAPE<10% and test_MAPE<10% and test_R2>0.95 and complexity<=18 and sign_ok",
    }
    (MODELS_DIR / "pysr_stacking_metrics.json").write_text(json.dumps(out_metrics, indent=2))

    stack_label  = f"#   Stacking R²={stack_r2:.4f}\n" if stack_r2 is not None else ""
    test_label   = (
        f"# Test  R²={test_r2:.4f}  RMSE={test_rmse:.4f}  MAE={test_mae:.4f}  MAPE={test_mape:.2f}%\n"
        if test_r2 is not None else ""
    )
    cv_label = (
        f"# 5-fold CV  R²={cv['cv_R2_mean']}±{cv['cv_R2_std']}  "
        f"MAPE={cv['cv_MAPE_mean']}±{cv['cv_MAPE_std']}%\n"
    )
    (EQ_DIR / "best_equation_stacking.txt").write_text(
        "# Best Equation from PySR log-space distillation: Mmax = M0 * exp(z)\n"
        "# M0 = As_corr * fy * (d - a_corr/2) / 1e6  [kN·m]  (ACI with corroded area)\n"
        f"# TrainVal R²={r2:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}  MAPE={mape:.2f}%\n"
        f"{test_label}"
        f"{cv_label}"
        f"{stack_label}"
        "# Features: eta=mass_loss/100, cr=1-eta,\n"
        "#            rho=rho_t/100, d_b=d/b, a_d=a_corr/d,\n"
        "#            fc=fc_MPa/40, fy=fy_MPa/500, fy_fc=fy/fc,\n"
        "#            rho_g=As/(b*d)*100, cr_rho=cr*rho,\n"
        "#            csi=corr_severity_idx, ri=reinf_index\n"
        "# z = log-correction to physics baseline\n\n"
        f"z = {best.equation}\n"
        "Mmax = M0 * exp(z)   [kN·m]\n"
    )

    latex_eq = sp.latex(expr_sp)
    (EQ_DIR / "best_equation_stacking.latex").write_text(
        "% Best Equation from PySR log-space distillation\n"
        f"z = {latex_eq}\n"
        r"M_{\max} = M_0 \cdot e^{z} \quad [\mathrm{kN{\cdot}m}]" + "\n"
        r"M_0 = \frac{A_{s,c} f_y (d - a_c/2)}{10^6}, \quad a_c = \frac{A_{s,c} f_y}{0.85 f'_c b}" + "\n"
    )

    # Figure 1: predicted vs true
    fig1 = FIG_DIR / "pysr_stacking_scatter.png"
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, s=16, alpha=0.6)
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    plt.plot([lo, hi], [lo, hi], "r--", lw=1.5)
    plt.xlabel("Experimental $M_{max}$ (kN·m)")
    plt.ylabel("Equation $M_{max} = M_0 e^z$ (kN·m)")
    plt.title(f"PySR log-space | R²={r2:.4f} | MAPE={mape:.2f}%")
    plt.tight_layout()
    plt.savefig(fig1, dpi=250)
    plt.close()

    # Figure 2: z(η) trend — should be near 0 and monotonically non-increasing
    med = {k: float(np.median(v)) for k, v in data_dict.items()}
    eta_grid = np.linspace(0.0, 0.70, 120)
    z_grid = []
    for e in eta_grid:
        subs = _apply_eta_subs(dict(med), float(e))
        try:
            z_grid.append(float(expr_sp.evalf(subs=subs)))
        except Exception:
            z_grid.append(float("nan"))
    z_grid = np.asarray(z_grid, dtype=float)

    fig2 = FIG_DIR / "pysr_stacking_endpoints.png"
    plt.figure(figsize=(7, 4))
    plt.plot(eta_grid * 100.0, z_grid, lw=2, label="z(η) correction")
    plt.axhline(0.0, color="g", ls="--", lw=1, label="z = 0 (no correction)")
    plt.xlabel("Mass Loss η (%)")
    plt.ylabel("z = log(M / M₀)")
    plt.title("Log-correction z(η) Diagnostic")
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


def _refit_constants(
    eq_str: str,
    data_dict: Dict[str, np.ndarray],
    y_true: np.ndarray,
    M0: np.ndarray,
) -> str:
    """Refit numeric constants in a PySR equation to minimize MAPE directly."""
    from scipy.optimize import minimize

    try:
        expr_sp = safe_sympify(eq_str)
        consts = sorted(
            [a for a in expr_sp.atoms(sp.Number) if abs(float(a)) > 1e-12],
            key=lambda x: float(x),
        )
        if not consts:
            logger.info("Constant refitting: no numeric constants found — skipping")
            return eq_str

        symbols_c = [sp.Symbol(f"__c{i}") for i in range(len(consts))]
        expr_sub = expr_sp
        for orig, sym in zip(consts, symbols_c):
            expr_sub = expr_sub.subs(orig, sym)

        feat_keys = list(data_dict.keys())
        all_syms  = feat_keys + [str(s) for s in symbols_c]
        fn = sp.lambdify(all_syms, expr_sub, modules=["numpy"])
        feat_vals = [data_dict[k] for k in feat_keys]

        x0 = np.array([float(c) for c in consts])

        def mape_loss(x):
            try:
                z = np.asarray(fn(*feat_vals, *x), dtype=float)
                z = np.clip(np.nan_to_num(z, nan=0.0, posinf=1.0, neginf=-1.0), -1.0, 1.0)
                yp = M0 * np.exp(z)
                return float(np.mean(np.abs((y_true - yp) / np.maximum(y_true, 1e-9))))
            except Exception:
                return 1.0

        mape_before = mape_loss(x0) * 100
        res = minimize(mape_loss, x0, method="Nelder-Mead",
                       options={"maxiter": 8000, "xatol": 1e-6, "fatol": 1e-6})
        mape_after = res.fun * 100

        expr_opt = expr_sub
        for sym, val in zip(symbols_c, res.x):
            expr_opt = expr_opt.subs(sym, sp.Float(round(float(val), 6)))

        eq_refitted = str(expr_opt)
        logger.info(
            f"Constant refitting: MAPE {mape_before:.2f}% → {mape_after:.2f}% "
            f"({len(consts)} constants optimized)"
        )
        return eq_refitted
    except Exception as exc:
        logger.warning(f"Constant refitting failed: {exc} — using original equation")
        return eq_str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stacking to PySR symbolic distillation with MOEA/D-style selection"
    )
    p.add_argument("--niterations", type=int, default=6000)
    p.add_argument("--populations", type=int, default=40)
    p.add_argument("--maxsize",     type=int, default=35)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ref-vectors", type=int, default=8,
                   help="NSGA-III Das-Dennis partitions (default=8 → ~1287 ref-points for 8 obj)")
    p.add_argument(
        "--w-accuracy", type=float, default=0.70,
        help="Weight for accuracy objectives (R2, MAPE, RMSE). Default=0.70"
    )
    p.add_argument(
        "--w-physics", type=float, default=0.20,
        help="Weight for physics objectives (endpoints, monotonicity). Default=0.20"
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

    # Always start fresh — delete saved PySR model to avoid warm-start contamination
    if PYSR_MODEL_PATH.exists():
        PYSR_MODEL_PATH.unlink()
        logger.info("Deleted cached pysr_model.pkl — starting fresh search")

    df, X_scaled, y_true, m_aci = prepare_full_dataframe()
    m_stack = get_stacking_predictions(X_scaled)

    X_sym, data_dict, M0_raw = build_symbolic_inputs(df)
    M0_uncalib = np.maximum(M0_raw, 1e-9)

    # ── 3-way split: 60% train / 20% val / 20% test ─────────────────────────
    n = len(y_true)
    trainval_idx, test_idx = train_test_split(np.arange(n), test_size=0.20, random_state=args.seed)
    train_idx, val_idx     = train_test_split(trainval_idx, test_size=0.25, random_state=args.seed)

    # M0 calibration: remove ACI systematic bias using train data only
    _log_ratios = np.log(np.maximum(y_true[train_idx], 1e-9) / M0_uncalib[train_idx])
    k_calib = float(np.exp(np.median(_log_ratios)))
    k_calib = float(np.clip(k_calib, 0.5, 2.0))
    M0 = M0_uncalib * k_calib
    logger.info(
        f"M0 calibration: k={k_calib:.4f} "
        f"(M0 MAPE before={np.mean(np.abs((y_true - M0_uncalib)/np.maximum(y_true,1e-9)))*100:.1f}%, "
        f"after={np.mean(np.abs((y_true - M0)/np.maximum(y_true,1e-9)))*100:.1f}%)"
    )

    # Target: z = log(M_exp / M0_calibrated)
    z_true = np.log(np.maximum(y_true, 1e-9) / M0)
    y_target = np.clip(z_true, -1.0, 1.0)
    logger.info(
        f"Log-space target z=log(M_exp/M0_calib): mean={y_target.mean():.3f}, "
        f"std={y_target.std():.3f}, range=[{y_target.min():.3f}, {y_target.max():.3f}]"
    )

    # PySR trains on 60% train only + physics anchors
    X_sym_train    = X_sym.iloc[train_idx].reset_index(drop=True)
    y_target_train = y_target[train_idx]
    # Anchors disabled: M0 already encodes corrosion physics; anchor a_d would need
    # full recomputation per-eta which is not done in augment_with_anchor_samples.
    X_sym_aug, y_target_aug = X_sym_train, y_target_train

    # Validation set for equation selection (never seen by PySR)
    data_dict_val = {k: v[val_idx] for k, v in data_dict.items()}
    y_true_val    = y_true[val_idx]
    M0_val        = M0[val_idx]
    z_true_val    = y_target[val_idx]

    # Test set: only touched for final reporting
    data_dict_test = {k: v[test_idx] for k, v in data_dict.items()}
    y_true_test    = y_true[test_idx]
    M0_test        = M0[test_idx]

    logger.info(
        f"Split: {len(train_idx)} train + {len(X_sym_aug)-len(train_idx)} anchors | "
        f"{len(val_idx)} val (selection) | {len(test_idx)} test (final report)"
    )

    _, eq_df = run_pysr(
        X_sym=X_sym_aug,
        y_target=y_target_aug,
        niterations=args.niterations,
        populations=args.populations,
        maxsize=args.maxsize,
        random_state=args.seed,
    )

    # Select equation using VALIDATION set only — no leakage
    cands = evaluate_candidates(eq_df, data_dict_val, y_true_val, z_true_val, M0_val)

    # KAN disabled: inconsistent output format with z-space pipeline
    kan_exprs: List[str] = []

    if not cands:
        raise RuntimeError(
            "No valid candidate equations generated. "
            "Try increasing --niterations or --populations."
        )

    shap_weights = load_shap_weights(MODELS_DIR / "shap_importance.csv")
    selected = nsga3_select(cands, n_partitions=args.ref_vectors, shap_obj_weights=shap_weights)
    best_idx = choose_final(
        cands,
        selected,
        w_accuracy=args.w_accuracy,
        w_physics=args.w_physics,
        w_complexity=args.w_complexity,
        shap_obj_weights=shap_weights,
    )

    # ── Scipy constant refitting: optimize numeric constants for MAPE ────────
    data_dict_trainval = {k: v[trainval_idx] for k, v in data_dict.items()}
    best_eq_refitted = _refit_constants(
        cands[best_idx].equation,
        data_dict_trainval,
        y_true[trainval_idx],
        M0[trainval_idx],
    )
    if best_eq_refitted != cands[best_idx].equation:
        cands[best_idx] = Candidate(
            equation=best_eq_refitted,
            complexity=cands[best_idx].complexity,
            metrics=cands[best_idx].metrics,
            objectives=cands[best_idx].objectives,
            score=cands[best_idx].score,
        )

    # Pass trainval (80%) for CV, test (20%) for final holdout report
    save_outputs(
        cands, best_idx,
        y_true[trainval_idx], M0[trainval_idx], data_dict_trainval,
        m_stack=m_stack[trainval_idx],
        y_true_test=y_true_test, M0_test=M0_test, data_dict_test=data_dict_test,
    )

    logger.success("Done. Best equation pipeline finished successfully.")
    import os; os._exit(0)  # kills Julia child process and exits cleanly


if __name__ == "__main__":
    main()
