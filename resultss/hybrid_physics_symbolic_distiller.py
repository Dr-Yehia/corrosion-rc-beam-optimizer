#!/usr/bin/env python3
"""
Hybrid Physics-Constrained Symbolic Distiller.

Goal:
    Derive a closed-form, publication-grade corrosion correction equation:

        M_pred = M_ACI * R_c

    R_c is a mixture of physics-constrained experts:

        R_c = sum_k w_k(x) * (1 - eta)^alpha_k(x) * exp(-eta * beta_k(x))

Design rules:
    - Experimental data is the primary target.
    - The Stacking model is only a teacher signal.
    - ACI remains the mechanical backbone.
    - All symbolic variables are dimensionless.
    - eta appears only in the physical corrosion envelope.
    - MAPE <= target_mape is a hard publication gate.

This file is intentionally conservative: it favors a defensible equation over a
high-R2 but physically invalid symbolic artifact.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from scipy.optimize import differential_evolution, least_squares
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config import FEATURE_COLS, CAT_COLS, TARGET_COL  # noqa: E402
from data_preprocessing import load_raw_data, clean_data, engineer_features  # noqa: E402
from aci_calculator import compute_aci_predictions  # noqa: E402

RESULTSS = ROOT / "resultss"
MODELS = RESULTSS / "models"
EQS = RESULTSS / "equations"
FIGS = RESULTSS / "figures"
LOGS = RESULTSS / "logs"
for d in (MODELS, EQS, FIGS, LOGS):
    d.mkdir(parents=True, exist_ok=True)


@dataclass
class Term:
    name: str
    values: np.ndarray


@dataclass
class CandidateResult:
    k: int
    terms: List[str]
    theta: List[float]
    metrics: Dict[str, float]
    teacher_metrics: Dict[str, float]
    physics: Dict[str, object]
    complexity: int
    score: float
    publishable: bool


def setup_logger() -> None:
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}")
    logger.add(str(LOGS / "run_log_hybrid_distiller.txt"), level="DEBUG")


def find_col(df: pd.DataFrame, names: List[str], optional: bool = False, default: float = 0.0) -> np.ndarray:
    for n in names:
        if n in df.columns:
            return df[n].to_numpy(float)
    norm = {c.lower().replace(" ", "").strip(): c for c in df.columns}
    for n in names:
        key = n.lower().replace(" ", "").strip()
        if key in norm:
            return df[norm[key]].to_numpy(float)
    for n in names:
        key = n.lower().replace(" ", "").strip()
        hits = [c for c in df.columns if key in c.lower().replace(" ", "")]
        if hits:
            logger.warning(f"Column {n!r} not found exactly; using {hits[0]!r}")
            return df[hits[0]].to_numpy(float)
    if optional:
        logger.warning(f"Optional column missing: {names}. Using constant {default}.")
        return np.full(len(df), default, dtype=float)
    raise KeyError(f"Missing required column. Tried {names}. Available columns: {list(df.columns)}")


def softplus(z: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(-np.abs(z))) + np.maximum(z, 0.0) + 1e-8


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.maximum(e.sum(axis=1, keepdims=True), 1e-12)


def safe_mape(y: np.ndarray, yhat: np.ndarray) -> float:
    den = np.maximum(np.abs(y), np.maximum(np.mean(np.abs(y)) * 0.05, 1.0))
    return float(np.mean(np.clip(np.abs((y - yhat) / den), 0.0, 5.0)) * 100.0)


def metrics(y: np.ndarray, yhat: np.ndarray) -> Dict[str, float]:
    return {
        "R2": float(r2_score(y, yhat)),
        "RMSE": float(np.sqrt(mean_squared_error(y, yhat))),
        "MAE": float(mean_absolute_error(y, yhat)),
        "MAPE": safe_mape(y, yhat),
    }


def encode_with_saved_mapping(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    enc_path = MODELS / "cat_encoders.json"
    if enc_path.exists():
        mapping = json.loads(enc_path.read_text())
        for c in CAT_COLS:
            if c in out.columns:
                idx = {v: i for i, v in enumerate(mapping.get(c, []))}
                out[c] = out[c].astype(str).map(idx).fillna(-1).astype(int)
        return out
    for c in CAT_COLS:
        if c in out.columns:
            out[c] = out[c].astype("category").cat.codes
    return out


def get_stacking_predictions(df_feat: pd.DataFrame) -> np.ndarray | None:
    model_path = MODELS / "model_stacking.pkl"
    scaler_path = MODELS / "scaler_X.pkl"
    if not model_path.exists() or not scaler_path.exists():
        logger.warning("Stacking model or scaler not found. Running without teacher signal.")
        return None
    df_enc = encode_with_saved_mapping(df_feat)
    engineered = ["eta_log", "corr_severity_idx", "d_b_ratio", "eta_d_interaction", "reinf_index"]
    cols = [c for c in FEATURE_COLS if c in df_enc.columns] + [c for c in CAT_COLS if c in df_enc.columns] + [c for c in engineered if c in df_enc.columns]
    X = df_enc[cols].copy()
    scaler = joblib.load(scaler_path)
    if X.shape[1] != scaler.n_features_in_:
        X = X.iloc[:, : scaler.n_features_in_]
    Xs = scaler.transform(X)
    model = joblib.load(model_path)
    pred = np.asarray(model.predict(Xs), dtype=float)
    summary = RESULTSS / "for_part2" / "part1_summary.json"
    if summary.exists():
        try:
            if json.loads(summary.read_text()).get("log_transform", False):
                pred = np.expm1(pred)
        except Exception:
            pass
    return np.maximum(pred, 0.0)


def prepare_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray | None, Dict[str, np.ndarray], Dict[str, float]]:
    logger.info("Loading data, ACI baseline, and optional Stacking teacher.")
    df = engineer_features(clean_data(load_raw_data()))
    y = df[TARGET_COL].to_numpy(float)
    maci = np.maximum(compute_aci_predictions(df)["MACI_pred"].to_numpy(float), 1e-9)
    mstack = get_stacking_predictions(df)

    eta_pct = find_col(df, ["Mass Loss (Tensile bars), ηm (%)", "Mass Loss (Tensile bars), eta_m (%)", "Mass Loss", "eta_m", "eta"])
    rho_pct = find_col(df, ["Tension Reinforcement Ratio, pten (%)", "pten (%)", "rho_t", "rho"])
    d = find_col(df, ["Depth (mm)", "d (mm)", "depth"])
    b = find_col(df, ["Width (mm)", "b (mm)", "width"])
    fc = find_col(df, ["f'c (MPa)", "fc (MPa)", "fc"])
    fy = find_col(df, ["fy Longitudinal Bars (Tensile), (MPa) ", "fy Longitudinal Bars (Tensile), (MPa)", "fy (MPa)", "fy"])
    db = find_col(df, ["Diameter Tensile Bars, db,t (mm)", "db,t (mm)", "db_t", "db"])
    nbar = find_col(df, ["# Tensile Bars", "n_bars", "num_bars"], optional=True, default=2.0)
    csi = df["corr_severity_idx"].to_numpy(float) if "corr_severity_idx" in df.columns else np.zeros(len(df))
    ri = df["reinf_index"].to_numpy(float) if "reinf_index" in df.columns else np.zeros(len(df))

    eta = np.clip(eta_pct / 100.0, 0.0, 0.95)
    rho = np.clip(rho_pct / 100.0, 1e-9, None)
    lam = np.clip(d / np.maximum(b, 1e-9), 1e-9, None)
    delta = np.clip(db / np.maximum(d, 1e-9), 1e-9, None)
    phi = np.clip(fy / np.maximum(fc, 1e-9), 1e-9, None)
    As = nbar * math.pi * (db / 2.0) ** 2
    asr = np.clip(As / np.maximum(b * d, 1e-9), 1e-9, None)

    med = {
        "rho_med": float(np.median(rho)),
        "lambda_med": float(np.median(lam)),
        "delta_med": float(np.median(delta)),
        "phi_med": float(np.median(phi)),
        "as_ratio_med": float(np.median(asr)),
        "csi_med": float(max(np.median(np.abs(csi)), 1e-9)),
        "ri_med": float(max(np.median(np.abs(ri)), 1e-9)),
    }
    X = {
        "eta": eta,
        "rho_n": rho / max(med["rho_med"], 1e-9),
        "lambda_n": lam / max(med["lambda_med"], 1e-9),
        "delta_n": delta / max(med["delta_med"], 1e-9),
        "phi_n": phi / max(med["phi_med"], 1e-9),
        "as_ratio_n": asr / max(med["as_ratio_med"], 1e-9),
        "csi_n": csi / med["csi_med"],
        "ri_n": ri / med["ri_med"],
    }
    logger.info(f"Dataset n={len(y)}. Teacher available={mstack is not None}.")
    return y, maci, mstack, X, med


def make_terms(X: Dict[str, np.ndarray], y_ratio: np.ndarray, teacher_ratio: np.ndarray | None, top: int) -> List[Term]:
    names = [k for k in X.keys() if k != "eta"]
    raw: List[Term] = []
    for n in names:
        v = np.asarray(X[n], float)
        raw += [Term(n, v), Term(f"sqrt({n})", np.sqrt(np.maximum(v, 1e-9))), Term(f"log1p({n})", np.log1p(np.maximum(v, 0.0)))]
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            raw.append(Term(f"{a}*{b}", X[a] * X[b]))

    target = y_ratio.copy()
    if teacher_ratio is not None:
        target = 0.75 * target + 0.25 * teacher_ratio
    scored = []
    for t in raw:
        v = np.nan_to_num(t.values, nan=0.0, posinf=0.0, neginf=0.0)
        if np.std(v) < 1e-12:
            continue
        c = abs(float(np.corrcoef(v, target)[0, 1])) if np.std(target) > 1e-12 else 0.0
        scored.append((c, Term(t.name, v)))
    scored.sort(key=lambda z: z[0], reverse=True)
    out = [z[1] for z in scored[:top]]
    logger.info("Selected symbolic terms: " + ", ".join(t.name for t in out))
    return out


def design_matrix(terms: List[Term], idx: np.ndarray | None = None) -> np.ndarray:
    if idx is None:
        cols = [np.ones_like(terms[0].values)] + [t.values for t in terms]
    else:
        cols = [np.ones(len(idx))] + [t.values[idx] for t in terms]
    return np.vstack(cols).T.astype(float)


def unpack(theta: np.ndarray, K: int, P: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    q = P + 1
    off = 0
    A = theta[off : off + K * q].reshape(K, q); off += K * q
    B = theta[off : off + K * q].reshape(K, q); off += K * q
    G = theta[off : off + K].reshape(1, K)
    return A, B, G


def ratio_pred(theta: np.ndarray, K: int, T: np.ndarray, eta: np.ndarray) -> np.ndarray:
    A, B, G = unpack(theta, K, T.shape[1] - 1)
    alpha = softplus(T @ A.T)
    beta = softplus(T @ B.T)
    w = softmax(np.tile(G, (len(eta), 1)))
    expert = np.maximum(1.0 - eta[:, None], 1e-9) ** alpha * np.exp(-eta[:, None] * beta)
    return np.clip(np.sum(w * expert, axis=1), 0.0, 2.5)


def pred_moment(theta: np.ndarray, K: int, T: np.ndarray, eta: np.ndarray, maci: np.ndarray) -> np.ndarray:
    return np.maximum(ratio_pred(theta, K, T, eta) * maci, 0.0)


def residual(theta: np.ndarray, K: int, T: np.ndarray, eta: np.ndarray, y: np.ndarray, maci: np.ndarray, teacher: np.ndarray | None, teacher_w: float, l2: float) -> np.ndarray:
    yhat = pred_moment(theta, K, T, eta, maci)
    den = np.maximum(np.abs(y), np.maximum(np.mean(np.abs(y)) * 0.05, 1.0))
    r = [(yhat - y) / den]
    if teacher is not None and teacher_w > 0:
        rt = (yhat - teacher) / np.maximum(np.abs(teacher), np.maximum(np.mean(np.abs(teacher)) * 0.05, 1.0))
        r.append(np.sqrt(teacher_w) * rt)
    r.append(np.sqrt(l2) * theta)
    return np.concatenate(r)


def fit_model(K: int, T: np.ndarray, eta: np.ndarray, y: np.ndarray, maci: np.ndarray, teacher: np.ndarray | None, seed: int, mode: str, teacher_w: float) -> np.ndarray:
    n = 2 * K * T.shape[1] + K
    bounds = [(-3.5, 3.5)] * n
    maxiter = {"fast": 120, "publish": 350, "explore": 650}.get(mode, 350)
    popsize = {"fast": 8, "publish": 12, "explore": 16}.get(mode, 12)

    def obj(th: np.ndarray) -> float:
        r = residual(th, K, T, eta, y, maci, teacher, teacher_w, 1e-3)
        return float(np.mean(np.minimum(r * r, 4.0)))

    logger.info(f"Fit K={K}, params={n}, DE maxiter={maxiter}, popsize={popsize}")
    de = differential_evolution(obj, bounds, seed=seed, maxiter=maxiter, popsize=popsize, tol=1e-7, polish=False, workers=1)
    lo = np.array([b[0] for b in bounds], float); hi = np.array([b[1] for b in bounds], float)
    ls = least_squares(lambda th: residual(th, K, T, eta, y, maci, teacher, teacher_w, 1e-3), de.x, bounds=(lo, hi), loss="soft_l1", f_scale=0.20, max_nfev=30000)
    return ls.x.astype(float)


def physics_checks(theta: np.ndarray, K: int, T_med: np.ndarray) -> Dict[str, object]:
    eta = np.linspace(0.0, 0.95, 250)
    T = np.tile(T_med, (len(eta), 1))
    r = ratio_pred(theta, K, T, eta)
    return {
        "R_eta0": float(r[0]),
        "R_eta095": float(r[-1]),
        "monotonic_decreasing": bool(np.all(np.diff(r) <= 1e-10)),
        "non_negative": bool(np.all(r >= -1e-12)),
        "max_ratio_grid": float(np.max(r)),
        "finite_grid": bool(np.all(np.isfinite(r))),
    }


def score_candidate(m: Dict[str, float], tm: Dict[str, float], p: Dict[str, object], complexity: int, target_mape: float) -> Tuple[float, bool]:
    physics_ok = bool(p["finite_grid"] and p["monotonic_decreasing"] and p["non_negative"] and abs(float(p["R_eta0"]) - 1.0) < 1e-8 and float(p["max_ratio_grid"]) <= 2.5)
    publish = physics_ok and m["MAPE"] <= target_mape
    score = 0.45 * (m["MAPE"] / 100.0) + 0.20 * max(0.0, 1.0 - m["R2"]) + 0.15 * (m["RMSE"] / 100.0) + 0.10 * (tm.get("MAPE", 50.0) / 100.0) + 0.10 * (complexity / 100.0)
    if not physics_ok:
        score += 10.0
    if m["MAPE"] > target_mape:
        score += (m["MAPE"] - target_mape) / 10.0
    return float(score), bool(publish)


def evaluate(theta: np.ndarray, K: int, terms: List[Term], y: np.ndarray, maci: np.ndarray, teacher: np.ndarray | None, target_mape: float) -> CandidateResult:
    T = design_matrix(terms)
    eta = XGLOBAL["eta"]
    yhat = pred_moment(theta, K, T, eta, maci)
    m = metrics(y, yhat)
    tm = metrics(teacher, yhat) if teacher is not None else {"R2": 0.0, "RMSE": 0.0, "MAE": 0.0, "MAPE": 0.0}
    T_med = np.array([1.0] + [float(np.median(t.values)) for t in terms])
    p = physics_checks(theta, K, T_med)
    complexity = K * (2 * len(terms) + 3) + len(terms)
    sc, pub = score_candidate(m, tm, p, complexity, target_mape)
    return CandidateResult(K, [t.name for t in terms], theta.tolist(), m, tm, p, complexity, sc, pub)


def kfold_validate(K: int, terms: List[Term], y: np.ndarray, maci: np.ndarray, teacher: np.ndarray | None, seed: int, mode: str, teacher_w: float) -> Dict[str, object]:
    k = 3 if mode == "fast" else 5
    eta = XGLOBAL["eta"]
    T_all = design_matrix(terms)
    rows = []
    for fold, (tr, te) in enumerate(KFold(n_splits=k, shuffle=True, random_state=seed).split(y), start=1):
        th = fit_model(K, T_all[tr], eta[tr], y[tr], maci[tr], None if teacher is None else teacher[tr], seed + fold * 17, mode, teacher_w)
        yp = pred_moment(th, K, T_all[te], eta[te], maci[te])
        rows.append({"fold": fold, "test": metrics(y[te], yp)})
        logger.info(f"Fold {fold}: MAPE={rows[-1]['test']['MAPE']:.2f}% R2={rows[-1]['test']['R2']:.4f}")
    keys = ["R2", "RMSE", "MAE", "MAPE"]
    mean = {k: float(np.mean([r["test"][k] for r in rows])) for k in keys}
    std = {k: float(np.std([r["test"][k] for r in rows], ddof=1)) if len(rows) > 1 else 0.0 for k in keys}
    return {"folds": rows, "mean": mean, "std": std}


def equation_text(theta: np.ndarray, K: int, terms: List[Term], med: Dict[str, float]) -> str:
    A, B, G = unpack(np.asarray(theta), K, len(terms))
    lines = ["M_pred = M_ACI * R_c", "R_c = sum_k w_k * (1 - eta)^alpha_k * exp(-eta * beta_k)", "w_k = softmax(g_k)", "alpha_k = softplus(A_k)", "beta_k = softplus(B_k)", "softplus(z)=ln(1+exp(z))", ""]
    names = ["1"] + [t.name for t in terms]
    for k in range(K):
        def lin(coefs):
            return " + ".join([f"({coefs[i]:.8g})*{names[i]}" for i in range(len(names))])
        lines += [f"A_{k+1} = {lin(A[k])}", f"B_{k+1} = {lin(B[k])}", f"g_{k+1} = {G[0,k]:.8g}", ""]
    lines += ["Definitions:", "eta = mass_loss_percent / 100", f"rho_n = (rho_tension_percent/100) / {med['rho_med']:.10g}", f"lambda_n = (d/b) / {med['lambda_med']:.10g}", f"delta_n = (db_t/d) / {med['delta_med']:.10g}", f"phi_n = (fy/fc) / {med['phi_med']:.10g}", f"as_ratio_n = (As/(b*d)) / {med['as_ratio_med']:.10g}", f"csi_n = csi / {med['csi_med']:.10g}", f"ri_n = ri / {med['ri_med']:.10g}"]
    return "\n".join(lines) + "\n"


def save_outputs(best: CandidateResult, candidates: List[CandidateResult], kfold: Dict[str, object], med: Dict[str, float], target_mape: float) -> None:
    EQS.mkdir(parents=True, exist_ok=True); MODELS.mkdir(parents=True, exist_ok=True)
    (EQS / "hybrid_best_equation.txt").write_text(equation_text(np.asarray(best.theta), best.k, [Term(n, np.array([])) for n in best.terms], med))
    payload = {
        "approach": "Hybrid physics-constrained symbolic distillation",
        "equation_family": "M_pred = M_ACI * sum_k softmax(g_k) * (1-eta)^alpha_k * exp(-eta*beta_k)",
        "target_mape_percent": target_mape,
        "best": best.__dict__,
        "kfold_validation": kfold,
        "publication_gate": {
            "full_data_mape_le_target": best.metrics["MAPE"] <= target_mape,
            "kfold_mean_mape_le_target": kfold["mean"]["MAPE"] <= target_mape,
            "physics_pass": bool(best.physics["monotonic_decreasing"] and best.physics["non_negative"]),
            "publishable_candidate_found": bool(best.publishable and kfold["mean"]["MAPE"] <= target_mape),
        },
    }
    (MODELS / "hybrid_metrics.json").write_text(json.dumps(payload, indent=2))
    ranked = sorted([c.__dict__ for c in candidates], key=lambda z: z["score"])
    (MODELS / "hybrid_candidates_ranked.json").write_text(json.dumps(ranked, indent=2))
    logger.success(f"Equation -> {EQS / 'hybrid_best_equation.txt'}")
    logger.success(f"Metrics  -> {MODELS / 'hybrid_metrics.json'}")
    logger.success(f"Ranked   -> {MODELS / 'hybrid_candidates_ranked.json'}")


XGLOBAL: Dict[str, np.ndarray] = {}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["fast", "publish", "explore"], default="publish")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-k", type=int, default=3)
    p.add_argument("--top-terms", type=int, default=7)
    p.add_argument("--target-mape", type=float, default=15.0)
    p.add_argument("--teacher-weight", type=float, default=0.20)
    args = p.parse_args()
    setup_logger()

    y, maci, teacher, X, med = prepare_data()
    globals()["XGLOBAL"] = X
    y_ratio = y / np.maximum(maci, 1e-9)
    teacher_ratio = None if teacher is None else teacher / np.maximum(maci, 1e-9)
    terms_all = make_terms(X, y_ratio, teacher_ratio, args.top_terms)

    candidates: List[CandidateResult] = []
    term_counts = [min(4, len(terms_all)), min(6, len(terms_all)), len(terms_all)] if args.mode != "fast" else [min(4, len(terms_all))]
    for K in range(1, args.max_k + 1):
        for tc in sorted(set(term_counts)):
            terms = terms_all[:tc]
            T = design_matrix(terms)
            th = fit_model(K, T, X["eta"], y, maci, teacher, args.seed + 101 * K + tc, args.mode, args.teacher_weight)
            cand = evaluate(th, K, terms, y, maci, teacher, args.target_mape)
            candidates.append(cand)
            logger.info(f"Candidate K={K}, terms={tc}: MAPE={cand.metrics['MAPE']:.2f}% R2={cand.metrics['R2']:.4f} score={cand.score:.4f} publishable={cand.publishable}")

    best = sorted(candidates, key=lambda c: c.score)[0]
    best_terms = [t for t in terms_all if t.name in best.terms]
    kfold = kfold_validate(best.k, best_terms, y, maci, teacher, args.seed + 999, args.mode, args.teacher_weight)
    save_outputs(best, candidates, kfold, med, args.target_mape)
    if not (best.publishable and kfold["mean"]["MAPE"] <= args.target_mape):
        logger.warning("No strict publication-grade equation found under the requested MAPE gate. Best candidate was saved for inspection.")
    else:
        logger.success("Publication gate passed under the requested MAPE target.")


if __name__ == "__main__":
    main()
