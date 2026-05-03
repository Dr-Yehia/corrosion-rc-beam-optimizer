#!/usr/bin/env python3
"""
UHPC Nano-Silica Compressive Strength — ML Ensemble Pipeline
─────────────────────────────────────────────────────────────
Preprocessing : KNN Imputation (k=5) + Log-Transform + StandardScaler
Models        : CatBoost | XGBoost | RF | GBR | MLP → Stacking (Ridge meta)
HPO           : Optuna TPE — 5-fold CV objective
Gates         : L1 R²>0.90 | L2 R²≥0.982  (auto-retry 2.5× trials if L2 fails)
Analysis      : SHAP importance | NS dosage-response curve
Target        : 28-Day Compressive Strength (MPa)
Split         : 80 / 20  (stratified by CS quantile)
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import shap
from catboost import CatBoostRegressor
from sklearn.ensemble import (GradientBoostingRegressor,
                               RandomForestRegressor, StackingRegressor)
from sklearn.impute import KNNImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                              r2_score)
from sklearn.model_selection import KFold, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

# ── Configuration ──────────────────────────────────────────────────────────────
SEED          = 42
TEST_SIZE     = 0.20
L1_GATE       = 0.90
L2_GATE       = 0.982
TRIALS_BASE   = 60
TRIALS_RETRY  = 150
KNN_K         = 5
OUT           = Path("outputs")
OUT.mkdir(exist_ok=True)

DATA_PATHS = [
    "UHPC Dataset  (Version-2).xlsx",
    "Nano silica data/UHPC Dataset  (Version-2).xlsx",
    "/kaggle/input/uhpc-nano-silica/UHPC Dataset  (Version-2).xlsx",
    "/kaggle/working/UHPC Dataset  (Version-2).xlsx",
]

# Keywords for auto-detecting the target and NS columns
TARGET_KW = ["28d", "28-d", "cs28", "fc28", "f'c28", "compressive strength 28",
             "28 day", "28day", "28 d"]
NS_KW     = ["nano-sio2", "nanosio2", "nano sio2", "nano silica",
             "ns ", " ns", "nano_sio", "nsio2", "nano-si"]

CV = KFold(n_splits=5, shuffle=True, random_state=SEED)

# ── Utilities ──────────────────────────────────────────────────────────────────
def _mape(y: np.ndarray, yp: np.ndarray) -> float:
    return float(np.mean(np.abs((y - yp) / np.maximum(np.abs(y), 1e-9))) * 100)

def _a20(y: np.ndarray, yp: np.ndarray) -> float:
    ratio = yp / np.maximum(y, 1e-9)
    return float(np.mean((ratio >= 0.8) & (ratio <= 1.2)))

def report(y: np.ndarray, yp: np.ndarray, label: str = "") -> dict:
    m = dict(
        R2   = round(r2_score(y, yp), 4),
        MAE  = round(mean_absolute_error(y, yp), 3),
        RMSE = round(float(np.sqrt(mean_squared_error(y, yp))), 3),
        MAPE = round(_mape(y, yp), 2),
        a20  = round(_a20(y, yp), 4),
    )
    print(f"  {label:12s}  R²={m['R2']:.4f}  MAE={m['MAE']:.2f}  "
          f"RMSE={m['RMSE']:.2f}  MAPE={m['MAPE']:.2f}%  a20={m['a20']:.3f}")
    return m

def find_col(df: pd.DataFrame, keywords: list[str]) -> str | None:
    """Case-insensitive keyword search across column names."""
    cols_lower = {c.lower().replace(" ", ""): c for c in df.columns}
    for kw in keywords:
        kw_clean = kw.lower().replace(" ", "")
        for cl, orig in cols_lower.items():
            if kw_clean in cl:
                return orig
    return None

# ── Data Loading ───────────────────────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    for path in DATA_PATHS:
        p = Path(path)
        if p.exists():
            print(f"Loading: {p}")
            xf = pd.ExcelFile(p)
            for sheet in xf.sheet_names:
                df = xf.parse(sheet)
                if len(df) > 50:
                    print(f"  Sheet='{sheet}'  shape={df.shape}")
                    return df
    raise FileNotFoundError(
        "UHPC Excel file not found.\n"
        "Place 'UHPC Dataset  (Version-2).xlsx' in the working directory "
        "or add it as a Kaggle dataset input."
    )

def prepare(df: pd.DataFrame) -> tuple:
    target = find_col(df, TARGET_KW)
    ns_col = find_col(df, NS_KW)

    if target is None:
        # Fallback: any numeric column containing '28'
        cands = [c for c in df.select_dtypes(include=[np.number]).columns if "28" in c]
        target = cands[0] if cands else None
    if target is None:
        print("Available columns:", list(df.columns))
        raise ValueError("28-day CS column not found. Check TARGET_KW or column names above.")

    print(f"  Target column : '{target}'")
    print(f"  NS column     : '{ns_col}'")

    # Keep numeric only; drop columns with >60% missing
    df = df.select_dtypes(include=[np.number])
    df = df.loc[:, df.isnull().mean() < 0.60]
    df = df.dropna(subset=[target]).reset_index(drop=True)

    y     = df[target].to_numpy(float)
    X     = df.drop(columns=[target])
    feats = list(X.columns)
    ns_i  = feats.index(ns_col) if (ns_col and ns_col in feats) else None

    print(f"  Dataset ready : {X.shape[0]} samples × {X.shape[1]} features")
    return X.to_numpy(float), y, feats, ns_i, ns_col

# ── Optuna model factories ─────────────────────────────────────────────────────
def _cb(t):
    return CatBoostRegressor(
        iterations=t.suggest_int("n", 300, 1500),
        learning_rate=t.suggest_float("lr", 0.01, 0.30, log=True),
        depth=t.suggest_int("d", 4, 10),
        l2_leaf_reg=t.suggest_float("l2", 1.0, 10.0),
        random_seed=SEED, verbose=0,
    )

def _xgb(t):
    return XGBRegressor(
        n_estimators=t.suggest_int("n", 200, 1500),
        learning_rate=t.suggest_float("lr", 0.01, 0.30, log=True),
        max_depth=t.suggest_int("d", 3, 10),
        subsample=t.suggest_float("ss", 0.6, 1.0),
        colsample_bytree=t.suggest_float("cs", 0.6, 1.0),
        random_state=SEED, verbosity=0,
    )

def _rf(t):
    return RandomForestRegressor(
        n_estimators=t.suggest_int("n", 200, 800),
        max_depth=t.suggest_int("d", 5, 30),
        min_samples_split=t.suggest_int("mss", 2, 10),
        random_state=SEED, n_jobs=-1,
    )

def _gbr(t):
    return GradientBoostingRegressor(
        n_estimators=t.suggest_int("n", 200, 800),
        learning_rate=t.suggest_float("lr", 0.01, 0.20, log=True),
        max_depth=t.suggest_int("d", 3, 8),
        subsample=t.suggest_float("ss", 0.6, 1.0),
        random_state=SEED,
    )

def _mlp(t):
    n_layers = t.suggest_int("nl", 1, 3)
    layers   = tuple(t.suggest_int(f"h{i}", 64, 256) for i in range(n_layers))
    return MLPRegressor(
        hidden_layer_sizes=layers,
        alpha=t.suggest_float("a", 1e-5, 1e-2, log=True),
        max_iter=600, random_state=SEED,
    )

MAKERS = {"CatBoost": _cb, "XGBoost": _xgb, "RF": _rf, "GBR": _gbr, "MLP": _mlp}

def tune(name: str, maker, X: np.ndarray, y: np.ndarray, n_trials: int):
    def objective(trial):
        m = maker(trial)
        scores = [
            r2_score(y[va], m.fit(X[tr], y[tr]).predict(X[va]))
            for tr, va in CV.split(X)
        ]
        return float(np.mean(scores))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = maker(study.best_trial)
    best.fit(X, y)
    print(f"  {name:10s}  CV R²={study.best_value:.4f}")
    return best

# ── Core pipeline ──────────────────────────────────────────────────────────────
def run_pipeline(X_raw: np.ndarray, y_raw: np.ndarray,
                 feats: list[str], n_trials: int) -> dict:
    """Full train/test pipeline. Returns dict with all artifacts."""
    # 1 — Split (stratify by CS quantile to preserve distribution)
    q = pd.qcut(y_raw, q=5, labels=False, duplicates="drop")
    Xtr_r, Xte_r, ytr_r, yte_r = train_test_split(
        X_raw, y_raw, test_size=TEST_SIZE, random_state=SEED, stratify=q
    )

    # 2 — KNN Imputation (fit on train only — no leakage)
    imp = KNNImputer(n_neighbors=KNN_K)
    Xtr_i = imp.fit_transform(Xtr_r)
    Xte_i = imp.transform(Xte_r)

    # 3 — Log-transform target if right-skewed
    log_y = bool(pd.Series(ytr_r).skew() > 0.5)
    ytr = np.log1p(ytr_r) if log_y else ytr_r.copy()
    if log_y:
        print("  Log1p applied to target (skew detected)")

    # 4 — Standardize features
    sc = StandardScaler()
    Xtr = sc.fit_transform(Xtr_i)
    Xte = sc.transform(Xte_i)

    # 5 — Train & tune each base model
    print(f"\nOptuna HPO — {n_trials} trials × {len(MAKERS)} models")
    models = {name: tune(name, maker, Xtr, ytr, n_trials)
              for name, maker in MAKERS.items()}

    # 6 — Stacking (CatBoost + XGBoost + RF + GBR as estimators)
    stack = StackingRegressor(
        estimators=[(k, v) for k, v in models.items() if k != "MLP"],
        final_estimator=Ridge(alpha=1.0),
        cv=5, n_jobs=-1,
    )
    stack.fit(Xtr, ytr)
    models["Stacking"] = stack

    # 7 — Evaluate on test set
    print("\nTest Set Results:")
    results = {}
    for name, m in models.items():
        p_log = m.predict(Xte)
        p = np.expm1(p_log) if log_y else p_log
        results[name] = report(yte_r, p, label=name)

    return dict(models=models, results=results, Xtr=Xtr, Xte=Xte,
                Xtr_raw=Xtr_r, ytr_r=ytr_r, yte_r=yte_r,
                imp=imp, sc=sc, log_y=log_y)

# ── SHAP Analysis ──────────────────────────────────────────────────────────────
def run_shap(model, Xte: np.ndarray, feats: list[str],
             ns_col: str | None, model_name: str) -> tuple:
    print(f"\nSHAP Analysis — {model_name}")
    try:
        sv = shap.TreeExplainer(model).shap_values(Xte)
    except Exception:
        sv = shap.KernelExplainer(model.predict,
                                   shap.sample(Xte, 80)).shap_values(Xte)

    mean_sv = np.abs(sv).mean(axis=0)
    df_s = (pd.DataFrame({"feature": feats, "shap": mean_sv})
            .sort_values("shap", ascending=False)
            .reset_index(drop=True))

    print(df_s.head(10).to_string(index=False))

    ns_rank = None
    if ns_col and ns_col in df_s.feature.values:
        ns_rank = int(df_s[df_s.feature == ns_col].index[0]) + 1
        print(f"  → Nano Silica SHAP rank: #{ns_rank}")

    # Bar chart
    top = df_s.head(12)
    colors = ["#FF8C00" if f == ns_col else "#4682B4" for f in top.feature[::-1]]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top.feature[::-1], top.shap[::-1], color=colors)
    ax.set_xlabel("Mean |SHAP Value| (MPa)")
    ax.set_title(f"Feature Importance — {model_name}\n"
                 f"(Orange = Nano Silica  |  rank #{ns_rank})")
    plt.tight_layout()
    plt.savefig(OUT / "shap_bar.png", dpi=200)
    plt.close()

    # Beeswarm summary
    plt.figure()
    shap.summary_plot(sv, Xte, feature_names=feats, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(OUT / "shap_summary.png", dpi=200, bbox_inches="tight")
    plt.close()

    return df_s, ns_rank

# ── NS Dosage-Response Curve ───────────────────────────────────────────────────
def ns_curve(model, Xtr_raw: np.ndarray, ns_i: int | None,
             imp: KNNImputer, sc: StandardScaler,
             log_y: bool, ns_col: str | None) -> float | None:
    if ns_i is None:
        print("NS column not found — skipping dosage curve")
        return None

    print("\nNS Dosage-Response Curve")
    med_raw = np.nanmedian(Xtr_raw, axis=0)   # median of RAW train features
    ns_range = np.linspace(0, 200, 200)       # 0 → 200 kg/m³
    cs_vals  = []

    for v in ns_range:
        x = med_raw.copy()
        x[ns_i] = v
        x_imp = imp.transform(x.reshape(1, -1))
        x_sc  = sc.transform(x_imp)
        p = model.predict(x_sc)[0]
        cs_vals.append(float(np.expm1(p) if log_y else p))

    cs_arr  = np.array(cs_vals)
    opt_ns  = float(ns_range[np.argmax(cs_arr)])
    opt_cs  = float(cs_arr.max())
    print(f"  Optimal NS dosage : {opt_ns:.1f} kg/m³  →  CS = {opt_cs:.1f} MPa")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ns_range, cs_arr, "b-", lw=2.5)
    ax.axvline(opt_ns, color="r", ls="--",
               label=f"Optimal = {opt_ns:.1f} kg/m³  ({opt_cs:.1f} MPa)")
    ax.fill_between(ns_range, cs_arr.min(), cs_arr, alpha=0.08, color="blue")
    ax.set_xlabel("Nano Silica dosage (kg/m³)")
    ax.set_ylabel("Predicted 28d CS (MPa)")
    ax.set_title("NS Dosage-Response Curve\n"
                 "(all other features fixed at training median)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "ns_effect_curve.png", dpi=200)
    plt.close()
    return opt_ns

# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    # ── Load & prepare
    df = load_data()
    X_raw, y_raw, feats, ns_i, ns_col = prepare(df)

    # ── First training pass
    print(f"\n{'─'*60}")
    print("PASS 1 — Training")
    print('─'*60)
    art = run_pipeline(X_raw, y_raw, feats, TRIALS_BASE)

    best_name = max(art["results"], key=lambda k: art["results"][k]["R2"])
    best_r2   = art["results"][best_name]["R2"]
    l1 = best_r2 > L1_GATE
    l2 = best_r2 >= L2_GATE
    print(f"\nBest : {best_name}  R²={best_r2:.4f}")
    print(f"Gate L1 (>{L1_GATE}) : {'PASS ✅' if l1 else 'FAIL ❌'}")
    print(f"Gate L2 (≥{L2_GATE}) : {'PASS ✅' if l2 else 'FAIL ❌'}")

    # ── Retry if L2 not met
    if not l2:
        print(f"\n{'─'*60}")
        print(f"PASS 2 — Re-tuning ({TRIALS_RETRY} trials — 2.5× base)")
        print('─'*60)
        art = run_pipeline(X_raw, y_raw, feats, TRIALS_RETRY)
        best_name = max(art["results"], key=lambda k: art["results"][k]["R2"])
        best_r2   = art["results"][best_name]["R2"]
        l1 = best_r2 > L1_GATE
        l2 = best_r2 >= L2_GATE
        print(f"\nBest : {best_name}  R²={best_r2:.4f}")
        print(f"Gate L1 (>{L1_GATE}) : {'PASS ✅' if l1 else 'FAIL ❌'}")
        print(f"Gate L2 (≥{L2_GATE}) : {'PASS ✅' if l2 else 'FAIL ❌'}")

    # ── Select best tree model for SHAP (not stacking — TreeExplainer compatible)
    tree_order = ["CatBoost", "XGBoost", "RF", "GBR"]
    shap_name  = max((n for n in tree_order if n in art["results"]),
                     key=lambda k: art["results"][k]["R2"])
    shap_model = art["models"][shap_name]
    best_model = art["models"][best_name]

    # ── SHAP
    shap_df, ns_rank = run_shap(
        shap_model, art["Xte"], feats, ns_col, shap_name
    )

    # ── NS curve
    opt_ns = ns_curve(
        shap_model, art["Xtr_raw"], ns_i,
        art["imp"], art["sc"], art["log_y"], ns_col
    )

    # ── Predicted vs Actual scatter
    p_log = best_model.predict(art["Xte"])
    p     = np.expm1(p_log) if art["log_y"] else p_log
    lo, hi = min(art["yte_r"].min(), p.min()), max(art["yte_r"].max(), p.max())
    plt.figure(figsize=(6, 6))
    plt.scatter(art["yte_r"], p, s=18, alpha=0.55, edgecolors="none")
    plt.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="Perfect fit")
    plt.xlabel("Experimental CS 28d (MPa)")
    plt.ylabel("Predicted CS 28d (MPa)")
    plt.title(f"{best_name}  |  R²={best_r2:.4f}  |  "
              f"MAPE={art['results'][best_name]['MAPE']:.2f}%")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "scatter.png", dpi=200)
    plt.close()

    # ── Save metrics JSON
    out_data = {
        "best_model":       best_name,
        "gate_L1_pass":     l1,
        "gate_L2_pass":     l2,
        "publication_targets": {
            "R2": f">= {L2_GATE}", "MAE_MPa": "<= 4.0",
            "RMSE_MPa": "<= 4.5",  "MAPE_pct": "<= 2.5", "a20": "= 1.00",
        },
        "achieved":         art["results"][best_name],
        "all_models":       art["results"],
        "ns_optimal_kg_m3": opt_ns,
        "ns_shap_rank":     ns_rank,
        "shap_model_used":  shap_name,
    }
    (OUT / "metrics.json").write_text(json.dumps(out_data, indent=2))

    # ── Publication summary
    m = art["results"][best_name]
    sep = "═" * 60
    print(f"\n{sep}")
    print("  PUBLICATION SUMMARY")
    print(sep)
    print(f"  Best Model  : {best_name}")
    print(f"  R²          : {m['R2']:.4f}   {'✅' if m['R2']  >= L2_GATE else '❌'}  (target ≥ {L2_GATE})")
    print(f"  MAE         : {m['MAE']:.3f} MPa  {'✅' if m['MAE']  <= 4.0   else '❌'}  (target ≤ 4.0)")
    print(f"  RMSE        : {m['RMSE']:.3f} MPa  {'✅' if m['RMSE'] <= 4.5   else '❌'}  (target ≤ 4.5)")
    print(f"  MAPE        : {m['MAPE']:.2f}%    {'✅' if m['MAPE'] <= 2.5   else '❌'}  (target ≤ 2.5%)")
    print(f"  a20-index   : {m['a20']:.4f}    {'✅' if m['a20']  >= 1.0   else '❌'}  (target = 1.00)")
    print(f"  Gate L1     : {'PASS ✅' if l1 else 'FAIL ❌'}   (R² > {L1_GATE})")
    print(f"  Gate L2     : {'PASS ✅' if l2 else 'FAIL ❌'}   (R² ≥ {L2_GATE})")
    if ns_rank:
        print(f"  NS SHAP rank: #{ns_rank}  {'✅ in top-5' if ns_rank <= 5 else '⚠ not in top-5'}")
    if opt_ns:
        print(f"  Optimal NS  : {opt_ns:.1f} kg/m³")
    print(sep)
    print(f"  Outputs → {OUT}/")
    print(f"    metrics.json | scatter.png | shap_bar.png | "
          f"shap_summary.png | ns_effect_curve.png")


if __name__ == "__main__":
    main()
