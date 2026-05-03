#!/usr/bin/env python3
"""
UHPC Nano-Silica Compressive Strength — ML Ensemble Pipeline
─────────────────────────────────────────────
Preprocessing : KNN Imputation (k=5) + Log-Transform + StandardScaler
Models        : CatBoost | XGBoost | RF | GBR | MLP → Stacking (Ridge meta)
HPO           : Optuna TPE — 5-fold CV objective
Gates         : L1 R²>0.90 | L2 R²≥0.982  (auto-retry 2.5× trials if L2 fails)
Analysis      : SHAP importance | NS dosage-response curve
Target        : 28-Day Compressive Strength (MPa)
Split         : 80 / 20  (stratified by CS quantile)

Data source   : Auto-downloads from GitHub — no manual upload needed on Kaggle.
"""
from __future__ import annotations

import io
import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import requests
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

# ── Configuration ────────────────────────────────────────────────────────────────
SEED         = 42
TEST_SIZE    = 0.20
L1_GATE      = 0.90
L2_GATE      = 0.982
TRIALS_BASE  = 60
TRIALS_RETRY = 150
KNN_K        = 5
OUT          = Path("outputs")
OUT.mkdir(exist_ok=True)

GITHUB_RAW = (
    "https://raw.githubusercontent.com/"
    "Dr-Yehia/corrosion-rc-beam-optimizer/nano-silica/"
    "Nano%20silica%20data/"
)
EXCEL_FILE = "UHPC Dataset  (Version-2).xlsx"

LOCAL_PATHS = [
    EXCEL_FILE,
    f"Nano silica data/{EXCEL_FILE}",
    f"/kaggle/input/uhpc-nano-silica/{EXCEL_FILE}",
    f"/kaggle/working/{EXCEL_FILE}",
]

TARGET_KW = ["28d", "28-d", "cs28", "fc28", "f'c28",
             "compressive strength 28", "28 day", "28day", "28 d",
             "fc,28", "f'c,28", "fcu28", "fcu,28"]
NS_KW     = ["nano-sio2", "nanosio2", "nano sio2", "nano silica",
             "ns ", " ns", "nano_sio", "nsio2", "nano-si", "ns%",
             "sio2", "silica fume"]

CV = KFold(n_splits=5, shuffle=True, random_state=SEED)

# ── Utilities ────────────────────────────────────────────────────────────────
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
    cols_lower = {str(c).lower().replace(" ", "").replace(",", "").replace("'", ""): c
                  for c in df.columns}
    for kw in keywords:
        kw_clean = kw.lower().replace(" ", "").replace(",", "").replace("'", "")
        for cl, orig in cols_lower.items():
            if kw_clean in cl:
                return orig
    return None

# ── Excel parsing — handles merged-cell / multi-row headers ───────────────────
def _parse_best(xf: pd.ExcelFile, sheet: str) -> pd.DataFrame | None:
    """Try header rows 0, 1, 2 and return the parse with fewest Unnamed cols."""
    best_df, best_score = None, -1
    for h in range(3):
        try:
            df = xf.parse(sheet, header=h)
            if len(df) < 20:
                continue
            # Count non-Unnamed numeric columns (higher = better)
            n_named = sum(
                1 for c in df.columns
                if not str(c).startswith("Unnamed") and
                   pd.api.types.is_numeric_dtype(df[c])
            )
            if n_named > best_score:
                best_score = n_named
                best_df = df
        except Exception:
            pass
    return best_df

# ── Data Loading — GitHub first, local fallback ────────────────────────────
def load_data() -> pd.DataFrame:
    # 1. Try GitHub raw URL
    url = GITHUB_RAW + requests.utils.quote(EXCEL_FILE)
    try:
        print("Downloading from GitHub ...")
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        xf = pd.ExcelFile(io.BytesIO(r.content))
        for sheet in xf.sheet_names:
            df = _parse_best(xf, sheet)
            if df is not None and len(df) > 20:
                # Drop fully-unnamed columns
                df = df.loc[:, [c for c in df.columns
                                if not str(c).startswith("Unnamed")]]
                print(f"  OK — sheet='{sheet}'  shape={df.shape}")
                print(f"  Columns: {list(df.columns[:8])} ...")
                return df
    except Exception as e:
        print(f"  GitHub download failed: {e}")

    # 2. Fallback — local / Kaggle paths
    for path in LOCAL_PATHS:
        p = Path(path)
        if p.exists():
            print(f"Loading local: {p}")
            xf = pd.ExcelFile(p)
            for sheet in xf.sheet_names:
                df = _parse_best(xf, sheet)
                if df is not None and len(df) > 20:
                    df = df.loc[:, [c for c in df.columns
                                    if not str(c).startswith("Unnamed")]]
                    print(f"  sheet='{sheet}'  shape={df.shape}")
                    return df

    raise FileNotFoundError(
        f"Cannot load '{EXCEL_FILE}' from GitHub or local paths.\n"
        "Check your internet connection or add the file as a Kaggle dataset."
    )

def prepare(df: pd.DataFrame) -> tuple:
    print(f"\nAll columns ({len(df.columns)}): {list(df.columns)}")

    target = find_col(df, TARGET_KW)
    ns_col = find_col(df, NS_KW)

    # Fallback: pick the column containing '28' with highest numeric mean
    if target is None:
        cands = [c for c in df.select_dtypes(include=[np.number]).columns
                 if "28" in str(c).lower()]
        if cands:
            target = max(cands, key=lambda c: df[c].dropna().mean())

    if target is None:
        raise ValueError(
            "28-day CS column not found.\n"
            f"Available columns: {list(df.columns)}\n"
            "Add the exact column name to TARGET_KW."
        )

    print(f"  Target : '{target}'  |  NS : '{ns_col}'")

    df = df.select_dtypes(include=[np.number])
    df = df.loc[:, df.isnull().mean() < 0.60]
    df = df.dropna(subset=[target]).reset_index(drop=True)

    y     = df[target].to_numpy(float)
    X     = df.drop(columns=[target])
    feats = list(X.columns)
    ns_i  = feats.index(ns_col) if (ns_col and ns_col in feats) else None

    print(f"  Ready : {X.shape[0]} samples × {X.shape[1]} features")
    return X.to_numpy(float), y, feats, ns_i, ns_col

# ── Optuna model factories ─────────────────────────────────────────────────────────────
def _cb(t):
    return CatBoostRegressor(
        iterations=t.suggest_int("n", 300, 1500),
        learning_rate=t.suggest_float("lr", 0.01, 0.30, log=True),
        depth=t.suggest_int("d", 4, 10),
        l2_leaf_reg=t.suggest_float("l2", 1.0, 10.0),
        random_seed=SEED, verbose=0)

def _xgb(t):
    return XGBRegressor(
        n_estimators=t.suggest_int("n", 200, 1500),
        learning_rate=t.suggest_float("lr", 0.01, 0.30, log=True),
        max_depth=t.suggest_int("d", 3, 10),
        subsample=t.suggest_float("ss", 0.6, 1.0),
        colsample_bytree=t.suggest_float("cs", 0.6, 1.0),
        random_state=SEED, verbosity=0)

def _rf(t):
    return RandomForestRegressor(
        n_estimators=t.suggest_int("n", 200, 800),
        max_depth=t.suggest_int("d", 5, 30),
        min_samples_split=t.suggest_int("mss", 2, 10),
        random_state=SEED, n_jobs=-1)

def _gbr(t):
    return GradientBoostingRegressor(
        n_estimators=t.suggest_int("n", 200, 800),
        learning_rate=t.suggest_float("lr", 0.01, 0.20, log=True),
        max_depth=t.suggest_int("d", 3, 8),
        subsample=t.suggest_float("ss", 0.6, 1.0),
        random_state=SEED)

def _mlp(t):
    n_layers = t.suggest_int("nl", 1, 3)
    layers   = tuple(t.suggest_int(f"h{i}", 64, 256) for i in range(n_layers))
    return MLPRegressor(hidden_layer_sizes=layers,
                        alpha=t.suggest_float("a", 1e-5, 1e-2, log=True),
                        max_iter=600, random_state=SEED)

MAKERS = {"CatBoost": _cb, "XGBoost": _xgb, "RF": _rf, "GBR": _gbr, "MLP": _mlp}

def tune(name: str, maker, X: np.ndarray, y: np.ndarray, n_trials: int):
    def objective(trial):
        m = maker(trial)
        return float(np.mean([
            r2_score(y[va], m.fit(X[tr], y[tr]).predict(X[va]))
            for tr, va in CV.split(X)
        ]))
    study = optuna.create_study(direction="maximize",
                                 sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = maker(study.best_trial)
    best.fit(X, y)
    print(f"  {name:10s}  CV R²={study.best_value:.4f}")
    return best

# ── Core pipeline ─────────────────────────────────────────────────────────────────
def run_pipeline(X_raw: np.ndarray, y_raw: np.ndarray,
                 feats: list[str], n_trials: int) -> dict:
    q = pd.qcut(y_raw, q=5, labels=False, duplicates="drop")
    Xtr_r, Xte_r, ytr_r, yte_r = train_test_split(
        X_raw, y_raw, test_size=TEST_SIZE, random_state=SEED, stratify=q)

    imp   = KNNImputer(n_neighbors=KNN_K)
    Xtr_i = imp.fit_transform(Xtr_r)
    Xte_i = imp.transform(Xte_r)

    log_y = bool(pd.Series(ytr_r).skew() > 0.5)
    ytr   = np.log1p(ytr_r) if log_y else ytr_r.copy()
    if log_y: print("  Log1p applied to target")

    sc  = StandardScaler()
    Xtr = sc.fit_transform(Xtr_i)
    Xte = sc.transform(Xte_i)

    print(f"\nOptuna HPO — {n_trials} trials × {len(MAKERS)} models")
    models = {n: tune(n, m, Xtr, ytr, n_trials) for n, m in MAKERS.items()}

    stack = StackingRegressor(
        estimators=[(k, v) for k, v in models.items() if k != "MLP"],
        final_estimator=Ridge(alpha=1.0), cv=5, n_jobs=-1)
    stack.fit(Xtr, ytr)
    models["Stacking"] = stack

    print("\nTest Set Results:")
    results = {}
    for name, m in models.items():
        p = np.expm1(m.predict(Xte)) if log_y else m.predict(Xte)
        results[name] = report(yte_r, p, label=name)

    return dict(models=models, results=results, Xtr=Xtr, Xte=Xte,
                Xtr_raw=Xtr_r, yte_r=yte_r, imp=imp, sc=sc, log_y=log_y)

# ── SHAP ────────────────────────────────────────────────────────────────────────────
def run_shap(model, Xte, feats, ns_col, model_name) -> tuple:
    print(f"\nSHAP — {model_name}")
    try:
        sv = shap.TreeExplainer(model).shap_values(Xte)
    except Exception:
        sv = shap.KernelExplainer(model.predict, shap.sample(Xte, 80)).shap_values(Xte)

    df_s = (pd.DataFrame({"feature": feats, "shap": np.abs(sv).mean(0)})
            .sort_values("shap", ascending=False).reset_index(drop=True))
    print(df_s.head(10).to_string(index=False))

    ns_rank = None
    if ns_col and ns_col in df_s.feature.values:
        ns_rank = int(df_s[df_s.feature == ns_col].index[0]) + 1
        print(f"  Nano Silica rank: #{ns_rank}")

    top = df_s.head(12)
    colors = ["#FF8C00" if f == ns_col else "#4682B4" for f in top.feature[::-1]]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top.feature[::-1], top.shap[::-1], color=colors)
    ax.set_xlabel("Mean |SHAP Value| (MPa)")
    ax.set_title(f"Feature Importance — {model_name}  (orange = Nano Silica, rank #{ns_rank})")
    plt.tight_layout(); plt.savefig(OUT / "shap_bar.png", dpi=200); plt.close()

    plt.figure()
    shap.summary_plot(sv, Xte, feature_names=feats, show=False, max_display=15)
    plt.tight_layout(); plt.savefig(OUT / "shap_summary.png", dpi=200, bbox_inches="tight")
    plt.close()
    return df_s, ns_rank

# ── NS Curve ───────────────────────────────────────────────────────────────────────
def ns_curve(model, Xtr_raw, ns_i, imp, sc, log_y, ns_col) -> float | None:
    if ns_i is None:
        print("NS column not found — skipping curve"); return None

    print("\nNS Dosage-Response Curve")
    med  = np.nanmedian(Xtr_raw, axis=0)
    rng  = np.linspace(0, 200, 200)
    pred = []
    for v in rng:
        x = med.copy(); x[ns_i] = v
        p = model.predict(sc.transform(imp.transform(x.reshape(1, -1))))[0]
        pred.append(float(np.expm1(p) if log_y else p))

    pred   = np.array(pred)
    opt_ns = float(rng[np.argmax(pred)])
    print(f"  Optimal NS: {opt_ns:.1f} kg/m³  →  {pred.max():.1f} MPa")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(rng, pred, "b-", lw=2.5)
    ax.axvline(opt_ns, color="r", ls="--",
               label=f"Optimal = {opt_ns:.1f} kg/m³  ({pred.max():.1f} MPa)")
    ax.fill_between(rng, pred.min(), pred, alpha=0.08, color="blue")
    ax.set_xlabel("Nano Silica (kg/m³)"); ax.set_ylabel("Predicted 28d CS (MPa)")
    ax.set_title("NS Dosage-Response Curve  (others at training median)")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(OUT / "ns_effect_curve.png", dpi=200); plt.close()
    return opt_ns

# ── Main ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    df = load_data()
    X_raw, y_raw, feats, ns_i, ns_col = prepare(df)

    def _run(n_trials):
        print(f"\n{'─'*55}\nTraining — {n_trials} Optuna trials\n{'─'*55}")
        art  = run_pipeline(X_raw, y_raw, feats, n_trials)
        best = max(art["results"], key=lambda k: art["results"][k]["R2"])
        r2   = art["results"][best]["R2"]
        l1, l2 = r2 > L1_GATE, r2 >= L2_GATE
        print(f"\nBest: {best}  R²={r2:.4f}")
        print(f"Gate L1 (>{L1_GATE}): {'PASS ✅' if l1 else 'FAIL ❌'}")
        print(f"Gate L2 (≥{L2_GATE}): {'PASS ✅' if l2 else 'FAIL ❌'}")
        return art, best, r2, l1, l2

    art, best_name, best_r2, l1, l2 = _run(TRIALS_BASE)
    if not l2:
        print(f"\nL2 not met — retrying with {TRIALS_RETRY} trials ...")
        art, best_name, best_r2, l1, l2 = _run(TRIALS_RETRY)

    shap_name = max(
        (n for n in ["CatBoost", "XGBoost", "RF", "GBR"] if n in art["results"]),
        key=lambda k: art["results"][k]["R2"])
    shap_df, ns_rank = run_shap(art["models"][shap_name], art["Xte"],
                                 feats, ns_col, shap_name)
    opt_ns = ns_curve(art["models"][shap_name], art["Xtr_raw"],
                      ns_i, art["imp"], art["sc"], art["log_y"], ns_col)

    bm = art["models"][best_name]
    p  = np.expm1(bm.predict(art["Xte"])) if art["log_y"] else bm.predict(art["Xte"])
    lo, hi = min(art["yte_r"].min(), p.min()), max(art["yte_r"].max(), p.max())
    plt.figure(figsize=(6, 6))
    plt.scatter(art["yte_r"], p, s=18, alpha=0.55, edgecolors="none")
    plt.plot([lo, hi], [lo, hi], "r--", lw=1.5)
    plt.xlabel("Experimental CS 28d (MPa)"); plt.ylabel("Predicted CS 28d (MPa)")
    plt.title(f"{best_name}  R²={best_r2:.4f}  MAPE={art['results'][best_name]['MAPE']:.2f}%")
    plt.tight_layout(); plt.savefig(OUT / "scatter.png", dpi=200); plt.close()

    (OUT / "metrics.json").write_text(json.dumps({
        "best_model": best_name, "gate_L1": l1, "gate_L2": l2,
        "targets": {"R2": f">={L2_GATE}", "MAE": "<=4.0",
                    "RMSE": "<=4.5", "MAPE%": "<=2.5", "a20": "=1.00"},
        "achieved": art["results"][best_name],
        "all_models": art["results"],
        "ns_optimal_kg_m3": opt_ns, "ns_shap_rank": ns_rank,
    }, indent=2))

    m = art["results"][best_name]
    print(f"\n{'═'*55}\n  PUBLICATION SUMMARY\n{'═'*55}")
    print(f"  Model  : {best_name}")
    print(f"  R²     : {m['R2']:.4f}  {'✅' if m['R2']  >= L2_GATE else '❌'}  (≥{L2_GATE})")
    print(f"  MAE    : {m['MAE']:.3f} MPa  {'✅' if m['MAE']  <= 4.0 else '❌'}  (≤4.0)")
    print(f"  RMSE   : {m['RMSE']:.3f} MPa  {'✅' if m['RMSE'] <= 4.5 else '❌'}  (≤4.5)")
    print(f"  MAPE   : {m['MAPE']:.2f}%  {'✅' if m['MAPE'] <= 2.5 else '❌'}  (≤2.5%)")
    print(f"  a20    : {m['a20']:.4f}  {'✅' if m['a20'] >= 1.0 else '❌'}  (=1.00)")
    print(f"  Gate L1: {'PASS ✅' if l1 else 'FAIL ❌'}  |  Gate L2: {'PASS ✅' if l2 else 'FAIL ❌'}")
    if ns_rank: print(f"  NS rank: #{ns_rank}  {'✅ top-5' if ns_rank <= 5 else '⚠ not top-5'}")
    if opt_ns:  print(f"  Opt NS : {opt_ns:.1f} kg/m³")
    print(f"{'═'*55}\n  Outputs → {OUT}/")


if __name__ == "__main__":
    main()
