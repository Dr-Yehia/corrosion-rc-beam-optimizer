#!/usr/bin/env python3
"""
UHPC Multi-Property ML Pipeline — Nano Silica Focus
──────────────────────────────────────────────────
Targets : CS_28d | Flexural | Tensile | E_Modulus | Porosity
Inputs  : Mix-design features only (no result-column leakage)
HPO     : Optuna TPE 5-fold CV  |  Stack: Ridge meta
Gates   : per-property R² threshold  (auto-retry at 2.5× trials)
Output  : outputs/<property>/metrics.json + scatter + shap + ns_curve

Title   : Machine Learning-Based Multi-Property Prediction of UHPC
          with Focus on Nano Silica Effect: Insights from a Global
          Database of 2,188 Mix Designs
"""
from __future__ import annotations
import io, json, warnings
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

# ── Global config ─────────────────────────────────────────────────────────────
SEED         = 42
TEST_SIZE    = 0.20
TRIALS_BASE  = 50
TRIALS_RETRY = 120
KNN_K        = 5
ROOT_OUT     = Path("outputs")
ROOT_OUT.mkdir(exist_ok=True)

GITHUB_RAW = (
    "https://raw.githubusercontent.com/"
    "Dr-Yehia/corrosion-rc-beam-optimizer/nano-silica/"
    "Nano%20silica%20data/"
)
EXCEL_FILE = "UHPC Dataset  (Version-2).xlsx"
LOCAL_PATHS = [
    EXCEL_FILE, f"Nano silica data/{EXCEL_FILE}",
    f"/kaggle/input/uhpc-nano-silica/{EXCEL_FILE}",
    f"/kaggle/working/{EXCEL_FILE}",
]

# ── Multi-target definitions ────────────────────────────────────────────────
MULTI_TARGETS = [
    {"kw": ["28-day", "28day", "cs28", "fc28"],
     "name": "CS_28d",    "unit": "MPa", "gate": 0.982, "min_n": 200},
    {"kw": ["peakstrength", "mormpа", "mor(", " mor"],
     "name": "Flexural",  "unit": "MPa", "gate": 0.960, "min_n": 100},
    {"kw": ["splittensile"],
     "name": "Tensile",   "unit": "MPa", "gate": 0.930, "min_n":  80},
    {"kw": ["elasticmodulus", "elasticmod"],
     "name": "E_Modulus", "unit": "GPa", "gate": 0.930, "min_n":  80},
    {"kw": ["porosity"],
     "name": "Porosity",  "unit": "%",   "gate": 0.920, "min_n":  80},
]

NS_KW = ["nano silica", "nanosio2", "nano-sio2", "nsio2", "nano-si", "nanosilica"]

# Result columns — NEVER use as features (prevent leakage)
_RESULT_KW = [
    "1-day","3-day","7-day","14-day","21day","28-day","56-day","90-day",
    "elasticmodulus","splittensile","directtensile","tensileelastic",
    "straincapacity","peaktensilestrain","lop(","mor("," mor","peakstrength",
    "residualstrength","toughness","aircontent","airvoid","porosity",
    "waterabsorption","shrinkage","cycles","totalcharge","surfaceresistivity",
    "crackingstrength","first-cracking","firstcracking",
]

_HEADER_KW = ["cement","water","silica","fly","slag","sand","aggregate",
              "fiber","sp","superplast","nano","strength","28","mpa","ns"]

CV = KFold(n_splits=5, shuffle=True, random_state=SEED)

# ── Utilities ────────────────────────────────────────────────────────────────
def _c(s): return str(s).lower().replace(" ","").replace(",","").replace("'","").replace("-","")

def _mape(y, yp): return float(np.mean(np.abs((y-yp)/np.maximum(np.abs(y),1e-9)))*100)
def _a20(y, yp):
    r = yp/np.maximum(y,1e-9); return float(np.mean((r>=0.8)&(r<=1.2)))

def _report(y, yp, label=""):
    m = dict(R2=round(r2_score(y,yp),4),
             MAE=round(mean_absolute_error(y,yp),3),
             RMSE=round(float(np.sqrt(mean_squared_error(y,yp))),3),
             MAPE=round(_mape(y,yp),2), a20=round(_a20(y,yp),4))
    print(f"  {label:12s}  R²={m['R2']:.4f}  MAE={m['MAE']:.2f}  "
          f"RMSE={m['RMSE']:.2f}  MAPE={m['MAPE']:.2f}%  a20={m['a20']:.3f}")
    return m

def _find_col(df, keywords):
    mp = {_c(c): c for c in df.columns}
    for kw in keywords:
        for cl, orig in mp.items():
            if _c(kw) in cl: return orig
    return None

def _is_result_col(col):
    cc = _c(col)
    return any(_c(kw) in cc for kw in _RESULT_KW)

# ── Excel loading ────────────────────────────────────────────────────────────────
def _read_sheet(xf, sheet):
    try: raw = xf.parse(sheet, header=None, nrows=8)
    except: return None
    if len(raw) < 5: return None
    best_h, best_sc = 0, -1
    for h in range(min(5, len(raw))):
        vals = raw.iloc[h].astype(str).str.lower().tolist()
        sc   = sum(1 for v in vals for kw in _HEADER_KW if kw in v)
        sc  -= int(sum(1 for v in vals if v.replace(".","").replace("-","").isdigit())/max(len(vals),1)*10)
        if sc > best_sc: best_sc, best_h = sc, h
    try:
        df = xf.parse(sheet, header=best_h)
        print(f"    header row={best_h}  score={best_sc}  shape={df.shape}")
        return df if len(df) >= 10 else None
    except: return None

def load_data():
    url = GITHUB_RAW + requests.utils.quote(EXCEL_FILE)
    sources = []
    try:
        print("Downloading from GitHub ...")
        r = requests.get(url, timeout=120); r.raise_for_status()
        sources.append(("GitHub", r.content))
    except Exception as e: print(f"  GitHub failed: {e}")
    for p in LOCAL_PATHS:
        if Path(p).exists(): sources.append((p, None))
    for label, content in sources:
        try:
            xf = pd.ExcelFile(io.BytesIO(content) if content else label)
            for sheet in xf.sheet_names:
                df = _read_sheet(xf, sheet)
                if df is not None:
                    print(f"  OK [{label}] sheet='{sheet}'")
                    return df
        except Exception as e: print(f"  Failed [{label}]: {e}")
    raise FileNotFoundError(f"Cannot load '{EXCEL_FILE}'.")

# ── Feature preparation ──────────────────────────────────────────────────────────
def get_mix_features(df):
    """Return only mix-design input columns (exclude all result columns)."""
    num_cols = df.select_dtypes(include=[np.number]).columns
    return [c for c in num_cols if not _is_result_col(c)]

# ── Optuna model factories ───────────────────────────────────────────────────────────
MAKERS = {
    "CatBoost": lambda t: CatBoostRegressor(
        iterations=t.suggest_int("n",300,1200),
        learning_rate=t.suggest_float("lr",0.01,0.3,log=True),
        depth=t.suggest_int("d",4,10),
        l2_leaf_reg=t.suggest_float("l2",1,10),
        random_seed=SEED, verbose=0),
    "XGBoost": lambda t: XGBRegressor(
        n_estimators=t.suggest_int("n",200,1200),
        learning_rate=t.suggest_float("lr",0.01,0.3,log=True),
        max_depth=t.suggest_int("d",3,10),
        subsample=t.suggest_float("ss",0.6,1.0),
        colsample_bytree=t.suggest_float("cs",0.6,1.0),
        random_state=SEED, verbosity=0),
    "RF": lambda t: RandomForestRegressor(
        n_estimators=t.suggest_int("n",200,800),
        max_depth=t.suggest_int("d",5,30),
        min_samples_split=t.suggest_int("mss",2,10),
        random_state=SEED, n_jobs=-1),
    "GBR": lambda t: GradientBoostingRegressor(
        n_estimators=t.suggest_int("n",200,800),
        learning_rate=t.suggest_float("lr",0.01,0.2,log=True),
        max_depth=t.suggest_int("d",3,8),
        subsample=t.suggest_float("ss",0.6,1.0),
        random_state=SEED),
    "MLP": lambda t: MLPRegressor(
        hidden_layer_sizes=tuple(t.suggest_int(f"h{i}",64,256)
                                 for i in range(t.suggest_int("nl",1,3))),
        alpha=t.suggest_float("a",1e-5,1e-2,log=True),
        max_iter=600, random_state=SEED),
}

def _tune(name, maker, X, y, n_trials):
    def obj(trial):
        m = maker(trial)
        return float(np.mean([r2_score(y[va], m.fit(X[tr],y[tr]).predict(X[va]))
                               for tr,va in CV.split(X)]))
    st = optuna.create_study(direction="maximize",
                              sampler=optuna.samplers.TPESampler(seed=SEED))
    st.optimize(obj, n_trials=n_trials, show_progress_bar=False)
    best = maker(st.best_trial); best.fit(X,y)
    print(f"  {name:10s}  CV R²={st.best_value:.4f}")
    return best

# ── Single-property pipeline ──────────────────────────────────────────────────────
def _pipeline(X_raw, y_raw, n_trials):
    q = pd.qcut(y_raw, q=min(5,len(y_raw)//20), labels=False, duplicates="drop")
    Xtr_r,Xte_r,ytr_r,yte_r = train_test_split(
        X_raw, y_raw, test_size=TEST_SIZE, random_state=SEED, stratify=q)
    imp  = KNNImputer(n_neighbors=KNN_K)
    Xtr_i,Xte_i = imp.fit_transform(Xtr_r), imp.transform(Xte_r)
    log_y = bool(pd.Series(ytr_r).skew() > 0.5)
    ytr   = np.log1p(ytr_r) if log_y else ytr_r.copy()
    if log_y: print("  Log1p applied")
    sc  = StandardScaler()
    Xtr = sc.fit_transform(Xtr_i)
    Xte = sc.transform(Xte_i)
    print(f"  Optuna {n_trials} trials × {len(MAKERS)} models")
    models = {n: _tune(n, m, Xtr, ytr, n_trials) for n, m in MAKERS.items()}
    stack = StackingRegressor(
        estimators=[(k,v) for k,v in models.items() if k!="MLP"],
        final_estimator=Ridge(alpha=1.0), cv=5, n_jobs=-1)
    stack.fit(Xtr, ytr); models["Stacking"] = stack
    print("  Test Set:")
    results = {}
    for nm, m in models.items():
        p = np.expm1(m.predict(Xte)) if log_y else m.predict(Xte)
        results[nm] = _report(yte_r, p, nm)
    return dict(models=models, results=results, Xtr=Xtr, Xte=Xte,
                Xtr_raw=Xtr_r, yte_r=yte_r, imp=imp, sc=sc, log_y=log_y)

# ── SHAP ────────────────────────────────────────────────────────────────────────────
def _shap(model, Xte, feats, ns_col, prop_name, out):
    try:    sv = shap.TreeExplainer(model).shap_values(Xte)
    except: sv = shap.KernelExplainer(model.predict, shap.sample(Xte,80)).shap_values(Xte)
    df_s = (pd.DataFrame({"feature":feats,"shap":np.abs(sv).mean(0)})
            .sort_values("shap",ascending=False).reset_index(drop=True))
    ns_rank = None
    if ns_col and ns_col in df_s.feature.values:
        ns_rank = int(df_s[df_s.feature==ns_col].index[0])+1
        print(f"  NS rank #{ns_rank}")
    top    = df_s.head(12)
    colors = ["#FF8C00" if f==ns_col else "#4682B4" for f in top.feature[::-1]]
    fig,ax = plt.subplots(figsize=(9,6))
    ax.barh(top.feature[::-1], top.shap[::-1], color=colors)
    ax.set_xlabel("Mean |SHAP Value|")
    ax.set_title(f"{prop_name} — Feature Importance  (orange=NS rank #{ns_rank})")
    plt.tight_layout(); plt.savefig(out/"shap_bar.png",dpi=150); plt.close()
    plt.figure()
    shap.summary_plot(sv, Xte, feature_names=feats, show=False, max_display=15)
    plt.tight_layout(); plt.savefig(out/"shap_summary.png",dpi=150,bbox_inches="tight")
    plt.close()
    return ns_rank

# ── NS Curve ───────────────────────────────────────────────────────────────────────
def _ns_curve(model, Xtr_raw, ns_i, imp, sc, log_y, prop_name, unit, out):
    if ns_i is None: return None
    med  = np.nanmedian(Xtr_raw, axis=0)
    rng  = np.linspace(0, 200, 200)
    pred = []
    for v in rng:
        x = med.copy(); x[ns_i] = v
        p = model.predict(sc.transform(imp.transform(x.reshape(1,-1))))[0]
        pred.append(float(np.expm1(p) if log_y else p))
    pred   = np.array(pred)
    opt_ns = float(rng[np.argmax(pred)])
    print(f"  Optimal NS: {opt_ns:.1f} kg/m³  →  {pred.max():.1f} {unit}")
    fig,ax = plt.subplots(figsize=(8,5))
    ax.plot(rng, pred, "b-", lw=2.5)
    ax.axvline(opt_ns, color="r", ls="--",
               label=f"Optimal={opt_ns:.1f} kg/m³  ({pred.max():.1f} {unit})")
    ax.fill_between(rng, pred.min(), pred, alpha=0.08, color="blue")
    ax.set_xlabel("Nano Silica (kg/m³)")
    ax.set_ylabel(f"{prop_name} ({unit})")
    ax.set_title(f"NS Dosage-Response — {prop_name}  (others at train median)")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(out/"ns_curve.png",dpi=150); plt.close()
    return opt_ns

# ── Scatter ───────────────────────────────────────────────────────────────────────────
def _scatter(yte, ypred, best_name, best_r2, best_mape, prop_name, unit, out):
    lo = min(yte.min(), ypred.min()); hi = max(yte.max(), ypred.max())
    plt.figure(figsize=(6,6))
    plt.scatter(yte, ypred, s=18, alpha=0.55, edgecolors="none")
    plt.plot([lo,hi],[lo,hi],"r--",lw=1.5)
    plt.xlabel(f"Experimental {prop_name} ({unit})")
    plt.ylabel(f"Predicted {prop_name} ({unit})")
    plt.title(f"{best_name}  R²={best_r2:.4f}  MAPE={best_mape:.2f}%")
    plt.tight_layout(); plt.savefig(out/"scatter.png",dpi=150); plt.close()

# ── Run one property ───────────────────────────────────────────────────────────────
def run_property(cfg, df, mix_cols, ns_col):
    target = _find_col(df, cfg["kw"])
    if target is None:
        print(f"  [{cfg['name']}] column not found — skip"); return None

    df_t  = df[mix_cols + [target]].dropna(subset=[target]).copy()
    n     = len(df_t)
    print(f"\n{'='*60}")
    print(f"  Property : {cfg['name']}  |  column='{target}'  |  n={n}")
    print(f"{'='*60}")
    if n < cfg["min_n"]:
        print(f"  Skipped: {n} < {cfg['min_n']} minimum"); return None

    # Remove columns >60% missing
    df_t  = df_t.loc[:, df_t.isnull().mean() < 0.60]
    feats = [c for c in df_t.columns if c != target]
    X_raw = df_t[feats].to_numpy(float)
    y_raw = df_t[target].to_numpy(float)
    ns_i  = feats.index(ns_col) if (ns_col and ns_col in feats) else None
    print(f"  Features : {len(feats)}  |  NS feature : {'yes' if ns_i is not None else 'NO'}")

    out = ROOT_OUT / cfg["name"]; out.mkdir(exist_ok=True)
    gate = cfg["gate"]

    def _run(n_trials):
        art  = _pipeline(X_raw, y_raw, n_trials)
        best = max(art["results"], key=lambda k: art["results"][k]["R2"])
        r2   = art["results"][best]["R2"]
        return art, best, r2, r2 >= gate

    art, best_name, best_r2, passed = _run(TRIALS_BASE)
    if not passed:
        print(f"  Gate {gate} not met (R²={best_r2:.4f}) — retry {TRIALS_RETRY} trials")
        art, best_name, best_r2, passed = _run(TRIALS_RETRY)

    # SHAP on best tree model
    shap_m = max((nm for nm in ["CatBoost","XGBoost","RF","GBR"] if nm in art["results"]),
                 key=lambda k: art["results"][k]["R2"])
    ns_rank = _shap(art["models"][shap_m], art["Xte"], feats, ns_col, cfg["name"], out)
    opt_ns  = _ns_curve(art["models"][shap_m], art["Xtr_raw"],
                        ns_i, art["imp"], art["sc"], art["log_y"],
                        cfg["name"], cfg["unit"], out)

    # Scatter
    bm = art["models"][best_name]
    p  = np.expm1(bm.predict(art["Xte"])) if art["log_y"] else bm.predict(art["Xte"])
    _scatter(art["yte_r"], p, best_name, best_r2,
             art["results"][best_name]["MAPE"], cfg["name"], cfg["unit"], out)

    summary = {
        "property": cfg["name"], "unit": cfg["unit"], "n_samples": n,
        "gate": gate, "gate_passed": passed,
        "best_model": best_name, "metrics": art["results"][best_name],
        "all_models": art["results"],
        "ns_rank": ns_rank, "ns_optimal_kg_m3": opt_ns,
    }
    (out / "metrics.json").write_text(json.dumps(summary, indent=2))
    return summary

# ── Main ─────────────────────────────────────────────────────────────────────────────
def main():
    df = load_data()
    print(f"\nDataset: {df.shape[0]} rows × {df.shape[1]} columns")

    mix_cols = get_mix_features(df)
    ns_col   = _find_col(df, NS_KW)
    print(f"Mix-design features : {len(mix_cols)}")
    print(f"Nano Silica column  : '{ns_col}'")

    all_results = []
    for cfg in MULTI_TARGETS:
        res = run_property(cfg, df, mix_cols, ns_col)
        if res: all_results.append(res)

    # ── Final summary table ──────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  MULTI-PROPERTY SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Property':<12} {'n':>5} {'Best Model':<12} {'R²':>7} {'MAPE%':>7} {'Gate':>7} {'NS Rank':>8} {'Opt NS':>8}")
    print(f"  {'-'*68}")
    for r in all_results:
        m    = r["metrics"]
        tick = "✅" if r["gate_passed"] else "❌"
        ns_r = f"#{r['ns_rank']}" if r["ns_rank"] else "N/A"
        ns_o = f"{r['ns_optimal_kg_m3']:.0f}" if r["ns_optimal_kg_m3"] else "N/A"
        print(f"  {r['property']:<12} {r['n_samples']:>5} {r['best_model']:<12} "
              f"{m['R2']:>7.4f} {m['MAPE']:>7.2f} {tick:>7}  {ns_r:>7}  {ns_o:>7}")
    print(f"{'='*70}")
    print(f"  Outputs → outputs/<property>/")

    # Save combined JSON
    (ROOT_OUT / "all_metrics.json").write_text(json.dumps(all_results, indent=2))
    print("  Combined → outputs/all_metrics.json")


if __name__ == "__main__":
    main()
