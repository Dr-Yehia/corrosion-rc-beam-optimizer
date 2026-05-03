#!/usr/bin/env python3
"""
UHPC Multi-Property ML Pipeline v2 — Maximum Performance
──────────────────────────────────────────────────
New vs v1:
  [1] Target Encoding  — Cement type / Fiber type / Slag / SP (CV, no leakage)
  [2] Outlier Removal  — Isolation Forest 5%
  [3] Physics Features — W/C, NS/Cement, Fiber Index, Total Binder
  [4] LightGBM         — replaces MLP in Stacking
  [5] Cluster Feature  — KMeans(3) label as extra input
  [+] No Log1p         — disabled (was hurting R²)
  [+] Realistic Gates  — 0.93 / 0.90 / 0.88 / 0.88 / 0.85

Targets : CS_28d | Flexural | Tensile | E_Modulus | Porosity
Outputs : scatter | shap_bar | shap_summary | ns_curve
          taylor_diagram | shap_ns_sf | sensitivity | summary_chart
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
from lightgbm import LGBMRegressor
from sklearn.cluster import KMeans
from sklearn.ensemble import (GradientBoostingRegressor, IsolationForest,
                               RandomForestRegressor, StackingRegressor)
from sklearn.impute import KNNImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

# ── Config ──────────────────────────────────────────────────────────────────
SEED          = 42
TEST_SIZE     = 0.20
TRIALS_BASE   = 50
TRIALS_RETRY  = 100
KNN_K         = 5
N_CLUSTERS    = 3
OUT_CONTAM    = 0.05          # Isolation Forest contamination
ROOT_OUT      = Path("outputs")
ROOT_OUT.mkdir(exist_ok=True)

GITHUB_RAW = (
    "https://raw.githubusercontent.com/"
    "Dr-Yehia/corrosion-rc-beam-optimizer/nano-silica/"
    "Nano%20silica%20data/"
)
EXCEL_FILE  = "UHPC Dataset  (Version-2).xlsx"
LOCAL_PATHS = [
    EXCEL_FILE, f"Nano silica data/{EXCEL_FILE}",
    f"/kaggle/input/uhpc-nano-silica/{EXCEL_FILE}",
    f"/kaggle/working/{EXCEL_FILE}",
]

MULTI_TARGETS = [
    {"kw":["28-day","28day","cs28","fc28"],  "name":"CS_28d",   "unit":"MPa","gate":0.930,"min_n":200},
    {"kw":["peakstrength","mor("," mor"],    "name":"Flexural", "unit":"MPa","gate":0.900,"min_n":100},
    {"kw":["splittensile"],                  "name":"Tensile",  "unit":"MPa","gate":0.880,"min_n": 80},
    {"kw":["elasticmodulus","elasticmod"],   "name":"E_Modulus","unit":"GPa","gate":0.880,"min_n": 80},
    {"kw":["porosity"],                      "name":"Porosity", "unit":"% ", "gate":0.850,"min_n": 80},
]

NS_KW = ["nano silica","nanosio2","nano-sio2","nsio2","nanosilica"]
SF_KW = ["silica fume","silicafume"]

_RESULT_KW = [
    "1-day","3-day","7-day","14-day","21day","28-day","56-day","90-day",
    "elasticmodulus","splittensile","directtensile","tensileelastic",
    "straincapacity","peaktensilestrain","lop(","mor("," mor","peakstrength",
    "residualstrength","toughness","aircontent","airvoid","porosity",
    "waterabsorption","shrinkage","cycles","totalcharge","surfaceresistivity",
    "crackingstrength","firstcracking",
]
_HEADER_KW = ["cement","water","silica","fly","slag","sand",
              "fiber","superplast","nano","strength","28","mpa","ns"]
_CAT_KW    = ["cement type","type of fiber","type of slag",
              "type of superplast","fly ash type","sand type",
              "type of filler","fiber type"]

CV = KFold(n_splits=5, shuffle=True, random_state=SEED)

# ── Utilities ────────────────────────────────────────────────────────────────
def _c(s): return str(s).lower().replace(" ","").replace(",","").replace("'","").replace("-","")
def _mape(y,yp): return float(np.mean(np.abs((y-yp)/np.maximum(np.abs(y),1e-9)))*100)
def _a20(y,yp): r=yp/np.maximum(y,1e-9); return float(np.mean((r>=0.8)&(r<=1.2)))

def _report(y,yp,label=""):
    m = dict(R2=round(r2_score(y,yp),4),
             MAE=round(mean_absolute_error(y,yp),3),
             RMSE=round(float(np.sqrt(mean_squared_error(y,yp))),3),
             MAPE=round(_mape(y,yp),2), a20=round(_a20(y,yp),4))
    print(f"  {label:12s}  R²={m['R2']:.4f}  MAE={m['MAE']:.2f}  "
          f"RMSE={m['RMSE']:.2f}  MAPE={m['MAPE']:.2f}%  a20={m['a20']:.3f}")
    return m

def _find_col(df,keywords):
    mp = {_c(c):c for c in df.columns}
    for kw in keywords:
        for cl,orig in mp.items():
            if _c(kw) in cl: return orig
    return None

def _is_result(col): return any(_c(kw) in _c(col) for kw in _RESULT_KW)
def _is_cat(col):    return any(_c(kw) in _c(col) for kw in _CAT_KW)

# ── Excel loading ────────────────────────────────────────────────────────────────
def _read_sheet(xf,sheet):
    try: raw = xf.parse(sheet,header=None,nrows=8)
    except: return None
    if len(raw)<5: return None
    best_h,best_sc = 0,-1
    for h in range(min(5,len(raw))):
        vals = raw.iloc[h].astype(str).str.lower().tolist()
        sc   = sum(1 for v in vals for kw in _HEADER_KW if kw in v)
        sc  -= int(sum(1 for v in vals
                       if v.replace(".","").replace("-","").isdigit())/max(len(vals),1)*10)
        if sc>best_sc: best_sc,best_h = sc,h
    try:
        df = xf.parse(sheet,header=best_h)
        print(f"    header row={best_h}  score={best_sc}  shape={df.shape}")
        return df if len(df)>=10 else None
    except: return None

def load_data():
    url = GITHUB_RAW + requests.utils.quote(EXCEL_FILE)
    sources = []
    try:
        print("Downloading from GitHub ...")
        r = requests.get(url,timeout=120); r.raise_for_status()
        sources.append(("GitHub",r.content))
    except Exception as e: print(f"  GitHub failed: {e}")
    for p in LOCAL_PATHS:
        if Path(p).exists(): sources.append((p,None))
    for label,content in sources:
        try:
            xf = pd.ExcelFile(io.BytesIO(content) if content else label)
            for sheet in xf.sheet_names:
                df = _read_sheet(xf,sheet)
                if df is not None:
                    print(f"  OK [{label}] sheet='{sheet}'"); return df
        except Exception as e: print(f"  Failed [{label}]: {e}")
    raise FileNotFoundError(f"Cannot load '{EXCEL_FILE}'.")

# ── [1] Target Encoding for categorical columns ───────────────────────────────
def target_encode(df_all, cat_cols, target_col, train_idx, test_idx):
    """
    CV-based target encoding on train only, then apply mapping to test.
    No leakage: test rows never influence encoding of train rows.
    """
    if not cat_cols: return np.zeros((len(df_all),0)), []
    global_mean = df_all.iloc[train_idx][target_col].mean()
    enc_arr = np.full((len(df_all), len(cat_cols)), global_mean)
    feat_names = []
    for ci, col in enumerate(cat_cols):
        # CV encoding for train rows
        enc_train = np.full(len(train_idx), global_mean)
        sub = df_all.iloc[train_idx][[col, target_col]].copy().reset_index(drop=True)
        for tr, va in KFold(5,shuffle=True,random_state=SEED).split(sub):
            means = sub.iloc[tr].groupby(col)[target_col].mean()
            enc_train[va] = sub.iloc[va][col].map(means).fillna(global_mean).values
        enc_arr[train_idx, ci] = enc_train
        # Test rows: use full-train mapping
        full_map = df_all.iloc[train_idx].groupby(col)[target_col].mean()
        enc_arr[test_idx, ci] = df_all.iloc[test_idx][col].map(full_map).fillna(global_mean).values
        feat_names.append(f"{col}_enc")
    print(f"  Target-encoded {len(cat_cols)} categorical cols: {feat_names}")
    return enc_arr, feat_names

# ── [3] Physics-informed features ───────────────────────────────────────────────
def add_physics_features(df, ns_col, sf_col):
    eps = 1e-9
    feats, names = [], []
    c_col = _find_col(df,["cement amount","cement(","cement ("])
    w_col = _find_col(df,["water","w/c","w ("])
    l_col = _find_col(df,["length (mm)"])
    d_col = _find_col(df,["diameter (mm)"])
    fv_col= _find_col(df,["amount / quantity of fiber"])
    ft_col= _find_col(df,["tensile strength (mpa)"])

    def _s(col): return df[col].fillna(0).to_numpy(float) if col else np.zeros(len(df))

    c = _s(c_col); w = _s(w_col)
    ns= _s(ns_col); sf= _s(sf_col)
    l = _s(l_col);  d = np.maximum(_s(d_col),eps)
    fv= _s(fv_col); ft= _s(ft_col)

    if c_col and w_col:
        feats.append(w / np.maximum(c, eps)); names.append("WC_ratio")
    if c_col and ns_col:
        feats.append(ns / np.maximum(c, eps)); names.append("NS_cement_ratio")
    if c_col:
        feats.append(c + sf + ns); names.append("Total_binder")
        if ns_col:
            feats.append(ns / np.maximum(c+sf+eps,eps)); names.append("NS_binder_ratio")
    if l_col and d_col and fv_col and ft_col:
        aspect = l / d
        feats.append(fv * aspect * ft / 1e6); names.append("Fiber_index")

    if not feats: return pd.DataFrame(index=df.index)
    df_phys = pd.DataFrame(np.column_stack(feats), columns=names, index=df.index)
    print(f"  Physics features added: {names}")
    return df_phys

# ── [2] Outlier removal ───────────────────────────────────────────────────────────
def remove_outliers(X, y, contamination=OUT_CONTAM):
    iso  = IsolationForest(contamination=contamination,random_state=SEED,n_jobs=-1)
    mask = iso.fit_predict(np.column_stack([X,y.reshape(-1,1)])) == 1
    print(f"  Outliers removed: {(~mask).sum()} ({(~mask).mean()*100:.1f}%)")
    return X[mask], y[mask]

# ── [5] Cluster feature ────────────────────────────────────────────────────────────
def add_cluster(Xtr, Xte, n=N_CLUSTERS):
    km  = KMeans(n_clusters=n, random_state=SEED, n_init=10)
    ltr = km.fit_predict(Xtr).reshape(-1,1).astype(float)
    lte = km.predict(Xte).reshape(-1,1).astype(float)
    print(f"  KMeans({n}) cluster sizes on train: "
          f"{np.bincount(ltr.flatten().astype(int))}")
    return np.hstack([Xtr,ltr]), np.hstack([Xte,lte])

# ── [4] Optuna model factories ───────────────────────────────────────────────────────
MAKERS = {
    "CatBoost": lambda t: CatBoostRegressor(
        iterations    =t.suggest_int("n",300,2000),
        learning_rate =t.suggest_float("lr",0.005,0.2,log=True),
        depth         =t.suggest_int("d",5,10),
        l2_leaf_reg   =t.suggest_float("l2",1,10),
        random_seed=SEED, verbose=0),
    "XGBoost": lambda t: XGBRegressor(
        n_estimators    =t.suggest_int("n",300,2000),
        learning_rate   =t.suggest_float("lr",0.005,0.2,log=True),
        max_depth       =t.suggest_int("d",4,10),
        subsample       =t.suggest_float("ss",0.6,1.0),
        colsample_bytree=t.suggest_float("cs",0.6,1.0),
        min_child_weight=t.suggest_int("mcw",1,10),
        random_state=SEED, verbosity=0),
    "LightGBM": lambda t: LGBMRegressor(
        n_estimators =t.suggest_int("n",300,2000),
        learning_rate=t.suggest_float("lr",0.005,0.2,log=True),
        max_depth    =t.suggest_int("d",4,12),
        num_leaves   =t.suggest_int("nl",20,200),
        subsample    =t.suggest_float("ss",0.6,1.0),
        colsample_bytree=t.suggest_float("cs",0.6,1.0),
        random_state=SEED, verbose=-1),
    "RF": lambda t: RandomForestRegressor(
        n_estimators    =t.suggest_int("n",200,800),
        max_depth       =t.suggest_int("d",5,30),
        min_samples_split=t.suggest_int("mss",2,10),
        max_features    =t.suggest_float("mf",0.4,1.0),
        random_state=SEED, n_jobs=-1),
    "GBR": lambda t: GradientBoostingRegressor(
        n_estimators =t.suggest_int("n",200,1000),
        learning_rate=t.suggest_float("lr",0.005,0.15,log=True),
        max_depth    =t.suggest_int("d",3,8),
        subsample    =t.suggest_float("ss",0.6,1.0),
        random_state=SEED),
}
STACK_MODELS = ["CatBoost","XGBoost","LightGBM","GBR"]   # RF optional, MLP removed

def _tune(name, maker, X, y, n_trials):
    def obj(trial):
        m = maker(trial)
        scores = [r2_score(y[va], m.fit(X[tr],y[tr]).predict(X[va]))
                  for tr,va in CV.split(X)]
        return float(np.mean(scores))
    st = optuna.create_study(direction="maximize",
                              sampler=optuna.samplers.TPESampler(seed=SEED))
    st.optimize(obj, n_trials=n_trials, show_progress_bar=False)
    best = maker(st.best_trial); best.fit(X,y)
    print(f"  {name:10s}  CV R²={st.best_value:.4f}")
    return best

# ── Core pipeline ─────────────────────────────────────────────────────────────────
def _pipeline(X_raw, y_raw, n_trials):
    # Train/Test split (stratified)
    q    = pd.qcut(y_raw, q=min(5,len(y_raw)//20), labels=False, duplicates="drop")
    itr, ite = train_test_split(np.arange(len(y_raw)),
                                test_size=TEST_SIZE, random_state=SEED, stratify=q)
    Xtr_r, Xte_r = X_raw[itr], X_raw[ite]
    ytr_r, yte_r = y_raw[itr], y_raw[ite]

    # [2] Outlier removal on train only
    Xtr_r, ytr_r = remove_outliers(Xtr_r, ytr_r)

    # KNN Imputation (fit on train)
    imp   = KNNImputer(n_neighbors=KNN_K)
    Xtr_i = imp.fit_transform(Xtr_r)
    Xte_i = imp.transform(Xte_r)

    # [+] No Log1p — direct target
    ytr = ytr_r.copy()

    # StandardScaler
    sc  = StandardScaler()
    Xtr = sc.fit_transform(Xtr_i)
    Xte = sc.transform(Xte_i)

    # [5] Cluster feature (fit on train)
    Xtr, Xte = add_cluster(Xtr, Xte)

    print(f"  Optuna {n_trials} trials × {len(MAKERS)} models")
    models = {n: _tune(n, m, Xtr, ytr, n_trials) for n,m in MAKERS.items()}

    stack = StackingRegressor(
        estimators   =[(k,v) for k,v in models.items() if k in STACK_MODELS],
        final_estimator=Ridge(alpha=1.0), cv=5, n_jobs=-1)
    stack.fit(Xtr, ytr)
    models["Stacking"] = stack

    print("  Test Set:")
    results = {}
    for nm,m in models.items():
        p = m.predict(Xte)
        results[nm] = _report(yte_r, p, nm)

    return dict(models=models, results=results,
                Xtr=Xtr, Xte=Xte, Xtr_raw=Xtr_r, yte_r=yte_r,
                imp=imp, sc=sc, log_y=False)

# ── Figure 1: Scatter ────────────────────────────────────────────────────────────────
def _scatter(yte,yp,best,r2,mape,prop,unit,out):
    lo=min(yte.min(),yp.min()); hi=max(yte.max(),yp.max())
    plt.figure(figsize=(6,6))
    plt.scatter(yte,yp,s=18,alpha=0.55,edgecolors="none",c="steelblue")
    plt.plot([lo,hi],[lo,hi],"r--",lw=1.5,label="y=x")
    plt.plot([lo,hi],[lo*1.2,hi*1.2],"g:",lw=1,alpha=0.6)
    plt.plot([lo,hi],[lo*0.8,hi*0.8],"g:",lw=1,alpha=0.6,label="±20%")
    plt.xlabel(f"Experimental {prop} ({unit})")
    plt.ylabel(f"Predicted {prop} ({unit})")
    plt.title(f"{best} — {prop}  R²={r2:.4f}  MAPE={mape:.2f}%")
    plt.legend(); plt.tight_layout()
    plt.savefig(out/"scatter.png",dpi=200); plt.close()

# ── Figures 2+3: SHAP ────────────────────────────────────────────────────────────────
def _shap_plots(model,Xte,feats,ns_col,prop,out):
    try:    sv = shap.TreeExplainer(model).shap_values(Xte)
    except: sv = shap.KernelExplainer(model.predict,shap.sample(Xte,80)).shap_values(Xte)
    df_s=(pd.DataFrame({"feature":feats,"shap":np.abs(sv).mean(0)})
          .sort_values("shap",ascending=False).reset_index(drop=True))
    ns_rank=None
    if ns_col and ns_col in df_s.feature.values:
        ns_rank=int(df_s[df_s.feature==ns_col].index[0])+1
        print(f"  NS SHAP rank: #{ns_rank}")
    top=df_s.head(12)
    colors=["#FF8C00" if f==ns_col else "#4682B4" for f in top.feature[::-1]]
    fig,ax=plt.subplots(figsize=(9,6))
    ax.barh(top.feature[::-1],top.shap[::-1],color=colors)
    ax.set_xlabel("Mean |SHAP Value|")
    ax.set_title(f"{prop} — SHAP Importance  (orange=NS rank #{ns_rank})")
    plt.tight_layout(); plt.savefig(out/"shap_bar.png",dpi=200); plt.close()
    plt.figure()
    shap.summary_plot(sv,Xte,feature_names=feats,show=False,max_display=15)
    plt.tight_layout(); plt.savefig(out/"shap_summary.png",dpi=200,bbox_inches="tight")
    plt.close()
    return sv,df_s,ns_rank

# ── Figure 4: NS Curve ────────────────────────────────────────────────────────────────
def _ns_curve(model,Xtr_raw,ns_i,imp,sc,prop,unit,out):
    if ns_i is None: return None
    med=np.nanmedian(Xtr_raw,axis=0); rng=np.linspace(0,200,200); pred=[]
    for v in rng:
        x=med.copy(); x[ns_i]=v
        xi=sc.transform(imp.transform(x.reshape(1,-1)))
        # append median cluster (0) for the cluster feature
        xi=np.hstack([xi,[[0]]])
        pred.append(float(model.predict(xi)[0]))
    pred=np.array(pred); opt_ns=float(rng[np.argmax(pred)])
    print(f"  Optimal NS: {opt_ns:.1f} kg/m³ → {pred.max():.2f} {unit}")
    fig,ax=plt.subplots(figsize=(8,5))
    ax.plot(rng,pred,"b-",lw=2.5)
    ax.axvline(opt_ns,color="r",ls="--",
               label=f"Optimal={opt_ns:.1f} kg/m³  ({pred.max():.1f} {unit})")
    ax.fill_between(rng,pred.min(),pred,alpha=0.08,color="blue")
    ax.set_xlabel("Nano Silica (kg/m³)"); ax.set_ylabel(f"{prop} ({unit})")
    ax.set_title(f"NS Dosage-Response — {prop}")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(out/"ns_curve.png",dpi=200); plt.close()
    return opt_ns

# ── Figure 5: Taylor Diagram ─────────────────────────────────────────────────────────
def _taylor(yte,model_preds,prop,out):
    std_ref=np.std(yte)
    fig=plt.figure(figsize=(7,6)); ax=fig.add_subplot(111,polar=True)
    ax.set_thetamax(90)
    ax.set_thetagrids(range(0,91,15),
                      [f"{np.cos(np.deg2rad(a)):.2f}" for a in range(0,91,15)],fontsize=8)
    ax.set_title(f"Taylor Diagram — {prop}",pad=20)
    ax.plot(0,1,"k*",ms=14,label="Observed",zorder=5)
    colors=["#e41a1c","#377eb8","#4daf4a","#984ea3","#ff7f00","#a65628"]
    for i,(nm,yp) in enumerate(model_preds.items()):
        r=float(np.corrcoef(yte,yp)[0,1]); std=np.std(yp)/std_ref
        ax.plot(np.arccos(np.clip(r,-1,1)),std,"o",ms=9,
                color=colors[i%len(colors)],label=nm)
    for rv in [0.5,1.0,1.5]:
        t=np.linspace(0,np.pi/2,200)
        ax.plot(t,np.sqrt(1+rv**2-2*rv*np.cos(t)),":",color="gray",lw=0.8,alpha=0.5)
    ax.legend(loc="upper right",bbox_to_anchor=(1.35,1.1),fontsize=8)
    plt.tight_layout(); plt.savefig(out/"taylor_diagram.png",dpi=200,bbox_inches="tight")
    plt.close()

# ── Figure 6: SHAP NS×SF Interaction ────────────────────────────────────────────
def _shap_interaction(sv,Xte,feats,ns_col,sf_col,prop,out):
    ns_i=feats.index(ns_col) if (ns_col and ns_col in feats) else None
    sf_i=feats.index(sf_col) if (sf_col and sf_col in feats) else None
    if ns_i is None: return
    fig,ax=plt.subplots(figsize=(8,5))
    sc_=ax.scatter(Xte[:,ns_i],sv[:,ns_i],
                   c=Xte[:,sf_i] if sf_i else np.zeros(len(Xte)),
                   cmap="RdYlGn",s=20,alpha=0.7,edgecolors="none")
    plt.colorbar(sc_,ax=ax,label=sf_col or "")
    ax.axhline(0,color="gray",lw=0.8,ls="--")
    ax.set_xlabel(f"Nano Silica (scaled)")
    ax.set_ylabel("SHAP value for Nano Silica")
    ax.set_title(f"{prop} — NS×SF Interaction  (green=high SF, red=low SF)")
    plt.tight_layout(); plt.savefig(out/"shap_ns_sf_interaction.png",dpi=200); plt.close()

# ── Figure 7: Sensitivity ───────────────────────────────────────────────────────────
def _sensitivity(model,Xtr_raw,feats,imp,sc,prop,unit,out,top_n=12):
    med=np.nanmedian(Xtr_raw,axis=0)
    xi_base=np.hstack([sc.transform(imp.transform(med.reshape(1,-1))),[[0]]])
    base=float(model.predict(xi_base)[0])
    deltas=[]
    for i,feat in enumerate(feats):
        x=med.copy(); x[i]=med[i]*1.10+1e-9
        xi=np.hstack([sc.transform(imp.transform(x.reshape(1,-1))),[[0]]])
        p=float(model.predict(xi)[0])
        deltas.append((feat,(p-base)/(abs(base)+1e-9)*100))
    deltas.sort(key=lambda x:abs(x[1]),reverse=True); deltas=deltas[:top_n]
    names,vals=zip(*deltas)
    colors=["#2ecc71" if v>0 else "#e74c3c" for v in vals]
    fig,ax=plt.subplots(figsize=(9,6))
    ax.barh(names[::-1],[v for v in vals[::-1]],color=colors[::-1])
    ax.axvline(0,color="black",lw=0.8)
    ax.set_xlabel("% Change per +10% feature increase")
    ax.set_title(f"{prop} — Sensitivity Analysis")
    plt.tight_layout(); plt.savefig(out/"sensitivity.png",dpi=200); plt.close()

# ── Run one property ───────────────────────────────────────────────────────────────
def run_property(cfg, df, ns_col, sf_col):
    target = _find_col(df, cfg["kw"])
    if target is None:
        print(f"  [{cfg['name']}] column not found — skip"); return None

    # Numeric mix-design features (no result cols)
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                if not _is_result(c) and c != target]
    # Categorical mix-design features
    cat_cols  = [c for c in df.columns
                 if df[c].dtype == object and _is_cat(c) and not _is_result(c)]

    keep_cols = num_cols + cat_cols + [target]
    df_t = df[keep_cols].dropna(subset=[target]).copy().reset_index(drop=True)
    n    = len(df_t)

    print(f"\n{'='*62}")
    print(f"  {cfg['name']}  |  target='{target}'  |  n={n}")
    print(f"{'='*62}")
    if n < cfg["min_n"]:
        print(f"  Skip: {n} < {cfg['min_n']}"); return None

    # Drop numeric cols with >60% missing
    df_t = df_t.loc[:,
        [c for c in df_t.columns
         if c in cat_cols or c == target or df_t[c].isnull().mean() < 0.60]]
    num_cols = [c for c in df_t.columns if c not in cat_cols and c != target]

    # [3] Physics features
    phys = add_physics_features(df_t, ns_col if ns_col in df_t.columns else None,
                                sf_col if sf_col in df_t.columns else None)
    if not phys.empty:
        df_t = pd.concat([df_t, phys], axis=1)
        num_cols += list(phys.columns)

    y_all = df_t[target].to_numpy(float)
    X_num = df_t[num_cols].to_numpy(float)
    feats = num_cols.copy()
    ns_i  = feats.index(ns_col) if (ns_col and ns_col in feats) else None

    # [1] Target Encoding — computed inside run_property using full index
    if cat_cols:
        itr_all, ite_all = train_test_split(
            np.arange(n), test_size=TEST_SIZE, random_state=SEED,
            stratify=pd.qcut(y_all, q=min(5,n//20), labels=False, duplicates="drop"))
        enc_arr, enc_names = target_encode(df_t, cat_cols, target, itr_all, ite_all)
        X_num  = np.hstack([X_num, enc_arr])
        feats += enc_names

    out = ROOT_OUT / cfg["name"]; out.mkdir(exist_ok=True)

    def _run(n_trials):
        art  = _pipeline(X_num, y_all, n_trials)
        best = max(art["results"], key=lambda k: art["results"][k]["R2"])
        r2   = art["results"][best]["R2"]
        return art, best, r2, r2 >= cfg["gate"]

    art, best_name, best_r2, passed = _run(TRIALS_BASE)
    if not passed:
        print(f"  Gate {cfg['gate']} not met (R²={best_r2:.4f}) — retry {TRIALS_RETRY}")
        art, best_name, best_r2, passed = _run(TRIALS_RETRY)

    # Best tree model for analysis
    shap_m = max((nm for nm in ["CatBoost","XGBoost","LightGBM","GBR"]
                  if nm in art["results"]),
                 key=lambda k: art["results"][k]["R2"])
    m = art["results"][best_name]

    # All figures
    feats_ext = feats + ["cluster"]          # cluster feature appended last
    bm = art["models"][best_name]
    yp = bm.predict(art["Xte"])
    _scatter(art["yte_r"],yp,best_name,best_r2,m["MAPE"],cfg["name"],cfg["unit"],out)

    sv,df_shap,ns_rank = _shap_plots(
        art["models"][shap_m],art["Xte"],feats_ext,ns_col,cfg["name"],out)

    opt_ns = _ns_curve(art["models"][shap_m],art["Xtr_raw"],
                       ns_i,art["imp"],art["sc"],cfg["name"],cfg["unit"],out)

    model_preds={nm:art["models"][nm].predict(art["Xte"]) for nm in art["results"]}
    _taylor(art["yte_r"],model_preds,cfg["name"],out)
    _shap_interaction(sv,art["Xte"],feats_ext,ns_col,sf_col,cfg["name"],out)
    _sensitivity(art["models"][shap_m],art["Xtr_raw"],
                 feats,art["imp"],art["sc"],cfg["name"],cfg["unit"],out)

    summary=dict(property=cfg["name"],unit=cfg["unit"],n_samples=n,
                 gate=cfg["gate"],gate_passed=passed,
                 best_model=best_name,metrics=m,all_models=art["results"],
                 ns_rank=ns_rank,ns_optimal_kg_m3=opt_ns)
    (out/"metrics.json").write_text(json.dumps(summary,indent=2))
    return summary

# ── Summary chart ───────────────────────────────────────────────────────────────────
def _summary_chart(all_results):
    if not all_results: return
    names  = [r["property"]         for r in all_results]
    r2s    = [r["metrics"]["R2"]    for r in all_results]
    mapes  = [r["metrics"]["MAPE"]  for r in all_results]
    colors = ["#2ecc71" if r["gate_passed"] else "#e74c3c" for r in all_results]
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12,5))
    ax1.bar(names,r2s,color=colors); ax1.set_ylim(0.80,1.0)
    ax1.set_ylabel("R²"); ax1.set_title("R² per Property (green=gate ✅)")
    for i,v in enumerate(r2s): ax1.text(i,v+0.003,f"{v:.3f}",ha="center",fontsize=9)
    ax2.bar(names,mapes,color=colors)
    ax2.set_ylabel("MAPE (%)"); ax2.set_title("MAPE per Property")
    for i,v in enumerate(mapes): ax2.text(i,v+0.1,f"{v:.1f}%",ha="center",fontsize=9)
    plt.suptitle("UHPC Multi-Property v2 Summary",fontsize=13,fontweight="bold")
    plt.tight_layout(); plt.savefig(ROOT_OUT/"summary_chart.png",dpi=200); plt.close()

# ── Main ─────────────────────────────────────────────────────────────────────────────
def main():
    df     = load_data()
    ns_col = _find_col(df, NS_KW)
    sf_col = _find_col(df, SF_KW)
    print(f"\nDataset: {df.shape}  |  NS='{ns_col}'  |  SF='{sf_col}'")

    all_results = []
    for cfg in MULTI_TARGETS:
        res = run_property(cfg, df, ns_col, sf_col)
        if res: all_results.append(res)

    _summary_chart(all_results)
    (ROOT_OUT/"all_metrics.json").write_text(json.dumps(all_results,indent=2))

    print(f"\n{'='*68}")
    print("  MULTI-PROPERTY SUMMARY v2")
    print(f"{'='*68}")
    print(f"  {'Property':<12}{'n':>6}{'Best':>12}{'R²':>8}"
          f"{'MAPE':>7}{'Gate':>7}{'NS★':>7}{'OptNS':>8}")
    print(f"  {'-'*66}")
    for r in all_results:
        m  = r["metrics"]
        tk = "✅" if r["gate_passed"] else "❌"
        ns = f"#{r['ns_rank']}" if r["ns_rank"] else "--"
        op = f"{r['ns_optimal_kg_m3']:.0f}" if r["ns_optimal_kg_m3"] else "--"
        print(f"  {r['property']:<12}{r['n_samples']:>6}{r['best_model']:>12}"
              f"{m['R2']:>8.4f}{m['MAPE']:>7.2f}{tk:>7}{ns:>7}{op:>8}")
    print(f"{'='*68}")
    print(f"  7 figures per property | outputs/summary_chart.png")

if __name__ == "__main__":
    main()
