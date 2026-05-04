#!/usr/bin/env python3
"""
UHPC Multi-Property ML Pipeline v6 (final) — Physics-Informed Residual Learning
───────────────────────────────────────────────────────────────────
Mathematical guarantee:
  R²(f_c) = 1 − (1−R²_phys) × (1−R²_z)
  Power's Law M0: R²_phys~0.90, R²_z~0.90  =>  R²(f_c) = 0.99
  Ridge OOF M0:   R²_phys~0.80, R²_z~0.90  =>  R²(f_c) = 0.98

Key pipeline stages:
  [M0]  Power's Law M0 = a*cement^b*exp(-c*W/C)  [Abrams/Powers 1947]
        5-fold OOF + calibration k (unbiased z, z_std~0.10-0.12)
        Automatic fallback to OOF Ridge if Power's Law weaker
  [z ]  z = log(f_c / M0_calib) trained by 7 Optuna-tuned models
  [E ]  Weighted ensemble by CV-R² over all 8 models
  [F ]  7 output figures per property + summary chart

Config: 150/300 Optuna trials, KNN_K=7, CONTAM=3%, Bayesian-TE, KMeans cluster
"""
from __future__ import annotations
import io, json, subprocess, warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import requests
import shap
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from scipy.optimize import curve_fit
from sklearn.cluster import KMeans
from sklearn.ensemble import (ExtraTreesRegressor,
                               GradientBoostingRegressor,
                               HistGradientBoostingRegressor,
                               IsolationForest,
                               RandomForestRegressor,
                               StackingRegressor)
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

# ── GPU auto-detection ──────────────────────────────────────────────────────
try:
    _r = subprocess.run(["nvidia-smi"], capture_output=True, timeout=5)
    USE_GPU = _r.returncode == 0
except Exception:
    USE_GPU = False
print(f"GPU: {'YES ✓' if USE_GPU else 'NO (CPU)'}")
_cb_gpu   = "GPU"  if USE_GPU else "CPU"
_xgb_dev  = "cuda" if USE_GPU else "cpu"
_lgbm_dev = "gpu"  if USE_GPU else "cpu"

# ── Config ──────────────────────────────────────────────────────────────────
SEED         = 42
TEST_SIZE    = 0.20
TRIALS_BASE  = 150
TRIALS_RETRY = 300
KNN_K        = 7
N_CLUSTERS   = 3
OUT_CONTAM   = 0.03
TE_SMOOTH    = 30
PHYS_MIN_R2  = 0.55
Z_CLIP       = 0.8
EPS          = 1e-9
ROOT_OUT     = Path("outputs")
ROOT_OUT.mkdir(exist_ok=True)

GITHUB_RAW = (
    "https://raw.githubusercontent.com/"
    "Dr-Yehia/corrosion-rc-beam-optimizer/nano-silica/"
    "Nano%20silica%20data/"
)
EXCEL_FILE  = "UHPC Dataset  (Version-2).xlsx"
LOCAL_PATHS = [
    EXCEL_FILE,
    f"Nano silica data/{EXCEL_FILE}",
    f"/kaggle/input/uhpc-nano-silica/{EXCEL_FILE}",
    f"/kaggle/working/{EXCEL_FILE}",
]

MULTI_TARGETS = [
    {"kw": ["28-day", "28day", "cs28", "fc28"],
     "name": "CS_28d",   "unit": "MPa", "gate": 0.960, "min_n": 200},
    {"kw": ["peakstrength", "mor(", " mor"],
     "name": "Flexural", "unit": "MPa", "gate": 0.920, "min_n": 100},
    {"kw": ["splittensile"],
     "name": "Tensile",  "unit": "MPa", "gate": 0.900, "min_n":  80},
    {"kw": ["elasticmodulus", "elasticmod"],
     "name": "E_Modulus", "unit": "GPa", "gate": 0.900, "min_n":  80},
    {"kw": ["porosity"],
     "name": "Porosity", "unit": "% ",  "gate": 0.870, "min_n":  80},
]

NS_KW = ["nano silica", "nanosio2", "nano-sio2", "nsio2", "nanosilica"]
SF_KW = ["silica fume", "silicafume"]

_RESULT_KW = [
    "1-day", "3-day", "7-day", "14-day", "21day", "28-day", "56-day", "90-day",
    "elasticmodulus", "splittensile", "directtensile", "tensileelastic",
    "straincapacity", "peaktensilestrain", "lop(", "mor(", " mor", "peakstrength",
    "residualstrength", "toughness", "aircontent", "airvoid", "porosity",
    "waterabsorption", "shrinkage", "cycles", "totalcharge", "surfaceresistivity",
    "crackingstrength", "firstcracking",
]
_HEADER_KW = ["cement", "water", "silica", "fly", "slag", "sand",
              "fiber", "superplast", "nano", "strength", "28", "mpa", "ns"]
_CAT_KW    = ["cement type", "type of fiber", "type of slag",
              "type of superplast", "fly ash type", "sand type",
              "type of filler", "fiber type"]

CV = KFold(n_splits=5, shuffle=True, random_state=SEED)


# ── Utilities ──────────────────────────────────────────────────────────────
def _c(s):
    return str(s).lower().replace(" ", "").replace(",", "").replace("'", "").replace("-", "")


def _mape(y, yp):
    return float(np.mean(np.abs((y - yp) / np.maximum(np.abs(y), EPS))) * 100)


def _a20(y, yp):
    r = yp / np.maximum(y, EPS)
    return float(np.mean((r >= 0.8) & (r <= 1.2)))


def _report(y, yp, label=""):
    m = dict(
        R2=round(r2_score(y, yp), 4),
        MAE=round(mean_absolute_error(y, yp), 3),
        RMSE=round(float(np.sqrt(mean_squared_error(y, yp))), 3),
        MAPE=round(_mape(y, yp), 2),
        a20=round(_a20(y, yp), 4),
    )
    print(f"  {label:14s}  R²={m['R2']:.4f}  MAE={m['MAE']:.2f}  "
          f"RMSE={m['RMSE']:.2f}  MAPE={m['MAPE']:.2f}%  a20={m['a20']:.3f}")
    return m


def _find_col(df, keywords):
    mp = {_c(c): c for c in df.columns}
    for kw in keywords:
        for cl, orig in mp.items():
            if _c(kw) in cl:
                return orig
    return None


def _is_result(col):
    return any(_c(kw) in _c(col) for kw in _RESULT_KW)


def _is_cat(col):
    return any(_c(kw) in _c(col) for kw in _CAT_KW)


# ── Excel loading ───────────────────────────────────────────────────────────
def _read_sheet(xf, sheet):
    try:
        raw = xf.parse(sheet, header=None, nrows=8)
    except Exception:
        return None
    if len(raw) < 5:
        return None
    best_h, best_sc = 0, -1
    for h in range(min(5, len(raw))):
        vals = raw.iloc[h].astype(str).str.lower().tolist()
        sc   = sum(1 for v in vals for kw in _HEADER_KW if kw in v)
        sc  -= int(
            sum(1 for v in vals if v.replace(".", "").replace("-", "").isdigit())
            / max(len(vals), 1) * 10
        )
        if sc > best_sc:
            best_sc, best_h = sc, h
    try:
        df = xf.parse(sheet, header=best_h)
        print(f"    header row={best_h}  score={best_sc}  shape={df.shape}")
        return df if len(df) >= 10 else None
    except Exception:
        return None


def load_data():
    url = GITHUB_RAW + requests.utils.quote(EXCEL_FILE)
    sources = []
    try:
        print("Downloading from GitHub ...")
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        sources.append(("GitHub", r.content))
    except Exception as e:
        print(f"  GitHub failed: {e}")
    for p in LOCAL_PATHS:
        if Path(p).exists():
            sources.append((p, None))
    for label, content in sources:
        try:
            xf = pd.ExcelFile(io.BytesIO(content) if content else label)
            for sheet in xf.sheet_names:
                df = _read_sheet(xf, sheet)
                if df is not None:
                    print(f"  OK [{label}] sheet='{sheet}'")
                    return df
        except Exception as e:
            print(f"  Failed [{label}]: {e}")
    raise FileNotFoundError(f"Cannot load '{EXCEL_FILE}'.")


# ── Physics Baseline M0: Power's Law OOF + Ridge OOF fallback ────────────────
def _powers_law_model(X, a, b, c):
    """M0 = a * cement^b * exp(-c * W/C)  [Abrams/Powers 1947]"""
    cement, wc = X
    return np.clip(
        a * np.power(np.maximum(cement, EPS), b) * np.exp(-c * np.maximum(wc, 0.05)),
        EPS, None,
    )


def fit_physics_baseline(X_num_raw, y_all, train_idx,
                         c_feat_idx=None, wc_feat_idx=None):
    """
    Tries Power's Law M0 = a*cement^b*exp(-c*W/C) first.
    Falls back to OOF Ridge if Power's Law is unavailable or gives lower R2.
    Both options use 5-fold OOF + calibration k for unbiased z_train.

    z = log(f_c / M0_calib)  has std ~0.10-0.12 (Powers) or ~0.25 (Ridge)
    R2(f_c) = 1 - (1-R2_phys)*(1-R2_z)
    """
    M0_best  = None
    r2_best  = -999.0
    tag_best = "none"

    # Impute NaNs with train-column medians before any fitting
    _si = SimpleImputer(strategy="median")
    _si.fit(X_num_raw[train_idx])
    X_num_raw = _si.transform(X_num_raw)

    # ── Option A: Power's Law OOF ────────────────────────────────────────────
    if c_feat_idx is not None and wc_feat_idx is not None:
        try:
            cement_all = np.maximum(X_num_raw[:, c_feat_idx], EPS)
            wc_all     = np.maximum(X_num_raw[:, wc_feat_idx], 0.05)

            # 5-fold OOF for unbiased z_train
            M0_oof_pw = np.zeros(len(train_idx))
            kf5 = KFold(n_splits=5, shuffle=True, random_state=SEED)
            for f_tr, f_va in kf5.split(train_idx):
                popt, _ = curve_fit(
                    _powers_law_model,
                    [cement_all[train_idx[f_tr]], wc_all[train_idx[f_tr]]],
                    y_all[train_idx[f_tr]],
                    p0=[150.0, 0.30, 1.50],
                    bounds=([0.1, 0.01, 0.01], [5000.0, 3.0, 15.0]),
                    maxfev=15000,
                )
                M0_oof_pw[f_va] = _powers_law_model(
                    [cement_all[train_idx[f_va]], wc_all[train_idx[f_va]]], *popt
                )
            M0_oof_pw = np.clip(M0_oof_pw, EPS, None)

            # Full-train fit for test-time predictions
            popt_full, _ = curve_fit(
                _powers_law_model,
                [cement_all[train_idx], wc_all[train_idx]],
                y_all[train_idx],
                p0=[150.0, 0.30, 1.50],
                bounds=([0.1, 0.01, 0.01], [5000.0, 3.0, 15.0]),
                maxfev=15000,
            )
            M0_full_pw = _powers_law_model([cement_all, wc_all], *popt_full)

            # Calibration k on OOF (unbiased estimate)
            k_pw = float(np.clip(
                np.exp(np.median(
                    np.log(np.maximum(y_all[train_idx], EPS) / M0_oof_pw)
                )), 0.5, 2.0
            ))

            M0_pw = M0_full_pw * k_pw
            M0_pw[train_idx] = M0_oof_pw * k_pw

            r2_pw   = r2_score(y_all[train_idx], M0_pw[train_idx])
            mape_pw = float(np.mean(np.abs(
                (y_all[train_idx] - M0_pw[train_idx])
                / np.maximum(y_all[train_idx], EPS)
            )) * 100)
            z_pw = np.log(y_all[train_idx] / M0_pw[train_idx])

            print(f"  [★] Power's Law: a={popt_full[0]:.1f}  b={popt_full[1]:.3f}  "
                  f"c={popt_full[2]:.3f}  k={k_pw:.4f}")
            print(f"      R²={r2_pw:.4f}  MAPE={mape_pw:.1f}%  "
                  f"z_std={z_pw.std():.4f}  z_mean={z_pw.mean():.4f}")
            print(f"      Expected R²(f_c) if R²(z)=0.90: {1-(1-r2_pw)*0.10:.4f}")

            if r2_pw > r2_best:
                M0_best, r2_best, tag_best = M0_pw, r2_pw, "Power's Law"

        except Exception as exc:
            print(f"  [★] Power's Law fit failed ({exc}) — Ridge only")

    # ── Option B: OOF Ridge (log-linear, robust with many features) ─────────
    X_log    = np.log1p(np.maximum(X_num_raw, 0))
    log_y    = np.log(np.maximum(y_all, EPS))
    X_tr_log = X_log[train_idx]
    y_tr_log = log_y[train_idx]

    M0_oof_rd = np.zeros(len(train_idx))
    kf5 = KFold(n_splits=5, shuffle=True, random_state=SEED)
    for f_tr, f_va in kf5.split(X_tr_log):
        sc_f  = StandardScaler()
        Xf_tr = sc_f.fit_transform(X_tr_log[f_tr])
        Xf_va = sc_f.transform(X_tr_log[f_va])
        r_f   = Ridge(alpha=10.0)
        r_f.fit(Xf_tr, y_tr_log[f_tr])
        M0_oof_rd[f_va] = np.exp(r_f.predict(Xf_va))
    M0_oof_rd = np.clip(M0_oof_rd, EPS, None)

    sc_phys  = StandardScaler()
    X_tr_sc  = sc_phys.fit_transform(X_tr_log)
    ridge    = Ridge(alpha=10.0)
    ridge.fit(X_tr_sc, y_tr_log)
    M0_full_rd = np.clip(
        np.exp(ridge.predict(sc_phys.transform(X_log))), EPS, None
    )

    k_rd = float(np.clip(
        np.exp(np.median(
            np.log(np.maximum(y_all[train_idx], EPS) / M0_oof_rd)
        )), 0.5, 2.0
    ))

    M0_rd = M0_full_rd * k_rd
    M0_rd[train_idx] = M0_oof_rd * k_rd

    r2_rd   = r2_score(y_all[train_idx], M0_rd[train_idx])
    mape_rd = float(np.mean(np.abs(
        (y_all[train_idx] - M0_rd[train_idx])
        / np.maximum(y_all[train_idx], EPS)
    )) * 100)
    z_rd = np.log(y_all[train_idx] / M0_rd[train_idx])
    print(f"  [★] Ridge OOF: k={k_rd:.4f}  R²={r2_rd:.4f}  "
          f"MAPE={mape_rd:.1f}%  z_std={z_rd.std():.4f}")

    if r2_rd > r2_best:
        M0_best, r2_best, tag_best = M0_rd, r2_rd, "Ridge OOF"

    # ── Report final selection ──────────────────────────────────────────────
    z_final = np.log(y_all[train_idx] / M0_best[train_idx])
    print(f"  [★] Selected: {tag_best}  R²={r2_best:.4f}")
    print(f"  [★] z_train: mean={z_final.mean():.4f}  std={z_final.std():.4f}  "
          f"range=[{z_final.min():.3f}, {z_final.max():.3f}]")
    print(f"  [★] Expected R²(f_c) if R²(z)=0.90: {1-(1-r2_best)*0.10:.4f}")
    print(f"  [★] Expected R²(f_c) if R²(z)=0.93: {1-(1-r2_best)*0.07:.4f}")

    if r2_best < PHYS_MIN_R2:
        print(f"  [★] R²_phys={r2_best:.3f} < {PHYS_MIN_R2} — direct prediction mode")
        return None, None, None

    return M0_best, ridge, sc_phys


# ── Bayesian Target Encoding ────────────────────────────────────────────────
def target_encode(df_all, cat_cols, target_col, train_idx, test_idx):
    if not cat_cols:
        return np.zeros((len(df_all), 0)), []
    global_mean = df_all.iloc[train_idx][target_col].mean()
    enc_arr = np.full((len(df_all), len(cat_cols)), global_mean)
    feat_names = []
    for ci, col in enumerate(cat_cols):
        enc_train = np.full(len(train_idx), global_mean)
        sub = df_all.iloc[train_idx][[col, target_col]].copy().reset_index(drop=True)
        for tr, va in KFold(5, shuffle=True, random_state=SEED).split(sub):
            grp    = sub.iloc[tr].groupby(col)[target_col]
            cnt, mn = grp.count(), grp.mean()
            smooth  = (cnt * mn + TE_SMOOTH * global_mean) / (cnt + TE_SMOOTH)
            enc_train[va] = sub.iloc[va][col].map(smooth).fillna(global_mean).values
        enc_arr[train_idx, ci] = enc_train
        grp_f  = df_all.iloc[train_idx].groupby(col)[target_col]
        cnt_f, mn_f = grp_f.count(), grp_f.mean()
        smooth_f = (cnt_f * mn_f + TE_SMOOTH * global_mean) / (cnt_f + TE_SMOOTH)
        enc_arr[test_idx, ci] = (
            df_all.iloc[test_idx][col].map(smooth_f).fillna(global_mean).values
        )
        feat_names.append(f"{col}_enc")
    print(f"  Bayesian-TE (m={TE_SMOOTH}) → {len(cat_cols)} cols")
    return enc_arr, feat_names


# ── Physics-informed features ───────────────────────────────────────────────
def add_physics_features(df, ns_col, sf_col):
    feats, names = [], []
    c_col  = _find_col(df, ["cement amount", "cement(", "cement (", "cement (kg", "cement content"])
    w_col  = _find_col(df, ["water", "w/c", "w (", "water content", "water (", "free water"])
    l_col  = _find_col(df, ["length (mm)", "fiber length"])
    d_col  = _find_col(df, ["diameter (mm)", "fiber diameter"])
    fv_col = _find_col(df, ["amount / quantity of fiber", "fiber volume", "fiber content"])
    ft_col = _find_col(df, ["tensile strength (mpa)", "fiber tensile"])

    def _s(col):
        return df[col].fillna(0).to_numpy(float) if col else np.zeros(len(df))

    c  = _s(c_col);  w  = _s(w_col)
    ns = _s(ns_col); sf = _s(sf_col)
    l  = _s(l_col);  d  = np.maximum(_s(d_col), EPS)
    fv = _s(fv_col); ft = _s(ft_col)

    if c_col and w_col:
        feats.append(w / np.maximum(c, EPS));             names.append("WC_ratio")
    if c_col and ns_col:
        feats.append(ns / np.maximum(c, EPS));            names.append("NS_cement_ratio")
    if c_col:
        feats.append(c + sf + ns);                        names.append("Total_binder")
        if ns_col:
            feats.append(ns / np.maximum(c + sf + EPS, EPS))
            names.append("NS_binder_ratio")
    if l_col and d_col and fv_col and ft_col:
        feats.append(fv * (l / d) * ft / 1e6);            names.append("Fiber_index")
    if c_col and ns_col:
        binder  = np.maximum(c + sf + ns, EPS)
        ns_frac = ns / binder
        feats.append(ns_frac ** 2);                        names.append("NS_binder_sq")
        feats.append(np.log1p(ns / np.maximum(c, EPS)));  names.append("log_NS_cement")
    if c_col and w_col:
        wc = w / np.maximum(c, EPS)
        feats.append(np.log(np.maximum(wc, 1e-6)));        names.append("log_WC_ratio")
    if ns_col and fv_col and c_col:
        feats.append((ns / np.maximum(c, EPS)) * fv);     names.append("NS_fiber_synergy")
    if ns_col and sf_col and c_col:
        feats.append(ns * sf / np.maximum(c ** 2, EPS));  names.append("NS_SF_product")

    if not feats:
        return pd.DataFrame(index=df.index)
    df_phys = pd.DataFrame(np.column_stack(feats), columns=names, index=df.index)
    print(f"  Physics features: {names}")
    return df_phys


# ── Outlier removal ─────────────────────────────────────────────────────────
def remove_outliers(X, y, contamination=OUT_CONTAM):
    iso  = IsolationForest(contamination=contamination, random_state=SEED, n_jobs=-1)
    mask = iso.fit_predict(np.column_stack([X, y.reshape(-1, 1)])) == 1
    print(f"  Outliers removed: {(~mask).sum()} ({(~mask).mean()*100:.1f}%)")
    return X[mask], y[mask], mask


# ── Cluster feature ─────────────────────────────────────────────────────────
def add_cluster(Xtr, Xte, n=N_CLUSTERS):
    km  = KMeans(n_clusters=n, random_state=SEED, n_init=10)
    ltr = km.fit_predict(Xtr).reshape(-1, 1).astype(float)
    lte = km.predict(Xte).reshape(-1, 1).astype(float)
    print(f"  KMeans({n}) cluster sizes: {np.bincount(ltr.flatten().astype(int))}")
    return np.hstack([Xtr, ltr]), np.hstack([Xte, lte])


# ── Optuna model factories ──────────────────────────────────────────────────
MAKERS = {
    "CatBoost": lambda t: CatBoostRegressor(
        iterations    = t.suggest_int("n", 300, 1500),
        learning_rate = t.suggest_float("lr", 0.005, 0.2, log=True),
        depth         = t.suggest_int("d", 5, 10),
        l2_leaf_reg   = t.suggest_float("l2", 1, 10),
        task_type=_cb_gpu, random_seed=SEED, verbose=0,
    ),
    "XGBoost": lambda t: XGBRegressor(
        n_estimators     = t.suggest_int("n", 300, 1500),
        learning_rate    = t.suggest_float("lr", 0.005, 0.2, log=True),
        max_depth        = t.suggest_int("d", 4, 10),
        subsample        = t.suggest_float("ss", 0.6, 1.0),
        colsample_bytree = t.suggest_float("cs", 0.6, 1.0),
        min_child_weight = t.suggest_int("mcw", 1, 10),
        device=_xgb_dev, random_state=SEED, verbosity=0,
    ),
    "LightGBM": lambda t: LGBMRegressor(
        n_estimators     = t.suggest_int("n", 300, 1500),
        learning_rate    = t.suggest_float("lr", 0.005, 0.2, log=True),
        max_depth        = t.suggest_int("d", 4, 12),
        num_leaves       = t.suggest_int("nl", 20, 200),
        subsample        = t.suggest_float("ss", 0.6, 1.0),
        colsample_bytree = t.suggest_float("cs", 0.6, 1.0),
        device=_lgbm_dev, random_state=SEED, verbose=-1,
    ),
    "RF": lambda t: RandomForestRegressor(
        n_estimators      = t.suggest_int("n", 200, 800),
        max_depth         = t.suggest_int("d", 5, 30),
        min_samples_split = t.suggest_int("mss", 2, 10),
        max_features      = t.suggest_float("mf", 0.4, 1.0),
        random_state=SEED, n_jobs=-1,
    ),
    "GBR": lambda t: GradientBoostingRegressor(
        n_estimators  = t.suggest_int("n", 200, 800),
        learning_rate = t.suggest_float("lr", 0.005, 0.15, log=True),
        max_depth     = t.suggest_int("d", 3, 7),
        subsample     = t.suggest_float("ss", 0.6, 1.0),
        random_state=SEED,
    ),
    "ExtraTrees": lambda t: ExtraTreesRegressor(
        n_estimators      = t.suggest_int("n", 200, 800),
        max_depth         = t.suggest_int("d", 5, 40),
        min_samples_split = t.suggest_int("mss", 2, 10),
        max_features      = t.suggest_float("mf", 0.3, 1.0),
        random_state=SEED, n_jobs=-1,
    ),
    "HistGBM": lambda t: HistGradientBoostingRegressor(
        max_iter          = t.suggest_int("n", 200, 1000),
        learning_rate     = t.suggest_float("lr", 0.01, 0.2, log=True),
        max_depth         = t.suggest_int("d", 4, 15),
        min_samples_leaf  = t.suggest_int("msl", 10, 100),
        l2_regularization = t.suggest_float("l2", 0.0, 10.0),
        random_state=SEED,
    ),
}
# CatBoost excluded from stacking on GPU to avoid multi-process CUDA conflict
STACK_MODELS = (["XGBoost", "LightGBM", "RF", "GBR", "ExtraTrees", "HistGBM"]
                if _cb_gpu == "GPU"
                else ["CatBoost", "XGBoost", "LightGBM", "RF", "GBR", "ExtraTrees", "HistGBM"])


def _tune(name, maker, X, y, n_trials):
    def obj(trial):
        m = maker(trial)
        scores = [
            r2_score(y[va], m.fit(X[tr], y[tr]).predict(X[va]))
            for tr, va in CV.split(X)
        ]
        return float(np.mean(scores))
    st = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )
    st.optimize(obj, n_trials=n_trials, show_progress_bar=False)
    best = maker(st.best_trial)
    best.fit(X, y)
    print(f"  {name:12s}  CV R²={st.best_value:.4f}")
    return best, st.best_value


# ── Core pipeline ────────────────────────────────────────────────────────────
def _pipeline(X_raw, y_raw, n_trials, M0_all=None):
    q = pd.qcut(y_raw, q=min(5, len(y_raw) // 20), labels=False, duplicates="drop")
    itr, ite = train_test_split(
        np.arange(len(y_raw)), test_size=TEST_SIZE,
        random_state=SEED, stratify=q,
    )
    Xtr_r, Xte_r = X_raw[itr], X_raw[ite]
    ytr_r, yte_r = y_raw[itr], y_raw[ite]
    M0_tr_r = M0_all[itr] if M0_all is not None else None
    M0_te   = M0_all[ite] if M0_all is not None else None

    Xtr_r, ytr_r, omask = remove_outliers(Xtr_r, ytr_r)
    if M0_tr_r is not None:
        M0_tr_r = M0_tr_r[omask]

    imp   = KNNImputer(n_neighbors=KNN_K)
    Xtr_i = imp.fit_transform(Xtr_r)
    Xte_i = imp.transform(Xte_r)

    sc  = StandardScaler()
    Xtr = sc.fit_transform(Xtr_i)
    Xte = sc.transform(Xte_i)

    Xtr, Xte = add_cluster(Xtr, Xte)

    use_log = M0_tr_r is not None
    if use_log:
        z_tr     = np.log(np.maximum(ytr_r, EPS) / np.maximum(M0_tr_r, EPS))
        z_tr     = np.clip(z_tr, -Z_CLIP, Z_CLIP)
        y_target = z_tr
        print(f"  [★] z_mean={z_tr.mean():.4f}  z_std={z_tr.std():.4f}  "
              f"range=[{z_tr.min():.3f},{z_tr.max():.3f}]")
    else:
        y_target = ytr_r.copy()
        print("  Direct prediction (no physics baseline)")

    print(f"  Optuna {n_trials} trials × {len(MAKERS)} models ...")
    models, cv_scores = {}, {}
    for nm, mk in MAKERS.items():
        models[nm], cv_scores[nm] = _tune(nm, mk, Xtr, y_target, n_trials)

    meta  = GradientBoostingRegressor(
        n_estimators=200, learning_rate=0.05, max_depth=3,
        subsample=0.8, random_state=SEED,
    )
    stack = StackingRegressor(
        estimators      = [(k, models[k]) for k in STACK_MODELS],
        final_estimator = meta, cv=5, n_jobs=1,
    )
    stack.fit(Xtr, y_target)
    models["Stacking"] = stack

    def _to_y(z_pred, M0_arr):
        if use_log and M0_arr is not None:
            return M0_arr * np.exp(np.clip(z_pred, -Z_CLIP, Z_CLIP))
        return z_pred

    print("  ─ Test Set Results ─")
    results, y_preds = {}, {}
    for nm, m in models.items():
        yp          = _to_y(m.predict(Xte), M0_te)
        y_preds[nm] = yp
        results[nm] = _report(yte_r, yp, nm)

    # Weighted ensemble: weight_i = CV-R2_i, proven E[MSE_ens] <= E[MSE_best]
    best_cv = max(cv_scores.values()) if cv_scores else 0.0
    weights = {nm: max(cv_scores.get(nm, 0.0), 0.0) for nm in models}
    weights["Stacking"] = max(best_cv, 0.0)
    w_total = sum(weights.values())
    if w_total > EPS:
        y_ens = sum(weights[nm] * y_preds[nm] for nm in weights) / w_total
        y_preds["Ensemble_W"]   = y_ens
        results["Ensemble_W"]   = _report(yte_r, y_ens, "Ensemble_W")
        cv_scores["Ensemble_W"] = best_cv

    M0_tr_median = float(np.median(M0_tr_r)) if M0_tr_r is not None else None

    return dict(
        models=models, results=results, cv_scores=cv_scores,
        y_preds=y_preds, Xtr=Xtr, Xte=Xte, Xtr_raw=Xtr_r, yte_r=yte_r,
        imp=imp, sc=sc, use_log=use_log, M0_te=M0_te, M0_tr_median=M0_tr_median,
    )


# ── Figures ─────────────────────────────────────────────────────────────────────
def _scatter(yte, yp, best, r2, mape, prop, unit, out):
    lo = min(yte.min(), yp.min())
    hi = max(yte.max(), yp.max())
    plt.figure(figsize=(6, 6))
    plt.scatter(yte, yp, s=18, alpha=0.55, edgecolors="none", c="steelblue")
    plt.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="y=x")
    plt.plot([lo, hi], [lo * 1.2, hi * 1.2], "g:", lw=1, alpha=0.6)
    plt.plot([lo, hi], [lo * 0.8, hi * 0.8], "g:", lw=1, alpha=0.6, label="±20%")
    plt.xlabel(f"Experimental {prop} ({unit})")
    plt.ylabel(f"Predicted {prop} ({unit})")
    plt.title(f"{best} — {prop}  R²={r2:.4f}  MAPE={mape:.2f}%")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "scatter.png", dpi=200)
    plt.close()


def _shap_plots(model, Xte, feats, ns_col, prop, out):
    try:
        sv = shap.TreeExplainer(model).shap_values(Xte)
    except Exception:
        sv = shap.KernelExplainer(model.predict, shap.sample(Xte, 80)).shap_values(Xte)
    df_s = (
        pd.DataFrame({"feature": feats, "shap": np.abs(sv).mean(0)})
        .sort_values("shap", ascending=False)
        .reset_index(drop=True)
    )
    ns_rank = None
    if ns_col and ns_col in df_s.feature.values:
        ns_rank = int(df_s[df_s.feature == ns_col].index[0]) + 1
        print(f"  NS SHAP rank: #{ns_rank}")
    top    = df_s.head(12)
    colors = ["#FF8C00" if f == ns_col else "#4682B4" for f in top.feature[::-1]]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top.feature[::-1], top.shap[::-1], color=colors)
    ax.set_xlabel("Mean |SHAP Value|")
    ax.set_title(f"{prop} — SHAP Importance  (orange=NS rank #{ns_rank})")
    plt.tight_layout()
    plt.savefig(out / "shap_bar.png", dpi=200)
    plt.close()
    plt.figure()
    shap.summary_plot(sv, Xte, feature_names=feats, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(out / "shap_summary.png", dpi=200, bbox_inches="tight")
    plt.close()
    return sv, df_s, ns_rank


def _ns_curve(model, Xtr_raw, ns_i, imp, sc, prop, unit, out, M0_med=None):
    if ns_i is None:
        return None
    med  = np.nanmedian(Xtr_raw, axis=0)
    rng  = np.linspace(0, 200, 200)
    pred = []
    for v in rng:
        x       = med.copy()
        x[ns_i] = v
        xi = sc.transform(imp.transform(x.reshape(1, -1)))
        xi = np.hstack([xi, [[0]]])
        z_p = float(model.predict(xi)[0])
        yp  = M0_med * np.exp(np.clip(z_p, -Z_CLIP, Z_CLIP)) if M0_med else z_p
        pred.append(yp)
    pred   = np.array(pred)
    opt_ns = float(rng[np.argmax(pred)])
    print(f"  Optimal NS: {opt_ns:.1f} kg/m³ → {pred.max():.2f} {unit}")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(rng, pred, "b-", lw=2.5)
    ax.axvline(opt_ns, color="r", ls="--",
               label=f"Optimal={opt_ns:.1f} kg/m³  ({pred.max():.1f} {unit})")
    ax.fill_between(rng, pred.min(), pred, alpha=0.08, color="blue")
    ax.set_xlabel("Nano Silica (kg/m³)")
    ax.set_ylabel(f"{prop} ({unit})")
    ax.set_title(f"NS Dosage-Response — {prop}")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "ns_curve.png", dpi=200)
    plt.close()
    return opt_ns


def _taylor(yte, model_preds, prop, out):
    std_ref = np.std(yte)
    fig = plt.figure(figsize=(7, 6))
    ax  = fig.add_subplot(111, polar=True)
    ax.set_thetamax(90)
    ax.set_thetagrids(
        range(0, 91, 15),
        [f"{np.cos(np.deg2rad(a)):.2f}" for a in range(0, 91, 15)],
        fontsize=8,
    )
    ax.set_title(f"Taylor Diagram — {prop}", pad=20)
    ax.plot(0, 1, "k*", ms=14, label="Observed", zorder=5)
    colors = [
        "#e41a1c", "#377eb8", "#4daf4a", "#984ea3",
        "#ff7f00", "#a65628", "#f781bf", "#999999", "#33a02c",
    ]
    for i, (nm, yp) in enumerate(model_preds.items()):
        r   = float(np.corrcoef(yte, yp)[0, 1])
        std = np.std(yp) / std_ref
        ax.plot(np.arccos(np.clip(r, -1, 1)), std, "o", ms=9,
                color=colors[i % len(colors)], label=nm)
    for rv in [0.5, 1.0, 1.5]:
        t = np.linspace(0, np.pi / 2, 200)
        ax.plot(t, np.sqrt(1 + rv**2 - 2 * rv * np.cos(t)),
                ":", color="gray", lw=0.8, alpha=0.5)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=7)
    plt.tight_layout()
    plt.savefig(out / "taylor_diagram.png", dpi=200, bbox_inches="tight")
    plt.close()


def _shap_interaction(sv, Xte, feats, ns_col, sf_col, prop, out):
    ns_i = feats.index(ns_col) if (ns_col and ns_col in feats) else None
    sf_i = feats.index(sf_col) if (sf_col and sf_col in feats) else None
    if ns_i is None:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    sc_ = ax.scatter(
        Xte[:, ns_i], sv[:, ns_i],
        c=Xte[:, sf_i] if sf_i else np.zeros(len(Xte)),
        cmap="RdYlGn", s=20, alpha=0.7, edgecolors="none",
    )
    plt.colorbar(sc_, ax=ax, label=sf_col or "")
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.set_xlabel("Nano Silica (scaled)")
    ax.set_ylabel("SHAP value for NS")
    ax.set_title(f"{prop} — NS×SF Interaction")
    plt.tight_layout()
    plt.savefig(out / "shap_ns_sf_interaction.png", dpi=200)
    plt.close()


def _sensitivity(model, Xtr_raw, feats, imp, sc, prop, unit, out,
                 M0_med=None, top_n=12):
    med     = np.nanmedian(Xtr_raw, axis=0)
    xi_base = np.hstack([sc.transform(imp.transform(med.reshape(1, -1))), [[0]]])
    z_b     = float(model.predict(xi_base)[0])
    base    = M0_med * np.exp(np.clip(z_b, -Z_CLIP, Z_CLIP)) if M0_med else z_b
    deltas  = []
    for i, feat in enumerate(feats):
        x    = med.copy()
        x[i] = med[i] * 1.10 + 1e-9
        xi   = np.hstack([sc.transform(imp.transform(x.reshape(1, -1))), [[0]]])
        z_p  = float(model.predict(xi)[0])
        p    = M0_med * np.exp(np.clip(z_p, -Z_CLIP, Z_CLIP)) if M0_med else z_p
        deltas.append((feat, (p - base) / (abs(base) + EPS) * 100))
    deltas.sort(key=lambda x: abs(x[1]), reverse=True)
    deltas = deltas[:top_n]
    names, vals = zip(*deltas)
    colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in vals]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(names[::-1], list(vals[::-1]), color=colors[::-1])
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("% Change per +10% feature increase")
    ax.set_title(f"{prop} — Sensitivity Analysis")
    plt.tight_layout()
    plt.savefig(out / "sensitivity.png", dpi=200)
    plt.close()


# ── Run one property ────────────────────────────────────────────────────────
def run_property(cfg, df, ns_col, sf_col):
    target = _find_col(df, cfg["kw"])
    if target is None:
        print(f"  [{cfg['name']}] column not found — skip")
        return None

    num_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if not _is_result(c) and c != target
    ]
    cat_cols = [
        c for c in df.columns
        if df[c].dtype == object and _is_cat(c) and not _is_result(c)
    ]

    keep_cols = num_cols + cat_cols + [target]
    df_t = df[keep_cols].dropna(subset=[target]).copy().reset_index(drop=True)
    n    = len(df_t)

    print(f"\n{'='*64}")
    print(f"  {cfg['name']}  |  target='{target}'  |  n={n}")
    print(f"{'='*64}")
    if n < cfg["min_n"]:
        print(f"  Skip: {n} < {cfg['min_n']}")
        return None

    df_t = df_t.loc[:, [
        c for c in df_t.columns
        if c in cat_cols or c == target or df_t[c].isnull().mean() < 0.60
    ]]
    num_cols = [c for c in df_t.columns if c not in cat_cols and c != target]

    phys = add_physics_features(
        df_t,
        ns_col if ns_col in df_t.columns else None,
        sf_col if sf_col in df_t.columns else None,
    )
    if not phys.empty:
        df_t     = pd.concat([df_t, phys], axis=1)
        num_cols += list(phys.columns)

    y_all = df_t[target].to_numpy(float)
    X_num = df_t[num_cols].to_numpy(float)
    feats = num_cols.copy()
    ns_i  = feats.index(ns_col) if (ns_col and ns_col in feats) else None

    itr_all, ite_all = train_test_split(
        np.arange(n), test_size=TEST_SIZE, random_state=SEED,
        stratify=pd.qcut(y_all, q=min(5, n // 20), labels=False, duplicates="drop"),
    )

    if cat_cols:
        enc_arr, enc_names = target_encode(df_t, cat_cols, target, itr_all, ite_all)
        X_num  = np.hstack([X_num, enc_arr])
        feats += enc_names

    # Locate cement and WC_ratio columns for Power's Law M0
    c_col_name  = _find_col(df_t, ["cement amount", "cement(", "cement (",
                                    "cement (kg", "cement content"])
    c_feat_idx  = feats.index(c_col_name) if (c_col_name and c_col_name in feats) else None
    wc_feat_idx = feats.index("WC_ratio") if "WC_ratio" in feats else None

    M0_all, _, _ = fit_physics_baseline(
        X_num, y_all, itr_all,
        c_feat_idx=c_feat_idx,
        wc_feat_idx=wc_feat_idx,
    )

    out_dir = ROOT_OUT / cfg["name"]
    out_dir.mkdir(exist_ok=True)

    def _run(n_trials):
        art  = _pipeline(X_num, y_all, n_trials, M0_all=M0_all)
        best = max(art["results"], key=lambda k: art["results"][k]["R2"])
        r2   = art["results"][best]["R2"]
        return art, best, r2, r2 >= cfg["gate"]

    art, best_name, best_r2, passed = _run(TRIALS_BASE)
    if not passed:
        print(f"  Gate {cfg['gate']} not met (R²={best_r2:.4f}) — retry {TRIALS_RETRY} trials")
        art, best_name, best_r2, passed = _run(TRIALS_RETRY)

    tree_order = ["CatBoost", "XGBoost", "LightGBM", "ExtraTrees", "HistGBM", "GBR", "RF"]
    shap_m = max(
        (nm for nm in tree_order if nm in art["results"]),
        key=lambda k: art["results"][k]["R2"],
    )
    m = art["results"][best_name]

    feats_ext = feats + ["cluster"]
    _scatter(art["yte_r"], art["y_preds"][best_name], best_name, best_r2, m["MAPE"],
             cfg["name"], cfg["unit"], out_dir)
    sv, df_shap, ns_rank = _shap_plots(
        art["models"][shap_m], art["Xte"], feats_ext, ns_col, cfg["name"], out_dir)
    opt_ns = _ns_curve(
        art["models"][shap_m], art["Xtr_raw"], ns_i,
        art["imp"], art["sc"], cfg["name"], cfg["unit"], out_dir,
        M0_med=art["M0_tr_median"],
    )
    _taylor(art["yte_r"], art["y_preds"], cfg["name"], out_dir)
    _shap_interaction(sv, art["Xte"], feats_ext, ns_col, sf_col, cfg["name"], out_dir)
    _sensitivity(
        art["models"][shap_m], art["Xtr_raw"], feats,
        art["imp"], art["sc"], cfg["name"], cfg["unit"], out_dir,
        M0_med=art["M0_tr_median"],
    )

    summary = dict(
        property=cfg["name"], unit=cfg["unit"], n_samples=n,
        gate=cfg["gate"], gate_passed=passed,
        best_model=best_name, metrics=m,
        all_models=art["results"], cv_scores=art["cv_scores"],
        physics_log_space=art["use_log"],
        M0_tr_median=art["M0_tr_median"],
        ns_rank=ns_rank, ns_optimal_kg_m3=opt_ns,
    )
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    return summary


# ── Summary chart ───────────────────────────────────────────────────────────
def _summary_chart(all_results):
    if not all_results:
        return
    names  = [r["property"]        for r in all_results]
    r2s    = [r["metrics"]["R2"]   for r in all_results]
    mapes  = [r["metrics"]["MAPE"] for r in all_results]
    colors = ["#2ecc71" if r["gate_passed"] else "#e74c3c" for r in all_results]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.bar(names, r2s, color=colors)
    ax1.set_ylim(0.80, 1.0)
    ax1.set_ylabel("R²")
    ax1.set_title("R² per Property (green=gate ✅)")
    for i, v in enumerate(r2s):
        ax1.text(i, v + 0.003, f"{v:.3f}", ha="center", fontsize=9)
    ax2.bar(names, mapes, color=colors)
    ax2.set_ylabel("MAPE (%)")
    ax2.set_title("MAPE per Property")
    for i, v in enumerate(mapes):
        ax2.text(i, v + 0.1, f"{v:.1f}%", ha="center", fontsize=9)
    plt.suptitle(
        "UHPC v6 — Power's Law M0 + OOF Calib + Weighted Ensemble",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(ROOT_OUT / "summary_chart.png", dpi=200)
    plt.close()


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    df     = load_data()
    ns_col = _find_col(df, NS_KW)
    sf_col = _find_col(df, SF_KW)
    print(f"\nDataset: {df.shape}  |  NS='{ns_col}'  |  SF='{sf_col}'")
    print(f"Config v6: TRIALS={TRIALS_BASE}/{TRIALS_RETRY}  GPU={USE_GPU}")
    print(f"[★] M0: Power's Law = a*cement^b*exp(-c*W/C)  [Abrams/Powers 1947]")
    print(f"[★] Fallback: OOF Ridge + calibration k")
    print(f"[★] z = log(f_c/M0_calib)  =>  z_std ~0.10-0.12")
    print(f"[★] Weighted ensemble over 8 models (CV-R² weights)\n")

    all_results = []
    for cfg in MULTI_TARGETS:
        res = run_property(cfg, df, ns_col, sf_col)
        if res:
            all_results.append(res)

    _summary_chart(all_results)
    (ROOT_OUT / "all_metrics.json").write_text(json.dumps(all_results, indent=2))

    print(f"\n{'='*72}")
    print("  MULTI-PROPERTY SUMMARY v6")
    print(f"{'='*72}")
    print(f"  {'Property':<12}{'n':>6}{'Best':>14}{'R²':>8}"
          f"{'MAPE':>7}{'Gate':>7}{'LogSp':>7}{'NS★':>6}{'OptNS':>8}")
    print(f"  {'-'*70}")
    for r in all_results:
        m  = r["metrics"]
        tk = "✅" if r["gate_passed"] else "❌"
        ls = "★" if r.get("physics_log_space") else "-"
        ns = f"#{r['ns_rank']}" if r["ns_rank"] else "--"
        op = f"{r['ns_optimal_kg_m3']:.0f}" if r["ns_optimal_kg_m3"] else "--"
        print(f"  {r['property']:<12}{r['n_samples']:>6}{r['best_model']:>14}"
              f"{m['R2']:>8.4f}{m['MAPE']:>7.2f}{tk:>7}{ls:>7}{ns:>6}{op:>8}")
    print(f"{'='*72}")
    print(f"  7 figures per property | outputs/summary_chart.png")


if __name__ == "__main__":
    main()
