# ============================================================
# app/streamlit_app.py
# Corrosion RC Beam Optimizer — Professional Streamlit UI
# v7 — 100% clean Streamlit 1.35+ API, premium design,
#      fast loading, zero DeltaGenerator leaks
# ============================================================

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import joblib
import plotly.graph_objects as go
import plotly.express as px

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from config import (
    APP_TITLE, APP_ICON, APP_LAYOUT,
    MODELS_DIR, FIGURES_DIR, EQ_DIR, RESULTS_DIR,
    FEATURE_COLS, TARGET_COL,
    L1_TARGET_R2, L2_TARGET_R2,
)

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout=APP_LAYOUT,
    initial_sidebar_state="expanded",
)

# ============================================================
# PREMIUM CSS THEME
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Header bar */
.app-header {
    background: linear-gradient(135deg, #0D1B2A 0%, #1B3A5C 50%, #2E5C8A 100%);
    padding: 2rem 2.5rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    color: white;
    box-shadow: 0 8px 32px rgba(13, 27, 42, 0.3);
}
.app-header h1 {
    font-size: 2rem;
    font-weight: 800;
    margin: 0 0 0.3rem 0;
    color: white;
    letter-spacing: -0.5px;
}
.app-header p {
    font-size: 1rem;
    color: #B0C8E8;
    margin: 0;
    font-weight: 400;
}

/* Section headers */
.section-title {
    font-size: 1.4rem;
    font-weight: 700;
    color: #0D1B2A;
    margin: 1.5rem 0 0.8rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 3px solid #2E5C8A;
    display: inline-block;
}

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #F8FAFD 0%, #EDF2F9 100%);
    border: 1px solid #D0DCF0;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    transition: transform 0.2s, box-shadow 0.2s;
}
.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(0,0,0,0.08);
}
.metric-card .label {
    font-size: 0.75rem;
    font-weight: 600;
    color: #6B7B8D;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 0.3rem;
}
.metric-card .value {
    font-size: 2rem;
    font-weight: 800;
    color: #0D1B2A;
}
.metric-card .delta {
    font-size: 0.8rem;
    font-weight: 500;
    margin-top: 0.2rem;
}
.delta-positive { color: #2E7D32; }
.delta-negative { color: #C62828; }

/* Verdict badges */
.verdict-pass {
    background: linear-gradient(135deg, #E8F5E9, #C8E6C9);
    border-left: 5px solid #2E7D32;
    padding: 1rem 1.5rem;
    border-radius: 10px;
    color: #1B5E20;
    font-weight: 600;
    font-size: 1rem;
    margin: 1rem 0;
}
.verdict-fail {
    background: linear-gradient(135deg, #FFEBEE, #FFCDD2);
    border-left: 5px solid #C62828;
    padding: 1rem 1.5rem;
    border-radius: 10px;
    color: #B71C1C;
    font-weight: 600;
    margin: 1rem 0;
}

/* Equation cards */
.eq-card {
    background: linear-gradient(135deg, #F8FAFD, #EDF2F9);
    border: 1px solid #D0DCF0;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 0.8rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}
.eq-best {
    background: linear-gradient(135deg, #E8F5E9, #C8E6C9);
    border: 2px solid #2E7D32;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 0.8rem;
    box-shadow: 0 4px 16px rgba(46,125,50,0.15);
}
.eq-label {
    font-size: 0.75rem;
    font-weight: 700;
    color: #2E7D32;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 0.5rem;
}

/* Stat validation table */
.stat-pass { color: #2E7D32; font-weight: 600; }
.stat-fail { color: #C62828; font-weight: 600; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0D1B2A 0%, #1B3A5C 100%);
}
section[data-testid="stSidebar"] * {
    color: #E0E8F0 !important;
}
section[data-testid="stSidebar"] .stMarkdown h2 {
    color: white !important;
    font-weight: 700;
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: #F0F4F8;
    padding: 6px;
    border-radius: 12px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    font-weight: 600;
    font-size: 0.85rem;
    padding: 8px 16px;
}
.stTabs [aria-selected="true"] {
    background: #0D1B2A !important;
    color: white !important;
    border-radius: 8px;
}

/* Button */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #1A3A5C, #2E5C8A);
    border: none;
    border-radius: 10px;
    font-weight: 600;
    padding: 0.7rem 2rem;
    transition: all 0.3s;
}
.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #2E5C8A, #3A7CB8);
    box-shadow: 0 4px 16px rgba(26,58,92,0.3);
    transform: translateY(-1px);
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# STATIC EQUATION CATALOGUE
# ============================================================
EQ_CATALOGUE = [
    {"id":1, "complexity":3, "name":"Linear depth only",
     "sympy":"0.136*d", "latex":r"M = 0.136\,d",
     "variables":["d"], "corrosion_aware":False, "fy_aware":False,
     "rho_aware":False, "fc_aware":False,
     "pros":"Simplest — 1 variable, instant hand calculation.",
     "cons":"Ignores corrosion, reinforcement and strength.",
     "use_case":"Quick order-of-magnitude estimate.", "r2_approx":0.51},
    {"id":2, "complexity":5, "name":"Linear depth with offset",
     "sympy":"0.312*d - 39.22", "latex":r"M = 0.312\,d - 39.22",
     "variables":["d"], "corrosion_aware":False, "fy_aware":False,
     "rho_aware":False, "fc_aware":False,
     "pros":"Better calibrated linear form.",
     "cons":"Negative for small d. No corrosion.",
     "use_case":"Slight improvement over Eq.1.", "r2_approx":0.58},
    {"id":3, "complexity":7, "name":"Depth & corrosion (additive)",
     "sympy":"0.313*(d - eta_m) - 36.47",
     "latex":r"M = 0.313(d - \eta_m) - 36.47",
     "variables":["d","eta_m"], "corrosion_aware":True, "fy_aware":False,
     "rho_aware":False, "fc_aware":False,
     "pros":"First equation with corrosion. Physically interpretable.",
     "cons":"Corrosion enters additively — incorrect coupling. No strength.",
     "use_case":"Educational demonstration only.", "r2_approx":0.62},
    {"id":4, "complexity":8, "name":"√ρt × depth",
     "sympy":"sqrt(rho_t)*(0.330*d - 42.26)",
     "latex":r"M = \sqrt{\rho_t}(0.330\,d - 42.26)",
     "variables":["rho_t","d"], "corrosion_aware":False, "fy_aware":False,
     "rho_aware":True, "fc_aware":False,
     "pros":"First to include reinforcement ratio.",
     "cons":"No corrosion, no yield strength.",
     "use_case":"Proof of concept for ρ importance.", "r2_approx":0.69},
    {"id":5, "complexity":9, "name":"(fy×ρt) power-law with depth",
     "sympy":"(fy*rho_t)**log(log(0.025*d))",
     "latex":r"M = (f_y \rho_t)^{\log\log(0.025\,d)}",
     "variables":["fy","rho_t","d"], "corrosion_aware":False,
     "fy_aware":True, "rho_aware":True, "fc_aware":False,
     "pros":"Double-log scaling — dimensionally elegant.",
     "cons":"No corrosion. log(log()) undefined for small d.",
     "use_case":"Baseline for uncorroded beams.", "r2_approx":0.78},
    {"id":6, "complexity":11, "name":"(fy×ρt) power / db",
     "sympy":"(fy*rho_t)**log(log(0.029*d)) / d_b",
     "latex":r"M = \frac{(f_y\rho_t)^{\log\log(0.029\,d)}}{d_b}",
     "variables":["fy","rho_t","d","d_b"], "corrosion_aware":False,
     "fy_aware":True, "rho_aware":True, "fc_aware":False,
     "pros":"Bar diameter in denominator — physically correct.",
     "cons":"No corrosion. Unstable for large bars.",
     "use_case":"Uncorroded beams, variable bar size.", "r2_approx":0.80},
    {"id":7, "complexity":17, "name":"fy power × ρt (corrosion in exponent)",
     "sympy":"0.074*b + rho_t*(fy**log(log(0.023*(d-eta_m))) - 2.824)",
     "latex":r"M = 0.074b + \rho_t\left(f_y^{\log\log[0.023(d-\eta_m)]} - 2.82\right)",
     "variables":["fy","rho_t","d","b","eta_m"], "corrosion_aware":True,
     "fy_aware":True, "rho_aware":True, "fc_aware":False,
     "pros":"Corrosion embedded in depth argument.",
     "cons":"Corrosion reduces effective depth — slight physical mismatch.",
     "use_case":"Corroded beams when db unavailable.", "r2_approx":0.84},
    {"id":8, "complexity":19,
     "name":"★ RECOMMENDED — fy power × corrosion × reinforcement",
     "sympy":"fy**log(log(0.182*sqrt(d))) * (14.5-sqrt(eta_m)) * (-d_b+rho_t+2.69)",
     "latex":r"M_{max} = f_y^{\log\log(0.182\sqrt{d})}\cdot(14.5-\sqrt{\eta_m})\cdot(\rho_t - d_b + 2.69)",
     "variables":["fy","d","eta_m","rho_t","d_b"],
     "corrosion_aware":True, "fy_aware":True, "rho_aware":True, "fc_aware":False,
     "pros":"Best complexity/accuracy trade-off. √ηm physically meaningful — damage grows as √mass loss. Only 5 variables.",
     "cons":"No f'c. log(log()) needs calculator. Constants need re-calibration for different steel grades.",
     "use_case":"Primary design-code candidate. Best for journal publication.",
     "r2_approx":0.9328, "rmse":6.261, "mae":4.104, "mape":28.63},
    {"id":9, "complexity":25,
     "name":"★ MOST PHYSICAL — includes f'c",
     "sympy":"(6.58-sqrt(eta_m/sqrt(fc)))*(rho_t+3.85*exp(-d_b))*(fy**log(log(0.209*sqrt(d))))**1.177",
     "latex":r"M_{max} = \left(6.58-\sqrt{\frac{\eta_m}{\sqrt{f'_c}}}\right)(\rho_t+3.85e^{-d_b})\left[f_y^{\log\log(0.209\sqrt{d})}\right]^{1.177}",
     "variables":["fy","d","eta_m","rho_t","d_b","fc"],
     "corrosion_aware":True, "fy_aware":True, "rho_aware":True, "fc_aware":True,
     "pros":"Only equation with f'c — most physically complete. ηm/√f'c is physically meaningful.",
     "cons":"Most complex (25 nodes). Three empirical exponents. Requires all 6 variables.",
     "use_case":"Next-gen design code. Recommended for journal with f'c data.",
     "r2_approx":0.938},
]


# ============================================================
# CACHED LOADERS — fast, safe
# ============================================================
@st.cache_resource(show_spinner=False)
def _load_model():
    p = MODELS_DIR / "best_model.pkl"
    return joblib.load(p) if p.exists() else None

@st.cache_resource(show_spinner=False)
def _load_scaler_X():
    p = MODELS_DIR / "scaler_X.pkl"
    return joblib.load(p) if p.exists() else None

@st.cache_resource(show_spinner=False)
def _load_scaler_y():
    p = MODELS_DIR / "scaler_y.pkl"
    return joblib.load(p) if p.exists() else None

@st.cache_resource(show_spinner=False)
def _load_cat_encoders():
    p = MODELS_DIR / "cat_encoders.json"
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)

@st.cache_data(show_spinner=False)
def _load_json(path_str: str) -> dict:
    p = Path(path_str)
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)


def _label_encode(col_name, value, encoders):
    classes = encoders.get(col_name, [])
    return classes.index(value) if value in classes else 0


# ============================================================
# SIDEBAR
# ============================================================
def _sidebar():
    st.sidebar.markdown(f"## {APP_ICON} {APP_TITLE}")
    st.sidebar.markdown(
        "**PhD Research Pipeline**\n\n"
        "CatBoost + SHAP + PySR\n\n"
        "Predicts **Mmax (kN·m)** of corroded RC beams."
    )
    st.sidebar.markdown("---")

    model = _load_model()
    if model:
        st.sidebar.success("✅ Model (CatBoost): Loaded")
    else:
        st.sidebar.error("❌ Model not found")

    st.sidebar.markdown("### Benchmark Targets")
    st.sidebar.markdown(
        f"🏅 **L1** (ACI 318-19): R² > {L1_TARGET_R2}\n\n"
        f"🏆 **L2** (Zhang 2025): R² > {L2_TARGET_R2}"
    )
    st.sidebar.markdown("---")
    st.sidebar.caption(f"Results: `{RESULTS_DIR.relative_to(ROOT)}`")


# ============================================================
# TAB 1 — PREDICT
# ============================================================
def _tab_predict():
    st.markdown('<div class="section-title">🏗️ Beam Mmax Predictor</div>', unsafe_allow_html=True)
    st.markdown("Enter corroded RC beam parameters. Model: **CatBoost (R² = 0.987)**.")

    model    = _load_model()
    scaler_X = _load_scaler_X()
    scaler_y = _load_scaler_y()
    encoders = _load_cat_encoders()
    if model is None:
        st.error("⚠️ Model file not found. Run the pipeline first.")
        return

    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Section Geometry**")
        b  = st.number_input("Width b (mm)",     100, 350, 150, step=5)
        d  = st.number_input("Depth d (mm)",     100, 500, 300, step=5)
        L  = st.number_input("Test Length (mm)", 500, 5000, 2500, step=50)
        cv = st.number_input("Bottom Cover (mm)", 15,  60,  25,  step=1)
    with c2:
        st.markdown("**Reinforcement**")
        n_bars = st.number_input("# Tensile Bars",   1, 8, 3, step=1)
        db     = st.number_input("Bar Diameter (mm)", 6, 32, 16, step=2)
        pten   = st.number_input("ρ tension (%)",    0.1, 5.0, 1.5, step=0.1)
        fy     = st.number_input("fy (MPa)",        226, 650, 460, step=5)
    with c3:
        st.markdown("**Concrete & Corrosion**")
        fc       = st.number_input("f'c (MPa)",           20,  80, 32, step=1)
        wc       = st.number_input("W/C Ratio",          0.30, 0.70, 0.45, step=0.01)
        eta_m    = st.number_input("ηm — Mass Loss (%)", 0.0, 64.0, 10.0, step=0.5)
        s_stirr  = st.number_input("Stirrup Spacing (mm)", 50, 300, 150, step=10)
        ds_stirr = st.number_input("Stirrup Dia. (mm)",   6,  16,  8,  step=2)
        fy_s     = st.number_input("fy stirrups (MPa)",  226, 650, 420, step=5)
        shear_x  = st.number_input("Shear Span x (mm)", 100, 2000, 800, step=50)

    st.markdown("---")
    st.markdown("**Bar & Test Configuration**")
    cc1, cc2, cc3 = st.columns(3)
    with cc1:
        bar_type_classes = encoders.get("Longitudinal Bar Type", ["D", "P"])
        bar_type_label   = st.selectbox("Longitudinal Bar Type", options=bar_type_classes, index=0)
    with cc2:
        test_classes = encoders.get("Test Type and Configuration", ["SS_FPB_MONO", "SS_TPB_MONO"])
        test_label   = st.selectbox("Test Configuration", options=test_classes,
                                     index=min(2, len(test_classes)-1))
    with cc3:
        corr_classes = encoders.get("Corrosion Method", ["IC", "EI", "C", "N"])
        corr_label   = st.selectbox("Corrosion Method", options=corr_classes,
                                     index=min(2, len(corr_classes)-1))

    st.markdown("---")
    if st.button("🔍 Predict Mmax (kN·m)", type="primary", use_container_width=True):
        num_vals = {
            "Width (mm)"                              : b,
            "Depth (mm)"                              : d,
            "Test Length (mm)"                        : L,
            "Bottom Cover to Ctr of Tension Bar (mm)" : cv,
            "# Tensile Bars"                          : n_bars,
            "Diameter Tensile Bars, db,t (mm)"        : db,
            "Tension Reinforcement Ratio, pten (%)"   : pten,
            "fy Longitudinal Bars (Tensile), (MPa) "  : fy,
            "f'c (MPa)"                               : fc,
            "W/C Ratio"                               : wc,
            "Stirrup Spacing, s (mm) "                : s_stirr,
            "Stirrup Diameter, ds (mm)"               : ds_stirr,
            "fy,s Stirrup Bars"                       : fy_s,
            "Mass Loss (Tensile bars), \u03b7m (%)"   : eta_m,
            "Shear Span, x (mm)"                      : shear_x,
        }
        cat_vals = {
            "Longitudinal Bar Type"       : _label_encode("Longitudinal Bar Type", bar_type_label, encoders),
            "Test Type and Configuration" : _label_encode("Test Type and Configuration", test_label, encoders),
            "Corrosion Method"            : _label_encode("Corrosion Method", corr_label, encoders),
        }
        eta_log  = np.log1p(eta_m)
        As_proxy = n_bars * np.pi * (db / 2.0) ** 2
        eng_vals = {
            "eta_log"           : eta_log,
            "corr_severity_idx" : eta_m  * (fy / max(fc, 1)),
            "d_b_ratio"         : d      / max(b, 1),
            "eta_d_interaction" : eta_log * d,
            "reinf_index"       : As_proxy * fy / (fc * b * d),
        }
        all_cols = (
            FEATURE_COLS
            + ["Longitudinal Bar Type", "Test Type and Configuration", "Corrosion Method"]
            + ["eta_log", "corr_severity_idx", "d_b_ratio", "eta_d_interaction", "reinf_index"]
        )
        all_vals = {**num_vals, **cat_vals, **eng_vals}
        try:
            row    = np.array([all_vals.get(c, 0.0) for c in all_cols], dtype=float).reshape(1, -1)
            row_sc = scaler_X.transform(row) if scaler_X else row
            mmax_pred = float(model.predict(row_sc)[0]) if hasattr(model.predict(row_sc), '__len__') else float(model.predict(row_sc))
            
            from aci_calculator import aci_moment_capacity
            mn_aci = aci_moment_capacity(b=b, d=d, n_bars=n_bars, db_mm=db, fy=fy, fc=fc, eta_m=eta_m)

            # Display results
            m1, m2, m3 = st.columns(3)
            m1.metric("📊 Predicted Mmax (kN·m)", f"{mmax_pred:.2f}",
                       f"{mmax_pred-mn_aci:+.2f} vs ACI")
            m2.metric("📏 ACI 318-19 Mn (kN·m)", f"{mn_aci:.2f}")
            m3.metric("🧠 Corrosion Severity", f"{eng_vals['corr_severity_idx']:.1f}")

            # Gauge chart
            ax_max = max(mmax_pred, mn_aci) * 1.6
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=round(mmax_pred, 2),
                delta={"reference": mn_aci, "valueformat": ".2f"},
                title={"text": "Predicted Mmax (kN·m)"},
                gauge={
                    "axis": {"range": [0, ax_max]},
                    "bar":  {"color": "#1A3A5C"},
                    "steps": [
                        {"range": [0, mn_aci*0.5],  "color": "#FFCDD2"},
                        {"range": [mn_aci*0.5, mn_aci], "color": "#FFF9C4"},
                        {"range": [mn_aci, ax_max],  "color": "#C8E6C9"},
                    ],
                    "threshold": {"line": {"color": "orange", "width": 3}, "value": mn_aci},
                },
            ))
            fig.update_layout(height=300, margin=dict(t=30, b=10, l=20, r=20))
            st.plotly_chart(fig, use_container_width=True)

            ratio = mmax_pred / max(mn_aci, 1)
            if ratio >= 0.90:
                st.success(f"✅ Mmax = {mmax_pred:.2f} kN·m — Low corrosion impact.")
            elif ratio >= 0.60:
                st.warning(f"⚠️ Mmax = {mmax_pred:.2f} kN·m — Moderate impact. Inspect beam.")
            else:
                st.error(f"❌ Mmax = {mmax_pred:.2f} kN·m — Severe degradation!")
        except Exception as e:
            st.error(f"Prediction error: {e}")


# ============================================================
# TAB 2 — ENSEMBLE
# ============================================================
def _tab_ensemble():
    st.markdown('<div class="section-title">🧩 Ensemble Models Comparison</div>', unsafe_allow_html=True)
    st.markdown("All 5 models trained on 804 beams. **CatBoost** selected as final predictor.")

    ens = _load_json(str(MODELS_DIR / "ensemble_metrics.json"))
    mlp = _load_json(str(MODELS_DIR / "mlp_metrics.json"))
    if not ens:
        st.info("ensemble_metrics.json not found.")
        return

    rows = []

    # Optional: Read MLP first
    if mlp and "test" in mlp:
        rows.append({
            "Model"    : "MLP Baseline",
            "Train R²" : round(mlp.get("train", {}).get("R2", 0), 4),
            "Test R²"  : round(mlp.get("test", {}).get("R2", 0), 4),
            "RMSE"     : round(mlp.get("test", {}).get("RMSE", 0), 4),
            "MAE"      : round(mlp.get("test", {}).get("MAE", 0), 4),
            "L1 ✓"    : "✅" if mlp.get("test", {}).get("L1_broken") else "❌",
            "L2 ✓"    : "✅" if mlp.get("test", {}).get("L2_broken") else "❌",
        })

    for name, m in ens.get("models", {}).items():
        rows.append({
            "Model"    : ("⭐ " if name == "CatBoost" else "") + name,
            "Train R²" : round(m.get("train_R2",  0), 4),
            "Test R²"  : round(m.get("test_R2",   0), 4),
            "RMSE"     : round(m.get("test_RMSE", 0), 4),
            "MAE"      : round(m.get("test_MAE",  0), 4),
            "L1 ✓"    : "✅" if m.get("L1_broken") else "❌",
            "L2 ✓"    : "✅" if m.get("L2_broken") else "❌",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("---")

    # CV Folds chart
    cv_folds = ens.get("cv_folds", [])
    if cv_folds:
        st.markdown("### CatBoost 10-Fold CV R²")
        c1, c2, c3 = st.columns(3)
        c1.metric("CV R² Mean", f"{ens.get('cv_R2_mean',0):.4f}")
        c2.metric("CV R² Std",  f"{ens.get('cv_R2_std',0):.4f}")
        c3.metric("L2 Broken",  "✅ Yes" if ens.get("L2_broken") else "❌ No")

        fig = go.Figure(go.Bar(
            x=[f"Fold {i+1}" for i in range(len(cv_folds))],
            y=cv_folds,
            marker_color=["#2E7D32" if v >= L2_TARGET_R2 else "#1A3A5C" for v in cv_folds],
            text=[f"{v:.4f}" for v in cv_folds], textposition="outside",
        ))
        fig.add_hline(y=L2_TARGET_R2, line_dash="dash", line_color="red",
                      annotation_text=f"L2 target ({L2_TARGET_R2})")
        fig.update_layout(title="CatBoost — R² per CV Fold",
                          yaxis=dict(range=[0.93, 1.0]), height=350,
                          margin=dict(t=40, b=20, l=40, r=20))
        st.plotly_chart(fig, use_container_width=True)

    # All models bar chart
    st.markdown("### Test R² — All Models")
    names = [r["Model"] for r in rows]
    r2s   = [r["Test R²"] for r in rows]
    fig2  = go.Figure(go.Bar(
        x=names, y=r2s,
        marker_color=["#2E7D32" if v >= L2_TARGET_R2 else "#1A3A5C" for v in r2s],
        text=[f"{v:.4f}" for v in r2s], textposition="outside",
    ))
    fig2.add_hline(y=L1_TARGET_R2, line_dash="dot", line_color="#F57C00",
                   annotation_text=f"L1 ({L1_TARGET_R2})")
    fig2.add_hline(y=L2_TARGET_R2, line_dash="dash", line_color="red",
                   annotation_text=f"L2 ({L2_TARGET_R2})")
    fig2.update_layout(yaxis=dict(range=[0.94, 1.0]), height=350,
                       margin=dict(t=40, b=20, l=40, r=20))
    st.plotly_chart(fig2, use_container_width=True)


# ============================================================
# TAB 3 — RESULTS
# ============================================================
def _tab_results():
    st.markdown('<div class="section-title">📊 Model Performance & Benchmarks</div>', unsafe_allow_html=True)

    ens = _load_json(str(MODELS_DIR / "ensemble_metrics.json"))
    aci = _load_json(str(MODELS_DIR / "aci_benchmark_metrics.json"))
    if not ens and not aci:
        st.info("No results found.")
        return

    cat_m = ens.get("models", {}).get("CatBoost", {}) if ens else {}

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### ACI 318-19 Baseline")
        if aci:
            st.markdown(f"""
<div class="metric-card">
    <div class="label">R²</div>
    <div class="value">{aci.get('R2', '—')}</div>
</div>
""", unsafe_allow_html=True)
            st.metric("RMSE (kN·m)", f"{aci.get('RMSE', '—')}")
            st.metric("MAE (kN·m)", f"{aci.get('MAE', '—')}")
            st.metric("MAPE", f"{aci.get('MAPE', '—')}%")
            st.metric("Exp/Pred Ratio", f"{aci.get('ratio_mean', '—')}")

    with col2:
        st.markdown("### CatBoost — Best Model")
        if cat_m:
            tr2 = cat_m.get("test_R2", 0)
            st.markdown(f"""
<div class="metric-card">
    <div class="label">R²</div>
    <div class="value" style="color:#2E7D32">{tr2}</div>
    <div class="delta delta-positive">+{tr2 - aci.get('R2', 0):.4f} vs ACI</div>
</div>
""", unsafe_allow_html=True)
            rmse_d = cat_m.get("test_RMSE", 0) - aci.get("RMSE", 0)
            mae_d  = cat_m.get("test_MAE", 0) - aci.get("MAE", 0)
            st.metric("RMSE (kN·m)", f"{cat_m.get('test_RMSE', '—')}", f"{rmse_d:+.4f} vs ACI")
            st.metric("MAE (kN·m)",  f"{cat_m.get('test_MAE', '—')}",  f"{mae_d:+.4f} vs ACI")

            l1 = cat_m.get("L1_broken", False)
            l2 = cat_m.get("L2_broken", False)
            if l1 and l2:
                st.markdown('<div class="verdict-pass">🏆 Both benchmarks broken ✓</div>',
                            unsafe_allow_html=True)
            elif l1:
                st.markdown('<div class="verdict-pass">✓ L1 beaten</div>',
                            unsafe_allow_html=True)

    # Statistical validation section
    st.markdown("---")
    stat = _load_json(str(MODELS_DIR / "statistical_validation.json"))
    if stat:
        st.markdown("### 📈 Statistical Validation")
        stat_rows = []

        w = stat.get("wilcoxon", {})
        if w:
            stat_rows.append({
                "Test": "Wilcoxon Signed-Rank",
                "Statistic": f"W = {w.get('statistic', '—')}",
                "p-value": f"{w.get('p_value', '—')}",
                "Result": "✅ Significant" if w.get("significant") else "❌ Not significant",
            })

        bs = stat.get("bootstrap", {})
        if bs:
            ci = bs.get("R2_CI", [0, 0])
            stat_rows.append({
                "Test": "Bootstrap 95% CI",
                "Statistic": f"R² = {bs.get('R2', '—')}",
                "p-value": f"[{ci[0]:.4f}, {ci[1]:.4f}]",
                "Result": "✅ Stable" if ci[0] > 0.95 else "⚠️ Variable",
            })

        cd = stat.get("cohens_d", {})
        if cd:
            stat_rows.append({
                "Test": "Cohen's d",
                "Statistic": f"d = {cd.get('cohens_d', '—'):.4f}",
                "p-value": cd.get("magnitude", "—"),
                "Result": f"Effect: {cd.get('magnitude', '—')}",
            })

        mc = stat.get("mcnemar", {})
        if mc:
            stat_rows.append({
                "Test": "McNemar",
                "Statistic": f"χ² = {mc.get('chi2_statistic', '—'):.4f}",
                "p-value": f"{mc.get('p_value', '—')}",
                "Result": f"Model {mc.get('model_accuracy','—')}% vs ACI {mc.get('aci_accuracy','—')}%",
            })

        if stat_rows:
            st.dataframe(pd.DataFrame(stat_rows), use_container_width=True, hide_index=True)

        verdict = stat.get("verdict", "")
        if "PASSED" in verdict:
            st.markdown(f'<div class="verdict-pass">{verdict}</div>', unsafe_allow_html=True)
        elif verdict:
            st.markdown(f'<div class="verdict-fail">{verdict}</div>', unsafe_allow_html=True)


# ============================================================
# TAB 4 — SHAP
# ============================================================
def _tab_shap():
    st.markdown('<div class="section-title">🤖 SHAP Feature Importance</div>', unsafe_allow_html=True)

    top5_path = MODELS_DIR / "top5_shap_features.json"
    if top5_path.exists():
        top5 = _load_json(str(top5_path))
        feats = top5.get("top5_features", [])
        if feats:
            st.success("📌 Top-5 features: " + ", ".join(feats))

    c1, c2 = st.columns(2)

    imp_path = FIGURES_DIR / "shap_importance.png"
    bee_path = FIGURES_DIR / "shap_beeswarm.png"

    with c1:
        if imp_path.exists():
            st.image(str(imp_path), caption="Mean |SHAP|", use_container_width=True)
        else:
            st.info("shap_importance.png not found.")

    with c2:
        if bee_path.exists():
            st.image(str(bee_path), caption="Beeswarm Plot", use_container_width=True)
        else:
            st.info("shap_beeswarm.png not found.")

    # Dependence plots
    dep_plots = sorted(FIGURES_DIR.glob("shap_dependence_*.png"))
    if dep_plots:
        st.markdown("### Dependence Plots")
        for dep in dep_plots:
            st.image(str(dep), caption=f"SHAP Dependence: {dep.stem.replace('shap_dependence_', '')}",
                     use_container_width=True)

    # SHAP importance table
    shap_csv = MODELS_DIR / "shap_importance.csv"
    if shap_csv.exists():
        st.markdown("### Feature Importance Ranking")
        df = pd.read_csv(shap_csv)
        st.dataframe(df, use_container_width=True, hide_index=True)


# ============================================================
# TAB 5 — EQUATIONS
# ============================================================
def _tab_equation():
    st.markdown('<div class="section-title">📝 Symbolic Regression — Discovered Equations</div>',
                unsafe_allow_html=True)
    st.markdown(
        "**PySR** evolved **closed-form equations** over 200 iterations × 40 populations "
        "on 804 specimens. Each equation is a Pareto-optimal trade-off between "
        "*complexity* and *accuracy*."
    )

    # Pareto frontier chart
    st.markdown("---")
    st.markdown("### 📈 Pareto Frontier — Complexity vs Accuracy")
    comp_vals = [e["complexity"] for e in EQ_CATALOGUE]
    r2_vals   = [e["r2_approx"]  for e in EQ_CATALOGUE]
    colors_eq = [
        "#2E7D32" if e["id"] in (8, 9) else
        "#1A3A5C" if e["corrosion_aware"] else "#B0BEC5"
        for e in EQ_CATALOGUE
    ]

    fig_pareto = go.Figure()
    fig_pareto.add_trace(go.Scatter(
        x=comp_vals, y=r2_vals, mode="lines",
        line=dict(color="#CFD8DC", dash="dot", width=1), showlegend=False))
    fig_pareto.add_trace(go.Scatter(
        x=comp_vals, y=r2_vals, mode="markers+text",
        text=[f"Eq.{e['id']}" for e in EQ_CATALOGUE],
        textposition="top center",
        marker=dict(size=10, color=colors_eq, line=dict(width=1, color="white")),
        hovertext=[f"Eq.{e['id']}: {e['name'][:35]}" for e in EQ_CATALOGUE],
        hoverinfo="text+y", showlegend=False,
    ))
    fig_pareto.add_hline(y=0.8839, line_dash="dot", line_color="#F57C00",
                         annotation_text="ACI 318-19 (R²=0.884)")
    # Recommended stars
    for eid, col, label in [(8, "#2E7D32", "★ Recommended"), (9, "#9C27B0", "★ Most Physical")]:
        eq = next(e for e in EQ_CATALOGUE if e["id"] == eid)
        fig_pareto.add_trace(go.Scatter(
            x=[eq["complexity"]], y=[eq["r2_approx"]],
            mode="markers", marker=dict(size=18, symbol="star", color=col),
            name=label,
        ))
    fig_pareto.update_layout(
        xaxis_title="Equation Complexity (nodes)",
        yaxis_title="R²",
        yaxis=dict(range=[0.0, 1.02]),
        height=420, margin=dict(t=30, b=40, l=50, r=30),
        legend=dict(x=0.02, y=0.98),
    )
    st.plotly_chart(fig_pareto, use_container_width=True)

    # Comparison table
    st.markdown("---")
    st.markdown("### 📋 Equation Comparison Table")
    table_rows = []
    for e in EQ_CATALOGUE:
        delta = e["r2_approx"] - 0.8839
        table_rows.append({
            "Eq.": f"{'★ ' if e['id'] in (8,9) else ''}Eq.{e['id']}",
            "Complexity": e["complexity"],
            "Name": e["name"][:40],
            "R²": round(e["r2_approx"], 4),
            "ΔR² vs ACI": f"{delta:+.4f}",
            "ηm": "✅" if e["corrosion_aware"] else "—",
            "fy": "✅" if e["fy_aware"] else "—",
            "ρ": "✅" if e["rho_aware"] else "—",
            "f'c": "✅" if e["fc_aware"] else "—",
        })
    st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

    # Detailed equation cards
    st.markdown("---")
    st.markdown("### 🔬 Detailed Equation Cards")

    show_corr = st.checkbox("Show corrosion-aware only", value=False)
    filtered = [e for e in EQ_CATALOGUE if e["corrosion_aware"]] if show_corr else EQ_CATALOGUE

    for eq in filtered:
        is_rec = eq["id"] in (8, 9)
        card_class = "eq-best" if is_rec else "eq-card"
        label = ""
        if eq["id"] == 8:
            label = '<div class="eq-label">⭐ RECOMMENDED — Best complexity/accuracy trade-off</div>'
        elif eq["id"] == 9:
            label = '<div class="eq-label">⭐ MOST PHYSICAL — Only equation with f\'c</div>'

        st.markdown(f'<div class="{card_class}">{label}'
                    f'<b>Eq.{eq["id"]} — Complexity {eq["complexity"]} — '
                    f'{eq["name"][:50]}</b></div>', unsafe_allow_html=True)

        col_l, col_r = st.columns([3, 2])
        with col_l:
            st.latex(eq["latex"])
            st.markdown(f"**Variables:** `{'`, `'.join(eq['variables'])}`")
            badges = ""
            for flag, lbl in [("corrosion_aware","ηm"), ("fy_aware","fy"),
                               ("rho_aware","ρ"), ("fc_aware","f'c")]:
                badges += ("✅ " if eq[flag] else "❌ ") + lbl + "  "
            st.markdown(badges)
        with col_r:
            r2_v = eq.get("r2_approx")
            if r2_v:
                st.metric("R²", f"{r2_v:.4f}", f"{r2_v-0.8839:+.4f} vs ACI")
            rmse = eq.get("rmse")
            if rmse:
                st.metric("RMSE (kN·m)", f"{rmse:.3f}")

        st.markdown(f"✅ **Pros:** {eq['pros']}")
        st.markdown(f"❌ **Cons:** {eq['cons']}")
        st.markdown(f"💡 **Use case:** {eq['use_case']}")
        st.markdown("---")

    # Interactive calculator
    st.markdown("### 🧮 Hand Calculator")
    st.markdown("Select any equation to compute Mmax — no model needed.")

    eq_options = {f"Eq.{e['id']}: {e['name'][:50]}": e for e in EQ_CATALOGUE}
    chosen_label = st.selectbox("Choose equation", list(eq_options.keys()),
                                index=list(eq_options.keys()).index(
                                    next(k for k in eq_options if "Eq.8" in k)))
    chosen_eq = eq_options[chosen_label]
    needed = chosen_eq["variables"]

    calc_cols = st.columns(min(len(needed), 4))
    calc_inputs = {}
    defaults = {"d": 300, "eta_m": 10.0, "fy": 460, "rho_t": 1.5,
                "d_b": 16.0, "b": 150, "fc": 32}
    labels   = {"d": "d (mm)", "eta_m": "ηm (%)", "fy": "fy (MPa)",
                "rho_t": "ρt (%)", "d_b": "db (mm)", "b": "b (mm)", "fc": "f'c (MPa)"}
    for i, var in enumerate(needed):
        with calc_cols[i % min(len(needed), 4)]:
            calc_inputs[var] = st.number_input(
                labels.get(var, var),
                value=float(defaults.get(var, 1.0)),
                key=f"calc_{chosen_eq['id']}_{var}")

    if st.button("⚡ Compute", use_container_width=True):
        try:
            v = calc_inputs
            d_v = v.get("d", 300); eta_v = v.get("eta_m", 10); fy_v = v.get("fy", 460)
            rho_v = v.get("rho_t", 1.5); db_v = v.get("d_b", 16)
            b_v = v.get("b", 150); fc_v = v.get("fc", 32)
            
            # Note: PySR was trained on RAW features, but `d_b` inside PySR actually refers
            # to the `d_b_ratio` (d/b) column, not the reinforcement bar diameter (db,t)!
            # We must map this correctly if the equation uses the variable `d_b`
            py_db = d_v / max(b_v, 1)

            eid = chosen_eq["id"]
            M = float("nan")

            if eid == 1:
                M = 0.136 * d_v
            elif eid == 2:
                M = 0.312 * d_v - 39.22
            elif eid == 3:
                M = 0.313 * (d_v - eta_v) - 36.47
            elif eid == 4:
                M = np.sqrt(rho_v) * (0.330 * d_v - 42.26)
            elif eid == 5:
                arg = np.log(0.025 * d_v)
                M = (fy_v * rho_v) ** np.log(arg) if arg > 0 else float("nan")
            elif eid == 6:
                arg = np.log(0.029 * d_v)
                M = (fy_v * rho_v) ** np.log(arg) / py_db if arg > 0 else float("nan")
            elif eid == 7:
                arg = 0.023 * (d_v - eta_v)
                if arg > 0:
                    M = 0.074 * b_v + rho_v * (fy_v ** np.log(np.log(arg)) - 2.824)
            elif eid == 8:
                arg = np.log(0.182 * np.sqrt(d_v))
                if arg > 0:
                    M = fy_v ** np.log(arg) * (14.5 - np.sqrt(eta_v)) * (-py_db + rho_v + 2.69)
            elif eid == 9:
                arg = np.log(0.209 * np.sqrt(d_v))
                if arg > 0:
                    M = ((6.58 - np.sqrt(eta_v / np.sqrt(fc_v))) *
                         (rho_v + 3.85 * np.exp(-py_db)) *
                         (fy_v ** np.log(arg)) ** 1.177)

            if np.isnan(M) or np.isinf(M) or M < 0:
                st.error("⚠️ Undefined for these inputs. Result may evaluate to negative or NaN depending on bounds. Are measurements realistic?")
            else:
                st.success(f"**Mmax ≈ {M:.2f} kN·m** (Eq.{eid})")
                st.info(f"💡 Note: `d_b` in the equation denotes depth/width ratio (`d/b = {py_db:.2f}`), *not* bar diameter.")
                st.caption(f"CatBoost R²=0.987 | This equation R²≈{chosen_eq['r2_approx']:.4f} | ACI R²=0.884")
        except Exception as ex:
            st.error(f"Calculation error: {ex}")

    # Recommendation
    st.markdown("---")
    st.markdown("### 🏁 Recommendation")
    rec_data = [
        {"Goal": "Journal / Design code", "Equation": "Eq.8 (complexity 19)",
         "Why": "Best R²/simplicity; √ηm physically motivated; 5 variables"},
        {"Goal": "Maximum physical completeness", "Equation": "Eq.9 (complexity 25)",
         "Why": "Only one with f'c; ηm/√f'c dimensionally sound"},
        {"Goal": "Quick estimate", "Equation": "Eq.3 (complexity 7)",
         "Why": "2 variables; immediately interpretable"},
    ]
    st.dataframe(pd.DataFrame(rec_data), use_container_width=True, hide_index=True)


# ============================================================
# TAB 6 — REPORT
# ============================================================
def _tab_report():
    st.markdown('<div class="section-title">📄 Download PDF Report</div>', unsafe_allow_html=True)

    report = RESULTS_DIR / "Final_Report.pdf"
    if report.exists():
        with open(report, "rb") as f:
            st.download_button(
                label="⬇️ Download Final_Report.pdf",
                data=f, file_name="Corrosion_RC_Beam_Optimizer_Report.pdf",
                mime="application/pdf", use_container_width=True,
            )
        st.caption(f"Last modified: {pd.Timestamp(report.stat().st_mtime, unit='s')}")
    else:
        st.info("📔 Report not generated yet. Run `python src/main.py`.")

    st.markdown("---")
    if st.button("🔄 Regenerate Report", use_container_width=True):
        with st.spinner("Building PDF…"):
            try:
                from report_generator import generate_report
                path = generate_report()
                st.success(f"✅ Saved to: {path}")
                st.cache_data.clear()
                st.rerun()
            except Exception as e:
                st.error(f"Report generation failed: {e}")


# ============================================================
# MAIN ENTRY
# ============================================================
def main():
    _sidebar()

    # Premium header
    st.markdown(f"""
<div class="app-header">
    <h1>{APP_ICON} {APP_TITLE}</h1>
    <p>PhD Research — CatBoost (R² = 0.987) × SHAP × PySR Symbolic Regression</p>
</div>
""", unsafe_allow_html=True)

    tabs = st.tabs([
        "🏗️  Predict", "🧩  Ensemble", "📊  Results",
        "🤖  SHAP",   "📝  Equations", "📄  Report",
    ])
    with tabs[0]:
        _tab_predict()
    with tabs[1]:
        _tab_ensemble()
    with tabs[2]:
        _tab_results()
    with tabs[3]:
        _tab_shap()
    with tabs[4]:
        _tab_equation()
    with tabs[5]:
        _tab_report()


if __name__ == "__main__":
    main()
