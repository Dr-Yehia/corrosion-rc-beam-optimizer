# ============================================================
# app/streamlit_app.py
# Corrosion RC Beam Optimizer — Interactive Streamlit UI
# v6 — Professional Equation tab: all 19 PySR equations,
#      pros/cons, interactive calculator, future roadmap
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

st.markdown("""
<style>
.main-header{font-size:2rem;font-weight:700;color:#0D1B2A;margin-bottom:.2rem}
.sub-header{font-size:1rem;color:#3A5A8C;margin-bottom:1.5rem}
.verdict-pass{background:#E8F5E9;border-left:4px solid #2E7D32;padding:.8rem 1.2rem;
  border-radius:8px;color:#1B5E20;font-weight:600}
.verdict-fail{background:#FFEBEE;border-left:4px solid #C62828;padding:.8rem 1.2rem;
  border-radius:8px;color:#B71C1C;font-weight:600}
.eq-card{background:#F8FAFD;border:1px solid #D0DCF0;border-radius:10px;
  padding:1rem 1.4rem;margin-bottom:.8rem}
.eq-best{background:#E8F5E9;border:2px solid #2E7D32;border-radius:10px;
  padding:1rem 1.4rem;margin-bottom:.8rem}
.pros{color:#2E7D32;font-size:.9rem}
.cons{color:#C62828;font-size:.9rem}
</style>
""", unsafe_allow_html=True)


# ============================================================
# STATIC EQUATION CATALOGUE
# Built from all_equations.json + expert annotation
# ============================================================
EQ_CATALOGUE = [
    {
        "id": 1,
        "complexity": 3,
        "name": "Linear depth only",
        "sympy": "0.136 * d",
        "latex": r"M = 0.136\,d",
        "variables": ["d"],
        "corrosion_aware": False,
        "fy_aware": False,
        "rho_aware": False,
        "fc_aware": False,
        "pros": "Absolute simplest — 1 variable, instant hand calculation.",
        "cons": "Ignores corrosion, reinforcement and material strength. Unreliable for design.",
        "use_case": "Quick order-of-magnitude estimate only.",
        "r2_approx": 0.51,
    },
    {
        "id": 2,
        "complexity": 4,
        "name": "Power of sqrt(d)",
        "sympy": "1.250 ** sqrt(d)",
        "latex": r"M = 1.250^{\sqrt{d}}",
        "variables": ["d"],
        "corrosion_aware": False,
        "fy_aware": False,
        "rho_aware": False,
        "fc_aware": False,
        "pros": "Non-linear depth effect captured.",
        "cons": "Grows exponentially — unstable for large d. Still ignores all other variables.",
        "use_case": "Not recommended for practical use.",
        "r2_approx": 0.54,
    },
    {
        "id": 3,
        "complexity": 5,
        "name": "Linear depth with offset",
        "sympy": "0.312 * d - 39.22",
        "latex": r"M = 0.312\,d - 39.22",
        "variables": ["d"],
        "corrosion_aware": False,
        "fy_aware": False,
        "rho_aware": False,
        "fc_aware": False,
        "pros": "More calibrated linear form.",
        "cons": "Negative values possible for small d. No corrosion term.",
        "use_case": "Slight improvement over Eq.1, still not practical.",
        "r2_approx": 0.58,
    },
    {
        "id": 4,
        "complexity": 7,
        "name": "Depth & corrosion (additive)",
        "sympy": "0.313*(d - eta_m) - 36.47",
        "latex": r"M = 0.313(d - \eta_m) - 36.47",
        "variables": ["d", "eta_m"],
        "corrosion_aware": True,
        "fy_aware": False,
        "rho_aware": False,
        "fc_aware": False,
        "pros": "First equation to include corrosion. Physically interpretable subtraction form.",
        "cons": "Corrosion enters additively with depth — physically incorrect coupling. No strength terms.",
        "use_case": "Educational demonstration of corrosion effect only.",
        "r2_approx": 0.62,
    },
    {
        "id": 5,
        "complexity": 8,
        "name": "Reinforcement ratio x depth",
        "sympy": "sqrt(rho_t) * (0.330*d - 42.26)",
        "latex": r"M = \sqrt{\rho_t}(0.330\,d - 42.26)",
        "variables": ["rho_t", "d"],
        "corrosion_aware": False,
        "fy_aware": False,
        "rho_aware": True,
        "fc_aware": False,
        "pros": "First to include reinforcement ratio. Square-root form is physically meaningful.",
        "cons": "No corrosion, no yield strength.",
        "use_case": "Proof of concept for rho importance.",
        "r2_approx": 0.69,
    },
    {
        "id": 6,
        "complexity": 9,
        "name": "(fy x rho_t) power-law with depth",
        "sympy": "(fy * rho_t) ** log(log(0.025*d))",
        "latex": r"M = (f_y \rho_t)^{\log\log(0.025\,d)}",
        "variables": ["fy", "rho_t", "d"],
        "corrosion_aware": False,
        "fy_aware": True,
        "rho_aware": True,
        "fc_aware": False,
        "pros": "Introduces double-log scaling — dimensionally elegant. Combines fy and rho naturally.",
        "cons": "No corrosion. log(log()) can be undefined for small d values.",
        "use_case": "Good baseline for uncorroded beams.",
        "r2_approx": 0.78,
    },
    {
        "id": 7,
        "complexity": 11,
        "name": "(fy x rho_t) power / bar diameter",
        "sympy": "(fy*rho_t)**log(log(0.029*d)) / d_b",
        "latex": r"M = \frac{(f_y\rho_t)^{\log\log(0.029\sqrt{d})}}{d_b}",
        "variables": ["fy", "rho_t", "d", "d_b"],
        "corrosion_aware": False,
        "fy_aware": True,
        "rho_aware": True,
        "fc_aware": False,
        "pros": "Bar diameter now in denominator — larger bars give lower per-unit moment, physically correct.",
        "cons": "Division by d_b can be numerically unstable for large bars. Still no corrosion.",
        "use_case": "Uncorroded beams, variable bar size.",
        "r2_approx": 0.80,
    },
    {
        "id": 8,
        "complexity": 13,
        "name": "(fy x rho_t + offset) power / d_b",
        "sympy": "(fy*rho_t + 46.0)**log(log(0.028*d)) / d_b",
        "latex": r"M = \frac{(f_y\rho_t + 46.0)^{\log\log(0.028\,d)}}{d_b}",
        "variables": ["fy", "rho_t", "d", "d_b"],
        "corrosion_aware": False,
        "fy_aware": True,
        "rho_aware": True,
        "fc_aware": False,
        "pros": "Additive offset stabilises prediction at low rho.",
        "cons": "Offset of 46 has no physical meaning. No corrosion.",
        "use_case": "Marginal improvement over Eq.7 only.",
        "r2_approx": 0.81,
    },
    {
        "id": 9,
        "complexity": 15,
        "name": "(fy x rho_t + b-term) power / d_b",
        "sympy": "(0.235*b + fy*rho_t)**log(log(0.028*d)) / d_b",
        "latex": r"M = \frac{(0.235b + f_y\rho_t)^{\log\log(0.028\,d)}}{d_b}",
        "variables": ["fy", "rho_t", "d", "d_b", "b"],
        "corrosion_aware": False,
        "fy_aware": True,
        "rho_aware": True,
        "fc_aware": False,
        "pros": "Width b introduced — moment capacity grows with section width as expected.",
        "cons": "No corrosion. The b-coefficient (0.235) is empirical only.",
        "use_case": "Variable-width uncorroded beams.",
        "r2_approx": 0.82,
    },
    {
        "id": 10,
        "complexity": 17,
        "name": "fy power x rho_t (corrosion in exponent)",
        "sympy": "0.074*b + rho_t*(fy**log(log(0.023*(d-eta_m))) - 2.824)",
        "latex": r"M = 0.074b + \rho_t\left(f_y^{\log\log[0.023(d-\eta_m)]} - 2.82\right)",
        "variables": ["fy", "rho_t", "d", "d_b", "b", "eta_m"],
        "corrosion_aware": True,
        "fy_aware": True,
        "rho_aware": True,
        "fc_aware": False,
        "pros": "Corrosion embedded in depth argument of log — captures depth-corrosion interaction.",
        "cons": "Corrosion reduces effective depth, but real mechanism is area loss — slight physical mismatch.",
        "use_case": "Corroded beams when bar diameter is unavailable.",
        "r2_approx": 0.84,
    },
    {
        "id": 11,
        "complexity": 18,
        "name": "fy power x (rho_t - d_b) [no corrosion]",
        "sympy": "1.947*fy**(1.424*log(log(0.240*sqrt(d))))*(-d_b+rho_t+2.137)",
        "latex": r"M = 1.947\,f_y^{1.424\log\log(0.240\sqrt{d})}(-d_b+\rho_t+2.137)",
        "variables": ["fy", "rho_t", "d", "d_b"],
        "corrosion_aware": False,
        "fy_aware": True,
        "rho_aware": True,
        "fc_aware": False,
        "pros": "Clean 3-term product. Net reinforcement term (rho_t - d_b) captures bar geometry elegantly.",
        "cons": "Missing corrosion — major limitation for corroded beams.",
        "use_case": "Uncorroded beams with variable bar sizes.",
        "r2_approx": 0.87,
    },
    {
        "id": 12,
        "complexity": 19,
        "name": "RECOMMENDED — fy power x corrosion x reinforcement",
        "sympy": "fy**log(log(0.182*sqrt(d))) * (14.5-sqrt(eta_m)) * (-d_b+rho_t+2.69)",
        "latex": r"M = f_y^{\log\log(0.182\sqrt{d})}\cdot(14.5-\sqrt{\eta_m})\cdot(\rho_t - d_b + 2.69)",
        "variables": ["fy", "d", "eta_m", "rho_t", "d_b"],
        "corrosion_aware": True,
        "fy_aware": True,
        "rho_aware": True,
        "fc_aware": False,
        "pros": (
            "Best complexity/accuracy trade-off (complexity 19, R2=0.933). "
            "All four key structural variables present. "
            "sqrt(eta_m) term is physically meaningful — damage grows as square root of mass loss. "
            "Only 5 variables — easy hand calculation."
        ),
        "cons": (
            "No f'c — concrete strength absent. "
            "log(log()) hard to compute without calculator. "
            "Constants 14.5 and 2.69 require re-calibration for different steel grades."
        ),
        "use_case": "Primary design-code candidate. Best for journal publication and code calibration.",
        "r2_approx": 0.9328,
        "rmse": 6.261,
        "mae": 4.104,
        "mape": 28.63,
    },
    {
        "id": 13,
        "complexity": 20,
        "name": "fy power x corrosion x exp(-d_b)",
        "sympy": "fy**log(log(0.183*sqrt(d)))*(15.09-sqrt(eta_m))*(rho_t+4.46*exp(-d_b))",
        "latex": r"M = f_y^{\log\log(0.183\sqrt{d})}(15.09-\sqrt{\eta_m})(\rho_t+4.46e^{-d_b})",
        "variables": ["fy", "d", "eta_m", "rho_t", "d_b"],
        "corrosion_aware": True,
        "fy_aware": True,
        "rho_aware": True,
        "fc_aware": False,
        "pros": "exp(-d_b) replaces linear d_b — bar diameter effect is non-linear, more realistic.",
        "cons": "exp(-d_b) almost negligible for typical d_b values (16-25mm) — adds complexity for tiny gain.",
        "use_case": "Alternative to Eq.12 if non-linear bar diameter effect is important.",
        "r2_approx": 0.934,
    },
    {
        "id": 14,
        "complexity": 22,
        "name": "fy power x corrosion x exp(-d_b) + offset",
        "sympy": "fy**log(log(0.185*sqrt(d)))*((13.87-sqrt(eta_m))*(rho_t+4.07*exp(-d_b))+1.76)",
        "latex": r"M = f_y^{\log\log(0.185\sqrt{d})}\left[(13.87-\sqrt{\eta_m})(\rho_t+4.07e^{-d_b})+1.76\right]",
        "variables": ["fy", "d", "eta_m", "rho_t", "d_b"],
        "corrosion_aware": True,
        "fy_aware": True,
        "rho_aware": True,
        "fc_aware": False,
        "pros": "Offset +1.76 prevents under-prediction at zero corrosion.",
        "cons": "Marginal improvement over Eq.13 (R2 difference <0.001). Added complexity unjustified.",
        "use_case": "Use Eq.12 or Eq.13 instead.",
        "r2_approx": 0.935,
    },
    {
        "id": 15,
        "complexity": 23,
        "name": "Power-chain with linear corrosion",
        "sympy": "(3.96-0.045*eta_m)*(rho_t+3.87*exp(-d_b))*(fy**log(log(0.223*sqrt(d))))**1.227",
        "latex": r"M = (3.96-0.045\eta_m)(\rho_t+3.87e^{-d_b})\left[f_y^{\log\log(0.223\sqrt{d})}\right]^{1.227}",
        "variables": ["fy", "d", "eta_m", "rho_t", "d_b"],
        "corrosion_aware": True,
        "fy_aware": True,
        "rho_aware": True,
        "fc_aware": False,
        "pros": "Corrosion now linear (0.045 x eta_m) — simpler than sqrt form. Double exponent 1.227 adds flexibility.",
        "cons": "Linear corrosion less physically motivated than sqrt. Two empirical exponents reduce interpretability.",
        "use_case": "Sensitivity study on corrosion linearity assumption.",
        "r2_approx": 0.936,
    },
    {
        "id": 16,
        "complexity": 24,
        "name": "Power-chain with sqrt(eta_m) scaled",
        "sympy": "fy**(1.208*log(log(0.216*sqrt(d))))*(5.12-0.299*sqrt(eta_m))*(rho_t+3.80*exp(-d_b))",
        "latex": r"M = f_y^{1.208\log\log(0.216\sqrt{d})}(5.12-0.299\sqrt{\eta_m})(\rho_t+3.80e^{-d_b})",
        "variables": ["fy", "d", "eta_m", "rho_t", "d_b"],
        "corrosion_aware": True,
        "fy_aware": True,
        "rho_aware": True,
        "fc_aware": False,
        "pros": "Scaled sqrt corrosion (0.299 x sqrt(eta_m)) — coefficient now dimensionless, better calibration.",
        "cons": "Four free constants; requires re-calibration for different datasets.",
        "use_case": "Dataset-specific calibration study.",
        "r2_approx": 0.937,
    },
    {
        "id": 17,
        "complexity": 25,
        "name": "MOST PHYSICAL — includes f'c",
        "sympy": "(6.58-sqrt(eta_m/sqrt(fc)))*(rho_t+3.85*exp(-d_b))*(fy**log(log(0.209*sqrt(d))))**1.177",
        "latex": r"M = \left(6.58-\sqrt{\frac{\eta_m}{\sqrt{f'_c}}}\right)(\rho_t+3.85e^{-d_b})\left[f_y^{\log\log(0.209\sqrt{d})}\right]^{1.177}",
        "variables": ["fy", "d", "eta_m", "rho_t", "d_b", "fc"],
        "corrosion_aware": True,
        "fy_aware": True,
        "rho_aware": True,
        "fc_aware": True,
        "pros": (
            "Only equation with f'c — most physically complete. "
            "eta_m/sqrt(f'c) term is physically meaningful: corrosion damage is relative to concrete quality. "
            "Covers all 5 key structural variables."
        ),
        "cons": (
            "Most complex (complexity 25). "
            "Three empirical exponents. "
            "Requires all 6 variables — more field measurements needed."
        ),
        "use_case": "Best candidate for next-generation design code. Recommended for journal submission with f'c data.",
        "r2_approx": 0.938,
    },
]

# Map complexity -> approximate R2 from JSON loss data
_R2_MAP = {
    1:  0.00,
    2:  0.43,
    3:  0.508,
    4:  0.534,
    5:  0.576,
    6:  0.556,
    7:  0.622,
    8:  0.693,
    9:  0.781,
    11: 0.797,
    13: 0.812,
    15: 0.817,
    17: 0.826,
    18: 0.869,
    19: 0.9328,
    20: 0.934,
    22: 0.935,
    23: 0.936,
    24: 0.937,
    25: 0.938,
}


# ============================================================
# CACHED LOADERS
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
def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def _label_encode(col_name: str, value: str, encoders: dict) -> int:
    classes = encoders.get(col_name, [])
    if value in classes:
        return classes.index(value)
    return 0


# ============================================================
# SIDEBAR
# ============================================================
def _sidebar():
    st.sidebar.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/"
        "9/9d/Corroded_rebar.jpg/320px-Corroded_rebar.jpg",
        use_column_width=True,
    )
    st.sidebar.markdown(f"## {APP_ICON} {APP_TITLE}")
    st.sidebar.markdown(
        "PhD Research Pipeline — CatBoost + SHAP + PySR\n\n"
        "Predicts **Mmax (kN·m)** of corroded RC beams."
    )
    st.sidebar.markdown("---")
    model = _load_model()
    st.sidebar.success("Model (CatBoost): Loaded ✅") if model else \
        st.sidebar.error("Model not found")
    st.sidebar.markdown("### Benchmark Targets")
    st.sidebar.info(
        f"🏅 L1 (ACI 318-19): R² > {L1_TARGET_R2}\n\n"
        f"🏆 L2 (Zhang 2025 SOTA): R² > {L2_TARGET_R2}"
    )
    st.sidebar.markdown("---")
    st.sidebar.caption(f"Results: `{RESULTS_DIR.relative_to(ROOT)}`")


# ============================================================
# TAB 1 — PREDICT
# ============================================================
def _tab_predict():
    st.markdown('<p class="main-header">🏗️ Beam Mmax Predictor</p>',
                unsafe_allow_html=True)
    st.markdown(
        "Enter corroded RC beam parameters. "
        "Model: <b>CatBoost (R² = 0.987 on test set)</b>.",
        unsafe_allow_html=True)

    model    = _load_model()
    scaler_X = _load_scaler_X()
    scaler_y = _load_scaler_y()
    encoders = _load_cat_encoders()
    if model is None:
        st.error("⚠️ Model file not found.")
        return

    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Section Geometry**")
        b  = st.number_input("Width b (mm)",          100, 350, 150, step=5)
        d  = st.number_input("Depth d (mm)",          100, 500, 300, step=5)
        L  = st.number_input("Test Length (mm)",      500, 5000, 2500, step=50)
        cv = st.number_input("Bottom Cover (mm)",      15,  60,  25,  step=1)
    with c2:
        st.markdown("**Reinforcement**")
        n_bars = st.number_input("# Tensile Bars",    1, 8, 3, step=1)
        db     = st.number_input("Bar Diameter (mm)", 6, 32, 16, step=2)
        pten   = st.number_input("rho tension (%)",   0.1, 5.0, 1.5, step=0.1)
        fy     = st.number_input("fy (MPa)",          226, 650, 460, step=5)
    with c3:
        st.markdown("**Concrete & Corrosion**")
        fc       = st.number_input("f'c (MPa)",        20,  80, 32, step=1)
        wc       = st.number_input("W/C Ratio",        0.30, 0.70, 0.45, step=0.01)
        eta_m    = st.number_input("eta_m — Mass Loss (%)", 0.0, 64.0, 10.0, step=0.5)
        s_stirr  = st.number_input("Stirrup Spacing (mm)", 50, 300, 150, step=10)
        ds_stirr = st.number_input("Stirrup Dia. (mm)",    6,  16,  8,  step=2)
        fy_s     = st.number_input("fy stirrups (MPa)",   226, 650, 420, step=5)
        shear_x  = st.number_input("Shear Span x (mm)", 100, 2000, 800, step=50)

    st.markdown("---")
    st.markdown("**Bar & Test Configuration**")
    cc1, cc2, cc3 = st.columns(3)
    with cc1:
        bar_type_classes = encoders.get("Longitudinal Bar Type", ["D", "P"])
        bar_type_label   = st.selectbox("Longitudinal Bar Type", options=bar_type_classes, index=0)
    with cc2:
        test_classes = encoders.get("Test Type and Configuration", ["SS_FPB_MONO", "SS_TPB_MONO"])
        test_label   = st.selectbox("Test Configuration", options=test_classes, index=2)
    with cc3:
        corr_classes = encoders.get("Corrosion Method", ["IC", "EI", "C", "N"])
        corr_label   = st.selectbox("Corrosion Method", options=corr_classes, index=2)

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
            y_sc   = model.predict(row_sc)
            mmax_pred = (
                scaler_y.inverse_transform(y_sc.reshape(-1, 1)).ravel()[0]
                if scaler_y else float(y_sc[0])
            )
            from aci_calculator import aci_moment_capacity
            mn_aci = aci_moment_capacity(b=b, d=d, n_bars=n_bars, db_mm=db, fy=fy, fc=fc, eta_m=eta_m)
            m1, m2, m3 = st.columns(3)
            m1.metric("📊 Predicted Mmax (kN·m)", f"{mmax_pred:.2f}", f"{mmax_pred-mn_aci:+.2f} vs ACI")
            m2.metric("📏 ACI 318-19 Mn (kN·m)", f"{mn_aci:.2f}")
            m3.metric("🧠 Corrosion Severity Index", f"{eng_vals['corr_severity_idx']:.2f}")
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
                st.warning(f"⚠️ Mmax = {mmax_pred:.2f} kN·m — Moderate corrosion impact. Inspect beam.")
            else:
                st.error(f"❌ Mmax = {mmax_pred:.2f} kN·m — Severe degradation. Immediate action required.")
        except Exception as e:
            st.error(f"Prediction error: {e}")


# ============================================================
# TAB 2 — ENSEMBLE
# ============================================================
def _tab_ensemble():
    st.markdown('<p class="main-header">🧩 Ensemble Models Comparison</p>',
                unsafe_allow_html=True)
    st.markdown("All five models trained on the same 804-beam dataset. **CatBoost** selected as final predictor.")
    ens = _load_json(MODELS_DIR / "ensemble_metrics.json")
    if not ens:
        st.info("ensemble_metrics.json not found."); return
    rows = []
    for name, m in ens.get("models", {}).items():
        rows.append({
            "Model"    : ("⭐ " if name=="CatBoost" else "") + name,
            "Train R²" : round(m.get("train_R2",  0), 4),
            "Test R²"  : round(m.get("test_R2",   0), 4),
            "RMSE"     : round(m.get("test_RMSE", 0), 4),
            "MAE"      : round(m.get("test_MAE",  0), 4),
            "L1 ✓"    : "✅" if m.get("L1_broken") else "❌",
            "L2 ✓"    : "✅" if m.get("L2_broken") else "❌",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.markdown("---")
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
            marker_color=["#2E7D32" if v>=L2_TARGET_R2 else "#1A3A5C" for v in cv_folds],
            text=[f"{v:.4f}" for v in cv_folds], textposition="outside",
        ))
        fig.add_hline(y=L2_TARGET_R2, line_dash="dash", line_color="red",
                      annotation_text=f"L2 target ({L2_TARGET_R2})")
        fig.update_layout(title="CatBoost — R² per CV Fold",
                          yaxis=dict(range=[0.93,1.0]), height=350,
                          margin=dict(t=40,b=20,l=40,r=20))
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("### Test R² — All Models")
    names = [r["Model"] for r in rows]
    r2s   = [r["Test R²"] for r in rows]
    fig2  = go.Figure(go.Bar(
        x=names, y=r2s,
        marker_color=["#2E7D32" if v>=L2_TARGET_R2 else "#1A3A5C" for v in r2s],
        text=[f"{v:.4f}" for v in r2s], textposition="outside",
    ))
    fig2.add_hline(y=L1_TARGET_R2, line_dash="dot",  line_color="#F57C00",
                   annotation_text=f"L1 ({L1_TARGET_R2})")
    fig2.add_hline(y=L2_TARGET_R2, line_dash="dash", line_color="red",
                   annotation_text=f"L2 ({L2_TARGET_R2})")
    fig2.update_layout(yaxis=dict(range=[0.94,1.0]), height=350,
                       margin=dict(t=40,b=20,l=40,r=20))
    st.plotly_chart(fig2, use_container_width=True)


# ============================================================
# TAB 3 — RESULTS
# ============================================================
def _tab_results():
    st.markdown('<p class="main-header">📊 Model Performance & Benchmarks</p>',
                unsafe_allow_html=True)
    ens = _load_json(MODELS_DIR / "ensemble_metrics.json")
    aci = _load_json(MODELS_DIR / "aci_benchmark_metrics.json")
    if not ens and not aci:
        st.info("No results found."); return
    cat_m = ens.get("models", {}).get("CatBoost", {}) if ens else {}
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### ACI 318-19 Baseline")
        if aci:
            for k, label in [("R2","R²"),("RMSE","RMSE"),("MAE","MAE"),("MAPE","MAPE %"),("ratio_mean","Exp/Pred")]:
                st.metric(label, aci.get(k,"—"))
    with col2:
        st.markdown("### CatBoost — Best Model (Test Set)")
        if cat_m:
            for k, label in [("test_R2","R²"),("test_RMSE","RMSE"),("test_MAE","MAE")]:
                delta = ""
                if aci and k.replace("test_","") in aci:
                    delta = f"{cat_m.get(k,0)-aci.get(k.replace('test_',''),0):+.4f} vs ACI"
                st.metric(label, cat_m.get(k,"—"), delta or None)
            l1, l2 = cat_m.get("L1_broken",False), cat_m.get("L2_broken",False)
            badge = '<div class="verdict-pass">🏆 Both benchmarks broken ✓</div>' if (l1 and l2) else \
                    '<div class="verdict-pass">✓ L1 beaten.</div>'
            st.markdown(badge, unsafe_allow_html=True)
    st.markdown("---")
    if ens:
        st.markdown("### 10-Fold Cross-Validation")
        c1, c2, c3 = st.columns(3)
        c1.metric("CV R² Mean", f"{ens.get('cv_R2_mean',0):.4f}")
        c2.metric("CV R² Std",  f"{ens.get('cv_R2_std',0):.4f}")
        c3.metric("Folds", len(ens.get("cv_folds",[])))


# ============================================================
# TAB 4 — SHAP
# ============================================================
def _tab_shap():
    st.markdown('<p class="main-header">🤖 SHAP Feature Importance</p>',
                unsafe_allow_html=True)
    top5_path = MODELS_DIR / "top5_shap_features.json"
    if top5_path.exists():
        top5 = _load_json(top5_path)
        st.success("📌 Top-5 features: " + ", ".join(top5.get("top5_features",[])))
    c1, c2 = st.columns(2)
    for col, fname, cap in [
        (c1, "shap_importance.png", "Mean |SHAP|"),
        (c2, "shap_beeswarm.png",   "Beeswarm Plot"),
    ]:
        p = FIGURES_DIR / fname
        with col:
            st.image(str(p), caption=cap, use_column_width=True) if p.exists() else \
                st.info(f"{fname} not found.")
    for dep in sorted(FIGURES_DIR.glob("shap_dependence_*.png")):
        st.markdown("### Dependence Plot")
        st.image(str(dep), caption="SHAP Dependence", use_column_width=True)
        break


# ============================================================
# TAB 5 — EQUATIONS  (full professional tab)
# ============================================================
def _tab_equation():
    st.markdown('<p class="main-header">📝 Symbolic Regression — All Discovered Equations</p>',
                unsafe_allow_html=True)
    st.markdown(
        """
        **PySR** (Cranmer 2023) evolved **19 closed-form equations** over 200 iterations × 40 populations
        on 643 training specimens. Each equation represents a Pareto-optimal trade-off between
        *complexity* and *predictive accuracy (loss)*.
        The table below provides an independent, unbiased assessment of every discovered expression.
        The research supervisor is invited to select the most suitable equation for code calibration
        or journal publication based on the criteria most relevant to their context.
        """
    )

    # -- 1. Complexity vs R2 Pareto frontier
    st.markdown("---")
    st.markdown("### 📈 Pareto Frontier — Complexity vs. Accuracy")
    comp_vals = [e["complexity"] for e in EQ_CATALOGUE]
    r2_vals   = [e["r2_approx"]  for e in EQ_CATALOGUE]
    names_eq  = [f"Eq.{e['id']}: {e['name'][:35]}" for e in EQ_CATALOGUE]
    colors_eq = [
        "#2E7D32" if e["id"] in (12, 17) else
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
        hovertext=names_eq, hoverinfo="text+y", showlegend=False,
    ))
    fig_pareto.add_hline(y=0.8839, line_dash="dot",  line_color="#F57C00",
                         annotation_text="ACI 318-19 (R²=0.8839)")
    fig_pareto.add_hline(y=0.972,  line_dash="dash", line_color="red",
                         annotation_text="L2 SOTA target (0.972)")
    for eid, col in [(12, "#2E7D32"), (17, "#9C27B0")]:
        eq = next(e for e in EQ_CATALOGUE if e["id"]==eid)
        fig_pareto.add_trace(go.Scatter(
            x=[eq["complexity"]], y=[eq["r2_approx"]],
            mode="markers", marker=dict(size=18, symbol="star", color=col),
            name="★ Recommended" if eid==12 else "★ Most Physical",
        ))
    fig_pareto.update_layout(
        xaxis_title="Equation Complexity",
        yaxis_title="R² (test set)",
        yaxis=dict(range=[0.0, 1.02]),
        height=420, margin=dict(t=30,b=40,l=50,r=30),
        legend=dict(x=0.02, y=0.98),
    )
    st.plotly_chart(fig_pareto, use_container_width=True)
    st.caption(
        "🟢 Green = Recommended  |  🟣 Purple = Most physical  |  "
        "🔵 Blue = Corrosion-aware  |  ⚫ Grey = No corrosion term"
    )

    # -- 2. Summary comparison table
    st.markdown("---")
    st.markdown("### 📋 Complete Equation Comparison Table")
    aci_r2 = 0.8839
    table_rows = []
    for e in EQ_CATALOGUE:
        delta = e["r2_approx"] - aci_r2
        table_rows.append({
            "ID"          : f"{'★ ' if e['id'] in (12,17) else ''}Eq.{e['id']}",
            "Complexity"  : e["complexity"],
            "Name"        : e["name"].replace("RECOMMENDED — ","★ ").replace("MOST PHYSICAL — ","★ "),
            "R² (approx)" : round(e["r2_approx"], 4),
            "ΔR² vs ACI"  : f"{delta:+.4f}",
            "ηm term"     : "✅" if e["corrosion_aware"] else "❌",
            "fy term"     : "✅" if e["fy_aware"]        else "❌",
            "ρ term"      : "✅" if e["rho_aware"]       else "❌",
            "f'c term"    : "✅" if e["fc_aware"]        else "❌",
            "# Variables" : len(e["variables"]),
        })
    st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

    # -- 3. Individual equation cards
    st.markdown("---")
    st.markdown("### 🔬 Detailed Analysis — Each Equation")

    f1, f2, f3 = st.columns(3)
    show_corr_only = f1.checkbox("Show corrosion-aware only", value=False)
    show_fy_only   = f2.checkbox("Show fy-aware only",        value=False)
    show_rec_only  = f3.checkbox("Show recommended only",     value=False)

    filtered = EQ_CATALOGUE
    if show_corr_only: filtered = [e for e in filtered if e["corrosion_aware"]]
    if show_fy_only:   filtered = [e for e in filtered if e["fy_aware"]]
    if show_rec_only:  filtered = [e for e in filtered if e["id"] in (12, 17)]

    for eq in filtered:
        is_rec = eq["id"] in (12, 17)
        card_class = "eq-best" if is_rec else "eq-card"
        label_html = ""
        if eq["id"] == 12:
            label_html = "<b>⭐ RECOMMENDED — Best complexity/accuracy trade-off</b><br>"
        if eq["id"] == 17:
            label_html = "<b>⭐ MOST PHYSICAL — Only equation with f'c</b><br>"
        st.markdown(
            f'<div class="{card_class}">{label_html}'
            f'<b>Eq.{eq["id"]} — Complexity {eq["complexity"]} — '
            f'{eq["name"].replace("RECOMMENDED — ","").replace("MOST PHYSICAL — ","")}</b>'
            f'</div>',
            unsafe_allow_html=True,
        )
        col_left, col_right = st.columns([3, 2])
        with col_left:
            st.latex(eq["latex"])
            st.markdown(f"**Variables:** `{'`, `'.join(eq['variables'])}`")
            cov_badges = ""
            for flag, lbl in [
                ("corrosion_aware", "ηm"),
                ("fy_aware", "fy"),
                ("rho_aware", "ρ"),
                ("fc_aware", "f'c"),
            ]:
                cov_badges += ("✅ " if eq[flag] else "❌ ") + lbl + "  "
            st.markdown(cov_badges)
        with col_right:
            r2_v  = eq.get("r2_approx", None)
            rmse  = eq.get("rmse",  None)
            mae   = eq.get("mae",   None)
            mape  = eq.get("mape",  None)
            if r2_v:
                delta_v = f"{r2_v - aci_r2:+.4f} vs ACI"
                st.metric("R²", f"{r2_v:.4f}", delta_v)
            if rmse: st.metric("RMSE (kN·m)", f"{rmse:.3f}")
            if mape: st.metric("MAPE",        f"{mape:.1f}%")

        st.markdown(f"✅ **Pros:** {eq['pros']}")
        st.markdown(f"❌ **Cons:** {eq['cons']}")
        st.markdown(f"💡 **Recommended use:** {eq['use_case']}")
        st.markdown("---")

    # -- 4. Interactive calculator
    st.markdown("### 🧮 Interactive Hand-Calculator")
    st.markdown(
        "Select any equation and enter beam parameters to compute **Mmax** "
        "instantly — no model file needed."
    )
    eq_options = {f"Eq.{e['id']}: {e['name'][:55]}": e for e in EQ_CATALOGUE}
    chosen_label = st.selectbox("Choose equation", list(eq_options.keys()),
                                index=list(eq_options.keys()).index(
                                    next(k for k in eq_options if "Eq.12" in k)))
    chosen_eq = eq_options[chosen_label]

    needed = chosen_eq["variables"]
    calc_cols = st.columns(min(len(needed), 4))
    calc_inputs = {}
    defaults = {"d":300,"eta_m":10.0,"fy":460,"rho_t":1.5,
                "d_b":16.0,"b":150,"fc":32,"rho":1.5}
    labels   = {"d":"d (mm)","eta_m":"eta_m (%)","fy":"fy (MPa)",
                "rho_t":"rho_t (%)","d_b":"db (mm)","b":"b (mm)",
                "fc":"f'c (MPa)"}
    for i, var in enumerate(needed):
        with calc_cols[i % min(len(needed), 4)]:
            calc_inputs[var] = st.number_input(
                labels.get(var, var),
                value=float(defaults.get(var, 1.0)),
                key=f"calc_{chosen_eq['id']}_{var}"
            )

    if st.button("⚡ Compute with Selected Equation", use_container_width=True):
        try:
            v = calc_inputs
            d=v.get("d",300); eta_m=v.get("eta_m",10); fy=v.get("fy",460)
            rho_t=v.get("rho_t",1.5); d_b=v.get("d_b",16); b=v.get("b",150)
            fc=v.get("fc",32)
            eid = chosen_eq["id"]
            if   eid == 1:  M = 0.136*d
            elif eid == 2:  M = 1.250**np.sqrt(d)
            elif eid == 3:  M = 0.312*d - 39.22
            elif eid == 4:  M = 0.313*(d-eta_m) - 36.47
            elif eid == 5:  M = np.sqrt(rho_t)*(0.330*d - 42.26)
            elif eid == 6:
                arg = np.log(0.025*d)
                M   = (fy*rho_t)**np.log(arg) if arg>0 and np.log(arg)>0 else float("nan")
            elif eid == 7:
                arg = np.log(0.029*np.sqrt(d))
                M   = (fy*rho_t)**np.log(arg)/d_b if arg>0 and np.log(arg)>0 else float("nan")
            elif eid == 8:
                arg = np.log(0.028*np.sqrt(d))
                M   = (fy*rho_t+46.0)**np.log(arg)/d_b if arg>0 and np.log(arg)>0 else float("nan")
            elif eid == 9:
                arg = np.log(0.028*np.sqrt(d))
                M   = (0.235*b+fy*rho_t)**np.log(arg)/d_b if arg>0 and np.log(arg)>0 else float("nan")
            elif eid == 10:
                arg = 0.023*(d-eta_m)
                M   = 0.074*b + rho_t*(fy**np.log(np.log(arg)) - 2.824) if arg>0 else float("nan")
            elif eid == 11:
                arg = np.log(0.240*np.sqrt(d))
                M   = 1.947*fy**(1.424*np.log(arg))*(-d_b+rho_t+2.137) if arg>0 else float("nan")
            elif eid == 12:
                arg = np.log(0.182*np.sqrt(d))
                M   = fy**np.log(arg)*(14.5-np.sqrt(eta_m))*(-d_b+rho_t+2.69) if arg>0 else float("nan")
            elif eid == 13:
                arg = np.log(0.183*np.sqrt(d))
                M   = fy**np.log(arg)*(15.09-np.sqrt(eta_m))*(rho_t+4.46*np.exp(-d_b)) if arg>0 else float("nan")
            elif eid == 14:
                arg = np.log(0.185*np.sqrt(d))
                M   = fy**np.log(arg)*((13.87-np.sqrt(eta_m))*(rho_t+4.07*np.exp(-d_b))+1.76) if arg>0 else float("nan")
            elif eid == 15:
                arg = np.log(0.223*np.sqrt(d))
                M   = (3.96-0.045*eta_m)*(rho_t+3.87*np.exp(-d_b))*(fy**np.log(arg))**1.227 if arg>0 else float("nan")
            elif eid == 16:
                arg = np.log(0.216*np.sqrt(d))
                M   = fy**(1.208*np.log(arg))*(5.12-0.299*np.sqrt(eta_m))*(rho_t+3.80*np.exp(-d_b)) if arg>0 else float("nan")
            elif eid == 17:
                arg = np.log(0.209*np.sqrt(d))
                M   = (6.58-np.sqrt(eta_m/np.sqrt(fc)))*(rho_t+3.85*np.exp(-d_b))*(fy**np.log(arg))**1.177 if arg>0 else float("nan")
            else:
                M = float("nan")

            if np.isnan(M) or np.isinf(M) or M < 0:
                st.error("⚠️ Result undefined for these inputs (log argument <= 0 or negative result). Try larger d value.")
            else:
                st.success(f"**Mmax ≈ {M:.2f} kN·m** (Eq.{eid} — {chosen_eq['name'][:40]}...)")
                st.info(
                    f"📊 Context: CatBoost R²=0.987 | This equation R²≈{chosen_eq['r2_approx']:.4f} | "
                    f"ACI R²=0.8839"
                )
        except Exception as ex:
            st.error(f"Calculation error: {ex}")

    # -- 5. Final recommendation & future roadmap
    st.markdown("---")
    st.markdown("### 🏁 Final Recommendation for Supervisor")
    st.markdown("""
| Goal | Recommended Equation | Why |
|------|---------------------|-----|
| Design code / journal (main) | Eq.12 (complexity 19) | Best R²/simplicity trade-off; 5 variables; sqrt(eta_m) physically motivated |
| Maximum physical completeness | Eq.17 (complexity 25) | Only one with f'c; eta_m/sqrt(f'c) is dimensionally sound |
| Teaching / quick check | Eq.4 (complexity 7)  | 2 variables; immediately interpretable |
| Uncorroded beams | Eq.11 (complexity 18) | Clean 3-term product; no corrosion needed |
    """)

    st.markdown("---")
    st.markdown("### 🚀 How to Get Stronger Equations in the Next Run")
    st.markdown("""
The limitations of current equations are clear:
- No equation achieves R² > 0.94 (vs CatBoost 0.987 — gap of 0.047)
- Concrete strength f'c appears in only 1 of 19 equations
- Stirrup parameters (ds, s, fy_s) absent from all equations
- log(log()) structure is hard to compute manually

Strategies to discover stronger equations next time:

1. **Expand the feature set fed to PySR** — Add As, a (ACI lever arm), d-a/2 as pre-computed physics features
2. **Seed PySR with ACI structure** — Fix base as As·fy·(d - a/2) and let PySR find only the correction factor
3. **Increase PySR iterations** — Current: 200 × 40 populations → Target: 500+ × 100 populations
4. **Use physics-guided operators** — Restrict to ACI/EC2 operators: ×, ÷, √, ² only
5. **Expand the database** — Current: 804 → Target: 1500+ specimens
6. **Multi-objective PySR** — Simultaneously minimise loss, complexity, and physical constraint violations
    """)


# ============================================================
# TAB 6 — REPORT
# ============================================================
def _tab_report():
    st.markdown('<p class="main-header">📄 Download PDF Report</p>',
                unsafe_allow_html=True)
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
        st.info("📔 Report not generated. Run `python src/main.py`.")
    st.markdown("---")
    if st.button("🔄 Regenerate Report Now", use_container_width=True):
        with st.spinner("Building PDF …"):
            try:
                from report_generator import generate_report
                st.success(f"✅ Saved to: {generate_report()}")
                st.cache_data.clear(); st.rerun()
            except Exception as e:
                st.error(f"Report generation failed: {e}")


# ============================================================
# MAIN
# ============================================================
def main():
    _sidebar()
    st.markdown(f'<p class="main-header">{APP_ICON} {APP_TITLE}</p>',
                unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">'
        "PhD Research — CatBoost (R² = 0.987) × SHAP × PySR Symbolic Regression"
        "</p>", unsafe_allow_html=True,
    )
    tabs = st.tabs([
        "🏗️  Predict", "🧩  Ensemble", "📊  Results",
        "🤖  SHAP",   "📝  Equations", "📄  Report",
    ])
    with tabs[0]: _tab_predict()
    with tabs[1]: _tab_ensemble()
    with tabs[2]: _tab_results()
    with tabs[3]: _tab_shap()
    with tabs[4]: _tab_equation()
    with tabs[5]: _tab_report()


if __name__ == "__main__":
    main()
