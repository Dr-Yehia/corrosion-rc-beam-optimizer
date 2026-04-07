# ============================================================
# app/streamlit_app.py
# Corrosion RC Beam Optimizer — Interactive Streamlit UI
#
# Tabs:
#   1. 🏗️  Predict    ─ single beam Mmax (kN·m) prediction
#   2. 🧬  GA Run     ─ live NSGA-III optimisation dashboard
#   3. 📊  Results    ─ model metrics & benchmark comparison
#   4. 🤖  SHAP       ─ feature importance viewer
#   5. 📝  Equation   ─ PySR discovered equation
#   6. 📄  Report     ─ download PDF report
# ============================================================

import sys
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import joblib
import plotly.graph_objects as go
import plotly.express as px

# ─ ensure src/ is importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from config import (
    APP_TITLE, APP_ICON, APP_LAYOUT,
    MODELS_DIR, FIGURES_DIR, EQ_DIR, RESULTS_DIR,
    FEATURE_COLS, TARGET_COL,
    L1_TARGET_R2, L2_TARGET_R2,
    GENE_BOUNDS,
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title = APP_TITLE,
    page_icon  = APP_ICON,
    layout     = APP_LAYOUT,
    initial_sidebar_state = "expanded",
)


# ============================================================
# CSS THEME
# ============================================================

st.markdown("""
<style>
    .main-header {
        font-size: 2rem; font-weight: 700;
        color: #0D1B2A; margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem; color: #3A5A8C; margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #F0F4FA; border-radius: 10px;
        padding: 1rem 1.2rem; margin-bottom: 0.8rem;
        border-left: 4px solid #1A3A5C;
    }
    .verdict-pass {
        background: #E8F5E9; border-left: 4px solid #2E7D32;
        padding: 0.8rem 1.2rem; border-radius: 8px;
        color: #1B5E20; font-weight: 600;
    }
    .verdict-fail {
        background: #FFEBEE; border-left: 4px solid #C62828;
        padding: 0.8rem 1.2rem; border-radius: 8px;
        color: #B71C1C; font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# CACHED LOADERS
# ============================================================

@st.cache_resource(show_spinner=False)
def _load_model():
    path = MODELS_DIR / "best_model.pkl"
    if not path.exists():
        return None
    return joblib.load(path)

@st.cache_resource(show_spinner=False)
def _load_scaler_X():
    path = MODELS_DIR / "scaler_X.pkl"
    return joblib.load(path) if path.exists() else None

@st.cache_resource(show_spinner=False)
def _load_scaler_y():
    path = MODELS_DIR / "scaler_y.pkl"
    return joblib.load(path) if path.exists() else None

@st.cache_data(show_spinner=False)
def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)

@st.cache_data(show_spinner=False)
def _load_clean_data() -> pd.DataFrame:
    from data_preprocessing import run_preprocessing
    return run_preprocessing(save_clean=False)["df_clean"]


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
        "PhD Research Pipeline — Neural Network × NSGA-III\n"
        "Predicts **Mmax (kN·m)** of corroded RC beams."
    )
    st.sidebar.markdown("---")

    # Model status
    model = _load_model()
    if model:
        st.sidebar.success("Model: Loaded ✅")
    else:
        st.sidebar.error("Model not found — run main.py first.")

    # Benchmark targets
    st.sidebar.markdown("### Benchmark Targets")
    st.sidebar.info(
        f"🏅 L1 (ACI 318-19): R² > {L1_TARGET_R2}\n\n"
        f"🏆 L2 (SOTA):        R² > {L2_TARGET_R2}"
    )
    st.sidebar.markdown("---")
    st.sidebar.caption(
        f"Results dir: `{RESULTS_DIR.relative_to(ROOT)}`"
    )


# ============================================================
# TAB 1 ─ PREDICT
# ============================================================

def _tab_predict():
    st.markdown(
        '<p class="main-header">🏗️ Beam Mmax (kN·m) Predictor</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        "Enter your corroded RC beam parameters below. "
        "The model predicts the <b>Maximum Flexural Capacity Mmax (kN·m)</b>.",
        unsafe_allow_html=True,
    )

    model    = _load_model()
    scaler_X = _load_scaler_X()
    scaler_y = _load_scaler_y()

    if model is None:
        st.error("⚠️ Model not found. Run `python src/main.py` first.")
        return

    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Section Geometry**")
        b  = st.number_input("Width b (mm)",          100, 350, 150, step=5)
        d  = st.number_input("Depth d (mm)",          100, 500, 300, step=5)
        L  = st.number_input("Test Length (mm)",      500, 5000, 2500, step=50)
        cv = st.number_input("Bottom Cover (mm)",      15,  60,  25,  step=1)

    with col2:
        st.markdown("**Reinforcement**")
        n_bars = st.number_input("# Tensile Bars",    1, 8, 3, step=1)
        db     = st.number_input("Bar Diameter (mm)", 6, 32, 16, step=2)
        pten   = st.number_input("ρ tension (%)",    0.1, 5.0, 1.5, step=0.1)
        fy     = st.number_input("fy (MPa)",          226, 650, 460, step=5)

    with col3:
        st.markdown("**Concrete & Corrosion**")
        fc       = st.number_input("f'c (MPa)",        20,  80, 32, step=1)
        wc       = st.number_input("W/C Ratio",        0.30, 0.70, 0.45, step=0.01)
        eta_m    = st.number_input("ηm — Mass Loss (%)", 0.0, 64.0, 10.0, step=0.5)
        s_stirr  = st.number_input("Stirrup Spacing (mm)", 50, 300, 150, step=10)
        ds_stirr = st.number_input("Stirrup Dia. (mm)",     6,  16,   8, step=2)
        fy_s     = st.number_input("fy stirrups (MPa)",   226, 650, 420, step=5)
        shear_x  = st.number_input("Shear Span x (mm)",   100, 2000, 800, step=50)

    st.markdown("---")

    # ── Categorical inputs (shown as selectboxes) ─────────────
    st.markdown("**Bar & Test Configuration**")
    cc1, cc2, cc3 = st.columns(3)
    with cc1:
        bar_type = st.selectbox(
            "Longitudinal Bar Type",
            options=["Deformed (D)", "Plain (P)"],
            index=0,
        )
    with cc2:
        test_type = st.selectbox(
            "Test Configuration",
            options=["SS_FPB_MONO", "SS_TPB", "Other"],
            index=0,
        )
    with cc3:
        corr_method = st.selectbox(
            "Corrosion Method",
            options=["Impressed Current (IC)", "Accelerated (AC)", "Natural (C)"],
            index=0,
        )

    st.markdown("---")
    if st.button("🔍  Predict Mmax (kN·m)", type="primary", use_container_width=True):

        # ── 15 numeric features ───────────────────────────────
        input_dict = {
            "Width (mm)"                                     : b,
            "Depth (mm)"                                     : d,
            "Test Length (mm)"                               : L,
            "Bottom Cover to Ctr of Tension Bar (mm)"        : cv,
            "# Tensile Bars"                                 : n_bars,
            "Diameter Tensile Bars, db,t (mm)"               : db,
            "Tension Reinforcement Ratio, pten (%)"          : pten,
            "fy Longitudinal Bars (Tensile), (MPa) "         : fy,
            "f'c (MPa)"                                      : fc,
            "W/C Ratio"                                      : wc,
            "Stirrup Spacing, s (mm) "                       : s_stirr,
            "Stirrup Diameter, ds (mm)"                      : ds_stirr,
            "fy,s Stirrup Bars"                              : fy_s,
            "Mass Loss (Tensile bars), \u03b7m (%)"           : eta_m,
            "Shear Span, x (mm)"                             : shear_x,
        }

        # ── 3 engineered features ─────────────────────────────
        input_dict["corr_severity_idx"] = eta_m * (fy / max(fc, 1))
        input_dict["d_b_ratio"]         = d / max(b, 1)
        input_dict["eta_d_interaction"] = eta_m * d

        # ── 5 one-hot categorical features ────────────────────
        input_dict["Longitudinal Bar Type_D"] = 1 if "D" in bar_type else 0
        input_dict["Longitudinal Bar Type_P"] = 1 if "P" in bar_type else 0

        input_dict["Test Type and Configuration_SS_FPB_MONO"] = 1 if test_type == "SS_FPB_MONO" else 0
        input_dict["Test Type and Configuration_SS_TPB"]      = 1 if test_type == "SS_TPB"      else 0

        input_dict["Corrosion Method_IC"] = 1 if "IC" in corr_method else 0
        input_dict["Corrosion Method_AC"] = 1 if "AC" in corr_method else 0
        input_dict["Corrosion Method_C"]  = 1 if corr_method == "Natural (C)" else 0

        # ── Build ordered feature vector (23 columns) ─────────
        model_cols = (
            FEATURE_COLS
            + ["corr_severity_idx", "d_b_ratio", "eta_d_interaction"]
            + [
                "Longitudinal Bar Type_D",
                "Longitudinal Bar Type_P",
                "Test Type and Configuration_SS_FPB_MONO",
                "Test Type and Configuration_SS_TPB",
                "Corrosion Method_IC",
                "Corrosion Method_AC",
                "Corrosion Method_C",
            ]
        )

        try:
            row = np.array(
                [input_dict.get(c, 0.0) for c in model_cols],
                dtype=float
            ).reshape(1, -1)

            if scaler_X:
                row_sc = scaler_X.transform(row)
            else:
                row_sc = row

            y_pred_sc = model.predict(row_sc)
            mmax_pred = (
                scaler_y.inverse_transform(y_pred_sc.reshape(-1, 1)).ravel()[0]
                if scaler_y else float(y_pred_sc[0])
            )

            # ── ACI benchmark prediction ───────────────────────
            from aci_calculator import aci_moment_capacity
            mn_aci = aci_moment_capacity(
                b=b, d=d, n_bars=n_bars, db_mm=db,
                fy=fy, fc=fc, eta_m=eta_m,
            )

            # ── Display metrics ────────────────────────────────
            col_r, col_m, col_c = st.columns(3)
            col_r.metric(
                label="📊 Predicted Mmax (kN·m)",
                value=f"{mmax_pred:.2f} kN·m",
                delta=f"{mmax_pred - mn_aci:+.2f} vs ACI",
            )
            col_m.metric(
                label="📏 ACI Mn (kN·m)",
                value=f"{mn_aci:.2f} kN·m",
            )
            col_c.metric(
                label="🧠 Corrosion Severity Index",
                value=f"{input_dict['corr_severity_idx']:.2f}",
            )

            # ── Gauge chart ────────────────────────────────────
            fig = go.Figure(go.Indicator(
                mode  = "gauge+number+delta",
                value = round(mmax_pred, 2),
                delta = {"reference": mn_aci, "valueformat": ".2f"},
                title = {"text": "Predicted Mmax (kN·m)"},
                gauge = {
                    "axis"  : {"range": [0, max(mmax_pred * 1.5, mn_aci * 1.5, 50)]},
                    "bar"   : {"color": "#1A3A5C"},
                    "steps" : [
                        {"range": [0,   mn_aci * 0.5], "color": "#FFCDD2"},
                        {"range": [mn_aci * 0.5, mn_aci], "color": "#FFF9C4"},
                        {"range": [mn_aci, mn_aci * 1.5], "color": "#C8E6C9"},
                    ],
                    "threshold": {
                        "line" : {"color": "orange", "width": 3},
                        "value": mn_aci,
                    },
                },
            ))
            fig.update_layout(height=320, margin=dict(t=30, b=10, l=20, r=20))
            st.plotly_chart(fig, use_container_width=True)

            # ── Interpretation ─────────────────────────────────
            ratio = mmax_pred / max(mn_aci, 1)
            if ratio >= 0.90:
                st.success(f"✅ Predicted Mmax ({mmax_pred:.2f} kN·m) is close to ACI estimate — Low corrosion impact.")
            elif ratio >= 0.60:
                st.warning(f"⚠️ Predicted Mmax ({mmax_pred:.2f} kN·m) is significantly below ACI — Moderate damage.")
            else:
                st.error(f"❌ Predicted Mmax ({mmax_pred:.2f} kN·m) is far below ACI — Severe corrosion damage. Inspection required.")

        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.info(f"Debug — feature vector length: {len(model_cols)} | Expected by model: check model_cols list above.")


# ============================================================
# TAB 2 ─ GA LIVE DASHBOARD
# ============================================================

def _tab_ga_dashboard():
    st.markdown(
        '<p class="main-header">🧬 NSGA-III Live Optimisation</p>',
        unsafe_allow_html=True,
    )

    hof_path = MODELS_DIR / "hall_of_fame.json"
    if not hof_path.exists():
        st.info("📊 No optimisation results yet. Run `python src/main.py` to start.")
        return

    hof = _load_json(hof_path)
    if not hof:
        st.warning("Hall of Fame is empty.")
        return

    # Best individual
    best = hof[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best R\u00b2",    f"{best['metrics'].get('R2',   0):.4f}")
    c2.metric("Best RMSE",   f"{best['metrics'].get('RMSE', 0):.4f}")
    c3.metric("CV-R\u00b2",      f"{best['metrics'].get('CV_R2',0):.4f}")
    c4.metric("Fitness",     f"{best['fitness']:.4f}")

    l1_ok = best["metrics"].get("R2", 0) >= L1_TARGET_R2
    l2_ok = best["metrics"].get("R2", 0) >= L2_TARGET_R2

    if l1_ok and l2_ok:
        st.markdown('<div class="verdict-pass">🏆 Both benchmarks broken — L1 ✓ and L2 ✓</div>',
                    unsafe_allow_html=True)
    elif l1_ok:
        st.markdown('<div class="verdict-pass">✓ L1 (ACI) broken. L2 (SOTA) pending.</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<div class="verdict-fail">✗ Neither benchmark broken yet.</div>',
                    unsafe_allow_html=True)

    st.markdown("---")

    r2s = [e["metrics"].get("R2", 0) for e in hof]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(1, len(r2s)+1)), y=r2s,
        mode="lines+markers", name="Best R\u00b2 per Run",
        line=dict(color="#1A3A5C", width=2.5),
        marker=dict(size=7),
    ))
    fig.add_hline(y=L1_TARGET_R2, line_dash="dash",
                  line_color="#F57C00", annotation_text="L1 target")
    fig.add_hline(y=L2_TARGET_R2, line_dash="dash",
                  line_color="#1B5E20", annotation_text="L2 target")
    fig.update_layout(
        title="Best R\u00b2 Across GA Runs",
        xaxis_title="Run", yaxis_title="R\u00b2",
        height=380, margin=dict(t=40, b=30, l=40, r=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Hall of Fame")
    df_hof = pd.DataFrame([
        {
            "Rank"     : i + 1,
            "Run"      : e["run"],
            "Gen"      : e["generation"],
            "R\u00b2"    : round(e["metrics"].get("R2",    0), 4),
            "RMSE"     : round(e["metrics"].get("RMSE",  0), 4),
            "CV-R\u00b2" : round(e["metrics"].get("CV_R2", 0), 4),
            "Fitness"  : round(e["fitness"], 4),
            "Timestamp": e["timestamp"][:19],
        }
        for i, e in enumerate(hof)
    ])
    st.dataframe(df_hof, use_container_width=True, hide_index=True)


# ============================================================
# TAB 3 ─ RESULTS
# ============================================================

def _tab_results():
    st.markdown(
        '<p class="main-header">📊 Model Performance & Benchmarks</p>',
        unsafe_allow_html=True,
    )

    mlp_m  = _load_json(MODELS_DIR / "mlp_metrics.json")
    aci_m  = _load_json(MODELS_DIR / "aci_benchmark_metrics.json")

    if not mlp_m and not aci_m:
        st.info("No results found. Run the pipeline first.")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### ACI 318-19 Baseline")
        if aci_m:
            for k in ["R2", "RMSE", "MAE", "MAPE", "ratio_mean"]:
                st.metric(k, aci_m.get(k, "—"))

    with col2:
        st.markdown("### Optimised Model (Test Set)")
        if mlp_m and "test" in mlp_m:
            t = mlp_m["test"]
            for k in ["R2", "RMSE", "MAE", "MAPE"]:
                st.metric(
                    k, t.get(k, "—"),
                    delta=(
                        f"{t.get(k, 0) - aci_m.get(k, 0):+.4f} vs ACI"
                        if aci_m and k in aci_m else ""
                    ),
                )
            l1 = t.get("L1_broken", False)
            l2 = t.get("L2_broken", False)
            st.markdown(
                f"**L1 (ACI):** {'\u2705' if l1 else '\u274c'}    "
                f"**L2 (SOTA):** {'\u2705' if l2 else '\u274c'}"
            )

    st.markdown("---")

    if mlp_m and "cv" in mlp_m:
        st.markdown("### 10-Fold Cross-Validation")
        cv = mlp_m["cv"]
        c1, c2, c3 = st.columns(3)
        c1.metric("CV R\u00b2 Mean", cv.get("cv_R2_mean",   "—"))
        c2.metric("CV R\u00b2 Std",  cv.get("cv_R2_std",    "—"))
        c3.metric("CV RMSE Mean", cv.get("cv_RMSE_mean", "—"))

    if mlp_m and aci_m:
        st.markdown("### 📍 Performance Radar")
        categories = ["R\u00b2", "1-MAPE/100", "1-RMSE/50"]
        m_test = mlp_m.get("test", {})
        vals_model = [
            m_test.get("R2",   0),
            max(0, 1 - m_test.get("MAPE", 100) / 100),
            max(0, 1 - m_test.get("RMSE", 50)  / 50),
        ]
        vals_aci = [
            aci_m.get("R2",   0),
            max(0, 1 - aci_m.get("MAPE", 100) / 100),
            max(0, 1 - aci_m.get("RMSE", 50)  / 50),
        ]
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=vals_model + [vals_model[0]], theta=categories + [categories[0]],
            fill="toself", name="Optimised Model",
            line_color="#1A3A5C",
        ))
        fig.add_trace(go.Scatterpolar(
            r=vals_aci + [vals_aci[0]], theta=categories + [categories[0]],
            fill="toself", name="ACI 318-19",
            line_color="#F57C00", opacity=0.6,
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(range=[0, 1])),
            height=380, margin=dict(t=40, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)


# ============================================================
# TAB 4 ─ SHAP
# ============================================================

def _tab_shap():
    st.markdown(
        '<p class="main-header">🤖 SHAP Feature Importance</p>',
        unsafe_allow_html=True,
    )

    bar_path      = FIGURES_DIR / "shap_importance.png"
    beeswarm_path = FIGURES_DIR / "shap_beeswarm.png"
    top5_path     = MODELS_DIR  / "top5_shap_features.json"

    if top5_path.exists():
        top5 = _load_json(top5_path)
        st.success(
            f"📌 Top-5 most influential features: "
            f"{', '.join(top5.get('top5_features', []))}"
        )

    col1, col2 = st.columns(2)
    with col1:
        if bar_path.exists():
            st.image(str(bar_path), caption="Mean |SHAP| Feature Importance",
                     use_column_width=True)
        else:
            st.info("SHAP bar chart not generated yet.")
    with col2:
        if beeswarm_path.exists():
            st.image(str(beeswarm_path), caption="SHAP Beeswarm Plot",
                     use_column_width=True)
        else:
            st.info("SHAP beeswarm not generated yet.")

    dep_plots = sorted(FIGURES_DIR.glob("shap_dependence_*.png"))
    if dep_plots:
        st.markdown("### Dependence Plot")
        st.image(str(dep_plots[0]),
                 caption="SHAP Dependence — Top Feature",
                 use_column_width=True)


# ============================================================
# TAB 5 ─ EQUATION
# ============================================================

def _tab_equation():
    st.markdown(
        '<p class="main-header">📝 PySR Symbolic Equation</p>',
        unsafe_allow_html=True,
    )

    txt_path   = EQ_DIR / "best_equation.txt"
    latex_path = EQ_DIR / "best_equation.latex"
    json_path  = EQ_DIR / "all_equations.json"

    if not txt_path.exists():
        st.info("⏳ Symbolic regression not run yet. "
                "Run `python src/main.py` (without --skip-pysr).")
        return

    with open(txt_path) as f:
        eq_text = f.read()

    st.markdown("### Best Discovered Equation (plain text)")
    st.code(eq_text, language="python")

    if latex_path.exists():
        with open(latex_path) as f:
            eq_latex = f.read()
        st.markdown("### LaTeX Format")
        clean = eq_latex.replace("% ", "").strip()
        for line in clean.split("\n"):
            if line.startswith("M") or line.startswith("R"):
                st.latex(line)

    if json_path.exists():
        eq_data = _load_json(json_path)
        st.markdown("### Performance Metrics")
        m = eq_data.get("metrics", {})
        c1, c2, c3 = st.columns(3)
        c1.metric("R\u00b2",   m.get("R2",   "—"))
        c2.metric("RMSE", m.get("RMSE", "—"))
        c3.metric("MAPE", f"{m.get('MAPE', '?')} %")

        for label, ok in [
            ("🏅 L1 (ACI) ",   m.get("L1_broken", False)),
            ("🏆 L2 (SOTA)", m.get("L2_broken", False)),
        ]:
            if ok:
                st.success(f"{label}: Broken ✅")
            else:
                st.warning(f"{label}: Not yet broken")

        # Show all equations table
        all_eqs = eq_data.get("all_equations", [])
        if all_eqs:
            st.markdown("### All PySR Equations (Hall of Fame)")
            df_eq = pd.DataFrame([
                {
                    "Complexity": e.get("complexity"),
                    "Loss":       round(e.get("loss", 0), 4),
                    "Score":      round(e.get("score", 0), 4),
                    "Equation":   e.get("sympy_format", e.get("equation", "")),
                }
                for e in all_eqs
            ])
            st.dataframe(df_eq, use_container_width=True, hide_index=True)


# ============================================================
# TAB 6 ─ REPORT DOWNLOAD
# ============================================================

def _tab_report():
    st.markdown(
        '<p class="main-header">📄 Download PDF Report</p>',
        unsafe_allow_html=True,
    )

    report_path = RESULTS_DIR / "Final_Report.pdf"
    if report_path.exists():
        with open(report_path, "rb") as f:
            st.download_button(
                label     = "⬇️ Download Final_Report.pdf",
                data      = f,
                file_name = "Corrosion_RC_Beam_Optimizer_Report.pdf",
                mime      = "application/pdf",
                use_container_width=True,
            )
        st.caption(
            f"Last modified: "
            f"{pd.Timestamp(report_path.stat().st_mtime, unit='s')}"
        )
    else:
        st.info(
            "📔 Report not generated yet. "
            "Run `python src/main.py` to generate it."
        )

    st.markdown("---")
    if st.button("🔄 Regenerate Report Now", use_container_width=True):
        with st.spinner("Building PDF — please wait ..."):
            try:
                from report_generator import generate_report
                path = generate_report()
                st.success(f"✅ Report saved to: {path}")
                st.cache_data.clear()
                st.rerun()
            except Exception as e:
                st.error(f"Report generation failed: {e}")


# ============================================================
# MAIN APP
# ============================================================

def main():
    _sidebar()

    st.markdown(
        f'<p class="main-header">{APP_ICON} {APP_TITLE}</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="sub-header">'
        "PhD Research — Neural Network × NSGA-III × PySR × SHAP"
        "</p>",
        unsafe_allow_html=True,
    )

    tabs = st.tabs([
        "🏗️  Predict",
        "🧬  GA Dashboard",
        "📊  Results",
        "🤖  SHAP",
        "📝  Equation",
        "📄  Report",
    ])

    with tabs[0]: _tab_predict()
    with tabs[1]: _tab_ga_dashboard()
    with tabs[2]: _tab_results()
    with tabs[3]: _tab_shap()
    with tabs[4]: _tab_equation()
    with tabs[5]: _tab_report()


if __name__ == "__main__":
    main()
