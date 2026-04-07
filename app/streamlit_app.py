# ============================================================
# app/streamlit_app.py
# Corrosion RC Beam Optimizer — Interactive Streamlit UI
#
# Tabs:
#   1. 🏗️  Predict    ─ single beam R(%) prediction
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
        "Predicts Predicted Mmax (kN·m) of corroded RC beams."
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
        '<p class="main-header">🏗️ Beam R(%) Predictor</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        "Enter your corroded RC beam parameters below. "
        "The model predicts the <b>Residual Flexural Capacity R(%)</b>.",
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
        fc       = st.number_input("f’c (MPa)",        20,  80, 32, step=1)
        wc       = st.number_input("W/C Ratio",        0.30, 0.70, 0.45, step=0.01)
        eta_m    = st.number_input("ηm — Mass Loss (%)", 0.0, 64.0, 10.0, step=0.5)
        s_stirr  = st.number_input("Stirrup Spacing (mm)", 50, 300, 150, step=10)
        ds_stirr = st.number_input("Stirrup Dia. (mm)",     6,  16,   8, step=2)
        fy_s     = st.number_input("fy stirrups (MPa)",   226, 650, 420, step=5)
        shear_x  = st.number_input("Shear Span x (mm)",   100, 2000, 800, step=50)

    st.markdown("---")
    if st.button("🔍  Predict R(%)", type="primary", use_container_width=True):
        # Build input vector in the exact FEATURE_COLS order
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
        # Engineered features
        input_dict["corr_severity_idx"] = eta_m * (fy / max(fc, 1))
        input_dict["d_b_ratio"]         = d / max(b, 1)
        input_dict["eta_d_interaction"] = eta_m * d
                
        # Add categorical features with default values (most common in dataset)
        input_dict["Longitudinal Bar Type_D"] = 1  # Deformed
        input_dict["Longitudinal Bar Type_P"] = 0
        input_dict["Test Type and Configuration_SS_FPB_MONO"] = 1  # Most common
        input_dict["Test Type and Configuration_SS_TPB"] = 0
        input_dict["Corrosion Method_IC"] = 1  # Impressed current (most common)
        input_dict["Corrosion Method_AC"] = 0
        input_dict["Corrosion Method_C"] = 0

        # Align with training columns
        try:
            model_cols = (
                FEATURE_COLS +
                ["corr_severity_idx", "d_b_ratio", "eta_d_interaction"] + 
                ["Longitudinal Bar Type_D",                 "Test Type and Configuration_SS_FPB_MONO", "Test Type and Configuration_SS_TPB",
                 "Test Type and Configuration_SS_TPB",            )
                             "Corrosion Method_IC", "Corrosion Method_AC", "Corrosion Method_C"]
            row = np.array(
                [input_dict.get(c, 0.0) for c in model_cols],
                dtype=float
            ).reshape(1, -1)

            if scaler_X:
                row_sc = scaler_X.transform(row)
            else:
                row_sc = row

            y_pred_sc = model.predict(row_sc)
            r_pct     = (
                scaler_y.inverse_transform(y_pred_sc.reshape(-1, 1)).ravel()[0]
                if scaler_y else float(y_pred_sc[0])
            )

            # ACI benchmark prediction
            from aci_calculator import aci_moment_capacity
            mn = aci_moment_capacity(
                b=b, d=d, n_bars=n_bars, db_mm=db,
                fy=fy, fc=fc, eta_m=eta_m,
            )

            col_r, col_m, col_c = st.columns(3)
            col_r.metric(
                label="📊 Predicted R(%)",
                value=f"{r_pct:.1f} %",
                delta=(
                    f"{r_pct - 100:.1f}% vs Control"
                    if r_pct != 100 else ""
                ),
            )
            col_m.metric(
                label="📏 ACI Mn (kN·m)",
                value=f"{mn:.2f} kN·m",
            )
            col_c.metric(
                label="🧠 Corrosion Severity Index",
                value=f"{input_dict['corr_severity_idx']:.2f}",
            )

            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode  = "gauge+number+delta",
                value = round(r_pct, 1),
                delta = {"reference": 100, "valueformat": ".1f"},
                title = {"text": "Predicted Mmax (kN·m)"},
                gauge = {
                    "axis"  : {"range": [0, 130]},
                    "bar"   : {"color": "#1A3A5C"},
                    "steps" : [
                        {"range": [0,   50], "color": "#FFCDD2"},
                        {"range": [50,  80], "color": "#FFF9C4"},
                        {"range": [80, 130], "color": "#C8E6C9"},
                    ],
                    "threshold": {
                        "line" : {"color": "red", "width": 3},
                        "value": 100,
                    },
                },
            ))
            fig.update_layout(height=320, margin=dict(t=30,b=10,l=20,r=20))
            st.plotly_chart(fig, use_container_width=True)

            # Interpretation
            if r_pct >= 90:
                st.success(f"✅ Beam retains {r_pct:.1f}% of original capacity — Low risk.")
            elif r_pct >= 60:
                st.warning(f"⚠️ Beam retains {r_pct:.1f}% — Moderate corrosion damage.")
            else:
                st.error(f"❌ Beam retains only {r_pct:.1f}% — Severe damage. Inspection required.")

        except Exception as e:
            st.error(f"Prediction error: {e}")


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

    # Fitness progression across runs
    runs  = [e["run"]     for e in hof]
    fitns = [e["fitness"] for e in hof]
    r2s   = [e["metrics"].get("R2", 0) for e in hof]

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

    # Hall of Fame table
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

    # Load saved metrics
    mlp_m  = _load_json(MODELS_DIR / "mlp_metrics.json")
    aci_m  = _load_json(MODELS_DIR / "aci_benchmark_metrics.json")

    if not mlp_m and not aci_m:
        st.info("No results found. Run the pipeline first.")
        return

    # Side-by-side comparison
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
                f"**L1 (ACI):** {'\u2705' if l1 else '\u274c'}    "
                f"**L2 (SOTA):** {'\u2705' if l2 else '\u274c'}"
            )

    st.markdown("---")

    # K-Fold CV
    if mlp_m and "cv" in mlp_m:
        st.markdown("### 10-Fold Cross-Validation")
        cv = mlp_m["cv"]
        c1, c2, c3 = st.columns(3)
        c1.metric("CV R\u00b2 Mean", cv.get("cv_R2_mean",   "—"))
        c2.metric("CV R\u00b2 Std",  cv.get("cv_R2_std",    "—"))
        c3.metric("CV RMSE Mean", cv.get("cv_RMSE_mean", "—"))

    # Radar chart — model vs ACI
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

    bar_path     = FIGURES_DIR / "shap_importance.png"
    beeswarm_path= FIGURES_DIR / "shap_beeswarm.png"
    top5_path    = MODELS_DIR  / "top5_shap_features.json"

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

    st.markdown("### Discovered Equation (plain)")
    st.code(eq_text, language="python")

    if latex_path.exists():
        with open(latex_path) as f:
            eq_latex = f.read()
        st.markdown("### LaTeX Format")
        clean = eq_latex.replace("% ", "").strip()
        for line in clean.split("\n"):
            if line.startswith("R"):
                st.latex(line)

    if json_path.exists():
        eq_data = _load_json(json_path)
        st.markdown("### Performance Metrics")
        m = eq_data.get("metrics", {})
        c1, c2, c3 = st.columns(3)
        c1.metric("R\u00b2",   m.get("R2",   "—"))
        c2.metric("RMSE", m.get("RMSE", "—"))
        c3.metric("MAPE", f"{m.get('MAPE', '?')} %")

        sts = [
            ("🏅 L1 (ACI) ",   m.get("L1_broken", False)),
            ("🏆 L2 (SOTA)", m.get("L2_broken", False)),
        ]
        for label, ok in sts:
            if ok:
                st.success(f"{label}: Broken ✅")
            else:
                st.warning(f"{label}: Not yet broken")


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

    # Regenerate button
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
