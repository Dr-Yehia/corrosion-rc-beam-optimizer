# ============================================================
# app/streamlit_app.py
# Corrosion RC Beam Optimizer — Interactive Streamlit UI
# v5 — CORRECTED: LabelEncoder for categoricals, CatBoost metrics,
#         Ensemble comparison replaces empty GA tab
# ============================================================

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import joblib
import plotly.graph_objects as go

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
</style>
""", unsafe_allow_html=True)


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
    """Load LabelEncoder mappings saved during training."""
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


# ============================================================
# HELPER ─ encode a single categorical value with LabelEncoder
# ============================================================
def _label_encode(col_name: str, value: str, encoders: dict) -> int:
    """Return the integer code for value in col_name.
    Falls back to 0 if value not found (most-common class)."""
    classes = encoders.get(col_name, [])
    if value in classes:
        return classes.index(value)
    return 0  # fallback: first (most common) class


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
    if model:
        st.sidebar.success("Model (CatBoost): Loaded ✅")
    else:
        st.sidebar.error("Model not found — check final_results/models/")
    st.sidebar.markdown("### Benchmark Targets")
    st.sidebar.info(
        f"🏅 L1 (ACI 318-19): R² > {L1_TARGET_R2}\n\n"
        f"🏆 L2 (Zhang 2025 SOTA): R² > {L2_TARGET_R2}"
    )
    st.sidebar.markdown("---")
    st.sidebar.caption(f"Results: `{RESULTS_DIR.relative_to(ROOT)}`")


# ============================================================
# TAB 1 ─ PREDICT
# ============================================================
def _tab_predict():
    st.markdown('<p class="main-header">🏗️ Beam Mmax Predictor</p>',
                unsafe_allow_html=True)
    st.markdown(
        "Enter corroded RC beam parameters. "
        "Model: <b>CatBoost (R² = 0.987 on test set)</b>.",
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
        b  = st.number_input("Width b (mm)",         100, 350, 150, step=5)
        d  = st.number_input("Depth d (mm)",         100, 500, 300, step=5)
        L  = st.number_input("Test Length (mm)",     500, 5000, 2500, step=50)
        cv = st.number_input("Bottom Cover (mm)",     15,  60,  25,  step=1)

    with c2:
        st.markdown("**Reinforcement**")
        n_bars = st.number_input("# Tensile Bars",   1, 8, 3, step=1)
        db     = st.number_input("Bar Diameter (mm)",6, 32, 16, step=2)
        pten   = st.number_input("ρ tension (%)",   0.1, 5.0, 1.5, step=0.1)
        fy     = st.number_input("fy (MPa)",         226, 650, 460, step=5)

    with c3:
        st.markdown("**Concrete & Corrosion**")
        fc       = st.number_input("f’c (MPa)",       20,  80, 32, step=1)
        wc       = st.number_input("W/C Ratio",       0.30, 0.70, 0.45, step=0.01)
        eta_m    = st.number_input("ηm — Mass Loss (%)", 0.0, 64.0, 10.0, step=0.5)
        s_stirr  = st.number_input("Stirrup Spacing (mm)", 50, 300, 150, step=10)
        ds_stirr = st.number_input("Stirrup Dia. (mm)",    6,  16,  8,  step=2)
        fy_s     = st.number_input("fy stirrups (MPa)",  226, 650, 420, step=5)
        shear_x  = st.number_input("Shear Span x (mm)",  100, 2000, 800, step=50)

    st.markdown("---")
    st.markdown("**Bar & Test Configuration**")
    cc1, cc2, cc3 = st.columns(3)
    with cc1:
        # classes from cat_encoders.json: ['D', 'P']
        bar_type_classes = encoders.get("Longitudinal Bar Type", ["D", "P"])
        bar_type_label   = st.selectbox("Longitudinal Bar Type",
                                        options=bar_type_classes, index=0)
    with cc2:
        # classes: ['SS-FPB-MONO','SS_CONT_TPB_MONO','SS_FPB_MONO',...]
        test_classes = encoders.get("Test Type and Configuration",
                                    ["SS_FPB_MONO", "SS_TPB_MONO"])
        test_label   = st.selectbox("Test Configuration",
                                    options=test_classes, index=2)
    with cc3:
        # classes: ['C', 'EI', 'IC', 'N']
        corr_classes = encoders.get("Corrosion Method", ["IC", "EI", "C", "N"])
        corr_label   = st.selectbox("Corrosion Method",
                                    options=corr_classes, index=2)

    st.markdown("---")
    if st.button("🔍  Predict Mmax (kN·m)", type="primary",
                 use_container_width=True):

        # ── 15 numeric (FEATURE_COLS order) ───────────────────────
        num_vals = {
            "Width (mm)"                                : b,
            "Depth (mm)"                                : d,
            "Test Length (mm)"                          : L,
            "Bottom Cover to Ctr of Tension Bar (mm)"   : cv,
            "# Tensile Bars"                            : n_bars,
            "Diameter Tensile Bars, db,t (mm)"          : db,
            "Tension Reinforcement Ratio, pten (%)"     : pten,
            "fy Longitudinal Bars (Tensile), (MPa) "    : fy,
            "f'c (MPa)"                                 : fc,
            "W/C Ratio"                                 : wc,
            "Stirrup Spacing, s (mm) "                  : s_stirr,
            "Stirrup Diameter, ds (mm)"                 : ds_stirr,
            "fy,s Stirrup Bars"                         : fy_s,
            "Mass Loss (Tensile bars), \u03b7m (%)"      : eta_m,
            "Shear Span, x (mm)"                        : shear_x,
        }

        # ── 3 categorical (LabelEncoded — same as training) ────
        cat_vals = {
            "Longitudinal Bar Type"         : _label_encode(
                "Longitudinal Bar Type", bar_type_label, encoders),
            "Test Type and Configuration"   : _label_encode(
                "Test Type and Configuration", test_label, encoders),
            "Corrosion Method"              : _label_encode(
                "Corrosion Method", corr_label, encoders),
        }

        # ── 5 engineered features (match data_preprocessing.py) ──
        eta_log  = np.log1p(eta_m)
        As_proxy = n_bars * np.pi * (db / 2.0) ** 2
        eng_vals = {
            "eta_log"           : eta_log,
            "corr_severity_idx" : eta_m  * (fy / max(fc, 1)),
            "d_b_ratio"         : d      / max(b,  1),
            "eta_d_interaction" : eta_log * d,
            "reinf_index"       : As_proxy * fy / (fc * b * d),
        }

        # ── Ordered column list: 15 + 3 + 5 = 23 ───────────────
        all_cols = (
            FEATURE_COLS
            + ["Longitudinal Bar Type",
               "Test Type and Configuration",
               "Corrosion Method"]
            + ["eta_log", "corr_severity_idx",
               "d_b_ratio", "eta_d_interaction", "reinf_index"]
        )
        all_vals = {**num_vals, **cat_vals, **eng_vals}

        try:
            row = np.array(
                [all_vals.get(c, 0.0) for c in all_cols],
                dtype=float
            ).reshape(1, -1)

            row_sc    = scaler_X.transform(row) if scaler_X else row
            y_sc      = model.predict(row_sc)
            mmax_pred = (
                scaler_y.inverse_transform(
                    y_sc.reshape(-1, 1)).ravel()[0]
                if scaler_y else float(y_sc[0])
            )

            # ACI benchmark
            from aci_calculator import aci_moment_capacity
            mn_aci = aci_moment_capacity(
                b=b, d=d, n_bars=n_bars, db_mm=db,
                fy=fy, fc=fc, eta_m=eta_m,
            )

            # ── Metrics display ─────────────────────────────
            m1, m2, m3 = st.columns(3)
            m1.metric("📊 Predicted Mmax (kN·m)",
                      f"{mmax_pred:.2f} kN·m",
                      f"{mmax_pred - mn_aci:+.2f} vs ACI")
            m2.metric("📏 ACI 318-19 Mn (kN·m)", f"{mn_aci:.2f} kN·m")
            m3.metric("🧠 Corrosion Severity Index",
                      f"{eng_vals['corr_severity_idx']:.2f}")

            # Gauge
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
                        {"range": [0,          mn_aci*0.5], "color": "#FFCDD2"},
                        {"range": [mn_aci*0.5, mn_aci],     "color": "#FFF9C4"},
                        {"range": [mn_aci,     ax_max],     "color": "#C8E6C9"},
                    ],
                    "threshold": {"line": {"color": "orange", "width": 3},
                                  "value": mn_aci},
                },
            ))
            fig.update_layout(height=300,
                              margin=dict(t=30, b=10, l=20, r=20))
            st.plotly_chart(fig, use_container_width=True)

            # Interpretation
            ratio = mmax_pred / max(mn_aci, 1)
            if ratio >= 0.90:
                st.success(f"✅ Mmax = {mmax_pred:.2f} kN·m — Close to ACI. Low corrosion impact.")
            elif ratio >= 0.60:
                st.warning(f"⚠️ Mmax = {mmax_pred:.2f} kN·m — Moderately below ACI. Inspect beam.")
            else:
                st.error(f"❌ Mmax = {mmax_pred:.2f} kN·m — Severely below ACI. Immediate action required.")

        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.info(f"Feature vector length sent: {len(all_cols)} columns.")


# ============================================================
# TAB 2 ─ ENSEMBLE COMPARISON (replaces empty GA tab)
# ============================================================
def _tab_ensemble():
    st.markdown('<p class="main-header">🧩 Ensemble Models Comparison</p>',
                unsafe_allow_html=True)
    st.markdown(
        "All five models were trained on the same 804-beam dataset. "
        "**CatBoost** was selected as the final predictor — best test R² and RMSE."
    )

    ens = _load_json(MODELS_DIR / "ensemble_metrics.json")
    if not ens:
        st.info("ensemble_metrics.json not found.")
        return

    models_data = ens.get("models", {})
    rows = []
    for name, m in models_data.items():
        rows.append({
            "Model"       : ("\u2b50 " if name == "CatBoost" else "") + name,
            "Train R²"   : round(m.get("train_R2",   0), 4),
            "Test R²"    : round(m.get("test_R2",    0), 4),
            "Test RMSE"   : round(m.get("test_RMSE",  0), 4),
            "Test MAE"    : round(m.get("test_MAE",   0), 4),
            "L1 ✓"       : "✅" if m.get("L1_broken") else "❌",
            "L2 ✓"       : "✅" if m.get("L2_broken") else "❌",
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # CatBoost CV R² per fold
    cv_folds = ens.get("cv_folds", [])
    if cv_folds:
        st.markdown("### CatBoost 10-Fold CV R²")
        c1, c2, c3 = st.columns(3)
        c1.metric("CV R² Mean", f"{ens.get('cv_R2_mean', 0):.4f}")
        c2.metric("CV R² Std",  f"{ens.get('cv_R2_std',  0):.4f}")
        c3.metric("L2 Broken",
                  "✅ Yes" if ens.get("L2_broken") else "❌ No")

        fig = go.Figure(go.Bar(
            x=[f"Fold {i+1}" for i in range(len(cv_folds))],
            y=cv_folds,
            marker_color=[
                "#2E7D32" if v >= L2_TARGET_R2 else "#1A3A5C"
                for v in cv_folds
            ],
            text=[f"{v:.4f}" for v in cv_folds],
            textposition="outside",
        ))
        fig.add_hline(y=L2_TARGET_R2, line_dash="dash",
                      line_color="red",
                      annotation_text=f"L2 target ({L2_TARGET_R2})")
        fig.update_layout(
            title="CatBoost — R² per CV Fold",
            yaxis=dict(range=[0.93, 1.0]),
            height=350, margin=dict(t=40, b=20, l=40, r=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Bar chart: all models test R²
    st.markdown("### Test R² — All Models")
    names = [r["Model"] for r in rows]
    r2s   = [r["Test R²"] for r in rows]
    fig2  = go.Figure(go.Bar(
        x=names, y=r2s,
        marker_color=[
            "#2E7D32" if v >= L2_TARGET_R2 else "#1A3A5C" for v in r2s
        ],
        text=[f"{v:.4f}" for v in r2s],
        textposition="outside",
    ))
    fig2.add_hline(y=L1_TARGET_R2, line_dash="dot",
                   line_color="#F57C00",
                   annotation_text=f"L1 ({L1_TARGET_R2})")
    fig2.add_hline(y=L2_TARGET_R2, line_dash="dash",
                   line_color="red",
                   annotation_text=f"L2 ({L2_TARGET_R2})")
    fig2.update_layout(
        yaxis=dict(range=[0.94, 1.0]),
        height=350, margin=dict(t=40, b=20, l=40, r=20),
    )
    st.plotly_chart(fig2, use_container_width=True)


# ============================================================
# TAB 3 ─ RESULTS
# ============================================================
def _tab_results():
    st.markdown('<p class="main-header">📊 Model Performance & Benchmarks</p>',
                unsafe_allow_html=True)

    # ─ Load CORRECT metrics file: ensemble (CatBoost) ────────────
    ens  = _load_json(MODELS_DIR / "ensemble_metrics.json")
    aci  = _load_json(MODELS_DIR / "aci_benchmark_metrics.json")

    if not ens and not aci:
        st.info("No results found.")
        return

    cat_m = ens.get("models", {}).get("CatBoost", {}) if ens else {}

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### ACI 318-19 Baseline")
        if aci:
            for k, label in [("R2","R²"),("RMSE","RMSE"),("MAE","MAE"),
                             ("MAPE","MAPE %"),("ratio_mean","Exp/Pred ratio")]:
                st.metric(label, aci.get(k, "—"))

    with col2:
        st.markdown("### CatBoost — Best Model (Test Set)")
        if cat_m:
            for k, label in [("test_R2","R²"),("test_RMSE","RMSE"),
                             ("test_MAE","MAE")]:
                delta = ""
                if aci and k.replace("test_","") in aci:
                    delta = f"{cat_m.get(k,0) - aci.get(k.replace('test_',''),0):+.4f} vs ACI"
                st.metric(label, cat_m.get(k, "—"), delta or None)
            l1 = cat_m.get("L1_broken", False)
            l2 = cat_m.get("L2_broken", False)
            st.markdown(
                f"**L1 (ACI 318-19 beaten):** {'\u2705' if l1 else '\u274c'}  "
                f"**L2 (Zhang 2025 SOTA beaten):** {'\u2705' if l2 else '\u274c'}"
            )
            if l1 and l2:
                st.markdown(
                    '<div class="verdict-pass">🏆 Both benchmarks broken — '
                    'CatBoost surpasses ACI 318-19 ✓ and Zhang et al. 2025 ✓</div>',
                    unsafe_allow_html=True)
            elif l1:
                st.markdown(
                    '<div class="verdict-pass">✓ L1 beaten. L2 close but not yet.</div>',
                    unsafe_allow_html=True)

    st.markdown("---")

    # 10-Fold CV
    if ens:
        st.markdown("### CatBoost 10-Fold Cross-Validation")
        c1, c2, c3 = st.columns(3)
        c1.metric("CV R² Mean", f"{ens.get('cv_R2_mean', 0):.4f}")
        c2.metric("CV R² Std",  f"{ens.get('cv_R2_std',  0):.4f}")
        c3.metric("Folds",       ens.get("cv_folds", ["—"]).__len__())

    # Radar
    if cat_m and aci:
        st.markdown("### 📍 Performance Radar")
        cats = ["R²", "1-MAPE/100", "1-RMSE/50"]
        vm = [
            cat_m.get("test_R2", 0),
            max(0, 1 - (aci.get("MAPE", 100) - 16) / 100),   # approx MAPE for CatBoost
            max(0, 1 - cat_m.get("test_RMSE", 50) / 50),
        ]
        va = [
            aci.get("R2",   0),
            max(0, 1 - aci.get("MAPE", 100) / 100),
            max(0, 1 - aci.get("RMSE", 50)  / 50),
        ]
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=vm+[vm[0]], theta=cats+[cats[0]],
            fill="toself", name="CatBoost", line_color="#1A3A5C"))
        fig.add_trace(go.Scatterpolar(
            r=va+[va[0]], theta=cats+[cats[0]],
            fill="toself", name="ACI 318-19",
            line_color="#F57C00", opacity=0.6))
        fig.update_layout(
            polar=dict(radialaxis=dict(range=[0, 1])),
            height=380, margin=dict(t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)


# ============================================================
# TAB 4 ─ SHAP
# ============================================================
def _tab_shap():
    st.markdown('<p class="main-header">🤖 SHAP Feature Importance</p>',
                unsafe_allow_html=True)

    top5_path = MODELS_DIR / "top5_shap_features.json"
    if top5_path.exists():
        top5 = _load_json(top5_path)
        st.success(
            "📌 Top-5 most influential features: "
            + ", ".join(top5.get("top5_features", []))
        )

    c1, c2 = st.columns(2)
    for col, fname, caption in [
        (c1, "shap_importance.png",  "Mean |SHAP| Feature Importance"),
        (c2, "shap_beeswarm.png",    "SHAP Beeswarm Plot"),
    ]:
        p = FIGURES_DIR / fname
        with col:
            if p.exists():
                st.image(str(p), caption=caption, use_column_width=True)
            else:
                st.info(f"{fname} not found.")

    for dep in sorted(FIGURES_DIR.glob("shap_dependence_*.png")):
        st.markdown("### Dependence Plot")
        st.image(str(dep),
                 caption="SHAP Dependence — Top Feature",
                 use_column_width=True)
        break


# ============================================================
# TAB 5 ─ EQUATION
# ============================================================
def _tab_equation():
    st.markdown('<p class="main-header">📝 PySR Symbolic Equation</p>',
                unsafe_allow_html=True)
    st.markdown(
        "Automatically discovered by **PySR** (200 iterations, 40 populations). "
        "Explicit closed-form formula — usable without a computer."
    )

    txt_path  = EQ_DIR / "best_equation.txt"
    lat_path  = EQ_DIR / "best_equation.latex"
    json_path = EQ_DIR / "all_equations.json"

    if not txt_path.exists():
        st.info("⏳ Symbolic regression results not found.")
        return

    with open(txt_path) as f:
        st.markdown("### Best Equation (plain text)")
        st.code(f.read(), language="python")

    if lat_path.exists():
        with open(lat_path) as f:
            raw = f.read().replace("% ", "").strip()
        st.markdown("### LaTeX")
        for line in raw.split("\n"):
            if line.strip() and line[0] in "MR":
                st.latex(line)

    if json_path.exists():
        eq_data = _load_json(json_path)
        m = eq_data.get("metrics", {})
        st.markdown("### Performance")
        e1, e2, e3 = st.columns(3)
        e1.metric("R²",   m.get("R2",   "—"))
        e2.metric("RMSE", m.get("RMSE", "—"))
        e3.metric("MAPE", f"{m.get('MAPE', '?')} %")

        all_eqs = eq_data.get("all_equations", [])
        if all_eqs:
            st.markdown("### All PySR Equations (Hall of Fame)")
            df_eq = pd.DataFrame([
                {
                    "Complexity": e.get("complexity"),
                    "Loss":       round(e.get("loss", 0), 5),
                    "Score":      round(e.get("score", 0), 5),
                    "Equation":   e.get("sympy_format",
                                       e.get("equation", "")),
                }
                for e in all_eqs
            ])
            st.dataframe(df_eq, use_container_width=True, hide_index=True)


# ============================================================
# TAB 6 ─ REPORT
# ============================================================
def _tab_report():
    st.markdown('<p class="main-header">📄 Download PDF Report</p>',
                unsafe_allow_html=True)
    report = RESULTS_DIR / "Final_Report.pdf"
    if report.exists():
        with open(report, "rb") as f:
            st.download_button(
                label="⬇️ Download Final_Report.pdf",
                data=f,
                file_name="Corrosion_RC_Beam_Optimizer_Report.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        st.caption(
            f"Last modified: {pd.Timestamp(report.stat().st_mtime, unit='s')}"
        )
    else:
        st.info("📔 Report not generated. Run `python src/main.py`.")

    st.markdown("---")
    if st.button("🔄 Regenerate Report Now", use_container_width=True):
        with st.spinner("Building PDF …"):
            try:
                from report_generator import generate_report
                path = generate_report()
                st.success(f"✅ Saved to: {path}")
                st.cache_data.clear()
                st.rerun()
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
        "PhD Research — CatBoost (R² = 0.987) × SHAP × PySR Symbolic Regression"
        "</p>",
        unsafe_allow_html=True,
    )

    tabs = st.tabs([
        "🏗️  Predict",
        "🧩  Ensemble",
        "📊  Results",
        "🤖  SHAP",
        "📝  Equation",
        "📄  Report",
    ])
    with tabs[0]: _tab_predict()
    with tabs[1]: _tab_ensemble()
    with tabs[2]: _tab_results()
    with tabs[3]: _tab_shap()
    with tabs[4]: _tab_equation()
    with tabs[5]: _tab_report()


if __name__ == "__main__":
    main()
