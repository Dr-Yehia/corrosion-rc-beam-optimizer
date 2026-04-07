<!-- README.md — Corrosion RC Beam Optimizer -->

<h1 align="center">
  🏗️ Corrosion RC Beam Optimizer
</h1>

<p align="center">
  <b>PhD Research Pipeline</b> &nbsp;—&nbsp;
  Neural Network × NSGA-III Genetic Algorithm × PySR × SHAP<br/>
  Predicting the <b>Residual Flexural Capacity R(%)</b> of corroded reinforced concrete beams
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B?logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/PySR-0.18%2B-8A2BE2" />
  <img src="https://img.shields.io/badge/SHAP-0.44%2B-00897B" />
  <img src="https://img.shields.io/badge/license-MIT-green" />
</p>

---

## 📌 Overview

This repository contains the complete research pipeline for predicting the
**residual flexural capacity R(%)** of corroded reinforced concrete (RC) beams.
The study introduces a two-layer benchmarking strategy:

| Layer | Benchmark | Target R² | Reference |
|-------|-----------|------------|----------|
| L1 | ACI 318-19 analytical model | > **0.85** | ACI Committee 318 |
| L2 | State-of-the-art ML | > **0.970** | Zhang et al. (2025) |

Both layers must be surpassed simultaneously, with statistical significance
confirmed by Wilcoxon, Bootstrap CI, 10-Fold CV, Cohen’s d, and McNemar tests.

---

## 🏗️ Architecture

```
Raw Database (804 specimens)
        │
        ▼
┌────────────────────────────────────────────┐
│  Phase 0  ACI 318-19 Baseline Benchmark         │
└────────────────────────────────────────────┘
        │
        ▼
┌────────────────────────────────────────────┐
│  Phase 1  MLP Baseline  [64→32→1]              │
└────────────────────────────────────────────┘
        │
        ▼
┌────────────────────────────────────────────┐
│  Phase 2  NSGA-III GA   pop=100, gen=500        │
│           Fitness = W1·R² + W2·ACI − W3·P      │
│           Restart-on-convergence (max 10 runs)   │
└────────────────────────────────────────────┘
        │
   ┌───┴───┐
   ▼         ▼
┌───────────┐  ┌───────────┐
│ Phase 3   │  │ Phase 4   │
│ PySR SR   │  │ SHAP      │
│ Equation  │  │ Analysis  │
└───────────┘  └───────────┘
        │
        ▼
┌────────────────────────────────────────────┐
│  Phase 5  Statistical Validation               │
│           Wilcoxon · Bootstrap · 10-Fold CV     │
│           Cohen’s d · McNemar                   │
└────────────────────────────────────────────┘
        │
        ▼
┌────────────────────────────────────────────┐
│  Phase 6  PDF Report + Streamlit App            │
└────────────────────────────────────────────┘
```

---

## 📁 Repository Structure

```
corrosion-rc-beam-optimizer/
├── data/
│   ├── Database.csv          ← raw experimental database (804 specimens)
│   └── clean_data.csv        ← auto-generated after preprocessing
├── src/
│   ├── config.py             ← all constants, paths, hyperparameters
│   ├── data_preprocessing.py ← load → clean → engineer → scale → split
│   ├── aci_calculator.py     ← ACI 318-19 Mn equation (vectorised)
│   ├── neural_network.py     ← MLP build / train / evaluate / save
│   ├── genetic_algorithm.py  ← NSGA-III + fitness + hall of fame
│   ├── symbolic_regression.py← PySR closed-form equation discovery
│   ├── shap_analysis.py      ← SHAP bar + beeswarm + dependence
│   ├── statistical_validation.py ← Wilcoxon / Bootstrap / CV / Cohen
│   ├── report_generator.py   ← automated PDF scientific report
│   └── main.py               ← master pipeline orchestrator
├── app/
│   └── streamlit_app.py      ← 6-tab interactive Streamlit dashboard
├── results/                  ← auto-generated
│   ├── models/               ← saved models, scalers, metrics JSON
│   ├── figures/              ← SHAP plots, training curves (PNG)
│   ├── equations/            ← PySR equation (.txt, .latex, .json)
│   ├── logs/                 ← run_log.txt
│   └── Final_Report.pdf      ← automated scientific report
├── tests/
│   ├── test_preprocessing.py
│   ├── test_aci_calculator.py
│   └── test_neural_network.py
├── .github/workflows/ci.yml  ← GitHub Actions CI
├── requirements.txt
└── README.md
```

---

## ⚡ Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/Dr-Yehia/corrosion-rc-beam-optimizer.git
cd corrosion-rc-beam-optimizer
pip install -r requirements.txt
```

### 2. Add Database

Place your experimental CSV at:
```
data/Database.csv
```

### 3. Run Full Pipeline

```bash
python src/main.py
```

### 4. Run Specific Phases

```bash
# ACI benchmark + MLP only
python src/main.py --phase 0 1

# Skip PySR (faster run)
python src/main.py --skip-pysr

# Regenerate PDF report from saved results
python src/main.py --report-only
```

### 5. Launch Streamlit App

```bash
streamlit run app/streamlit_app.py
```

---

## 🧪 Statistical Validation Protocol

| Test | Purpose | Threshold |
|------|---------|----------|
| **Wilcoxon Signed-Rank** | Model errors < ACI errors | p < 0.05 |
| **Bootstrap CI** (n=1000) | 95% CI lower bound of R² | CI₀ ≥ L1_TARGET |
| **10-Fold CV** | Stability across all folds | All folds R² ≥ L1 |
| **Cohen’s d** | Practical effect size | d > 0.80 (large) |
| **McNemar Test** | Classification accuracy | p < 0.05 |

---

## 🧬 NSGA-III Fitness Function

$$
\text{FF} = W_1 \cdot R^2_{\text{test}} + W_2 \cdot \text{ACI\_score} - W_3 \cdot \text{penalty}
$$

| Weight | Value | Objective |
|--------|-------|----------|
| W₁ | 0.50 | Maximise R² |
| W₂ | 0.30 | Improve over ACI |
| W₃ | 0.20 | Physics constraint penalty |

**Stopping rule:** L1 ∧ L2 both broken → stop immediately. Convergence detected (30-gen window) → restart new run. Max 10 runs.

---

## 📊 Key Results (Expected)

| Model | R² | RMSE | L1 | L2 |
|-------|-----|------|----|----|s
| ACI 318-19 | ~0.50 | — | — | — |
| MLP Baseline | >0.85 | — | ✓ | — |
| GA-Optimised | >0.970 | — | ✓ | ✓ |

---

## 📦 Dependencies

| Package | Version | Purpose |
|---------|---------|--------|
| scikit-learn | ≥1.3 | MLP, preprocessing, CV |
| pymoo | ≥0.6 | NSGA-III reference points |
| pysr | ≥0.18 | Symbolic regression |
| shap | ≥0.44 | Feature importance |
| reportlab | ≥4.1 | PDF report generation |
| streamlit | ≥1.32 | Interactive dashboard |
| loguru | ≥0.7 | Structured logging |

---

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@phdthesis{attia2026corrosion,
  author  = {Attia, Yehia Abdelhamid},
  title   = {Predicting Residual Flexural Capacity of Corroded RC Beams
             Using Neural Network-NSGA-III Optimisation},
  year    = {2026},
  school  = {PhD Research},
  note    = {\url{https://github.com/Dr-Yehia/corrosion-rc-beam-optimizer}}
}
```

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">
  Built with ❤️ for structural engineering research<br/>
  <a href="https://linkedin.com/in/yehia-attia-b661101a2">LinkedIn</a>
  &nbsp;·&nbsp;
  <a href="https://github.com/Dr-Yehia">GitHub</a>
</p>
