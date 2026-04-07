# Corrosion RC Beam Optimizer

> **PhD Research** — AI-powered prediction of residual moment capacity  
> for corroded reinforced concrete beams using NSGA-III + CatBoost  

---

## Overview

This repository predicts **Mmax,exp (kN·m)** — the experimental residual moment capacity  
of corroded RC beams — using a two-layer benchmark strategy:

| Layer | Target | Reference |
|-------|--------|-----------|
| **L1** | R² ≥ 0.90 | Beat ACI 318-19 (R² ≈ 0.867) |
| **L2** | R² ≥ 0.972 | Beat Zhang et al. (2025) PSO-CatBoost SOTA |

**Dataset**: 804 experimental beam specimens from published literature.  
**Best model**: CatBoost + Optuna — Test R² = **0.9870** ✅ Both benchmarks broken.

---

## Repository Structure

```
corrosion-rc-beam-optimizer/
├── src/                    ← Python modules (pipeline core)
│   ├── config.py           ← All constants and hyperparameters
│   ├── data_preprocessing.py
│   ├── aci_calculator.py
│   ├── neural_network.py
│   ├── ensemble_models.py
│   ├── genetic_algorithm.py
│   ├── phase_5_validation.py
│   ├── shap_analysis.py
│   ├── pysr_equations.py
│   ├── report_generator.py
│   └── main.py
├── app/
│   └── streamlit_app.py    ← Interactive dashboard
├── data/
│   └── Database.csv        ← 804-specimen dataset
├── final_results/
│   ├── models/             ← Saved .pkl files + JSON metrics
│   ├── figures/            ← SHAP plots, residual plots
│   └── equations/          ← PySR symbolic equations
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone https://github.com/dr-yehia/corrosion-rc-beam-optimizer.git
cd corrosion-rc-beam-optimizer
pip install -r requirements.txt
```

---

## Running the Pipeline

```bash
# Full pipeline (all phases)
python src/main.py

# Interactive dashboard
streamlit run app/streamlit_app.py
```

---

## GA — NSGA-III Configuration

The following values are authoritative and match `src/config.py` exactly.

| Parameter | Value | Notes |
|-----------|-------|-------|
| Population size | 40 | `GA_POPULATION_SIZE` |
| Max generations | 80 | `GA_MAX_GENERATIONS` |
| Max runs | 3 | `GA_MAX_RUNS` |
| Elite size | 5 | `GA_ELITE_SIZE` |
| Crossover rate | 0.85 | BLX-α crossover |
| Mutation rate | 0.15 | Gaussian mutation |
| Convergence window | 15 | Gens without improvement |
| **W1** (R² weight) | **0.60** | Fitness = W1·R² + W2·ACI_score − W3·penalty |
| **W2** (ACI weight) | **0.25** | |
| **W3** (penalty) | **0.15** | |
| Objectives | 3 | R², RMSE, CV-R² |

---

## MLP Architecture

| Parameter | Value |
|-----------|-------|
| Hidden layers | [128, 64, 32] |
| Activation | ReLU |
| Optimizer | Adam (lr=0.001) |
| Epochs | 500 |
| Early stopping patience | 30 |
| L2 regularisation α | 1e-4 |
| Solver | sklearn MLPRegressor |

> Note: Dropout is not used — `sklearn.MLPRegressor` does not natively  
> support dropout. A PyTorch-based model can be substituted if needed.

---

## Results Summary

| Model | Train R² | Test R² | RMSE | MAE | L1 | L2 |
|-------|----------|---------|------|-----|----|----|  
| ACI 318-19 *(baseline)* | — | 0.8839 | 8.229 | 5.062 | ✅ | ❌ |
| MLP | 0.9882 | 0.9695 | 4.471 | 2.342 | ✅ | ❌ |
| XGBoost | 0.9949 | 0.9861 | 3.019 | 1.770 | ✅ | ✅ |
| Random Forest | 0.9875 | 0.9810 | 3.530 | 1.959 | ✅ | ✅ |
| GBR | 0.9992 | 0.9831 | 3.333 | 1.788 | ✅ | ✅ |
| **CatBoost ★** | **0.9931** | **0.9870** | **2.921** | **1.745** | ✅ | ✅ |
| Stacking | 0.9990 | 0.9845 | 3.191 | 1.660 | ✅ | ✅ |

**10-Fold CV (CatBoost):** Mean R² = 0.9713 ± 0.0086  
**Bootstrap CI (1000 iter):** R² ∈ [0.9791, 0.9917]  
**Wilcoxon vs ACI:** p < 0.0001 ✅  
**Cohen's d:** 0.5814 (medium–large effect)  
**McNemar accuracy:** 99.38% vs 85.09% (p = 4e-06) ✅  

---

## SHAP Feature Importance (Top 5)

| Rank | Feature | SHAP Value |
|------|---------|------------|
| 1 | Depth (mm) | 7.894 |
| 2 | db,t (mm) | 3.882 |
| 3 | ds (mm) | 2.697 |
| 4 | Width (mm) | 2.245 |
| 5 | Test Length | 1.418 |

> Key finding: ηm (mass loss %) ranks 10th — geometric dimensions  
> dominate residual capacity over corrosion degree alone.

---

## ACI 318-19 Nominal Moment Calculation

The ACI benchmark computes **nominal** Mn (no φ factor) for direct  
comparison with experimental results:

$$M_n = A_{s,corr} \cdot f_{y,corr} \cdot \left(d - \frac{a}{2}\right)$$

where:
- $A_{s,corr} = A_s \cdot (1 - \eta_m/100)$
- $f_{y,corr} = f_y \cdot (1 - \eta_m/100)$  
- $a = \frac{A_{s,corr} \cdot f_{y,corr}}{0.85 \cdot f'_c \cdot b}$

> φ = 0.90 is intentionally omitted — applying φ to predictions  
> compared against experimental data introduces systematic bias.

---

## Citation

If you use this work, please cite:  
*[To be updated upon publication]*

---

## License

MIT License — see `LICENSE` for details.
