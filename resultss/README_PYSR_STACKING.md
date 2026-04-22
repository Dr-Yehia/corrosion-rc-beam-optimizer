# Stacking → PySR Symbolic Equation Pipeline

Distills the trained Stacking ensemble into a physically-informed symbolic equation
using PySR with MOEA/D-style multi-objective candidate selection.

## Files

| File | Description |
|---|---|
| `resultss/pysr_stacking_moead_selector.py` | Main script (fixed, production-ready) |
| `resultss/find_best_equation_from_stacking.py` | Easy wrapper — same as above |

## Required inputs

| File | Location |
|---|---|
| Trained Stacking model | `resultss/models/model_stacking.pkl` |
| Feature scaler | `resultss/models/scaler_X.pkl` |
| Category encoders | `resultss/models/cat_encoders.json` |
| Dataset | `data/Database.csv` |

## Run on Kaggle / Colab

```bash
# Install dependencies
pip install pysr sympy loguru scikit-learn xgboost

# Clone repo
git clone https://github.com/Dr-Yehia/corrosion-rc-beam-optimizer
cd corrosion-rc-beam-optimizer

# Run (default settings)
python resultss/pysr_stacking_moead_selector.py

# Run with custom settings
python resultss/pysr_stacking_moead_selector.py \
    --niterations 400 \
    --populations 60 \
    --maxsize 20 \
    --seed 42 \
    --ref-vectors 128 \
    --w-accuracy 0.4 \
    --w-physics 0.4 \
    --w-complexity 0.2
```

## CLI Options

| Option | Default | Description |
|---|---|---|
| `--niterations` | 220 | PySR iterations |
| `--populations` | 40 | PySR populations |
| `--maxsize` | 16 | Max equation size |
| `--seed` | 42 | Random seed |
| `--ref-vectors` | 64 | MOEA/D reference vectors |
| `--w-accuracy` | 0.45 | Weight for accuracy objectives (R2, MAPE, RMSE) |
| `--w-physics` | 0.45 | Weight for physics objectives (endpoints, monotonicity) |
| `--w-complexity` | 0.10 | Weight for complexity objective |

## Expected outputs

| File | Description |
|---|---|
| `resultss/equations/best_equation_stacking.txt` | Best equation (plain text) |
| `resultss/equations/best_equation_stacking.latex` | Best equation (LaTeX) |
| `resultss/models/pysr_stacking_metrics.json` | Final metrics (R², RMSE, MAE, MAPE) |
| `resultss/models/pysr_candidates_ranked.json` | All candidates ranked by score |
| `resultss/figures/pysr_stacking_scatter.png` | Predicted vs experimental scatter |
| `resultss/figures/pysr_stacking_endpoints.png` | Endpoint behaviour diagnostic |
| `resultss/logs/run_log_pysr_moead.txt` | Full run log |

## How it works

1. Loads the full dataset and rebuilds preprocessing
2. Loads `model_stacking.pkl` and generates predictions
3. Builds dimensionless ratio target: `R = M_stack / M_ACI`
4. Runs PySR symbolic regression on 5 dimensionless features: `eta, rho, d_b, csi, ri`
5. Scores every candidate equation on 7 objectives:
   - Accuracy: `1-R²`, `MAPE`, `RMSE_norm`
   - Physics endpoints: `|R(η=0) - 1|`, `|R(η=100%) - 0|`
   - Monotonicity violation w.r.t. η
   - Complexity
6. Selects candidates via MOEA/D Tchebycheff scalarization
7. Chooses final equation via configurable weighted scoring
8. Saves equation, metrics, and diagnostic plots
