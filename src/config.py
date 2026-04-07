# ============================================================
# src/config.py  —  Corrosion RC Beam Optimizer
# ============================================================
# v4 — Target changed to Mmax,exp (kNm) to match Zhang et al.
#       Added CatBoost + Optuna configuration
# ============================================================
from pathlib import Path

ROOT_DIR    = Path(__file__).resolve().parent.parent
DATA_DIR    = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"
MODELS_DIR  = RESULTS_DIR / "models"
FIGURES_DIR = RESULTS_DIR / "figures"
EQ_DIR      = RESULTS_DIR / "equations"
LOG_DIR     = ROOT_DIR / "results" / "logs"

for _dir in [MODELS_DIR, FIGURES_DIR, EQ_DIR, LOG_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

DATA_RAW   = DATA_DIR / "Database.csv"
DATA_CLEAN = DATA_DIR / "clean_data.csv"
RANDOM_STATE = 42

# ── Target ───────────────────────────────────────────────────
# PRIMARY: Mmax,exp (kNm) — experimental moment capacity
#   This is what Zhang et al. (2025) and Abushanab (2023) predict.
#   Predicting Mmax directly is far more learnable than R(%).
# SECONDARY: R(%) — kept for reporting and comparison only.
TARGET_COL   = "Mmax,exp (kNm)"
TARGET_COL_R = "Residual Capacity, R (%)"

# ── Numeric features ─────────────────────────────────────────
FEATURE_COLS = [
    "Width (mm)",
    "Depth (mm)",
    "Test Length (mm)",
    "Bottom Cover to Ctr of Tension Bar (mm)",
    "# Tensile Bars",
    "Diameter Tensile Bars, db,t (mm)",
    "Tension Reinforcement Ratio, pten (%)",
    "fy Longitudinal Bars (Tensile), (MPa) ",
    "f'c (MPa)",
    "W/C Ratio",
    "Stirrup Spacing, s (mm) ",
    "Stirrup Diameter, ds (mm)",
    "fy,s Stirrup Bars",
    "Mass Loss (Tensile bars), \u03b7m (%)",
    "Shear Span, x (mm)",
]

# ── Categorical features ─────────────────────────────────────
CAT_COLS = [
    "Longitudinal Bar Type",          # D / P
    "Test Type and Configuration",    # SS_FPB_MONO / SS_TPB / ...
    "Corrosion Method",               # IC / AC / C
]

# ── ACI reference columns (for benchmark comparison) ─────────
ACI_COLS = [
    "Width (mm)",
    "Depth (mm)",
    "Diameter Tensile Bars, db,t (mm)",
    "# Tensile Bars",
    "fy Longitudinal Bars (Tensile), (MPa) ",
    "f'c (MPa)",
    "Mass Loss (Tensile bars), \u03b7m (%)",
    "Mmax,exp (kNm)",
]

TEST_SIZE        = 0.20
VALIDATION_SIZE  = 0.10

# ── Benchmark Targets ────────────────────────────────────────
# L1: Beat ACI 318-19 (R² ≈ 0.867 on Mmax prediction)
# L2: Beat Zhang et al. 2025 PSO-CatBoost (R² = 0.972 on Test)
ACI_R2_BASELINE  = 0.867
L1_TARGET_R2     = 0.90      # Comfortably beat ACI 318-19
L1_LABEL         = "ACI 318-19 Benchmark"
L2_TARGET_R2     = 0.972     # Beat Zhang et al. (2025) SOTA
L2_LABEL         = "Zhang et al. (2025) SOTA"
BREAK_BOTH       = True

# ── MLP (kept for GA fitness evaluation — lightweight) ───────
NN_HIDDEN_LAYERS   = [128, 64, 32]
NN_DROPOUT         = 0.2
NN_LEARNING_RATE   = 0.001
NN_EPOCHS          = 500
NN_BATCH_SIZE      = 32
NN_PATIENCE        = 30
NN_L2_ALPHA        = 1e-4
NN_VALIDATION_FRAC = 0.10

# ── Ensemble Models ──────────────────────────────────────────
# XGBoost
XGB_N_ESTIMATORS   = 1000
XGB_MAX_DEPTH      = 6
XGB_LEARNING_RATE  = 0.05
XGB_SUBSAMPLE      = 0.8
XGB_COLSAMPLE      = 0.8
XGB_REG_ALPHA      = 0.1
XGB_REG_LAMBDA     = 1.0
XGB_EARLY_STOP     = 50

# Random Forest
RF_N_ESTIMATORS    = 500
RF_MAX_DEPTH       = None
RF_MIN_SAMPLES     = 2

# Gradient Boosting
GBR_N_ESTIMATORS   = 500
GBR_MAX_DEPTH      = 5
GBR_LEARNING_RATE  = 0.05
GBR_SUBSAMPLE      = 0.8

# CatBoost (NEW — same algorithm Zhang et al. used)
CAT_ITERATIONS     = 2000
CAT_DEPTH           = 8
CAT_LEARNING_RATE  = 0.05
CAT_L2_REG         = 3.0
CAT_EARLY_STOP     = 100

# ── Optuna (NEW — automatic hyperparameter tuning) ───────────
OPTUNA_N_TRIALS    = 100     # number of tuning trials
OPTUNA_CV_FOLDS    = 5       # CV folds during tuning
OPTUNA_TIMEOUT     = 600     # max seconds for tuning (10 min)

# ── GA — NSGA-III (optimises ensemble hyperparams) ───────────
GA_POPULATION_SIZE    = 40
GA_MAX_GENERATIONS    = 80
GA_CONSISTENCY_WINDOW = 15
GA_ELITE_SIZE         = 5
GA_CROSSOVER_RATE     = 0.85
GA_MUTATION_RATE      = 0.15
GA_MAX_RUNS           = 3
GA_N_OBJECTIVES       = 3
GA_N_PARTITIONS       = 12

GENE_BOUNDS = {
    "Width (mm)"                              : (100, 350),
    "Depth (mm)"                              : (100, 500),
    "fy Longitudinal Bars (Tensile), (MPa) "  : (226, 650),
    "f'c (MPa)"                               : (20,  80),
    "Mass Loss (Tensile bars), \u03b7m (%)"   : (0,   64),
}

W1 = 0.60
W2 = 0.25
W3 = 0.15

# ── PySR ─────────────────────────────────────────────────────
PYSR_NITERATIONS   = 200
PYSR_MAXSIZE       = 25
PYSR_POPULATIONS   = 40
PYSR_BINARY_OPS    = ["+", "-", "*", "/", "^"]
PYSR_UNARY_OPS     = ["sqrt", "log", "exp"]
PYSR_OUTPUT_FILE   = EQ_DIR / "best_equation.txt"
PYSR_LATEX_FILE    = EQ_DIR / "best_equation.latex"

# ── SHAP / Validation / Report ───────────────────────────────
SHAP_N_SAMPLES     = 200
SHAP_FIGURE_DPI    = 300
KFOLD_N_SPLITS     = 10
BOOTSTRAP_N        = 1000
WILCOXON_ALPHA     = 0.05

REPORT_TITLE       = "Corrosion RC Beam Optimizer \u2014 Scientific Report"
REPORT_FILE        = RESULTS_DIR / "Final_Report.pdf"
REPORT_AUTHOR      = "PhD Research \u2014 Corrosion RC Beam Optimization"
LOG_FILE           = LOG_DIR / "run_log.txt"

APP_TITLE          = "Corrosion RC Beam Optimizer"
APP_ICON           = "\U0001f3d7\ufe0f"
APP_LAYOUT         = "wide"

MODEL_MLP_PKL      = MODELS_DIR / "best_mlp.pkl"
MODEL_MLP_PT       = MODELS_DIR / "best_mlp.pt"
MODEL_GA_PKL       = MODELS_DIR / "best_ga_model.pkl"
MODEL_BEST_PKL     = MODELS_DIR / "best_model.pkl"
MODEL_CATBOOST_PKL = MODELS_DIR / "model_catboost.pkl"
SCALER_X_PATH      = MODELS_DIR / "scaler_X.pkl"
SCALER_Y_PATH      = MODELS_DIR / "scaler_y.pkl"
HALL_OF_FAME_PATH  = MODELS_DIR / "hall_of_fame.json"
