# ============================================================
# src/config.py
# Corrosion RC Beam Optimizer — Central Configuration
# All constants, paths, hyperparameters, and benchmark targets
# ============================================================

from pathlib import Path

# ── Project Root ─────────────────────────────────────────────
ROOT_DIR    = Path(__file__).resolve().parent.parent
DATA_DIR    = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"
MODELS_DIR  = RESULTS_DIR / "models"
FIGURES_DIR = RESULTS_DIR / "figures"
EQ_DIR      = RESULTS_DIR / "equations"
LOG_DIR     = ROOT_DIR / "results" / "logs"

# Auto-create output directories if they don't exist
for _dir in [MODELS_DIR, FIGURES_DIR, EQ_DIR, LOG_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# ── Data Files ───────────────────────────────────────────────
DATA_RAW   = DATA_DIR / "Database.csv"
DATA_CLEAN = DATA_DIR / "clean_data.csv"

# ── Reproducibility ──────────────────────────────────────────
RANDOM_STATE = 42

# ── Target Variable & Features ───────────────────────────────
TARGET_COL = "Residual Capacity, R (%)"

# Input features used in model training
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
    "Mass Loss (Tensile bars), ηm (%)",
    "Shear Span, x (mm)",
]

# ACI 318-19 required columns (for benchmark calculation)
ACI_COLS = [
    "Width (mm)",
    "Depth (mm)",
    "Diameter Tensile Bars, db,t (mm)",
    "# Tensile Bars",
    "fy Longitudinal Bars (Tensile), (MPa) ",
    "f'c (MPa)",
    "Mass Loss (Tensile bars), ηm (%)",
    "Mmax,exp (kNm)",
]

# ── Train / Test Split ───────────────────────────────────────
TEST_SIZE        = 0.20
VALIDATION_SIZE  = 0.10

# ── ACI Benchmark Targets ────────────────────────────────────
# Layer 1: Beat ACI 318-19 with statistical significance
ACI_R2_BASELINE  = 0.50
L1_TARGET_R2     = 0.85
L1_LABEL         = "ACI 318-19 Benchmark"

# Layer 2: Beat best published ML model (Zhang et al., 2025)
L2_TARGET_R2     = 0.970
L2_LABEL         = "Zhang et al. (2025) SOTA"

# Both layers must be broken simultaneously
BREAK_BOTH       = True

# ── Neural Network (MLP) ─────────────────────────────────────
NN_HIDDEN_LAYERS  = [64, 32]
NN_DROPOUT        = 0.2
NN_LEARNING_RATE  = 0.001
NN_EPOCHS         = 500
NN_BATCH_SIZE     = 32
NN_PATIENCE       = 30

# ── Genetic Algorithm — NSGA-III ─────────────────────────────
GA_POPULATION_SIZE    = 100
GA_MAX_GENERATIONS    = 500
GA_CONSISTENCY_WINDOW = 30
GA_ELITE_SIZE         = 10
GA_CROSSOVER_RATE     = 0.90
GA_MUTATION_RATE      = 0.10
GA_MAX_RUNS           = 10
GA_N_OBJECTIVES       = 3
GA_N_PARTITIONS       = 12

# Chromosome gene bounds
GENE_BOUNDS = {
    "Width (mm)"                              : (100, 350),
    "Depth (mm)"                              : (100, 500),
    "fy Longitudinal Bars (Tensile), (MPa) "  : (226, 650),
    "f'c (MPa)"                               : (20,  80),
    "Mass Loss (Tensile bars), ηm (%)"        : (0,   64),
}

# ── Fitness Function Weights ──────────────────────────────────
W1 = 0.50   # R² accuracy
W2 = 0.30   # ACI improvement  (Mpred/MACI → 1.0)
W3 = 0.20   # Physics penalty

# ── PySR Symbolic Regression ─────────────────────────────────
PYSR_NITERATIONS   = 100
PYSR_MAXSIZE       = 20
PYSR_POPULATIONS   = 30
PYSR_BINARY_OPS    = ["+", "-", "*", "/", "^"]
PYSR_UNARY_OPS     = ["sqrt", "log", "exp"]
PYSR_OUTPUT_FILE   = EQ_DIR / "best_equation.txt"
PYSR_LATEX_FILE    = EQ_DIR / "best_equation.latex"

# ── SHAP Analysis ────────────────────────────────────────────
SHAP_N_SAMPLES     = 200
SHAP_FIGURE_DPI    = 300

# ── Statistical Validation ───────────────────────────────────
KFOLD_N_SPLITS     = 10
BOOTSTRAP_N        = 1000
WILCOXON_ALPHA     = 0.05

# ── PDF Report ───────────────────────────────────────────────
REPORT_TITLE       = "Corrosion RC Beam Optimizer — Scientific Report"
REPORT_FILE        = RESULTS_DIR / "Final_Report.pdf"
REPORT_AUTHOR      = "PhD Research — Corrosion RC Beam Optimization"
LOG_FILE           = LOG_DIR / "run_log.txt"

# ── Streamlit App ────────────────────────────────────────────
APP_TITLE          = "Corrosion RC Beam Optimizer"
APP_ICON           = "🏗️"
APP_LAYOUT         = "wide"

# ── Saved Model Paths ─────────────────────────────────────────
MODEL_MLP_PKL      = MODELS_DIR / "best_mlp.pkl"
MODEL_MLP_PT       = MODELS_DIR / "best_mlp.pt"
MODEL_GA_PKL       = MODELS_DIR / "best_ga_model.pkl"
SCALER_X_PATH      = MODELS_DIR / "scaler_X.pkl"
SCALER_Y_PATH      = MODELS_DIR / "scaler_y.pkl"
HALL_OF_FAME_PATH  = MODELS_DIR / "hall_of_fame.json"
