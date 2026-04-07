# ============================================================
# src/config.py
# Corrosion RC Beam Optimizer — Central Configuration
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

for _dir in [MODELS_DIR, FIGURES_DIR, EQ_DIR, LOG_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# ── Data Files ───────────────────────────────────────────────
DATA_RAW   = DATA_DIR / "Database.csv"
DATA_CLEAN = DATA_DIR / "clean_data.csv"

# ── Reproducibility ──────────────────────────────────────────
RANDOM_STATE = 42

# ── Target Variable & Features ───────────────────────────────
TARGET_COL = "Residual Capacity, R (%)"

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

# ── Train / Test Split ───────────────────────────────────────
TEST_SIZE        = 0.20
VALIDATION_SIZE  = 0.10

# ── Benchmark Targets ────────────────────────────────────────
ACI_R2_BASELINE  = 0.50
L1_TARGET_R2     = 0.85     # Beat ACI 318-19
L1_LABEL         = "ACI 318-19 Benchmark"
L2_TARGET_R2     = 0.970    # Beat Zhang et al. (2025) SOTA
L2_LABEL         = "Zhang et al. (2025) SOTA"
BREAK_BOTH       = True

# ── Neural Network (MLP) — UPGRADED ──────────────────────────
# Deeper architecture + stronger regularisation + early stopping
NN_HIDDEN_LAYERS  = [256, 128, 64, 32]   # deeper than before [64,32]
NN_DROPOUT        = 0.3                  # used if torch backend added
NN_LEARNING_RATE  = 0.0005               # slower = more stable
NN_EPOCHS         = 1000                 # more headroom
NN_BATCH_SIZE     = 32
NN_PATIENCE       = 50                   # early stop patience
NN_L2_ALPHA       = 5e-4                 # stronger L2 regularisation
NN_VALIDATION_FRAC= 0.15                 # 15% for early-stop validation

# ── Genetic Algorithm — NSGA-III ─────────────────────────────
# Smaller pop/gen for Phase-2 speed; GA role is hyperparameter search
GA_POPULATION_SIZE    = 50     # was 100 — halved for speed
GA_MAX_GENERATIONS    = 100    # was 500 — enough to converge
GA_CONSISTENCY_WINDOW = 20     # was 30
GA_ELITE_SIZE         = 5      # was 10
GA_CROSSOVER_RATE     = 0.85
GA_MUTATION_RATE      = 0.15   # slightly higher for diversity
GA_MAX_RUNS           = 5      # was 10
GA_N_OBJECTIVES       = 3
GA_N_PARTITIONS       = 12

# Chromosome gene bounds (what GA optimises)
GENE_BOUNDS = {
    "Width (mm)"                              : (100, 350),
    "Depth (mm)"                              : (100, 500),
    "fy Longitudinal Bars (Tensile), (MPa) "  : (226, 650),
    "f'c (MPa)"                               : (20,  80),
    "Mass Loss (Tensile bars), \u03b7m (%)"   : (0,   64),
}

# ── Fitness Function Weights ─────────────────────────────────
W1 = 0.60   # R² accuracy        (increased — main objective)
W2 = 0.25   # ACI improvement
W3 = 0.15   # Physics penalty    (reduced — less restrictive)

# ── PySR Symbolic Regression ─────────────────────────────────
PYSR_NITERATIONS   = 200
PYSR_MAXSIZE       = 25
PYSR_POPULATIONS   = 40
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
REPORT_TITLE       = "Corrosion RC Beam Optimizer \u2014 Scientific Report"
REPORT_FILE        = RESULTS_DIR / "Final_Report.pdf"
REPORT_AUTHOR      = "PhD Research \u2014 Corrosion RC Beam Optimization"
LOG_FILE           = LOG_DIR / "run_log.txt"

# ── Streamlit App ────────────────────────────────────────────
APP_TITLE          = "Corrosion RC Beam Optimizer"
APP_ICON           = "\U0001f3d7\ufe0f"
APP_LAYOUT         = "wide"

# ── Saved Model Paths ────────────────────────────────────────
MODEL_MLP_PKL      = MODELS_DIR / "best_mlp.pkl"
MODEL_MLP_PT       = MODELS_DIR / "best_mlp.pt"
MODEL_GA_PKL       = MODELS_DIR / "best_ga_model.pkl"
SCALER_X_PATH      = MODELS_DIR / "scaler_X.pkl"
SCALER_Y_PATH      = MODELS_DIR / "scaler_y.pkl"
HALL_OF_FAME_PATH  = MODELS_DIR / "hall_of_fame.json"
