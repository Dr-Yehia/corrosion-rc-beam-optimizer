# ============================================================
# src/genetic_algorithm.py
# Corrosion RC Beam Optimizer
# NSGA-III Multi-Objective Genetic Algorithm
# Objectives : maximise R², minimise RMSE, maximise CV-R²
# Fitness    : W1·R² + W2·(Mpred/MACI→1) − W3·penalty
# Strategy   : Elitism (top-10) + Crossover + Mutation
# Stopping   : Benchmark broken (L1+L2) OR convergence → restart
#              Max restarts = GA_MAX_RUNS
# ============================================================

import numpy as np
import json
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from loguru import logger
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.neural_network import MLPRegressor
import sys

sys.path.append(str(Path(__file__).resolve().parent))
from config import (
    GA_POPULATION_SIZE, GA_MAX_GENERATIONS, GA_CONSISTENCY_WINDOW,
    GA_ELITE_SIZE, GA_CROSSOVER_RATE, GA_MUTATION_RATE, GA_MAX_RUNS,
    GA_N_OBJECTIVES, GA_N_PARTITIONS,
    W1, W2, W3,
    L1_TARGET_R2, L2_TARGET_R2, BREAK_BOTH,
    RANDOM_STATE, MODELS_DIR, HALL_OF_FAME_PATH, MODEL_GA_PKL,
    GENE_BOUNDS, FEATURE_COLS,
)
from neural_network import build_mlp, train_mlp, evaluate_model, predict

# Optional: pymoo NSGA-III reference points
try:
    from pymoo.util.ref_dirs import get_reference_directions
    PYMOO_AVAILABLE = True
except ImportError:
    PYMOO_AVAILABLE = False
    logger.warning("pymoo not found — using uniform reference points.")

import joblib

np.random.seed(RANDOM_STATE)


# ============================================================
# DATA STRUCTURES
# ============================================================

class Individual:
    """
    Represents one chromosome in the population.
    genes   : dict  {feature_name: value}
    fitness : float (higher = better)
    metrics : dict  {R2, RMSE, CV_R2, penalty}
    _model  : trained MLPRegressor (cached after fitness evaluation)
    """
    __slots__ = ["genes", "fitness", "metrics", "rank", "crowding", "_model"]

    def __init__(self, genes: dict):
        self.genes    = genes
        self.fitness  = -np.inf
        self.metrics  = {}
        self.rank     = 0
        self.crowding = 0.0
        self._model   = None

    def __repr__(self):
        return (f"Individual(fitness={self.fitness:.4f}, "
                f"R2={self.metrics.get('R2', 0):.4f})")


# ============================================================
# 1. INITIALISATION
# ============================================================

def _random_genes() -> dict:
    """Sample a random chromosome within GENE_BOUNDS."""
    genes = {}
    for feature, (lo, hi) in GENE_BOUNDS.items():
        genes[feature] = float(np.random.uniform(lo, hi))
    return genes


def initialise_population(size: int = GA_POPULATION_SIZE) -> list:
    """
    Create a fresh random population of `size` individuals.
    Called at the start of every new Run.
    """
    population = [Individual(_random_genes()) for _ in range(size)]
    logger.info(f"Population initialised — {size} random individuals.")
    return population


# ============================================================
# 2. FITNESS FUNCTION
# ============================================================

def fitness_function(
    individual: Individual,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test:  np.ndarray,
    y_test:  np.ndarray,
    scaler_y,
    aci_rmse: float,
    aci_mae:  float,
) -> float:
    """
    Evaluate one individual’s fitness.

    FF = W1 · R²_test
       + W2 · clamp(Mpred/MACI_ratio, 0, 2) / 2     ← ACI improvement
       − W3 · physics_penalty                         ← constraint violation

    Physics penalties:
        +0.20  if R² < 0  (inverted predictions)
        +0.10  if RMSE > 2 × ACI_RMSE
        +0.05  if any predicted R(%) outside [0, 130]
    """
    model = build_mlp()
    try:
        model = train_mlp(model, X_train, y_train)
    except Exception as e:
        logger.warning(f"Training failed for individual: {e}")
        individual.fitness = -1.0
        individual.metrics = {"R2": 0.0, "RMSE": 999.0, "CV_R2": 0.0, "penalty": 1.0}
        individual._model  = None
        return -1.0

    # Predict on test set
    y_pred_sc = model.predict(X_test)
    y_pred    = scaler_y.inverse_transform(y_pred_sc.reshape(-1, 1)).ravel()
    y_true    = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()

    r2   = r2_score(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

    # CV generalisation score (3-fold fast)
    from sklearn.model_selection import cross_val_score
    cv_model  = build_mlp()
    X_all     = np.vstack([X_train, X_test])
    y_all     = np.concatenate([y_train, y_test])
    cv_scores = cross_val_score(cv_model, X_all, y_all, cv=3, scoring="r2", n_jobs=-1)
    cv_r2     = float(np.mean(cv_scores))

    # ACI improvement term
    aci_improvement = min(rmse / max(aci_rmse, 1e-6), 2.0)
    aci_score       = 1.0 - aci_improvement / 2.0

    # Physics penalty
    penalty = 0.0
    if r2 < 0:
        penalty += 0.20
    if rmse > 2.0 * aci_rmse:
        penalty += 0.10
    if np.any(y_pred < 0) or np.any(y_pred > 135):
        penalty += 0.05

    # Composite fitness
    fitness = W1 * max(r2, 0.0) + W2 * aci_score - W3 * penalty

    individual.fitness = fitness
    individual.metrics = {
        "R2"      : round(r2,    4),
        "RMSE"    : round(rmse,  4),
        "CV_R2"   : round(cv_r2, 4),
        "penalty" : round(penalty, 4),
        "fitness" : round(fitness,  4),
    }
    individual._model = model
    return fitness


# ============================================================
# 3. NSGA-III NON-DOMINATED SORTING
# ============================================================

def _dominates(a: Individual, b: Individual) -> bool:
    r2_a,  rmse_a, cv_a = (a.metrics.get("R2",   0),
                            a.metrics.get("RMSE", 999),
                            a.metrics.get("CV_R2",0))
    r2_b,  rmse_b, cv_b = (b.metrics.get("R2",   0),
                            b.metrics.get("RMSE", 999),
                            b.metrics.get("CV_R2",0))
    better_or_equal = (r2_a >= r2_b) and (rmse_a <= rmse_b) and (cv_a >= cv_b)
    strictly_better = (r2_a >  r2_b) or  (rmse_a <  rmse_b) or  (cv_a >  cv_b)
    return better_or_equal and strictly_better


def non_dominated_sort(population: list) -> list:
    n      = len(population)
    S      = [[] for _ in range(n)]
    n_dom  = [0] * n
    fronts = [[]]

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if _dominates(population[i], population[j]):
                S[i].append(j)
            elif _dominates(population[j], population[i]):
                n_dom[i] += 1
        if n_dom[i] == 0:
            population[i].rank = 1
            fronts[0].append(i)

    k = 0
    while fronts[k]:
        next_front = []
        for i in fronts[k]:
            for j in S[i]:
                n_dom[j] -= 1
                if n_dom[j] == 0:
                    population[j].rank = k + 2
                    next_front.append(j)
        k += 1
        fronts.append(next_front)

    return [f for f in fronts if f]


def crowding_distance(population: list, front: list) -> None:
    l = len(front)
    if l == 0:
        return
    for i in front:
        population[i].crowding = 0.0

    for obj_fn in [
        lambda ind: ind.metrics.get("R2",   0),
        lambda ind: -ind.metrics.get("RMSE", 0),
        lambda ind: ind.metrics.get("CV_R2", 0),
    ]:
        vals  = [(obj_fn(population[i]), i) for i in front]
        vals.sort(key=lambda x: x[0])
        f_min, f_max = vals[0][0], vals[-1][0]
        if f_max == f_min:
            continue
        population[vals[0][1]].crowding  = np.inf
        population[vals[-1][1]].crowding = np.inf
        for k in range(1, l - 1):
            population[vals[k][1]].crowding += (
                (vals[k+1][0] - vals[k-1][0]) / (f_max - f_min)
            )


def nsga3_selection(population: list, n_select: int) -> list:
    fronts   = non_dominated_sort(population)
    selected = []
    for front in fronts:
        crowding_distance(population, front)
        if len(selected) + len(front) <= n_select:
            selected.extend(front)
        else:
            remaining = n_select - len(selected)
            front.sort(key=lambda i: (-population[i].rank, -population[i].crowding))
            selected.extend(front[:remaining])
            break
    return [population[i] for i in selected]


# ============================================================
# 4. GENETIC OPERATORS
# ============================================================

def _blend_crossover(parent_a: Individual, parent_b: Individual,
                     alpha: float = 0.5) -> Individual:
    child_genes = {}
    for key in GENE_BOUNDS:
        a_val = parent_a.genes.get(key, 0.0)
        b_val = parent_b.genes.get(key, 0.0)
        lo_b, hi_b = GENE_BOUNDS[key]
        d    = abs(a_val - b_val)
        lo_c = max(lo_b, min(a_val, b_val) - alpha * d)
        hi_c = min(hi_b, max(a_val, b_val) + alpha * d)
        child_genes[key] = float(np.random.uniform(lo_c, hi_c))
    return Individual(child_genes)


def _gaussian_mutation(individual: Individual,
                       mutation_rate: float = GA_MUTATION_RATE,
                       sigma: float = 0.05) -> Individual:
    mutant = deepcopy(individual)
    for key, (lo, hi) in GENE_BOUNDS.items():
        if np.random.rand() < mutation_rate:
            noise = np.random.normal(0, sigma * (hi - lo))
            mutant.genes[key] = float(np.clip(mutant.genes[key] + noise, lo, hi))
    return mutant


def produce_offspring(
    parents: list,
    n_offspring: int,
    crossover_rate: float = GA_CROSSOVER_RATE,
    mutation_rate:  float = GA_MUTATION_RATE,
) -> list:
    offspring = []
    while len(offspring) < n_offspring:
        a, b = np.random.choice(len(parents), size=2, replace=False)
        if np.random.rand() < crossover_rate:
            child = _blend_crossover(parents[a], parents[b])
        else:
            child = deepcopy(parents[a])
        child = _gaussian_mutation(child, mutation_rate)
        offspring.append(child)
    return offspring


# ============================================================
# 5. CONVERGENCE DETECTOR
# ============================================================

def _is_converged(history: list, window: int = GA_CONSISTENCY_WINDOW) -> bool:
    if len(history) < window:
        return False
    recent = history[-window:]
    return (max(recent) - min(recent)) < 1e-5


# ============================================================
# 6. HALL OF FAME
# ============================================================

def _update_hall_of_fame(hof: list, best: Individual,
                         run_id: int, generation: int) -> list:
    entry = {
        "run"        : run_id,
        "generation" : generation,
        "fitness"    : best.fitness,
        "metrics"    : best.metrics,
        "genes"      : best.genes,
        "timestamp"  : str(datetime.now()),
    }
    hof.append(entry)
    hof.sort(key=lambda x: x["fitness"], reverse=True)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(HALL_OF_FAME_PATH, "w") as f:
        json.dump(hof, f, indent=2)
    logger.info(f"Hall of Fame updated — best overall fitness: "
                f"{hof[0]['fitness']:.4f} "
                f"(R²={hof[0]['metrics'].get('R2', 0):.4f})")
    return hof


# ============================================================
# 7. LIVE LOG PANEL
# ============================================================

def _log_generation(run_id, gen, best, l1_broken, l2_broken, elapsed) -> str:
    l1_sym = "\u2713" if l1_broken else "\u2717"
    l2_sym = "\u2713" if l2_broken else "\u2717"
    r2     = best.metrics.get("R2",    0.0)
    rmse   = best.metrics.get("RMSE",  0.0)
    cv     = best.metrics.get("CV_R2", 0.0)
    line   = (
        f"[Run {run_id:2d} | Gen {gen:4d}]  "
        f"Best R²={r2:.4f}  RMSE={rmse:.4f}  CV-R²={cv:.4f}  "
        f"Fitness={best.fitness:.4f}  "
        f"L1:{l1_sym}  L2:{l2_sym}  "
        f"({elapsed:.1f}s)"
    )
    logger.info(line)
    return line


# ============================================================
# 8. MAIN GA LOOP
# ============================================================

def run_nsga3(
    X_train   : np.ndarray,
    y_train   : np.ndarray,
    X_test    : np.ndarray,
    y_test    : np.ndarray,
    scaler_y,
    aci_rmse  : float,
    aci_mae   : float,
    log_lines : list = None,
) -> dict:
    if log_lines is None:
        log_lines = []

    hall_of_fame   = []
    global_best    = None
    success        = False
    t_start_global = time.time()

    logger.info("=" * 60)
    logger.info(" NSGA-III Optimisation — Starting")
    logger.info(f" Max runs        : {GA_MAX_RUNS}")
    logger.info(f" Max generations : {GA_MAX_GENERATIONS}")
    logger.info(f" Population size : {GA_POPULATION_SIZE}")
    logger.info(f" Elite size      : {GA_ELITE_SIZE}")
    logger.info(f" L1 target R²   : {L1_TARGET_R2}")
    logger.info(f" L2 target R²   : {L2_TARGET_R2}")
    logger.info("=" * 60)

    for run_id in range(1, GA_MAX_RUNS + 1):
        logger.info(f"\n{'\u2500'*50}")
        logger.info(f" Run {run_id}/{GA_MAX_RUNS} — New random population")
        logger.info(f"{'\u2500'*50}")

        population      = initialise_population(GA_POPULATION_SIZE)
        fitness_history = []
        run_best        = None
        t_run_start     = time.time()

        for gen in range(1, GA_MAX_GENERATIONS + 1):

            # Evaluate fitness
            for ind in population:
                if ind.fitness == -np.inf:
                    fitness_function(
                        ind, X_train, y_train, X_test, y_test,
                        scaler_y, aci_rmse, aci_mae
                    )

            # Best this generation
            population.sort(key=lambda x: x.fitness, reverse=True)
            best_this_gen = population[0]

            if run_best is None or best_this_gen.fitness > run_best.fitness:
                run_best = deepcopy(best_this_gen)

            fitness_history.append(best_this_gen.fitness)

            # Check benchmark
            r2_val    = best_this_gen.metrics.get("R2", 0.0)
            l1_broken = r2_val >= L1_TARGET_R2
            l2_broken = r2_val >= L2_TARGET_R2

            # Live log
            elapsed = time.time() - t_run_start
            if gen == 1 or gen % 10 == 0 or l1_broken or l2_broken:
                line = _log_generation(run_id, gen, best_this_gen,
                                       l1_broken, l2_broken, elapsed)
                log_lines.append(line)

            # STOP: both benchmarks broken
            if BREAK_BOTH and l1_broken and l2_broken:
                logger.success(
                    f"\n{'*'*60}\n BENCHMARK BROKEN \u2713\u2713\n"
                    f" Run={run_id} | Gen={gen} | R²={r2_val:.4f}\n{'*'*60}"
                )
                log_lines.append(
                    f"\n*** BENCHMARK BROKEN *** Run={run_id} Gen={gen} "
                    f"R²={r2_val:.4f} L1:\u2713 L2:\u2713\n"
                )
                hall_of_fame = _update_hall_of_fame(
                    hall_of_fame, run_best, run_id, gen
                )
                global_best = deepcopy(run_best)

                if run_best._model is not None:
                    joblib.dump(run_best._model, MODEL_GA_PKL)
                    logger.info(f"Best GA model saved → {MODEL_GA_PKL}")

                return {
                    "best_individual": global_best,
                    "hall_of_fame"   : hall_of_fame,
                    "log_lines"      : log_lines,
                    "success"        : True,
                    "best_run"       : run_id,
                    "best_gen"       : gen,
                    "total_runs"     : run_id,
                }

            # NSGA-III selection
            n_survive = GA_POPULATION_SIZE // 2
            survivors = nsga3_selection(population, n_survive)

            # Elitism
            elites = population[:GA_ELITE_SIZE]

            # Crossover + Mutation
            n_offspring = GA_POPULATION_SIZE - GA_ELITE_SIZE
            offspring   = produce_offspring(
                survivors, n_offspring,
                crossover_rate = GA_CROSSOVER_RATE,
                mutation_rate  = GA_MUTATION_RATE,
            )
            for child in offspring:
                child.fitness = -np.inf

            population = elites + offspring

            # Convergence check
            if _is_converged(fitness_history, GA_CONSISTENCY_WINDOW):
                logger.info(
                    f"[Run {run_id} | Gen {gen}] Convergence detected "
                    f"({GA_CONSISTENCY_WINDOW} gens no improvement). "
                    f"Best R²={r2_val:.4f}. Starting new run ..."
                )
                log_lines.append(
                    f"[Run {run_id} | Gen {gen}] Converged — "
                    f"R²={r2_val:.4f} — Starting Run {run_id+1}"
                )
                break

        # End of run
        if run_best is not None:
            hall_of_fame = _update_hall_of_fame(
                hall_of_fame, run_best, run_id, len(fitness_history)
            )
            if global_best is None or run_best.fitness > global_best.fitness:
                global_best = deepcopy(run_best)

    # All runs exhausted
    best_r2 = global_best.metrics.get("R2", 0.0) if global_best else 0.0
    msg = (
        f"\n{'!'*60}\n"
        f" MAX RUNS REACHED ({GA_MAX_RUNS}) — Target not achieved.\n"
        f" Best R² found: {best_r2:.4f}\n"
        f"{'!'*60}"
    )
    logger.warning(msg)
    log_lines.append(msg)

    if global_best and global_best._model is not None:
        joblib.dump(global_best._model, MODEL_GA_PKL)
        logger.info(f"Best model (partial) saved → {MODEL_GA_PKL}")

    total_time = time.time() - t_start_global
    logger.info(f"Total optimisation time: {total_time:.1f}s")

    return {
        "best_individual": global_best,
        "hall_of_fame"   : hall_of_fame,
        "log_lines"      : log_lines,
        "success"        : False,
        "best_run"       : hall_of_fame[0]["run"]        if hall_of_fame else None,
        "best_gen"       : hall_of_fame[0]["generation"] if hall_of_fame else None,
        "total_runs"     : GA_MAX_RUNS,
    }


# ============================================================
# 9. CLI ENTRY POINT
# ============================================================

if __name__ == "__main__":
    from data_preprocessing import run_preprocessing, load_raw_data as _load, clean_data as _clean
    from aci_calculator import compute_aci_predictions, evaluate_aci_benchmark

    data       = run_preprocessing(save_clean=True)
    df_raw     = _load()
    df_clean   = _clean(df_raw)
    df_aci     = compute_aci_predictions(df_clean)
    aci_metrics = evaluate_aci_benchmark(df_aci)

    log_lines = []
    results   = run_nsga3(
        data["X_train"], data["y_train"],
        data["X_test"],  data["y_test"],
        data["scaler_y"],
        aci_metrics["RMSE"], aci_metrics["MAE"],
        log_lines = log_lines,
    )

    print(f"\nSuccess : {results['success']}")
    if results['best_individual']:
        print(f"Best R² : {results['best_individual'].metrics.get('R2', 0):.4f}")
    print(f"Total runs : {results['total_runs']}")
