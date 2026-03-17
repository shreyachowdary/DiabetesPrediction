"""
Genetic Algorithm-based feature selection for diabetes prediction.

Uses DEAP to evolve a population of feature subsets. The fitness function
balances model accuracy with the number of features (parsimony).
"""

import logging
from typing import List, Tuple, Callable, Any

import numpy as np
from deap import base, creator, tools, algorithms
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

logger = logging.getLogger(__name__)

# GA hyperparameters
DEFAULT_POPULATION_SIZE = 30
DEFAULT_GENERATIONS = 15
DEFAULT_CX_PROB = 0.5
DEFAULT_MUT_PROB = 0.2
DEFAULT_CV_FOLDS = 5


def create_fitness_and_individual():
    """
    Create DEAP fitness (maximize) and Individual (binary list) types.
    Must be called before using the GA.
    """
    creator.create("FitnessMax", base.Fitness, weights=(1.0, -0.01))
    creator.create("Individual", list, fitness=creator.FitnessMax)


def evaluate_individual(
    individual: list,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
) -> Tuple[float, int]:
    """
    Fitness function: maximize cross-validated accuracy, penalize feature count.
    
    Returns:
        (fitness_score, num_features)
        fitness = accuracy - 0.01 * num_features to encourage smaller subsets
    """
    # Get selected feature indices
    indices = [i for i, bit in enumerate(individual) if bit == 1]
    
    if len(indices) == 0:
        return (0.0, 0)
    
    X_subset = X[:, indices]
    clf = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=8)
    
    try:
        scores = cross_val_score(clf, X_subset, y, cv=DEFAULT_CV_FOLDS, scoring="accuracy")
        mean_accuracy = float(np.mean(scores))
        # Slight penalty for using more features (encourages parsimony)
        penalty = 0.01 * len(indices)
        fitness = mean_accuracy - penalty
        return (fitness, len(indices))
    except Exception as e:
        logger.warning(f"Evaluation failed for individual: {e}")
        return (0.0, len(indices))


def run_ga_feature_selection(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: List[str],
    population_size: int = DEFAULT_POPULATION_SIZE,
    generations: int = DEFAULT_GENERATIONS,
    cx_prob: float = DEFAULT_CX_PROB,
    mut_prob: float = DEFAULT_MUT_PROB,
) -> Tuple[List[str], List[int], Any]:
    """
    Run the genetic algorithm for feature selection.
    
    Returns:
        selected_feature_names, selected_indices, best_individual
    """
    n_features = X_train.shape[1]
    
    # Initialize DEAP types (only once)
    try:
        creator.FitnessMax
    except AttributeError:
        create_fitness_and_individual()
    
    toolbox = base.Toolbox()
    
    # Gene: 0 or 1 (include or exclude feature)
    toolbox.register("attr_bool", np.random.randint, 0, 2)
    toolbox.register(
        "individual",
        tools.initRepeat,
        creator.Individual,
        toolbox.attr_bool,
        n_features,
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    # Ensure at least one feature is selected in each individual
    def init_individual():
        ind = toolbox.individual()
        if sum(ind) == 0:
            ind[np.random.randint(n_features)] = 1
        return ind
    
    toolbox.register("individual_ensured", init_individual)
    toolbox.register(
        "population",
        tools.initRepeat,
        list,
        toolbox.individual_ensured,
    )
    
    def evaluate(ind):
        return evaluate_individual(ind, X_train, y_train, feature_names)
    
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    population = toolbox.population(n=population_size)
    
    # Statistics
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("max", np.max)
    
    logger.info(f"Starting GA: {population_size} individuals, {generations} generations")
    
    population, logbook = algorithms.eaSimple(
        population,
        toolbox,
        cxpb=cx_prob,
        mutpb=mut_prob,
        ngen=generations,
        stats=stats,
        verbose=False,
    )
    
    # Get best individual
    best_individual = tools.selBest(population, k=1)[0]
    selected_indices = [i for i, bit in enumerate(best_individual) if bit == 1]
    selected_names = [feature_names[i] for i in selected_indices]
    
    logger.info(f"GA complete. Selected {len(selected_names)} features: {selected_names}")
    
    return selected_names, selected_indices, best_individual
