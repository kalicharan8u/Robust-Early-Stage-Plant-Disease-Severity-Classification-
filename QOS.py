import numpy as np
import time


def QSO(population, objective_function, lb, ub, max_iter):
    pop_size, dim = population.shape

    # Parameter Settings
    T_range = [0.2, 0.44]   # Temperature
    H_range = [0.3, 0.65]  # Humidity
    N_range = [0.0, 1.0]  # Nitrogen

    # Initialize parameters
    best_solution = population[0, :].copy()
    best_fitness = objective_function(best_solution)
    T = T_range[0] + (T_range[1] - T_range[0]) * np.random.rand()
    H = H_range[0] + (H_range[1] - H_range[0]) * np.random.rand()
    N = N_range[0] + (N_range[1] - N_range[0]) * np.random.rand()
    convergence = np.zeros(max_iter)

    start_time = time.time()
    for iteration in range(max_iter):
        fitness_values = np.array([objective_function(ind) for ind in population])

        min_fitness_idx = np.argmin(fitness_values)
        min_fitness = fitness_values[min_fitness_idx]

        if min_fitness < best_fitness:
            best_fitness = min_fitness
            best_solution = population[min_fitness_idx, :].copy()

        new_population = np.copy(population)

        for i in range(pop_size):
            r = np.random.rand()
            D = r * (population[i, :] - best_solution)
            new_population[i, :] = (population[i, :]
                                    - T * D
                                    + H * (best_solution - population[i, :])
                                    + N * (np.random.rand(dim) - 0.5))

            # Apply Boundary Constraints
            new_population[i, :] = np.clip(new_population[i, :], lb[i, :], ub[i, :])

        population = new_population

        fitness_values = np.array([objective_function(ind) for ind in population])
        min_fitness_idx = np.argmin(fitness_values)
        best_fitness = fitness_values[min_fitness_idx]
        best_solution = population[min_fitness_idx, :].copy()

        convergence[iteration] = best_fitness

    computation_time = time.time() - start_time
    return best_fitness, convergence, best_solution, computation_time
