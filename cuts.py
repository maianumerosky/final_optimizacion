import pickle
from util import *
from time import time

def hill_climbing(M, max_iter, poly_0, std_dev=0.01):
    poly = poly_0
    objective_function_for_poly = obj_function(M, poly)
    iters = 0
    rng = np.random.default_rng()
    angles = rng.uniform(low=0, high=2*np.pi, size=2)
    while iters <= max_iter:
        angles = random_around(angles, std_dev)
        candidate = cut(poly, angles)
        objective_function_for_candidate = obj_function(M, candidate)
        if objective_function_for_candidate <= objective_function_for_poly:
            poly, objective_function_for_poly = candidate, objective_function_for_candidate
        iters += 1
    return [poly, objective_function_for_poly]


def iterated_local_search(M, n_restarts, max_iter_hill_climb, s_0, std_dev_hill_climb):
    best_s = s_0
    objective_function_for_best_s = obj_function(M, best_s)
    evolution = []
    for n in range(n_restarts):
        solution, objective_function_for_s = hill_climbing(M, max_iter_hill_climb, s_0, std_dev_hill_climb)
        if objective_function_for_s < objective_function_for_best_s:
            best_s, objective_function_for_best_s = solution, objective_function_for_s
            evolution.append({'polygon': best_s, 'objective_function': objective_function_for_best_s})
    return best_s, evolution


if __name__ == "__main__":
    tries = 10
    results = []
    for i in range(tries):
        print(f"try {i}")
        potato = UnknownObject()
        M = potato.noisy_measurements(k=100)
        poly_0 = random_polygon()
        params = [M, 300, 200, poly_0, 0.1]
        experiment = Experiment(iterated_local_search, params, potato)
        beginning = time()
        poly, evolution = experiment.run()
        end = time()
        duration = end - beginning
        results.append(Result(experiment, duration, evolution, poly))

    with open('data.pickle', 'wb') as f:
        pickle.dump(results, open(f"results_{params[1]}_{params[2]}_1.pkl", 'wb'))
