import pickle
from util import *
from time import time

def hill_climbing(M, max_iter, s_0, std_dev=0.01):
    k = M.shape[1]
    t = np.linspace(-1, 1, k)
    s = s_0
    objective_function_for_s = obj_function(M, convex_hull(points_in_rays(s, t)))
    iters = 0
    while iters <= max_iter:
        candidate = random_around(s, std_dev)
        objective_function_for_candidate = obj_function(M, convex_hull(points_in_rays(candidate, t)))
        if objective_function_for_candidate <= objective_function_for_s:
            s, objective_function_for_s = candidate, objective_function_for_candidate
        iters += 1
    return [s, objective_function_for_s]


def iterated_local_search(M, n_restarts, max_iter_hill_climb, s_0, std_dev, std_dev_hill_climb):
    k = M.shape[1]
    t = np.linspace(-1, 1, k)
    best_s = s_0
    objective_function_for_best_s = obj_function(M, convex_hull(points_in_rays(best_s, t)))
    evolution = []
    for n in range(n_restarts):
        s_0 = random_around(best_s, std_dev)
        solution, objective_function_for_s = hill_climbing(M, max_iter_hill_climb, s_0, std_dev_hill_climb)
        if objective_function_for_s < objective_function_for_best_s:
            best_s, objective_function_for_best_s = solution, objective_function_for_s
            evolution.append({'objective_function': objective_function_for_best_s, 'polygon': convex_hull(points_in_rays(best_s, t))})
    return convex_hull(points_in_rays(best_s, t)), evolution


if __name__ == "__main__":
    tries = 10
    results = []
    for i in range(tries):
        print(f"try {i}")
        potato = UnknownObject()
        M = potato.noisy_measurements(k=100)
        k = M.shape[1]
        s_0 = random_s(k)
        params = [M, 200, 50, s_0, 0.1, 0.05]
        experiment = Experiment(iterated_local_search, params, potato)
        beginning = time()
        poly, evolution = experiment.run()
        end = time()
        duration = end - beginning
        results.append(Result(experiment, duration, evolution, poly))

    with open('data.pickle', 'wb') as f:
        pickle.dump(results, open(f"ils_{params[1]}_{params[2]}.pkl", 'wb'))
