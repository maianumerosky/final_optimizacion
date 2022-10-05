import pickle
from util import *
from time import time
from pypoman import compute_polytope_vertices


def hill_climbing(M, max_iter, A, std_dev=0.01):
    N = A.shape[0]
    rng = np.random.default_rng()
    b = rng.uniform(low=-1, high=1, size=N)
    objective_function_for_b = obj_function(M, Polygon(compute_polytope_vertices(A, -b)))
    iters = 0
    while iters <= max_iter:
        candidate = random_around(b, std_dev)
        try:
            objective_function_for_candidate = obj_function(M, Polygon(compute_polytope_vertices(A, -candidate)))
            if objective_function_for_candidate <= objective_function_for_b:            
                b, objective_function_for_b = candidate, objective_function_for_candidate
        except Exception:
            break
        iters += 1
    return [b, objective_function_for_b]


def iterated_local_search(M, n_restarts, max_iter_hill_climb, b_0, std_dev_hill_climb, N):
    thetas = [2*np.pi*m/N for m in range(N)]
    A = np.array([[np.cos(theta), np.sin(theta)] for theta in thetas])
    best_b = b_0
    objective_function_for_best_b = obj_function(M, Polygon(compute_polytope_vertices(A, -best_b)))
    evolution = []
    for n in range(n_restarts):
        b, objective_function_for_b = hill_climbing(M, max_iter_hill_climb, A, std_dev_hill_climb)
        if objective_function_for_b < objective_function_for_best_b:
            best_b, objective_function_for_best_b = b, objective_function_for_b
            evolution.append({'b': best_b, 'objective_function': objective_function_for_best_b})
    return best_b, evolution


if __name__ == "__main__":
    tries = 10
    results = []
    for i in range(tries):
        print(f"try {i}")
        potato = UnknownObject()
        M = potato.noisy_measurements(k=100)
        N = 10
        params = [M, 300, 200, b_0, 0.1, N]
        rng = np.random.default_rng()
        b_0 = rng.uniform(low=-1, high=1, size=N)
        experiment = Experiment(iterated_local_search, params, potato)
        beginning = time()
        poly, evolution = experiment.run()
        end = time()
        duration = end - beginning
        results.append(Result(experiment, duration, evolution, poly))

    with open('data.pickle', 'wb') as f:
        pickle.dump(results, open(f"results_{params[1]}_{params[2]}_1.pkl", 'wb'))
