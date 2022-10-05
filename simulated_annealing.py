import pickle
from util import *
from time import time

def simulated_annealing(M, s_0, f_decrease_temp, max_iters=1000, T_0=25000):
    '''
    M is a 4*k matrix. Each row represents parallel x-ray measurments along a 
    vector u[i] on the grid of k equally spaced points from -1 to 1.
    '''
    evolution = []
    k = M.shape[1]
    t = np.linspace(-1, 1, k)
    T = T_0
    best = s_0
    obj_function_for_best = obj_function(M, convex_hull(points_in_rays(best, t)))
    current, obj_function_for_current = best, obj_function_for_best
    iters = 0
    while (iters < max_iters) and (T > 0):
        candidate = random_around(current)
        obj_function_for_candidate = obj_function(M, convex_hull(points_in_rays(candidate, t)))
        if obj_function_for_candidate < obj_function_for_best:
            best, obj_function_for_best = candidate, obj_function_for_candidate
            evolution.append({'objective_function': obj_function_for_best, 'polygon': convex_hull(points_in_rays(best, t))})
        diff = obj_function_for_candidate - obj_function_for_current
        T = f_decrease_temp(T, iter)
        if diff < 0 or random_uniform() < np.exp(-diff / T):
            current, obj_function_for_current = candidate, obj_function_for_candidate
        iters += 1
    return convex_hull(points_in_rays(best, t)), evolution

# 800 pasos y después bajar la temperatura en un 0.99
def scalar_annealing(T, iter):
    return 0.8*T

def fast_annealing(T, iter):
    return T/(iter+1)


if __name__ == "__main__":
    tries = 10
    results = []
    for i in range(tries):
        print(f"try {i}")
        potato = UnknownObject()
        M = potato.noisy_measurements(k=100)
        k = M.shape[1]
        s_0 = random_s(k)
        params = [M, s_0, scalar_annealing, 10000, 25000]
        experiment = Experiment(simulated_annealing, params, potato)
        beginning = time()
        poly, evolution = experiment.run()
        end = time()
        duration = end - beginning
        results.append(Result(experiment, duration, evolution, poly))

    with open('data.pickle', 'wb') as f:
        pickle.dump(results, open(f"sa_{params[-1]}_{params[-2]}.pkl", 'wb'))
