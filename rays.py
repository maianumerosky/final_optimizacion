from util import *
import pickle
from time import time
def points_polar(thetas, r):
    N = len(r)
    radii_one = np.array([[np.cos(theta), np.sin(theta)] for theta in thetas])
    return [r[i]*radii_one[i] for i in range(N)]

def hill_climbing(M, max_iter, r_0, std_dev=0.01):
    N = len(r_0)
    r = r_0
    thetas = [2*np.pi*m/N for m in range(N)]
    objective_function_for_r = obj_function(M, convex_hull(points_polar(thetas, r), remove_repeated=False))
    iters = 0
    while iters <= max_iter:
        candidate = random_around(r, std_dev, [0, 1])
        polygon_for_candidate = convex_hull(points_polar(thetas, candidate), remove_repeated=False)
        #plt.plot(*polygon_for_candidate.exterior.xy)
        #plt.show()
        objective_function_for_candidate = obj_function(M, polygon_for_candidate)
        if objective_function_for_candidate < objective_function_for_r:
            r, objective_function_for_r = candidate, objective_function_for_candidate
            #plt.plot(*polygon_for_candidate.exterior.xy)
            #plt.show()
        iters += 1
    return [r, objective_function_for_r]


def iterated_local_search(M, n_restarts, max_iter_hill_climb, r_0, std_dev_hill_climb):
    N = len(r_0)
    best_r = r_0
    thetas = [2*np.pi*m/N for m in range(N)]
    points = points_polar(thetas, best_r)
    objective_function_for_best_r = obj_function(M, convex_hull(points, remove_repeated=False))
    evolution = []
    for n in range(n_restarts):
        r_0 = random_uniform(N)
        r, objective_function_for_r = hill_climbing(M, max_iter_hill_climb, r_0, std_dev_hill_climb)
        if objective_function_for_r < objective_function_for_best_r:
            best_r, objective_function_for_best_r = r, objective_function_for_r
            best_polygon = convex_hull(points_polar(thetas, best_r), remove_repeated=False)
            # points = points_polar(thetas, best_r)
            # rays = [LineString([(0, 0), point]) for point in points]
            # for ray in rays:
            #     plt.plot(*ray.xy)
            # plt.plot(*best_polygon.exterior.xy)
            # plt.show()
            evolution.append({'polygon': best_polygon, 'objective_function': objective_function_for_best_r})
    return convex_hull(points_polar(thetas, best_r), remove_repeated=False), evolution

if __name__ == "__main__":
    tries = 10
    results = []
    for i in range(tries):
        print(f"try {i}")
        potato = UnknownObject()
        M = potato.noisy_measurements(k=100)
        r_0 = random_uniform(50)
        params = [M, 50, 200, r_0, 0.1]
        experiment = Experiment(iterated_local_search, params, potato)
        beginning = time()
        poly, evolution = experiment.run()
        end = time()
        duration = end - beginning
        results.append(Result(experiment, duration, evolution, poly))

    with open('data.pickle', 'wb') as f:
        pickle.dump(results, open(f"rays_{params[1]}_{params[2]}_50.pkl", 'wb'))
