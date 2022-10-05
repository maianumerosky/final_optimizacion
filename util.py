from array import array
from matplotlib import pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon, Point, box, LineString
from shapely.affinity import rotate, scale
import warnings
from shapely.ops import split
from typing import Optional, Tuple, List
import time


# Fixed vectors

u = np.array([[1, 0], [2/np.sqrt(5), 1/np.sqrt(5)], [0, 1], [-1/np.sqrt(5), 2/np.sqrt(5)]])
v = np.array([[0, 1], [-1/np.sqrt(5), 2/np.sqrt(5)], [-1, 0], [-2/np.sqrt(5), -1/np.sqrt(5)]])


def points_in_rays(s: np.ndarray, t: np.ndarray) -> np.ndarray:
    '''
    Given s, returns a bunch of points in the form s*u+tj*v.
    '''
    k = s.shape[1]
    return np.array([[s[i,j,0]*u[i]+t[j]*v[i], s[i, j, 1]*u[i]+t[j]*v[i]] for j in range(k) for i in range(4)])
    
def convex_hull(q: np.ndarray, remove_repeated: bool=True) -> Polygon:
    if remove_repeated:
        q = remove_duplicates_from(q)
    hull = ConvexHull(q)
    points = hull.points
    vertices = [Point(points[i]) for i in hull.vertices]
    return Polygon(vertices)

def remove_duplicates_from(q: np.ndarray):
    non_repeated = []
    for pairs_of_points in q:
        if np.linalg.norm(pairs_of_points[0]-pairs_of_points[1]) >= np.exp(-5):
            non_repeated.append(pairs_of_points[0])
            non_repeated.append(pairs_of_points[0])
    return non_repeated

def line_parallel_to_ui_on_tj(i: int, t_j: float) -> LineString:
    point_1 = Point(t_j*v[i])
    point_2 = Point(u[i]+t_j*v[i])
    return LineString((point_1, point_2))

def extend(line: LineString) -> LineString:
    minx = miny = -2
    maxx = maxy = 2
    bounding_box = box(minx, miny, maxx, maxy)
    a, b = line.boundary.geoms
    if a.x == b.x:  # vertical line
        extended_line = LineString([(a.x, miny), (a.x, maxy)])
    elif a.y == b.y:  # horizonthal line
        extended_line = LineString([(minx, a.y), (maxx, a.y)])
    else:
        # linear equation: y = k*x + m
        k = (b.y - a.y) / (b.x - a.x)
        m = a.y - k * a.x
        y0 = k * minx + m
        y1 = k * maxx + m
        x0 = (miny - m) / k
        x1 = (maxy - m) / k
        points_on_boundary_lines = [Point(minx, y0), Point(maxx, y1), 
                                    Point(x0, miny), Point(x1, maxy)]
        points_sorted_by_distance = sorted(points_on_boundary_lines, key=bounding_box.distance)
        extended_line = LineString(points_sorted_by_distance[:2])
    return extended_line

def x_ray_for(i: int, figure: Polygon, t_j: float) -> float:
    line = line_parallel_to_ui_on_tj(i, t_j)
    extended_line = extend(line)
    intersection = extended_line.intersection(figure).boundary.geoms
    if len(intersection) == 0:
        return 0
    else:
        return intersection[0].distance(intersection[-1])

def obj_function(M: np.ndarray, polygon: Polygon) -> float:
    k = M.shape[1]
    t = np.linspace(-1, 1, k)
    sum_of_squares = 0
    for i in range(4):
        sum_of_squares_for_u_i = 0
        for j in range(k):
            sum_of_squares_for_u_i += (M[i, j] - x_ray_for(i, polygon, t[j]))**2
        sum_of_squares += sum_of_squares_for_u_i
    return sum_of_squares

def random_s(k: int) -> np.ndarray:
    rng = np.random.default_rng()
    return rng.uniform(low=-1, high=1, size=(4, k, 2))

def random_around(s: np.ndarray, std_dev: float=0.01, limit: Optional[List[float]]=None) -> np.ndarray:
    rng = np.random.default_rng()
    neighbor = rng.normal(s, std_dev)
    if limit:
        while (neighbor.flatten()>limit[-1]).any() or (neighbor.flatten()<limit[0]).any():
            neighbor = rng.normal(s, std_dev)
    return neighbor

def random_uniform(size: int=1) -> float:
    rng = np.random.default_rng()
    return rng.uniform(size=size)

def cut(polygon: Polygon, angles: Tuple[float, float]) -> Polygon:
    theta_1, theta_2 = angles
    point_1 = Point(np.cos(theta_1), np.sin(theta_1))
    point_2 = Point(np.cos(theta_2), np.sin(theta_2))
    line = LineString((point_1, point_2))
    splitted = split(polygon, line).geoms
    areas = np.array([poly.area for poly in splitted])
    return splitted[np.argmax(areas)]
    
def random_polygon(vertices: int=20):
    rng = np.random.default_rng()
    angles = rng.uniform(low=0, high=2*np.pi, size=20)
    q = [(np.cos(angle), np.sin(angle)) for angle in angles]
    return convex_hull(q, remove_repeated=False)

class UnknownObject:
    def __init__(self, std_dev=0.01):
        self._a = np.random.uniform(0, 0.99)
        self._b = np.random.uniform(0, 0.99)
        self._angle = np.random.uniform(0, 90)
        self._std_dev = std_dev

    def object(self):
        aux_circle = Point((0,0)).buffer(1)
        ellipse  = scale(aux_circle, self._a, self._b)
        ellipse_rotated = rotate(ellipse, self._angle)
        if any(abs(bound)>1 for bound in ellipse_rotated.bounds):
            warnings.warn("Object does not fit in S1")
        return ellipse_rotated
        
    def noisy_measurements(self, k: int) -> np.ndarray:
        noise = np.random.normal(0, self._std_dev**2)
        t = np.linspace(-1, 1, k)
        return np.array([[x_ray_for(i, self.object(), t[j]) + noise for j in range(k)] for i in range(4)])

class Experiment:
    def __init__(self, optimizer, params, ellipse) -> None:
        self.optimizer = optimizer
        self.params = params
        self.ellipse = ellipse

    def run(self):
        return self.optimizer(*self.params)

class Result:
    def __init__(self, experiment, duration, evolution, poly) -> None:
        self.experiment = experiment
        self.duration = duration
        self.evolution = [value['objective_function'] for value in evolution]
        self.polygons = [value['polygon'] for value in evolution]
        self.poly = poly
    
    def summary(self):
        time_in_min_sec = time.strftime("%H:%M:%S", time.gmtime(self.duration))
        print(f"Tiempo: {time_in_min_sec}")
        print("Objeto a aproximar:")
        plt.plot(*self.experiment.ellipse.object().exterior.xy)
        plt.show()
        print("Evolución de la función objetivo:")
        plt.plot(self.evolution)
        plt.show()
        print("Evolución:")
        for polygon in self.polygons:
            plt.plot(*polygon.exterior.xy)
            plt.show()