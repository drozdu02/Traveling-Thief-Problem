import random

import numpy as np

from scipy.spatial.distance import pdist, squareform

from ttp_problem.common.data_loader import DataLoader

class TravelingSalesmanProblemSimAnn:
    def __init__(
        self,
        cities : np.ndarray,
        seed : int,
        T_max : float,
        T_min : float,
        alpha : float,
        iters_per_temperature : int
    ):
        self.cities = np.asarray(cities)
        self.T_max = T_max
        self.T_min = T_min
        self.alpha = alpha
        self.rng = np.random.default_rng(seed)
        self.distances = self.compute_distances()
        self.best = None
        self.current = None
        self.new_route = None
        self.t = T_max
        self.iterations = 0
        self.current_distance = None
        self.best_distance = None
        self.iters_per_temperature = iters_per_temperature


    def compute_distances(self):
        distances = squareform(pdist(self.cities[:, 1:], metric='euclidean'))
        return np.ceil(distances)

    def compute_route_distance(self, route):
        return self.distances[route, np.roll(route, -1)].sum()


    def generate_random_route(self):
        random_route = self.rng.permutation(self.cities.shape[0])
        if self.best is None:
            self.best = random_route.copy()
            self.current = random_route.copy()
            self.current_distance = self.compute_route_distance(random_route)
            self.best_distance = self.current_distance
        return random_route

    def generate_random_neighbors(self):
        i, j = sorted(self.rng.choice(self.cities.shape[0], size=2, replace=False))
        return i, j

    def change_neighbors(self):
        i, j = self.generate_random_neighbors()
        new_route = self.current.copy()
        new_route[i:j+1] = new_route[i:j+1][::-1].copy()
        self.new_route = new_route
        return new_route


    def compute_delta(self):
        new_route = self.change_neighbors()
        new_route_distance = self.compute_route_distance(new_route)
        delta = new_route_distance - self.current_distance
        return delta

    @staticmethod
    def compute_probability(delta, t):
        return np.exp(-delta/t)

    def if_accept_new_route(self):
        delta = self.compute_delta()
        if delta <= 0 or self.rng.random() < self.compute_probability(delta, self.t):
            self.current = self.new_route
            self.current_distance += delta
            if self.current_distance < self.best_distance:
                self.best = self.current.copy()
                self.best_distance = self.current_distance

    def run(self):
        self.generate_random_route()
        self.t = self.T_max
        while self.t > self.T_min:
            for _ in range(self.iters_per_temperature):
                self.if_accept_new_route()
                self.iterations += 1
            self.t *= self.alpha

        return self.best, self.best_distance

if __name__ == '__main__':
    data_loader = DataLoader(
        data_path="../data/a280-ttp/a280_n279_bounded-strongly-corr_01.ttp"
    )
    cities = data_loader.get_cords()
    tsp_sim_ann = TravelingSalesmanProblemSimAnn(
        cities=cities,
        seed=42,
        T_max=100,
        T_min=0.01,
        alpha=0.99,
        iters_per_temperature=1000
    )
    best_route, best_distance = tsp_sim_ann.run()
    print(best_route)
    print(best_distance)








