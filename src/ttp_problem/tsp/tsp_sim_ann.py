
import numpy as np

from scipy.spatial.distance import pdist, squareform

from ttp_problem.tsp import tsp_greedy
from ttp_problem.tsp.tsp_greedy import TravelingSalesmanProblemGreedy
from ttp_problem.packers.pack_item_sa import PackItemSa

class TravelingSalesmanProblemSimAnn:
    def __init__(
        self,
        cities : np.ndarray,
        items : np.ndarray,
        capacity : float,
        seed : int,
        v_min : float,
        v_max : float,
        T_max : float,
        T_min : float,
        alpha : float,
        iters_per_temperature : int
    ):
        self.cities = np.asarray(cities)
        self.items = np.asarray(items)
        self.capacity = capacity
        self.T_max = T_max
        self.T_min = T_min
        self.v_min = v_min
        self.v_max = v_max
        self.alpha = alpha
        self.rng = np.random.default_rng(seed)
        self.distances = self.compute_distances()
        self.best = None
        self.best_g = None
        self.current = None
        self.current_G = None
        self.new_route = None
        self.t = T_max
        self.iterations = 0
        self.evaluations = 0
        self.iters_per_temperature = iters_per_temperature
        self.history = {
            "best" : [],
            "avg" : [],
            "worst" : [],
        }


    def evaluate(self, route):
        self.evaluations += 1
        _, _, Gs = PackItemSa(
            individual=route,
            city_distances=self.distances,
            items=self.items,
            capacity=self.capacity,
            v_min=self.v_min,
            v_max=self.v_max,
        ).pack()
        return Gs

    def compute_distances(self):
        distances = squareform(pdist(self.cities[:, 1:], metric='euclidean'))
        return np.ceil(distances)


    def generate_initial_route(self):
        route = self.rng.permutation(self.cities.shape[0])
        route = np.asarray(route)
        g0 = self.evaluate(route)
        self.best = route.copy()
        self.current = route.copy()
        self.current_G = g0
        self.best_g = g0

        return route

    def change_neighbors(self):
        i, j = sorted(self.rng.choice(self.cities.shape[0], size=2, replace=False))
        new_route = self.current.copy()
        new_route[i:j+1] = new_route[i:j+1][::-1].copy()
        self.new_route = new_route
        return new_route

    @staticmethod
    def compute_probability(delta, t):
        return np.exp(-delta/t)

    def if_accept_new_route(self):
        new_route = self.change_neighbors()
        new_G = self.evaluate(new_route)
        delta = self.current_G - new_G
        if delta <= 0 or self.rng.random() < self.compute_probability(delta, self.t):
            self.current = new_route.copy()
            self.current_G = new_G
            if new_G > self.best_g:
                self.best = new_route.copy()
                self.best_g = new_G

    def run(self):
        self.generate_initial_route()
        self.t = self.T_max
        while self.t > self.T_min:
            for _ in range(self.iters_per_temperature):
                self.if_accept_new_route()

            self.history["best"].append(float(self.best_g))
            self.history["avg"].append(float(self.current_G))
            self.history["worst"].append(float(self.current_G))
            self.t *= self.alpha

        return self.best, float(self.best_g)

# if __name__ == '__main__':
#     data_loader = DataLoader(
#         data_path="../data/a280-ttp/a280_n279_bounded-strongly-corr_01.ttp"
#     )
#     cities = data_loader.get_cords()
#     tsp_sim_ann = TravelingSalesmanProblemSimAnn(
#         cities=cities,
#         seed=42,
#         T_max=100,
#         T_min=0.01,
#         alpha=0.99,
#         iters_per_temperature=1000
#     )
#     best_route, best_distance = tsp_sim_ann.run()
#     print(best_route)
#     print(best_distance)








