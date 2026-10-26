import numpy as np

from scipy.spatial.distance import squareform, pdist

from ttp_problem.common.data_loader import DataLoader

class TravelingSalesmanProblemRandom:
    def __init__(self, cities : np.ndarray, seed : int):
        self.cities = np.asarray(cities)
        self.rng = np.random.default_rng(seed=seed)
        self.distances = self.compute_distances()
        self.new_route()

    def new_route(self):
        self.route = self.random_route()
        self.first_city = self.get_first_city()
        self.swapped_route = self.swap_route()

    def compute_distances(self):
        distances = squareform(pdist(self.cities[..., 1:], metric='euclidean'))
        return np.ceil(distances)

    def random_route(self):
        route = self.rng.permutation(len(self.cities))
        return route

    def get_first_city(self):
        first_city = int(np.flatnonzero(self.route == 0)[0])
        return first_city

    def swap_route(self):
        swapped_route = np.roll(self.route, -self.first_city)
        return swapped_route

    def swap_distance(self):
        new_swapped_distance = self.distances[self.swapped_route, np.roll(self.swapped_route, -1)]
        return new_swapped_distance



if __name__ == '__main__':
    data_loader = DataLoader("../data/a280-ttp/a280_n279_bounded-strongly-corr_01.ttp")
    cities = data_loader.get_cords()
    tsp_random = TravelingSalesmanProblemRandom(
        cities=cities,
    )
    print(tsp_random.swap_distance())




