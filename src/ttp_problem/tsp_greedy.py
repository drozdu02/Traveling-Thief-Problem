import numpy as np

from scipy.spatial.distance import squareform, pdist

class TravelingSalesmanProblemGreedy:
    def __init__(self, cities : np.ndarray):
        self.cities = cities

    def compute_distances(self):
        return squareform(pdist(self.cities, metric='euclidean'))


if __name__ == '__main__':
    cities = np.random.randint(0, 9, size=(5, 2))

    tsp_greedy = TravelingSalesmanProblemGreedy(
        cities=cities
    )
    print(tsp_greedy.compute_distances())

