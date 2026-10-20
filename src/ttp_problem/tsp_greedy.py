import numpy as np

from scipy.spatial.distance import squareform, pdist

from data_loader import DataLoader

class TravelingSalesmanProblemGreedy:
    def __init__(self, cities : np.ndarray, starting_point : int):
        self.cities = cities
        self.starting_point = starting_point
        self.distances = self.compute_distances()

    def compute_distances(self):
        distances = squareform(pdist(self.cities, metric='euclidean'))
        distances[distances == 0] = np.inf
        return distances

    def travel(self):

        distance = 0
        curr = self.starting_point
        history = [curr]
        seen = set()
        seen.add(self.starting_point)

        while len(seen) < len(self.cities):
            row = self.distances[curr].copy()
            row[list(seen)] = np.inf
            nearest_node = int(np.argmin(row))

            distance += self.distances[curr][nearest_node]
            seen.add(nearest_node)
            history.append(nearest_node)
            curr = nearest_node

        distance += self.distances[curr][self.starting_point]
        history.append(self.starting_point)

        return distance, history


if __name__ == '__main__':
    data_loader = DataLoader("./data/a280-ttp/a280_n279_bounded-strongly-corr_01.ttp")
    cities = data_loader.get_cords()
    starting_point = np.random.randint(0, len(cities) - 1)
    tsp_greedy = TravelingSalesmanProblemGreedy(
        cities=cities,
        starting_point=starting_point,
    )
    distance, history = tsp_greedy.travel()
    print(distance, history)



