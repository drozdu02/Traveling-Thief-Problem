import numpy as np
import random

from typing import Tuple
from scipy.spatial.distance import pdist, squareform

class TravelingSalesmanProblem:
    def __init__(
            self,
            cities : np.ndarray,
            starting_point : Tuple[int, int],
            population_size : int,
            num_of_generations : int,
    ):
        self.cities = cities
        self.starting_point = starting_point
        self.population_size = population_size
        self.num_of_generations = num_of_generations


    def compute_distances(self):
        return squareform(pdist(self.cities, metric='euclidean'))


    def initialize_population(self):
        """
        Inicjalizujemy najpierw liczbę osobników z losowymi trasami.
        Nalezy wziac pod uwage, ze startujemy od okreslonego punktu.
        Tutaj trzeba wybrac kombinację.
        :return:
        """
        indexes = list(range(len(self.cities)))
        population = [random.sample(indexes, len(indexes)) for _ in range(self.population_size)]
        return population

    @staticmethod
    def _compute_distance_for_single_generation(generation : list[list[int]], distances : np.ndarray):
        distance = 0
        for i in range(len(generation) - 1):
            distance += distances[generation[i]][generation[i + 1]]

        distance += distances[generation[-1]][generation[0]]
        return distance


    def compute_distances_for_generations(self):
        distances_matrix = self.compute_distances()
        population = self.initialize_population()
        distance_for_all_population = [self._compute_distance_for_single_generation(p, distances_matrix) for p in population]
        return distance_for_all_population


if __name__ == '__main__':
    cities = np.array([
        [0, 0],
        [4, 1],
        [6, 3],
        [2, 5],
        [7, 7],
    ]) #shape = (5, 2) -> 4 indeksy
    # print(range(len(cities)))
    tsp = TravelingSalesmanProblem(
        cities=cities,
        population_size=3,
        num_of_generations=5,
        starting_point=(0, 0),
    )
    print(tsp.initialize_population())
    print(tsp.compute_distances_for_generations())
    # entry_population = tsp.initialize_population()
    # print(entry_population)
    # distances = tsp.compute_distances()
    # print(distances)



