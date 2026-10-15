import numpy as np
import random

from typing import Tuple
from scipy.spatial.distance import pdist, squareform
from itertools import combinations

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
        Tutaj trzeba wybrac permutacje.
        :return:
        """
        indexes = list(range(len(self.cities)))
        population = [random.sample(indexes, len(indexes)) for _ in range(self.population_size)]
        return population


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
    result = tsp.initialize_population()
    print(result)




