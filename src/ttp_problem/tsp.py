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
            k : int,
    ):
        self.cities = cities
        self.starting_point = starting_point
        self.population_size = population_size
        self.num_of_generations = num_of_generations
        self.k = k
        self.rng = np.random.default_rng(42)


    def compute_distances(self):
        """
        Macierz odleglosci, liczymy euklidesem
        :return:
        """
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
        """
        Obliczamy dlugosc trasy jednego osobnika
        :param generation:
        :param distances:
        :return:
        """
        distance = 0
        for i in range(len(generation) - 1):
            distance += distances[generation[i]][generation[i + 1]]

        distance += distances[generation[-1]][generation[0]]
        return distance


    def compute_distances_for_generations(self):
        """
        Obliczamy dlugosc trasy kazdego osobnika
        :return:
        """
        distances_matrix = self.compute_distances()
        population = self.initialize_population()
        distance_for_all_population = [self._compute_distance_for_single_generation(p, distances_matrix) for p in population]
        return distance_for_all_population

    def compute_fitness(self):
        """
        Obliczamy fitness, czyli 1/dlugosc trasy dla kazdego osobnika
        :return:
        """
        distances_for_generations = self.compute_distances_for_generations()
        fitness = [1 / d for d in distances_for_generations]
        return fitness

    def _generate_random_indexes(self):
        """
        Losujemy k indeksow, aby pobrac potem k losowych osobnikow do procesu selekcji
        :return:
        """
        indexes = self.rng.choice(np.arange(0, self.population_size), size=self.k, replace=False)
        return indexes

    def select_by_tournament(self):
        """
        Domyślnie mamy ustawione 2 turnieje. W kazdym turnieju wybieramy k osobnikow i z niego wybieramy indeks, aby
        wiedziec ktory potem zostal wybrany.
        :return:
        """
        fitness = np.asarray(self.compute_fitness())
        indexes_1 = self._generate_random_indexes()
        indexes_2 = self._generate_random_indexes()

        tournament_1 = fitness[indexes_1]
        tournament_2 = fitness[indexes_2]
        first_winner = np.argmax(tournament_1)
        second_winner = np.argmax(tournament_2)
        return first_winner, second_winner




if __name__ == '__main__':
    cities = np.array([
        [0, 0],
        [4, 1],
        [6, 3],
        [2, 5],
        [7, 7],
    ])
    tsp = TravelingSalesmanProblem(
        cities=cities,
        population_size=15,
        num_of_generations=5,
        starting_point=(0, 0),
        k=5,
    )
    print(tsp.select_by_tournament())




