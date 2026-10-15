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
        self.population = self.initialize_population()


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
        population = self.population
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

    def _create_tournament(self, fitness):
        """
        Tworzymy turniej. Zwracamy indeks osobnika, ktor ma najwiekszy fitness.
        :param fitness:
        :return:
        """
        indexes = self._generate_random_indexes()
        return indexes[np.argmax(fitness[indexes])]

    def select_by_tournament(self):
        """
        Przeprowadznie 2 turniejow. Warto pamietac, ze nie mozemy wybrac tego samego osobnika.
        :return:
        """
        fitness = np.asarray(self.compute_fitness())
        first_winner = self._create_tournament(fitness)
        second_winner = self._create_tournament(fitness)
        while second_winner == first_winner:
            second_winner = self._create_tournament(fitness)
        return first_winner, second_winner


    def get_winners(self):
        """
        Pobieramy zwyciezcow.
        :return:
        """
        first_winner, second_winner = self.select_by_tournament()
        return self.population[first_winner], self.population[second_winner]

    def ox_crossover(self):
        """
        Pobieramy wygranych turnieju.
        Potem losujemy 2 indeksy z zakresu [0, len - 1]
        :return:
        """
        first_winner, second_winner = self.get_winners()
        a, b = np.sort(self.rng.choice(np.arange(len(first_winner) + 1), size=2, replace=False))



if __name__ == '__main__':
    cities = np.random.randint(0, 9, size=(5, 2))
    # cities = np.array([
    #     [0, 0],
    #     [4, 1],
    #     [6, 3],
    #     [2, 5],
    #     [7, 7],
    # ])
    tsp = TravelingSalesmanProblem(
        cities=cities,
        population_size=15,
        num_of_generations=5,
        starting_point=(0, 0),
        k=5,
    )
    print(tsp.ox_crossover())




