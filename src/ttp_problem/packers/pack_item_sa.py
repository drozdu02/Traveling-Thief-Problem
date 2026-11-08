import numpy as np

from ttp_problem.common.packer import Packer
from ttp_problem.knp.knp_greedy import KnapsackProblemGreedy

class PackItemSa(Packer):
    def __init__(
        self,
        individual : np.ndarray,
        city_distances : np.ndarray,
        items : np.ndarray,
        capacity : float,
        v_max : float,
        v_min: float,

    ):
        super().__init__()
        self.individual = individual
        self.city_distances = city_distances
        self.items = items
        self.capacity = capacity
        self.v_max = v_max
        self.v_min = v_min


    def pack(self):
        starting_city_index = list(self.individual).index(0)
        rotated_individual = np.array(list(self.individual[starting_city_index:]) + list(self.individual[:starting_city_index]))
        num_of_cities = len(rotated_individual)

        individual_distance = self.city_distances[rotated_individual, np.roll(rotated_individual, -1)]

        distance_to_end = np.zeros(num_of_cities)

        distance_to_end[rotated_individual] = np.cumsum(individual_distance[::-1])[::-1]

        delay = (self.v_max - self.v_min) / self.capacity

        adjusted_items = []

        for profit, weight, city in self.items:
            extra_time = distance_to_end[int(city) - 1] * (1 / (self.v_max - weight * delay) - 1 / self.v_max)
            value = profit - extra_time

            if value > 0:
                adjusted_items.append((value, weight, city, profit))

        _, chosen_items, _ = KnapsackProblemGreedy(
            capacity=self.capacity,
            items=adjusted_items
        ).greedy()

        city_weight = np.zeros(num_of_cities)

        for _, weight, city, _ in chosen_items:
            city_weight[int(city) - 1] += weight

        speed = self.v_max - np.cumsum(city_weight[rotated_individual]) * delay
        time = (individual_distance / speed).sum()

        total_profit = sum(item[3] for item in chosen_items)
        return rotated_individual, chosen_items, total_profit - time





