import numpy as np

from ttp_problem.common.packer import Packer
from ttp_problem.knp.knp_greedy import KnapsackProblemGreedy

class PackItemRandom(Packer):
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
        num_of_cities = len(self.individual)
        distance_to_end = self.city_distances[self.individual, np.roll(self.individual, -1)]
        delay = (self.v_max - self.v_min) / self.capacity

        city_weight = np.zeros(num_of_cities)

        for profit, weight, city in self.items:
            city_weight[int(city) - 1] += weight

        speed = self.v_max - np.cumsum(city_weight[self.individual]) * delay
        time = (distance_to_end / speed).sum().sum()

        total_profit = sum(item[0] for item in self.items)

        return total_profit - time





