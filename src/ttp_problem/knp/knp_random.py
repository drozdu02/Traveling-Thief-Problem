import numpy as np

from typing import List
from ttp_problem.common.data_loader import DataLoader

class KnapsackProblemRandom:
    def __init__(self, capacity: float, items: List, seed : int):
        self.items = items
        self.capacity = capacity
        self.rng = np.random.default_rng(seed=seed)

    def random_item_pickup(self):
        random_backpack_order = self.rng.permutation(len(self.items))
        total_value = 0
        total_weight = 0
        chosen_items = []

        for idx_item in random_backpack_order:
            item = self.items[idx_item]
            if total_weight + item[1] <= self.capacity:
                chosen_items.append(item)
                total_weight += item[1]
                total_value += item[0]

        return total_value, chosen_items, total_weight


if __name__ == "__main__":
    data_loader = DataLoader("../data/a280-ttp/a280_n279_bounded-strongly-corr_01.ttp")
    items = data_loader.get_items()
    parameters = data_loader.get_knapsack_parameters()
    print(parameters)
    capacity = parameters['CAPACITY OF KNAPSACK']
    knp_greedy = KnapsackProblemRandom(
        capacity=capacity,
        items=items
    )
    print(knp_greedy.random_item_pickup())
