from typing import List

from data_loader import DataLoader

class KnapsackProblemGreedy:
    def __init__(self, capacity : float, items : List):
        self.items = items
        self.capacity = capacity


    def greedy(self):
        items = self.items
        sorted_backpack = sorted(
            items,
            key=lambda item: item[0] / item[1] if item[1] > 0 else float('inf'),
            reverse=True)
        total_value = 0
        total_weight = 0
        chosen_items = []

        for item in sorted_backpack:
            if total_weight + item[1] <= self.capacity:
                chosen_items.append(item)
                total_weight += item[1]
                total_value += item[0]


        fitting = [item for item in items if item[1] <= self.capacity]
        if fitting:
            best_single = max(fitting, key=lambda item: item[0])
            if best_single[0] > total_value:
                return best_single[0], [best_single], best_single[1]

        return total_value, chosen_items, total_weight



if __name__ == '__main__':
    data_loader = DataLoader("./data/a280-ttp/a280_n279_bounded-strongly-corr_01.ttp")
    items = data_loader.get_items()
    parameters = data_loader.get_knapsack_parameters()
    capacity = parameters['CAPACITY OF KNAPSACK']
    knp_greedy = KnapsackProblemGreedy(
        capacity=capacity,
        items=items
    )
    print(knp_greedy.greedy())





