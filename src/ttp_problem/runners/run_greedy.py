import numpy as np
import statistics

from typing import Dict

from ttp_problem.common.data_loader import DataLoader
from ttp_problem.common.data_logger import DataLogger
from ttp_problem.tsp.tsp_greedy import TravelingSalesmanProblemGreedy
from ttp_problem.common.runner import Runner
from ttp_problem.packers.pack_item_greedy import PackItemGreedy


class GreedyRunner(Runner):
    def __init__(
        self,
        items : np.ndarray,
        cities : np.ndarray,
        params : Dict,
        log_path : str
    ):
        self.items = items
        self.cities = cities
        self.params = params
        self.log_path = log_path

    def run(
        self
    ):
        starts = range(len(self.cities))
        dists, Gs = [], []
        history = {
            "best": [],
            "avg": [],
            "worst": [],
        }

        for start in starts:
            tsp_greedy = TravelingSalesmanProblemGreedy(
                cities=self.cities,
                starting_point=start,
            )
            distance, route = tsp_greedy.travel()

            # route, _, G = ttp_greedy_solution(
            #     route,
            #     tsp_greedy.distances,
            #     self.items,
            #     capacity=self.params['CAPACITY OF KNAPSACK'],
            #     v_min=self.params['MIN SPEED'],
            #     v_max=self.params['MAX SPEED'],
            # )
            route, _, G = PackItemGreedy(
                individual=route,
                city_distances=tsp_greedy.distances,
                items=self.items,
                capacity=self.params["CAPACITY OF KNAPSACK"],
                v_max=self.params['MAX SPEED'],
                v_min=self.params['MIN SPEED'],
            ).pack()
            dists.append(distance)
            Gs.append(G)
            history["best"].append(max(Gs))
            history["avg"].append(float(np.mean(Gs)))
            history["worst"].append(float(min(Gs)))

        data_logger = DataLogger(f"{self.log_path}.csv").log_data(history)

        Gs = np.asarray(Gs, dtype=float)
        print("Data after N runs:")
        print("best", float(Gs.max()))
        print("worst", float(Gs.min()))
        print("avg", float(Gs.mean()))
        print("median", float(statistics.median(Gs)))
        print("std", float(Gs.std(ddof=1)))

        DataLogger(f"{self.log_path}_summary.csv").log_summary(Gs)

if __name__ == "__main__":
    data_loader = DataLoader("./data/a280-ttp/a280_n2790_bounded-strongly-corr_01.ttp")

    cities = data_loader.get_cords()
    params = data_loader.get_knapsack_parameters()
    items = data_loader.get_items()
    log_path_prefix =f"./logs/greedy/hard/2/history"

    runner = GreedyRunner(
        items=items,
        cities=cities,
        params=params,
        log_path=log_path_prefix
    )
    runner.run()