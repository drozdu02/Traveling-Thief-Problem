import numpy as np
import statistics
import os

from typing import Dict
from concurrent.futures import ProcessPoolExecutor

from ttp_problem.common.runner import Runner
from ttp_problem.knp.knp_random import KnapsackProblemRandom
from ttp_problem.tsp.tsp_random import TravelingSalesmanProblemRandom
from ttp_problem.common.data_logger import DataLogger
from ttp_problem.common.data_loader import DataLoader
from ttp_problem.packers.pack_item_random import PackItemRandom


class RandomRunner(Runner):

    def __init__(
        self,
        cities : np.ndarray,
        items : np.ndarray,
        params : Dict,
        n_trials : int,
        seed : int,
        log_path : str,
    ):
        super().__init__()
        self.cities = cities
        self.items = items
        self.params = params
        self.n_trials = n_trials
        self.seed = seed
        self.log_path = log_path


    def run(
        self,
    ):
        tsp_random = TravelingSalesmanProblemRandom(
            cities=self.cities,
            seed=self.seed,
        )
        knp_random = KnapsackProblemRandom(
            capacity=self.params["CAPACITY OF KNAPSACK"],
            items=self.items,
            seed=self.seed,
        )
        history = {
            "best": [],
            "avg": [],
            "worst": []
        }

        Gs = []
        best_so_far = -np.inf
        worst_so_far = np.inf
        sum_G = 0.0

        for t in range(self.n_trials):
            tsp_random.new_route()
            _, chosen_items, _ = knp_random.random_item_pickup()
            G = PackItemRandom(
                individual=tsp_random.swapped_route,
                city_distances=tsp_random.distances,
                items=chosen_items,
                capacity=self.params["CAPACITY OF KNAPSACK"],
                v_min=self.params["MIN SPEED"],
                v_max=self.params["MAX SPEED"],
            ).pack()
            Gs.append(G)
            sum_G += G
            if G > best_so_far:
                best_so_far = G
            if G < worst_so_far:
                worst_so_far = G
            history["best"].append(best_so_far)
            history["avg"].append(sum_G / (t + 1))
            history["worst"].append(worst_so_far)

        DataLogger(self.log_path).log_data(history)

        return Gs

def _run_random_seed(args):
    cities, items, params, n_trials, seed, log_path = args

    runner = RandomRunner(
        cities=cities,
        items=items,
        params=params,
        n_trials=n_trials,
        seed=seed,
        log_path=log_path,
    )
    Gs = runner.run()
    return float(max(Gs))

if __name__ == "__main__":
    data_loader = DataLoader("./data/a280-ttp/a280_n837_uncorr_01.ttp")
    cities = data_loader.get_cords()
    params = data_loader.get_knapsack_parameters()
    items = data_loader.get_items()
    n_trials = 1000 * 250
    run_bests = []
    log_prefix = "./logs/random/medium/1/history"

    jobs = [
        (
            cities,
            items,
            params,
            n_trials,
            seed,
            f"{log_prefix}_{seed}.csv",
        )
        for seed in range(10)
    ]

    workers = min(10, os.cpu_count() or 1)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        run_bests = list(ex.map(_run_random_seed, jobs))

    run_bests = np.asarray(run_bests, dtype=float)
    print("Data after n_trials runs:")
    print("best", float(run_bests.max()))
    print("worst", float(run_bests.min()))
    print("avg", float(run_bests.mean()))
    print("median", float(statistics.median(run_bests)))
    print("std", float(run_bests.std(ddof=1)))

    DataLogger("./logs/random/medium/1/summary.csv").log_summary(run_bests)

