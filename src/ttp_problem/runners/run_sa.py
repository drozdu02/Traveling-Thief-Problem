import numpy as np
import statistics
import os

from concurrent.futures import ProcessPoolExecutor
from typing import Dict

from ttp_problem.common.data_loader import DataLoader
from ttp_problem.common.data_logger import DataLogger
from ttp_problem.common.runner import Runner
from ttp_problem.tsp.tsp_sim_ann import TravelingSalesmanProblemSimAnn

class SARunner(Runner):
    def __init__(
        self,
        cities : np.ndarray,
        items : np.ndarray,
        params : Dict,
        T_max : int,
        T_min : float,
        alpha : float,
        iters_per_temperature : int,
        log_path : str,

    ):
        super(SARunner, self).__init__()
        self.cities = cities
        self.items = items
        self.params = params
        self.T_max = T_max
        self.T_min = T_min
        self.alpha = alpha
        self.iters_per_temperature = iters_per_temperature
        self.log_path = log_path

    def run(
        self
    ):
        jobs = [
            (
                self.cities,
                self.items,
                self.params["CAPACITY OF KNAPSACK"],
                self.params["MIN SPEED"],
                self.params["MAX SPEED"],
                self.T_max,
                self.T_min,
                self.alpha,
                self.iters_per_temperature,
                seed,
                self.log_path,
            )
            for seed in range(10)
        ]
        workers = min(10, os.cpu_count() or 1)
        with ProcessPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(_run_sa_seed, jobs))
        Gs = np.asarray([r[0] for r in results], dtype=float)
        evals = [r[1] for r in results]
        print("Data after 10 runs:")
        print("best", float(Gs.max()))
        print("worst", float(Gs.min()))
        print("avg", float(Gs.mean()))
        print("median", float(statistics.median(Gs)))
        print("std", float(Gs.std(ddof=1)))
        DataLogger(f"{self.log_path}_summary.csv").log_summary(Gs)

def _run_sa_seed(args):
    (
        cities, items, capacity, v_min, v_max,
        T_max, T_min, alpha, iters_per_temperature,
        seed, log_path,
    ) = args

    sa = TravelingSalesmanProblemSimAnn(
        cities=cities,
        items=items,
        capacity=capacity,
        seed=seed,
        v_min=v_min,
        v_max=v_max,
        T_max=T_max,
        T_min=T_min,
        alpha=alpha,
        iters_per_temperature=iters_per_temperature,
    )
    _, best_G = sa.run()
    DataLogger(f"{log_path}_seed_{seed}.csv").log_data(sa.history)
    return best_G, sa.evaluations


if __name__ == "__main__":
    data_loader = DataLoader("./data/a280-ttp/a280_n279_uncorr_01.ttp")

    cities = data_loader.get_cords()
    params = data_loader.get_knapsack_parameters()
    items = data_loader.get_items()
    log_path_prefix =f"./logs/sa/tuning/mint/mint01/history"
    runner = SARunner(
        cities=cities,
        items=items,
        params=params,
        T_max=200,
        T_min=0.1,
        alpha=0.999,
        iters_per_temperature=250,
        log_path=log_path_prefix,
    )
    runner.run()