import numpy as np
import statistics
import os

from typing import Dict
from concurrent.futures import ProcessPoolExecutor

from ttp_problem.common.data_loader import DataLoader
from ttp_problem.common.data_logger import DataLogger
from ttp_problem.tsp.tsp_ga import TravelingSalesmanProblem
from ttp_problem.common.runner import Runner
def _run_ga_seed(args):
    (
        cities, items, capacity, v_min, v_max,
        population_size, num_of_generations, k,
        mutation_rate, crossover_rate,
        seed, log_path,
    ) = args

    ga = TravelingSalesmanProblem(
        cities=cities,
        items=items,
        capacity=capacity,
        v_min=v_min,
        v_max=v_max,
        population_size=population_size,
        num_of_generations=num_of_generations,
        k=k,
        mutation_rate=mutation_rate,
        crossover_rate=crossover_rate,
        seed=seed,
    )
    _, best = ga.run()
    DataLogger(f"{log_path}_seed_{seed}.csv").log_data(ga.history)
    return best, ga.evaluations

class GaRunner(Runner):
    def __init__(
        self,
        cities: np.ndarray,
        params: Dict,
        items: np.ndarray,
        log_path: str,
        population_size: int,
        num_of_generations: int,
        k: int,
        mutation_rate: float,
        crossover_rate: float,

    ):
        super().__init__()
        self.cities = cities
        self.params = params
        self.items = items
        self.log_path = log_path
        self.population_size = population_size
        self.num_of_generations = num_of_generations
        self.k = k
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

    def run(
        self
    ):
        Gs, evals = [], []

        jobs = [
            (
                self.cities, self.items, self.params["CAPACITY OF KNAPSACK"], self.params['MIN SPEED'], self.params['MAX SPEED'],
                self.population_size, self.num_of_generations, self.k,
                self.mutation_rate, self.crossover_rate,
                seed, self.log_path,
            )
            for seed in range(10)
        ]

        workers = min(10, os.cpu_count() or 1)
        with ProcessPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(_run_ga_seed, jobs))

        Gs = np.asarray([r[0] for r in results], dtype=float)
        evals = [r[1] for r in results]

        Gs = np.asarray(Gs, dtype=float)

        print("Data after 10 runs:")
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
    log_path_prefix =f"./logs/ga/hard/2/history"

    runner = GaRunner(
        cities=cities,
        params=params,
        items=items,
        log_path=log_path_prefix,
        population_size=250,
        num_of_generations=1000,
        k=35,
        mutation_rate=0.35,
        crossover_rate=0.90,
    )

    runner.run()




