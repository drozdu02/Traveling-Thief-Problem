import numpy as np

from ttp_problem.common.data_loader import DataLoader
from ttp_problem.knp.knp_random import KnapsackProblemRandom
from ttp_problem.tsp.tsp_random import TravelingSalesmanProblemRandom
from ttp_problem.common.data_logger import DataLogger

def ttp_random_random_solution(
    route: np.ndarray,
    distances: np.ndarray,
    chosen_items: list,
    capacity : float,
    v_min: float,
    v_max: float,
):
    num_of_cities = len(route)
    distance = distances[route, np.roll(route, -1)]
    nu = (v_max - v_min) / capacity

    city_weight = np.empty(num_of_cities)

    for profit, weight, city in chosen_items:
        city_weight[int(city) - 1] += weight


    speed = v_max - np.cumsum(city_weight[route]) * nu
    time = (distance / speed).sum().sum()

    total_profit = sum(item[0] for item in chosen_items)

    return total_profit - time

def run(
    cities,
    params,
    items,
    n_trials,
    seed,
    log_path
):
    tsp_random = TravelingSalesmanProblemRandom(
        cities=cities,
        seed=seed,
    )
    knp_random = KnapsackProblemRandom(
        capacity=params["CAPACITY OF KNAPSACK"],
        items=items,
        seed=seed,
    )
    history = {
        "best": [],
        "avg": [],
        "worst" : []
    }

    dists, Gs = [], []
    for _ in range(n_trials):
        tsp_random.new_route()
        _, chosen_items, _ = knp_random.random_item_pickup()
        G = ttp_random_random_solution(
            route=tsp_random.swapped_route,
            distances=tsp_random.distances,
            chosen_items=chosen_items,
            capacity=params["CAPACITY OF KNAPSACK"],
            v_min=params["MIN SPEED"],
            v_max=params["MAX SPEED"],
        )
        Gs.append(G)

        history["best"].append(max(Gs))
        history["avg"].append(float(np.mean(Gs)))
        history["worst"].append(float(min(Gs)))

    DataLogger(log_path).log_data(history)
    return Gs


if __name__ == "__main__":
    data_loader = DataLoader("./data/a280-ttp/a280_n2790_bounded-strongly-corr_01.ttp")
    cities = data_loader.get_cords()
    params = data_loader.get_knapsack_parameters()
    items = data_loader.get_items()
    n_trials = 100 * 100
    run_bests = []
    for seed in range(10):
        Gs = run(
            cities=cities,
            params=params,
            items=items,
            n_trials=n_trials,
            seed=seed,
            log_path=f"logs/random_random/ttp_random_random_hard/2/ttp_random_random_history_hard_{seed}.csv",
        )
        best_G = float(max(Gs))
        run_bests.append(best_G)

    run_bests = np.asarray(run_bests, dtype=float)
    print("--- summary (10 runs, each best) ---")
    print("best", float(run_bests.max()))
    print("worst", float(run_bests.min()))
    print("avg", float(run_bests.mean()))
    print("std", float(run_bests.std(ddof=1)))