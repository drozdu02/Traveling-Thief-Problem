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

    city_weight = np.zeros(num_of_cities)

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
        distance = tsp_random.swap_distance().sum()
        _, chosen_items, _ = knp_random.random_item_pickup()
        G = ttp_random_random_solution(
            route=tsp_random.swapped_route,
            distances=tsp_random.distances,
            chosen_items=chosen_items,
            capacity=params["CAPACITY OF KNAPSACK"],
            v_min=params["MIN SPEED"],
            v_max=params["MAX SPEED"],
        )
        dists.append(distance)
        Gs.append(G)

        history["best"].append(max(Gs))
        history["avg"].append(float(np.mean(Gs)))
        history["worst"].append(float(min(Gs)))

    DataLogger("logs/ttp_random_random_hard/tsp_random_random_history_hard_2.csv").log_data(history)
    return dists, Gs


if __name__ == "__main__":
    data_loader = DataLoader("./data/a280-ttp/a280_n2790_uncorr_10.ttp")
    cities = data_loader.get_cords()
    params = data_loader.get_knapsack_parameters()
    items = data_loader.get_items()
    n_trials = 100 * 500
    all_dists, all_Gs = [], []
    for seed in range(10):
        dists, Gs = run(
            cities=cities,
            params=params,
            items=items,
            n_trials=n_trials,
            seed=seed
        )
        all_dists.append(dists)
        all_Gs.append(Gs)