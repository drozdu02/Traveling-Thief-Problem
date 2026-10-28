import numpy as np
from ttp_problem.common.data_logger import DataLogger
from ttp_problem.common.data_loader import DataLoader
from ttp_problem.tsp.tsp_sim_ann import TravelingSalesmanProblemSimAnn
from ttp_problem.knp.knp_sim_ann import KnapsackProblemSimAnn

def ttp_sa_sa_solution(
    sa_route : np.ndarray,
    city_distances : np.ndarray,
    items: np.ndarray,
    capacity: float,
    v_min: float,
    v_max: float,
    seed : int,
    T_max: float,
    T_min: float,
    alpha: float,
    iters_per_temperature : int,

):
    starting_city_index = list(sa_route).index(0)
    route = np.array(list(sa_route[starting_city_index:]) + list(sa_route[:starting_city_index]))
    num_of_cities = len(route)

    distance = city_distances[route, np.roll(route, -1)]

    distance_to_end = np.empty(num_of_cities)

    distance_to_end[route] = np.cumsum(distance[::-1])[::-1]

    nu = (v_max - v_min) / capacity
    adjusted = []

    for profit, weight, city in items:
        extra_time = distance_to_end[int(city) - 1] * (1 / (v_max - weight * nu) - 1 / v_max)
        value = profit - extra_time

        if value > 0:
            adjusted.append((value, weight, city, profit))

    city_weight = np.empty(num_of_cities)

    if len(adjusted) == 0:
        time = (distance / v_max).sum()
        return route, adjusted, 0 - time

    adjusted = np.asarray(adjusted)

    best_items, _ = KnapsackProblemSimAnn(
        capacity=capacity,
        items=adjusted,
        seed=seed,
        T_max=T_max,
        T_min=T_min,
        alpha=alpha,
        iters_per_temperature=iters_per_temperature,
    ).run()

    chosen_items = adjusted[best_items.astype(bool)]


    for _, weight, city, _ in chosen_items:
        city_weight[int(city) - 1] += weight

    speed = v_max - np.cumsum(city_weight[route]) * nu
    time = (distance / speed).sum()
    total_profit = sum(item[3] for item in chosen_items)

    return route, chosen_items, total_profit - time

if __name__ == "__main__":
    data_loader = DataLoader("./data/a280-ttp/a280_n279_uncorr-similar-weights_01.ttp")

    cities = data_loader.get_cords()
    params = data_loader.get_knapsack_parameters()
    items = data_loader.get_items()
    final_Gs = []

    for seed in range(10):
        tsp_sim_ann = TravelingSalesmanProblemSimAnn(
            cities=cities,
            seed=seed,
            T_max=200,
            T_min=1,
            alpha=0.99,
            iters_per_temperature=100
        )
        route, dist = tsp_sim_ann.run()
        _, chosen, G = ttp_sa_sa_solution(
            sa_route=route,
            city_distances=tsp_sim_ann.distances,
            items=items,
            capacity=params['CAPACITY OF KNAPSACK'],
            v_min=params['MIN SPEED'],
            v_max=params['MAX SPEED'],
            T_max=200,
            T_min=1,
            alpha=0.99,
            iters_per_temperature=100,
            seed=seed,

        )

        final_Gs.append(G)

        data_logger = DataLogger(
            f"logs/sa_sa/ttp_sa_sa_easy/3/ttp_sa_sa_history_easy_{seed}.csv"
        )
        data_logger.log_data({
            "best": [G],
            "avg" : [G],
            "worst" : [G],
        })

    final_Gs = np.asarray(final_Gs, dtype=float)
    print("Data after 10 runs:")
    print("best", float(final_Gs.max()))
    print("worst", float(final_Gs.min()))
    print("avg", float(final_Gs.mean()))
    print("std", float(final_Gs.std(ddof=1)))