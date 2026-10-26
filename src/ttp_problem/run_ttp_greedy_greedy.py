import numpy as np
from ttp_problem.common.data_logger import DataLogger
from ttp_problem.common.data_loader import DataLoader
from ttp_problem.knp.knp_greedy import KnapsackProblemGreedy
from ttp_problem.tsp.tsp_greedy import TravelingSalesmanProblemGreedy

def ttp_greedy_greedy_solution(
    ga_route : np.ndarray,
    city_distances : np.ndarray,
    items: np.ndarray,
    capacity: float,
    v_min: float,
    v_max: float,
):
    starting_city_index = list(ga_route).index(0)
    route = np.array(list(ga_route[starting_city_index:]) + list(ga_route[:starting_city_index]))
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

    _, chosen_items, _ = KnapsackProblemGreedy(
        capacity=capacity,
        items=adjusted
    ).greedy()

    city_weight = np.zeros(num_of_cities)

    for _, weight, city, _ in chosen_items:
        city_weight[int(city) - 1] += weight

    speed = v_max - np.cumsum(city_weight[route]) * nu
    time = (distance / speed).sum()
    total_profit = sum(item[3] for item in chosen_items)

    return route, chosen_items, total_profit - time

def run(
        data_loader : DataLoader,
        cities : np.ndarray,
        params : dict,
        items: np.ndarray,
):
    starting_point = np.random.randint(0, len(cities))

    starts = range(len(cities))
    dists, Gs = [], []
    history = {
        "best": [],
        "avg" : [],
        "worst": [],
    }

    for start in starts:
        tsp_greedy = TravelingSalesmanProblemGreedy(
            cities=cities,
            starting_point=start,
        )
        distance, route = tsp_greedy.travel()

        route, chosen, G = ttp_greedy_greedy_solution(
            route,
            tsp_greedy.distances,
            items,
            capacity=params['CAPACITY OF KNAPSACK'],
            v_min=params['MIN SPEED'],
            v_max=params['MAX SPEED'],
        )
        dists.append(distance)
        Gs.append(G)
        history["best"].append(max(Gs))
        history["avg"].append(float(np.mean(Gs)))
        history["worst"].append(float(min(Gs)))

    data_logger = DataLogger(
        "logs/ttp_greedy_greedy_hard/tsp_greedy_greedy_history_hard_2.csv"
    ).log_data(history)

if __name__ == "__main__":
    data_loader = DataLoader("./data/a280-ttp/a280_n2790_uncorr_10.ttp")
    cities = data_loader.get_cords()
    params = data_loader.get_knapsack_parameters()
    items = data_loader.get_items()
    run(
        data_loader=data_loader,
        cities=cities,
        params=params,
        items=items,
    )
