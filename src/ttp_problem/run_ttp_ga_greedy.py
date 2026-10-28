import numpy as np
from ttp_problem.common.data_logger import DataLogger
from ttp_problem.common.data_loader import DataLoader
from ttp_problem.knp.knp_greedy import KnapsackProblemGreedy
from ttp_problem.tsp.tsp_ga import TravelingSalesmanProblem

def ttp_ga_greedy_solution(
    ga_route : np.ndarray,
    city_distances : np.ndarray,
    items: np.ndarray,
    capacity: float,
    v_min: float,
    v_max: float,

):
    starting_city_index = list(ga_route).index(0) # wybieramy indeks 0
    route = np.array(list(ga_route[starting_city_index:]) + list(ga_route[:starting_city_index])) # startujemy od 0
    num_of_cities = len(route)

    distance = city_distances[route, np.roll(route, -1)] # odleglosci od sąsiadów

    distance_to_end = np.empty(num_of_cities)

    distance_to_end[route] = np.cumsum(distance[::-1])[::-1] # odległość do konca trasy np. [40, 30, 20, 10] -> miasto pierwsze ma do przejechania 40

    nu = (v_max - v_min) / capacity # wspolczynnik opoznienia (ze wzoru z labow)
    adjusted = [] # nowe itemy

    for profit, weight, city in items:
        extra_time = distance_to_end[int(city) - 1] * (1 / (v_max - weight * nu) - 1 / v_max) # korekta wartości przedmiotów
        value = profit - extra_time # nadpisujemy wartości

        if value > 0:
            adjusted.append((value, weight, city, profit)) # dodane do plecaka

    _, chosen_items, _ = KnapsackProblemGreedy(
        capacity=capacity,
        items=adjusted
    ).greedy()

    city_weight = np.zeros(num_of_cities) # waga w miastach

    for _, weight, city, _ in chosen_items:
        city_weight[int(city) - 1] += weight # dodajemy wagę do miast

    speed = v_max - np.cumsum(city_weight[route]) * nu # obliczanie predkosci, suma kumulatywna, kazda wartosc to predkosc na odcinku wychodzacym z miasta
    time = (distance / speed).sum() # czas ze wzoru droga / predkosc
    total_profit = sum(item[3] for item in chosen_items) # sumujemy profity

    return route, chosen_items, total_profit - time





if __name__ == "__main__":
    data_loader = DataLoader("./data/a280-ttp/a280_n279_uncorr-similar-weights_01.ttp")

    cities = data_loader.get_cords()
    params = data_loader.get_knapsack_parameters()
    items = data_loader.get_items()
    final_Gs = []

    for seed in range(10):

        tsp = TravelingSalesmanProblem(
            cities=cities,
            population_size=100,
            num_of_generations=100,
            k=5,
            mutation_rate=0.1,
            crossover_rate=0.9,
            seed=seed,
        )

        _, _, G = ttp_ga_greedy_solution(
            route,
            tsp.distances,
            items,
            capacity=params['CAPACITY OF KNAPSACK'],
            v_min=params['MIN SPEED'],
            v_max=params['MAX SPEED'],
        )


        route, G = tsp.run()
        final_Gs.append(G)



        data_logger = DataLogger(
            f"logs/ga_greedy/ttp_ga_greedy_easy/3/ttp_ga_greedy_history_easy_{seed}.csv"
        )
        data_logger.log_data(tsp.history)

    final_Gs = np.asarray(final_Gs, dtype=float)
    print("Data after 10 runs:")
    print("best", float(final_Gs.max()))
    print("worst", float(final_Gs.min()))
    print("avg", float(final_Gs.mean()))
    print("std", float(final_Gs.std(ddof=1)))


