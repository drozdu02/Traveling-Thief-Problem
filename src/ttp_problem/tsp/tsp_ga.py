import numpy as np

from scipy.spatial.distance import pdist, squareform

from ttp_problem.common.data_loader import DataLoader
from ttp_problem.tsp.tsp_greedy import TravelingSalesmanProblemGreedy
from ttp_problem.packers.pack_item_ga import PackItemGa

class TravelingSalesmanProblem:
    def __init__(
            self,
            cities : np.ndarray,
            items : np.ndarray,
            capacity : int,
            v_min: float,
            v_max: float,
            population_size : int,
            num_of_generations : int,
            k : int,
            mutation_rate : float,
            crossover_rate : float,
            seed : int,
    ):
        self.cities = np.asarray(cities)
        self.items = np.asarray(items)
        self.capacity = capacity
        self.v_min = v_min
        self.v_max = v_max
        self.population_size = population_size
        self.num_of_generations = num_of_generations
        self.k = k
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.rng = np.random.default_rng(seed)
        self.distances = self.compute_distances()
        self.population = self.initialize_population()
        self.evaluations = 0
        self.history = {
            "best": [],
            "avg" : [],
            "worst" : []
        }


    def compute_distances(self):
        distances = squareform(pdist(self.cities[:, 1:], metric='euclidean'))
        return np.ceil(distances)

    def initialize_population(self):
        n = len(self.cities)
        starting_cities = np.arange(n)
        self.rng.shuffle(starting_cities)
        population = []
        for i in range(self.population_size):
            start = int(starting_cities[i%n])
            tsp_greedy = TravelingSalesmanProblemGreedy(
                cities=self.cities,
                starting_point=start,
                distances=self.distances,
            )
            _, route = tsp_greedy.travel()
            individual = list(route)

            if i >= n:
                a, b = self.rng.choice(n, size=2, replace=False)
                individual[a], individual[b] = individual[b], individual[a]
            population.append(individual)

        return population

    def compute_fitness(self):
        fitness = []
        for p in self.population:
            _, _, G = PackItemGa(
                individual=np.array(p),
                city_distances=self.distances,
                items=self.items,
                capacity=self.capacity,
                v_max=self.v_max,
                v_min=self.v_min,
            ).pack()
            fitness.append(G)
        self.evaluations += len(self.population)
        return fitness


    def __log_data(self, fitness):
        fitness = np.asarray(fitness, dtype=float)
        self.history["best"].append(float(fitness.max()))
        self.history["avg"].append(float(fitness.mean()))
        self.history["worst"].append(float(fitness.min()))


    def _generate_random_indexes(self):
        indexes = self.rng.choice(np.arange(0, self.population_size), size=self.k, replace=False)
        return indexes

    def _create_tournament(self, fitness):
        indexes = self._generate_random_indexes()
        return indexes[np.argmax(fitness[indexes])]

    def select_by_tournament(self, fitness):
        first_winner = self._create_tournament(fitness)
        second_winner = self._create_tournament(fitness)
        while second_winner == first_winner:
            second_winner = self._create_tournament(fitness)
        return first_winner, second_winner


    def get_winners(self, fitness):
        first_winner, second_winner = self.select_by_tournament(fitness=fitness)
        return self.population[first_winner], self.population[second_winner]

    def ox_crossover(self, fitness):
        first_winner, second_winner = self.get_winners(fitness=fitness)
        if self.rng.random() >= self.crossover_rate:
            return first_winner.copy()
        a, b = np.sort(self.rng.choice(np.arange(len(first_winner)), size=2, replace=False))
        seen = set()
        child = [-1] * len(first_winner)
        child[a:b] = first_winner[a:b]
        seen.update(child[a:b])
        position = b % len(first_winner)
        for i in range(len(first_winner)):
            city = second_winner[(b + i) % len(first_winner)]
            if city not in seen:
                child[position] = city
                seen.add(city)
                position = (position + 1) % len(first_winner)
        return child

    def mutate(self, child):
        if self.rng.random() < self.mutation_rate:
            a, b = self.rng.choice(len(child), size=2, replace=False)
            child[a], child[b] = child[b], child[a]
        return child

    def next_generation(self):
        fitness = np.asarray(self.compute_fitness())
        best_individual = np.argmax(fitness)

        self.__log_data(fitness)

        new_population = [self.population[best_individual].copy()]

        while len(new_population) < self.population_size:
            child = self.mutate(self.ox_crossover(fitness=fitness))
            new_population.append(child)

        self.population = new_population

    def run(self):
        for _ in range(self.num_of_generations):
            self.next_generation()

        fitness = np.asarray(self.compute_fitness())
        self.__log_data(fitness)
        best = int(np.argmax(fitness))

        return self.population[best], float(fitness[best])


if __name__ == '__main__':
    data_loader = DataLoader("../data/a280-ttp/a280_n279_bounded-strongly-corr_01.ttp")
    cities = data_loader.get_cords()
    tsp = TravelingSalesmanProblem(
        cities=cities,
        population_size=100,
        num_of_generations=20,
        k=5,
        mutation_rate=0.3,
        crossover_rate=0.9,
        seed=42,
    )
    print(tsp.run())




