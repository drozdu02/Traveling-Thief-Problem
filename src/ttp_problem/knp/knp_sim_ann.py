import numpy as np

from ttp_problem.common.data_loader import DataLoader

class KnapsackProblemSimAnn:
    def __init__(
        self,
        items : np.ndarray,
        seed: int,
        T_max: float,
        T_min: float,
        alpha: float,
        iters_per_temperature: int,
        capacity: float,
    ):
        self.items = np.asarray(items)
        self.values = self.items[..., 0]
        self.weights = self.items[..., 1]
        self.seed = seed
        self.T_max = T_max
        self.T_min = T_min
        self.alpha = alpha
        self.iters_per_temperature = iters_per_temperature
        self.capacity = capacity
        self.t = T_max
        self.current = None
        self.best = None
        self.best_value = None
        self.current_value = None
        self.current_weight = None
        self.new_value = None
        self.new_weight = None
        self.rng = np.random.default_rng(seed=seed)
        self.k = None
        self.iterations = 0



    def generate_starting_point(self):
        self.current = np.zeros(self.items.shape[0], dtype=int)
        self.current_value = 0
        self.current_weight = 0
        order = self.rng.permutation(self.items.shape[0])
        for i in order:
            w = self.weights[i]
            if self.current_weight + w <= self.capacity:
                self.current[i] = 1
                self.current_weight += w
                self.current_value += self.values[i]

        self.best = self.current.copy()
        self.best_value = self.current_value

    def random_items(self):
        self.k = self.rng.integers(self.items.shape[0])
        return self.k

    def compute_delta(self):
        k = self.random_items()
        if self.current[k] == 0:
            self.new_weight = self.current_weight + self.weights[k]
            self.new_value = self.current_value + self.values[k]
        else:
            self.new_weight = self.current_weight - self.weights[k]
            self.new_value = self.current_value - self.values[k]

        return self.current_value - self.new_value

    @staticmethod
    def compute_probability(delta, t):
        return np.exp(-delta / t)


    def if_accept_new_item(self):
        delta = self.compute_delta()
        if self.new_weight > self.capacity:
            return
        if delta <= 0 or self.rng.random() < self.compute_probability(delta, self.t):
            self.current[self.k] ^= 1
            self.current_weight = self.new_weight
            self.current_value = self.new_value
            if self.current_value > self.best_value:
                self.best = self.current.copy()
                self.best_value = self.current_value



    def run(self):
        self.generate_starting_point()
        self.t = self.T_max
        while self.t > self.T_min:
            for _ in range(self.iters_per_temperature):
                self.if_accept_new_item()
                self.iterations += 1
            self.t *= self.alpha

        return self.best, self.best_value






if __name__ == '__main__':
    data_loader = DataLoader(
        data_path="../data/a280-ttp/a280_n279_bounded-strongly-corr_01.ttp"
    )
    items = data_loader.get_items()
    parameters = data_loader.get_knapsack_parameters()
    capacity = parameters['CAPACITY OF KNAPSACK']
    knp_sim_ann = KnapsackProblemSimAnn(
        items=items,
        seed=42,
        T_max=100,
        T_min=0.01,
        alpha=0.99,
        iters_per_temperature=1000,
        capacity=capacity,
    )
    best, val = knp_sim_ann.run()
    print(best, val)

