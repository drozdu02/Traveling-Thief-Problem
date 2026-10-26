import numpy as np

from ttp_problem.common.data_loader import DataLoader
class TravelingSalesmanProblemSimAnn:
    def __init__(
        self,
        cities : np.ndarray,
    ):
        self.cities = cities





if __name__ == '__main__':
    data_loader = DataLoader(
        data_path="../data/a280-ttp/a280_n279_bounded-strongly-corr_01.ttp"
    )
    cities = data_loader.get_cords()
    print(cities)