class DataLoader:
    def __init__(self, data_path : str):
        self.data_path = data_path
        self.data = self.load_data()


    def load_data(self):
        with open(self.data_path) as f:
            data = f.read().splitlines()
        return data

    def get_cords(self):
        data = self.data
        coordinates = []
        start = None

        for i, line in enumerate(data):
            if line.startswith("NODE_COORD_SECTION"):
                start = i + 1
                break

        if start is None:
            raise ValueError("Could not find start node")

        for line in data[start:]:
            parts = line.split()
            if not parts:
                break

            try:
                int(parts[0])
            except ValueError:
                break

            idx, x,y = float(parts[0]), float(parts[1]), float(parts[2])
            coordinates.append([idx, x,y])

        return coordinates

    def get_items(self):
        data = self.data
        items = []
        start = None
        for i, line in enumerate(data):
            if line.startswith("ITEMS SECTION"):
                start = i + 1
                break
        if start is None:
            raise ValueError("Could not find start item")

        for line in data[start:]:
            parts = line.split()
            if not parts:
                break

            try:
                int(parts[0])
            except ValueError:
                break

            profit, weight, assigned_node = float(parts[1]), float(parts[2]), float(parts[3])
            items.append([profit,weight,assigned_node])

        return items

    def get_knapsack_parameters(self):
        data = self.data
        knapsack_parameters = {}
        for line in data:
            if line.startswith("NODE_COORD_SECTION"):
                break
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip()
            try:
                value = float(value)
            except ValueError:
                continue
            knapsack_parameters[key.strip()] = value

        return knapsack_parameters




if __name__ == "__main__":
    loader = DataLoader("./data/a280-ttp/a280_n279_bounded-strongly-corr_01.ttp")
    print(loader.get_knapsack_parameters())
