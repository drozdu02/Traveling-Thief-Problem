import csv
import os

class DataLogger:
    def __init__(
        self,
        data_path: str,
    ):
        self.data_path = data_path

    def log_data(self, data):
        os.makedirs(os.path.dirname(self.data_path) or ".", exist_ok=True)
        with open(self.data_path, "w", newline="") as f:
            write = csv.writer(f)
            write.writerow(
                ["generation", "best", "avg", "worst"],
            )
            for g, (best, avg, worst) in enumerate(zip(
                data["best"],
                data["avg"],
                data["worst"],
            )):
                write.writerow([g, best, avg, worst])


