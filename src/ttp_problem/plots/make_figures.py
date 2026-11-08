import csv
import sys
from pathlib import Path
import matplotlib.pyplot as plt

def load_history(path):
    gens, bests, avgs = [], [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            gens.append(int(row["generation"]))
            bests.append(float(row["best"]))
            avgs.append(float(row["avg"]))
    return gens, bests, avgs


if __name__ == "__main__":

    gens, bests, avgs = load_history("logs/ga/easy/1/history_seed_0.csv")

    plt.figure(figsize=(8, 4))
    plt.plot(gens, bests, label="best G")
    plt.plot(gens, avgs, label="avg G", alpha=0.8)
    plt.xlabel("Generacja")
    plt.ylabel("G")
    plt.title("Przebieg przeszukiwania dla a280_n279_uncorr_01.ttp")
    plt.legend()
    plt.tight_layout()
    plt.savefig("charts/ga_easy1_convergence.png", dpi=150)
    plt.close()