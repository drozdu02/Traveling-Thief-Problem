import csv
import matplotlib.pyplot as plt

def load_summary(path):
    with open(path, newline="") as f:
        return [float(row["G"]) for row in csv.DictReader(f)]

if __name__ == "__main__":
    data = [
        load_summary("logs/ga/easy/1/history_summary.csv"),
        load_summary("logs/sa/easy/1/history_summary.csv"),
        load_summary("logs/greedy/easy/1/history_summary.csv"),
        load_summary("logs/random/easy/1/summary.csv"),
    ]
    labels = ["GA", "SA", "Greedy", "Random"]

    plt.figure(figsize=(7, 4))
    plt.boxplot(data, tick_labels=labels, showmeans=True)
    plt.ylabel("G")
    plt.title("Porównanie metod dla instancji a280_n279_uncorr_01.ttp")
    plt.tight_layout()
    plt.savefig("charts/boxplot_easy1.png", dpi=150)
    plt.close()