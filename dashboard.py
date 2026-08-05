import json
import matplotlib.pyplot as plt
import numpy as np


def load_history(filepath="eval_history.json"):
    with open(filepath, "r") as f:
        return json.load(f)


def plot_accuracy_trend(history):
    runs = [f"Run {i + 1}" for i in range(len(history))]
    scores = [h["accuracy"] for h in history]

    plt.figure(figsize=(8, 4))
    plt.plot(runs, scores, marker="o", linewidth=2, color="#2a78d6")
    plt.ylim(0, 105)
    plt.ylabel("Accuracy (%)")
    plt.title("Triage Agent — Accuracy Over Time")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("accuracy_trend.png")
    print("Saved accuracy_trend.png")
    plt.close()


def plot_category_breakdown(category_scores):
    """category_scores: dict like
    {"Documentation": {"precision": 100, "recall": 100, "f1": 100},
     "Bug": {"precision": 75, "recall": 100, "f1": 85.7}}
    These come from the manual per-category calculation in Day 2 —
    computed by hand against test_set.json, not auto-derived here.
    """
    categories = list(category_scores.keys())
    metrics = ["precision", "recall", "f1"]
    colors = {"precision": "#2a78d6", "recall": "#eb6834", "f1": "#1baf7a"}

    x = np.arange(len(categories))
    width = 0.25

    plt.figure(figsize=(8, 4.5))
    for i, metric in enumerate(metrics):
        values = [category_scores[c][metric] for c in categories]
        plt.bar(x + (i - 1) * width, values, width, label=metric.capitalize(), color=colors[metric])

    plt.ylim(0, 110)
    plt.ylabel("Score (%)")
    plt.title("Per-Category Precision, Recall, F1")
    plt.xticks(x, categories)
    plt.legend(loc="lower right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("category_breakdown.png")
    print("Saved category_breakdown.png")
    plt.close()


if __name__ == "__main__":
    history = load_history()
    plot_accuracy_trend(history)

    # from the Day 2 manual precision/recall/F1 calculation against test_set.json
    category_scores = {
        "Documentation": {"precision": 100, "recall": 100, "f1": 100},
        "Bug": {"precision": 75, "recall": 100, "f1": 85.7},
    }
    plot_category_breakdown(category_scores)