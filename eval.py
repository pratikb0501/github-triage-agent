import json
import datetime
import os
import mlflow
from agent import fetch_issues, triage_one_node
import sys

sys.stdout.reconfigure(encoding='utf-8')

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("github-triage-categorization")

HISTORY_FILE = "eval_history.json"


def save_result_to_history(accuracy):
    history = []
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            history = json.load(f)

    history.append({
        "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "accuracy": accuracy
    })

    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)

    return history


def load_test_set(filepath="test_set.json"):
    with open(filepath, "r") as f:
        return json.load(f)


def run_evaluation(test_set):
    results = []

    for expected in test_set:
        issue = {
            "number": expected["number"],
            "title": expected["title"],
            "body": expected.get("body", "")
        }

        output = triage_one_node({"issue": issue, "all_issues": []})
        actual_category = output["triaged_issues"][0]["category"]

        is_match = actual_category.strip().lower() == expected["correct_category"].strip().lower()

        results.append({
            "number": expected["number"],
            "title": expected["title"],
            "expected": expected["correct_category"],
            "actual": actual_category,
            "match": is_match
        })

    return results


def calculate_category_metrics(results, category):
    """Precision/recall/F1 for one category, same manual method from Day 2."""
    tp = sum(1 for r in results if r["actual"] == category and r["expected"] == category)
    fp = sum(1 for r in results if r["actual"] == category and r["expected"] != category)
    fn = sum(1 for r in results if r["actual"] != category and r["expected"] == category)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

    return {"precision": precision * 100, "recall": recall * 100, "f1": f1 * 100}


def print_report(results):
    correct = sum(1 for r in results if r["match"])
    total = len(results)
    accuracy = (correct / total) * 100 if total > 0 else 0

    history = save_result_to_history(accuracy)
    check_for_regression(history)

    # log this run to MLflow — parameters (config) + metrics (results) + artifacts (files)
    with mlflow.start_run():
        mlflow.log_param("model", "qwen2.5:7b")
        mlflow.log_param("temperature", 0)
        mlflow.log_param("test_set_size", total)

        mlflow.log_metric("accuracy", accuracy)

        # per-category precision/recall/F1, logged automatically instead of hand-typed
        categories = set(r["expected"] for r in results)
        for category in categories:
            metrics = calculate_category_metrics(results, category)
            safe_name = category.lower().replace(" ", "_").replace("/", "_")
            mlflow.log_metric(f"precision_{safe_name}", metrics["precision"])
            mlflow.log_metric(f"recall_{safe_name}", metrics["recall"])
            mlflow.log_metric(f"f1_{safe_name}", metrics["f1"])

        mlflow.log_artifact("test_set.json")

        print(f"\n{'=' * 60}")
        print(f"EVALUATION REPORT")
        print(f"{'=' * 60}")
        print(f"Accuracy: {correct}/{total} ({accuracy:.1f}%)\n")

        for r in results:
            status = "✅" if r["match"] else "❌"
            print(f"{status} #{r['number']}: {r['title'][:50]}")
            print(f"   Expected: {r['expected']}  |  Actual: {r['actual']}")

        print(f"\n{'=' * 60}")
        print(f"FINAL SCORE: {accuracy:.1f}%")
        print(f"{'=' * 60}")

        print("\nPer-category breakdown (also logged to MLflow):")
        for category in categories:
            m = calculate_category_metrics(results, category)
            print(f"  {category}: precision={m['precision']:.1f}%  recall={m['recall']:.1f}%  f1={m['f1']:.1f}%")


def check_for_regression(history):
    if len(history) < 2:
        print("\n(First run — no previous score to compare against)")
        return

    current = history[-1]["accuracy"]
    previous = history[-2]["accuracy"]

    print(f"\nPrevious score: {previous}%  |  Current score: {current}%")

    if current < previous:
        print(f"⚠️  REGRESSION DETECTED: score dropped by {previous - current:.1f} points")
    elif current > previous:
        print(f"✅ Improvement: score increased by {current - previous:.1f} points")
    else:
        print("✅ No change — score is stable")


if __name__ == "__main__":
    test_set = load_test_set()
    results = run_evaluation(test_set)
    print_report(results)