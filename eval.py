import json
import datetime
import os
from agent import fetch_issues, triage_one_node

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
        # build a minimal issue dict to pass into triage_one_node
        issue = {
            "number": expected["number"],
            "title": expected["title"],
            "body": expected.get("body", "")
        }

        # run the SAME triage logic your agent uses
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


def print_report(results):
    correct = sum(1 for r in results if r["match"])
    total = len(results)
    accuracy = (correct / total) * 100 if total > 0 else 0

    history = save_result_to_history(accuracy)
    check_for_regression(history)

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